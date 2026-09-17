"""Sequence-only ESM3 residue embeddings and ss8 logits for the 3Di head.

For every domain in the split manifest (``domain_family_split.py``) this reads the
sequence with ``dataset.utils.three_di_labels.domain_sequence`` - the same code the
3Di Dataset uses, so embeddings and labels index the same residues - and runs
ESM3-open on the sequence alone, with every other track masked.  That matches a
FoldMatch query, where only the sequence is known.

What is stored, per residue with BOS/EOS removed:

* ``embeddings/`` - the last transformer block's output *before* the final
  LayerNorm (what ``per_residue_embedding`` has always been), 1536-d.
* ``ss8_logits/`` - ESM3's secondary-structure head (11 logits), free from the
  same forward pass.

Both are packed stores (``dataset.utils.packed_embeddings``) keyed by the SHA-1
of the sequence, so identical sequences are embedded once.  ``domains.tsv`` maps
each domain to its key; ``unusable.tsv`` lists files that yield no clean chain;
``provenance.json`` records model weights, library versions and repo commit.

Batching.  ``ESM3.logits``/``forward_and_sample`` run one sequence at a time and
pass ``sequence_id=None``, which builds no attention mask, so they cannot be
padded.  Here each sequence's input tracks are built exactly as
``forward_and_sample`` builds them, padded into a batch, and ``sequence_id``
keeps real residues from attending to padding.  ``--verify N`` re-embeds N
sequences through ``forward_and_sample`` and reports the difference.

Multi-GPU runs start one process per device; batches are length-sorted and
balanced across devices by token count.

    python esm3_sequence_embeddings.py --manifest domain_split.tsv \\
        --out-dir esm3-sequence --devices 4 --verify 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import random
import shutil
import subprocess
import time
import warnings
from datetime import datetime, timezone
from multiprocessing import Pool

import attr
import numpy as np
import torch
import torch.multiprocessing as mp

from dataset.utils.packed_embeddings import DATA_NAME, FORMAT_VERSION, INDEX_NAME, META_NAME
from dataset.utils.three_di_labels import UnusableDomain, domain_sequence

logger = logging.getLogger("esm3_sequence_embeddings")

EMBEDDING_DIM = 1536
SS8_DIM = 11
STORES = {"embeddings": EMBEDDING_DIM, "ss8_logits": SS8_DIM}


# ---------------------------------------------------------------------------------------
# sequences
# ---------------------------------------------------------------------------------------

def sequence_key(sequence: str) -> str:
    return hashlib.sha1(sequence.encode()).hexdigest()


def read_manifest(path: str, splits: set[str] | None) -> list[dict]:
    with open(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        rows = [dict(zip(header, line.rstrip("\n").split("\t"))) for line in handle if line.strip()]
    return [r for r in rows if splits is None or r["split"] in splits]


def _read_sequence(row: dict) -> tuple[dict, str | None, str | None]:
    warnings.filterwarnings("ignore")
    try:
        return row, domain_sequence(row["path"]), None
    except UnusableDomain as error:
        return row, None, str(error).split(": ", 1)[-1]
    except Exception as error:
        return row, None, f"{type(error).__name__}: {error}"


def make_batches(items: list[tuple[str, str]], max_tokens: int, max_batch: int,
                 max_pairs: int) -> list[list[tuple[str, str]]]:
    """Group (key, sequence) pairs, longest first, under both batch limits.

    ``max_tokens`` bounds padded tokens (B x L), which is what the feed-forward
    layers cost.  ``max_pairs`` bounds B x L^2, which is what attention costs: a
    fixed token budget still runs out of memory on long sequences.
    """
    items = sorted(items, key=lambda kv: -len(kv[1]))
    batches, current = [], []
    for key, sequence in items:
        width = (len(current[0][1]) if current else len(sequence)) + 2
        if current and ((len(current) + 1) * width > max_tokens
                        or (len(current) + 1) * width * width > max_pairs
                        or len(current) == max_batch):
            batches.append(current)
            current = []
        current.append((key, sequence))
    if current:
        batches.append(current)
    return batches


def assign_batches(batches: list, n_ranks: int) -> list[list]:
    """Greedy balance by token count: next-largest batch to the least-loaded rank."""
    load = [0] * n_ranks
    shards = [[] for _ in range(n_ranks)]
    for batch in sorted(batches, key=lambda b: -len(b) * (len(b[0][1]) + 2)):
        rank = load.index(min(load))
        shards[rank].append(batch)
        load[rank] += len(batch) * (len(batch[0][1]) + 2)
    return shards


# ---------------------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------------------

class SequenceOnlyESM3:
    """Batched, sequence-only ESM3 forward pass returning embeddings and ss8 logits.

    ``precision`` follows ESM3's own behaviour by default ("bf16"): ``ESM3.logits``
    enables bfloat16 autocast whenever the device is CUDA, so every embedding this
    repo has produced on a GPU is bf16-computed. "fp32" turns that off, which matches
    FoldMatch's CPU deployment exactly at roughly an order of magnitude more precision
    and a slower forward pass.
    """

    def __init__(self, device: torch.device, precision: str = "bf16"):
        from esm.models.esm3 import ESM3
        from esm.utils.constants.models import ESM3_OPEN_SMALL

        self.device = device
        self.precision = precision
        self.model = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=device).eval()
        if precision == "fp32":
            # from_pretrained casts weights to bfloat16 off CPU; undo that for a true fp32 pass.
            self.model = self.model.to(torch.float32)
        self.tokenizers = self.model.tokenizers
        # Geometric attention (block 0 only, 256 vector heads) contributes exactly zero
        # without coordinates: the model is built with mask_and_zero_frameless=True, so an
        # all-false frame mask zeroes its output, and its projection has no bias. Running it
        # anyway costs a (B, 256, L, L, 3) tensor - 26 GB for a batch of 26 x 600 residues.
        # ``reference()`` switches it back on, so --verify tests that skipping it is exact.
        self._geom_blocks = [b for b in self.model.transformer.blocks if b.use_geom_attn]
        self._set_geom_attn(False)

    def _set_geom_attn(self, enabled: bool) -> None:
        for block in self._geom_blocks:
            block.use_geom_attn = enabled

    def _tracks(self, sequence: str):
        """Input tracks for one sequence, filled exactly as ``forward_and_sample`` fills them."""
        from esm.sdk.api import ESMProtein, ESMProteinTensor

        protein = self.model.encode(ESMProtein(sequence=sequence))
        empty = ESMProteinTensor.empty(len(protein) - 2, tokenizers=self.tokenizers, device="cpu")
        for track in attr.fields(ESMProteinTensor):
            if getattr(protein, track.name, None) is None:
                setattr(protein, track.name, getattr(empty, track.name, None))
        return protein

    @torch.inference_mode()
    def __call__(self, sequences: list[str]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        from esm.utils.misc import stack_variable_length_tensors

        proteins = [self._tracks(s) for s in sequences]
        t = self.tokenizers

        def stack(track: str, pad: int) -> torch.Tensor:
            return stack_variable_length_tensors(
                [getattr(p, track).cpu() for p in proteins], constant_value=pad
            ).to(self.device)

        sequence_tokens = stack("sequence", t.sequence.pad_token_id)
        batch, width = sequence_tokens.shape
        is_pad = sequence_tokens == t.sequence.pad_token_id

        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda" and self.precision == "bf16"
            else torch.autocast(device_type=self.device.type, enabled=False)
        )
        with autocast:
            output = self.model.forward(
                sequence_tokens=sequence_tokens,
                structure_tokens=stack("structure", t.structure.pad_token_id),
                ss8_tokens=stack("secondary_structure", t.secondary_structure.pad_token_id),
                sasa_tokens=stack("sasa", t.sasa.pad_token_id),
                function_tokens=stack("function", t.function.pad_token_id),
                residue_annotation_tokens=stack("residue_annotations", t.residue_annotations.pad_token_id),
                average_plddt=torch.tensor(1.0, device=self.device),
                per_res_plddt=torch.zeros((batch, width), device=self.device),
                structure_coords=torch.full((batch, width, 3, 3), float("nan"), device=self.device),
                chain_id=torch.zeros((batch, width), dtype=torch.long, device=self.device),
                # Real tokens (incl. BOS/EOS) share id 0, padding gets 1: no attention across them.
                sequence_id=is_pad.long(),
            )
        embeddings = output.embeddings.float().cpu()
        ss8 = output.secondary_structure_logits.float().cpu()
        return [
            (embeddings[i, 1:len(s) + 1].clone(), ss8[i, 1:len(s) + 1].clone())
            for i, s in enumerate(sequences)
        ]

    @torch.inference_mode()
    def reference(self, sequence: str) -> torch.Tensor:
        """The unbatched path used for every earlier ESM3 embedding in this repo."""
        from esm.sdk.api import ESMProtein, SamplingConfig

        protein = self.model.encode(ESMProtein(sequence=sequence))
        self._set_geom_attn(True)
        try:
            output = self.model.forward_and_sample(protein, SamplingConfig(return_per_residue_embeddings=True))
        finally:
            self._set_geom_attn(False)
        return output.per_residue_embedding.float().cpu()[1:-1]


def _verify(embedder: SequenceOnlyESM3, sequences: list[str]) -> list[dict]:
    rows = []
    batched = embedder(sequences)
    for sequence, (embedding, _) in zip(sequences, batched):
        reference = embedder.reference(sequence)
        cosine = torch.nn.functional.cosine_similarity(embedding, reference, dim=-1)
        rows.append(dict(
            length=len(sequence),
            max_abs_diff=float((embedding - reference).abs().max()),
            reference_max_abs=float(reference.abs().max()),
            min_cosine=float(cosine.min()),
        ))
    return rows


# ---------------------------------------------------------------------------------------
# workers and store
# ---------------------------------------------------------------------------------------

def _worker(rank: int, device_name: str, batches: list, part_dir: str, dtype: str,
            verify: list[str], log_every: int, precision: str = "bf16") -> None:
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s [rank {rank}] %(message)s")
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "4"))))
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    embedder = SequenceOnlyESM3(device, precision=precision)
    np_dtype = np.dtype(dtype)

    if verify:
        report = _verify(embedder, verify)
        with open(os.path.join(part_dir, "verify.json"), "w") as handle:
            json.dump(report, handle, indent=2)
        for row in report:
            logger.info(f"verify L={row['length']}: max |diff| {row['max_abs_diff']:.3g} "
                        f"(max |value| {row['reference_max_abs']:.3g}), min cosine {row['min_cosine']:.6f}")

    n_done = n_tokens = 0
    max_abs = 0.0
    start = time.time()
    files = {name: open(os.path.join(part_dir, f"part-{rank}.{name}.dat"), "wb") for name in STORES}
    with open(os.path.join(part_dir, f"part-{rank}.index.tsv"), "w") as index:
        for step, batch in enumerate(batches):
            outputs = embedder([s for _, s in batch])
            for (key, sequence), (embedding, ss8) in zip(batch, outputs):
                max_abs = max(max_abs, float(embedding.abs().max()))
                for name, tensor in (("embeddings", embedding), ("ss8_logits", ss8)):
                    array = tensor.numpy().astype(np_dtype)
                    if not np.isfinite(array).all():
                        raise FloatingPointError(
                            f"{key}: non-finite values in {name} after casting to {dtype}; rerun with --dtype float32"
                        )
                    array.tofile(files[name])
                index.write(f"{key}\t{len(sequence)}\n")
                n_done += 1
                n_tokens += len(sequence)
            if step % log_every == 0 or step == len(batches) - 1:
                elapsed = time.time() - start
                peak = (f", peak GPU {torch.cuda.max_memory_allocated(device) / 2**30:.1f} GiB"
                        if device.type == "cuda" else "")
                logger.info(f"batch {step + 1}/{len(batches)} (size {len(batch)} x {len(batch[0][1])}): "
                            f"{n_done:,} sequences, {n_tokens / max(elapsed, 1e-9):,.0f} residues/s, "
                            f"max |embedding| {max_abs:.1f}{peak}")
    for handle in files.values():
        handle.close()
    with open(os.path.join(part_dir, f"part-{rank}.done"), "w") as handle:
        json.dump(dict(sequences=n_done, residues=n_tokens, seconds=time.time() - start,
                       max_abs_embedding=max_abs), handle)


def merge_parts(part_dir: str, out_dir: str, n_ranks: int, dtype: str) -> dict:
    """Concatenate per-rank parts into one packed store per output."""
    keys, lengths = [], []
    for rank in range(n_ranks):
        with open(os.path.join(part_dir, f"part-{rank}.index.tsv")) as handle:
            for line in handle:
                key, length = line.split("\t")
                keys.append(key)
                lengths.append(int(length))
    lengths = np.array(lengths, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(np.int64)

    for name, dim in STORES.items():
        store = os.path.join(out_dir, name)
        os.makedirs(store, exist_ok=True)
        with open(os.path.join(store, DATA_NAME + ".tmp"), "wb") as out:
            for rank in range(n_ranks):
                with open(os.path.join(part_dir, f"part-{rank}.{name}.dat"), "rb") as part:
                    shutil.copyfileobj(part, out, length=64 << 20)
        expected = int(lengths.sum()) * dim * np.dtype(dtype).itemsize
        size = os.path.getsize(os.path.join(store, DATA_NAME + ".tmp"))
        if size != expected:
            raise RuntimeError(f"{name}: {size} bytes written, expected {expected}")
        os.replace(os.path.join(store, DATA_NAME + ".tmp"), os.path.join(store, DATA_NAME))
        with open(os.path.join(store, INDEX_NAME), "wb") as handle:
            np.savez(handle, domains=np.array(keys), offsets=offsets, lengths=lengths)
        with open(os.path.join(store, META_NAME), "w") as handle:
            json.dump(dict(format_version=FORMAT_VERSION, dim=dim, dtype=np.dtype(dtype).name,
                           total_rows=int(lengths.sum()), count=len(keys)), handle, indent=2)
    return dict(sequences=len(keys), residues=int(lengths.sum()))


def provenance(args, devices: list[str]) -> dict:
    import esm
    from esm.utils.constants.models import ESM3_OPEN_SMALL
    from esm.utils.constants.esm3 import data_root

    def git(*command):
        try:
            return subprocess.check_output(["git", *command], cwd=os.path.dirname(os.path.abspath(__file__)),
                                           text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    weights = str(data_root("esm3"))
    return dict(
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model=ESM3_OPEN_SMALL,
        weights_path=weights,
        weights_snapshot=os.path.basename(weights.rstrip("/")),
        # HuggingFace cache blobs are named by their SHA-256, so this identifies the exact weights.
        weights_sha256=os.path.basename(os.path.realpath(os.path.join(weights, "data/weights/esm3_sm_open_v1.pth"))),
        esm_version=getattr(esm, "__version__", None),
        torch_version=torch.__version__,
        python=platform.python_version(),
        repo_commit=git("rev-parse", "HEAD"),
        repo_dirty=bool(git("status", "--porcelain")),
        embedding="last transformer block output before the final LayerNorm, BOS/EOS removed",
        inputs="sequence only; structure, ss8, sasa, function and residue tracks masked as in forward_and_sample",
        geometric_attention="skipped (exactly zero without coordinates); --verify compares against running it",
        autocast=("bfloat16" if any(d.startswith("cuda") for d in devices) and args.precision == "bf16" else "none"),
        stored_dtype=args.dtype,
        manifest=os.path.abspath(args.manifest),
        splits=args.splits,
        limit=args.limit,
        seed=args.seed,
        max_tokens=args.max_tokens,
    )


# ---------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, help="domain_split.tsv from domain_family_split.py")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--splits", nargs="+", default=None, help="only these splits (default: all)")
    parser.add_argument("--limit", type=int, default=None, help="random sample of domains, for tests")
    parser.add_argument("--devices", default="auto",
                        help="number of CUDA devices, or comma-separated device names such as cpu / mps "
                             "(default: all GPUs)")
    parser.add_argument("--max-tokens", type=int, default=16000, help="padded tokens per batch")
    parser.add_argument("--max-batch", type=int, default=128)
    parser.add_argument("--max-pairs", type=int, default=40_000_000,
                        help="padded tokens x width per batch, which is what attention memory scales with")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16",
                        help="stored dtype (float16 costs 4.9e-4 relative error, well under bf16 compute error)")
    parser.add_argument("--precision", choices=["bf16", "fp32"], default="bf16",
                        help="GPU compute precision; bf16 is what ESM3 itself uses on CUDA (default)")
    parser.add_argument("--verify", type=int, default=0,
                        help="compare N sequences with the unbatched forward_and_sample path")
    parser.add_argument("--cpu-workers", type=int, default=16, help="processes reading PDB files")
    parser.add_argument("--keep-parts", action="store_true")
    parser.add_argument("--log-every", type=int, default=50, help="batches between progress lines")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.devices == "auto":
        devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]
    elif args.devices.isdigit():
        devices = [f"cuda:{i}" for i in range(int(args.devices))]
    else:
        devices = args.devices.split(",")   # e.g. "cpu", "mps", or "cpu,cpu" to test multi-process locally

    rows = read_manifest(args.manifest, set(args.splits) if args.splits else None)
    if args.limit is not None and args.limit < len(rows):
        rows = random.Random(args.seed).sample(rows, args.limit)
    logger.info(f"{len(rows):,} domains from {args.manifest}; devices {devices}")

    os.makedirs(args.out_dir, exist_ok=True)
    part_dir = os.path.join(args.out_dir, "parts")
    shutil.rmtree(part_dir, ignore_errors=True)
    os.makedirs(part_dir)

    # ---- sequences, deduplicated ------------------------------------------------------
    unique: dict[str, str] = {}
    with Pool(args.cpu_workers) as pool, \
            open(os.path.join(args.out_dir, "domains.tsv"), "w") as domains, \
            open(os.path.join(args.out_dir, "unusable.tsv"), "w") as unusable:
        domains.write("domain_id\tsplit\tsequence_key\tlength\tsequence\n")
        unusable.write("domain_id\tsplit\treason\n")
        n_unusable = 0
        for row, sequence, error in pool.imap(_read_sequence, rows, chunksize=32):
            if sequence is None:
                unusable.write(f"{row['domain_id']}\t{row['split']}\t{error}\n")
                n_unusable += 1
                continue
            key = sequence_key(sequence)
            unique[key] = sequence
            domains.write(f"{row['domain_id']}\t{row['split']}\t{key}\t{len(sequence)}\t{sequence}\n")
    lengths = np.array([len(s) for s in unique.values()])
    logger.info(f"{len(rows) - n_unusable:,} usable domains, {n_unusable:,} unusable; "
                f"{len(unique):,} unique sequences, {lengths.sum():,} residues "
                f"(length median {int(np.median(lengths))}, max {lengths.max()})")

    # ---- embed --------------------------------------------------------------------------
    batches = make_batches(list(unique.items()), args.max_tokens, args.max_batch, args.max_pairs)
    shards = assign_batches(batches, len(devices))
    verify = [s for _, s in random.Random(args.seed).sample(list(unique.items()), min(args.verify, len(unique)))]
    logger.info(f"{len(batches):,} batches over {len(devices)} device(s)")

    start = time.time()
    if len(devices) == 1:
        _worker(0, devices[0], shards[0], part_dir, args.dtype, verify, args.log_every, args.precision)
    else:
        context = mp.get_context("spawn")
        processes = []
        for rank, device in enumerate(devices):
            process = context.Process(target=_worker, args=(
                rank, device, shards[rank], part_dir, args.dtype, verify if rank == 0 else [], args.log_every,
                args.precision))
            process.start()
            processes.append(process)
        for process in processes:
            process.join()
        failed = [rank for rank, p in enumerate(processes) if p.exitcode != 0]
        if failed:
            raise SystemExit(f"ranks {failed} failed; parts left in {part_dir}")

    # ---- store --------------------------------------------------------------------------
    summary = merge_parts(part_dir, args.out_dir, len(devices), args.dtype)
    ranks = []
    for rank in range(len(devices)):
        with open(os.path.join(part_dir, f"part-{rank}.done")) as handle:
            ranks.append(json.load(handle))
    record = provenance(args, devices)
    record.update(summary, domains=len(rows) - n_unusable, unusable=n_unusable,
                  seconds=round(time.time() - start, 1), ranks=ranks)
    if verify:
        with open(os.path.join(part_dir, "verify.json")) as handle:
            record["verify"] = json.load(handle)
    with open(os.path.join(args.out_dir, "provenance.json"), "w") as handle:
        json.dump(record, handle, indent=2)
    if not args.keep_parts:
        shutil.rmtree(part_dir)
    logger.info(f"wrote {summary['sequences']:,} sequences / {summary['residues']:,} residues "
                f"to {args.out_dir} in {record['seconds']:.0f}s")


if __name__ == "__main__":
    main()
