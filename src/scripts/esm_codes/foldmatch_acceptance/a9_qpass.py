#!/usr/bin/env python
"""P8 A9 (long sequences), step 2: the production esm-codes QUERY PASS of one FASTA, kept for Stage 2.
NERSC GPU node, one A100 (CUDA_VISIBLE_DEVICES=<k>, SLURM variables removed, as a 1-GPU 'fm-search query sequences').

The steps are those of FoldMatch's own 'fm-search query sequences --scoring esm-codes'
(cli/search.py _query_sequences_esm_codes at cc55c01), with the same functions, arguments and CLI defaults
(--batch-size 1, --num-workers 0, --min-res-n 0):
  _fasta_preflight(strict=True) -> prepare_scoring_system(esm-codes-v1, residue_output=query_encoding)
  -> EmbeddingComputer.compute_from_fasta(residue_output=QUERY_ENCODING, scoring_system=prepared)
     (FoldMatch's ESM3 pass: sequence input, so geometric attention is off (G13), fp32 weights + bf16 autocast on
     CUDA, the production aggregator and the production query encoder f_q -> q fp16 [L, 64])
  -> _release_pass_models() -> EmbeddingComputer.build_query_encodings() (the per-run fp16 memmap).
Two things differ from the CLI, both outside the computation:
  * the CLI resolves the system from the database it queries (resolve_esm_codes_database); here the system is
    esm-codes-v1 (the only registry entry), and a9_stage2.py checks that the database names that same system;
  * the CLI deletes the shards and the memmap after Stage 2. Here they are kept, together with
      entries.json      {id: [first row, rows, seq_sha1 hex]}, rows, width, the system name, the memmap sha256
      embeddings.npz    the chain embeddings in input order (EmbeddingComputer.in_input_order), for Stage 1
      sequences.json    EmbeddingComputer.input_sequences(): the normalised sequences that ESM3 encoded
    so that a9_stage2.py (CPU node) can run Stage 1 and Stage 2 on exactly these encodings.
Output: <out>/qpass.json (counts, wall time, GPU peak memory, checks, provenance incl. torch / CUDA).
Usage:  python a9_qpass.py --fasta FILE --out DIR
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acc_common as C  # noqa: E402  (first: it sets the BLAS thread variables before numpy loads)
import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--accelerator", default="gpu", choices=("gpu", "cpu", "auto"),
                    help="gpu (the A9 runs); cpu / auto only for a local smoke test of this script")
    a = ap.parse_args()
    from pathlib import Path
    from foldmatch.cli import search as S
    from foldmatch.cli.args_utils import arg_devices, set_log_level
    from foldmatch.search import scoring_system as ss
    from foldmatch.search.embedding_computer import EmbeddingComputer
    from foldmatch.search.residue_shards import QUERY_ENCODING
    from foldmatch.inference.chain_complete_inference import prepare_scoring_system
    from foldmatch.types.api_types import Accelerator, LogLevel, Strategy
    from foldmatch.utils.sequence import seq_sha1
    import torch

    set_log_level(LogLevel.info)
    out = Path(a.out)
    (out / "shards").mkdir(parents=True, exist_ok=False)
    fasta = Path(a.fasta).resolve()
    prov = C.provenance(torch_info=True)
    cuda = a.accelerator == "gpu"
    rec = {"title": "P8 A9 production query pass", "fasta": str(fasta), "fasta_md5": C.md5_file(str(fasta)),
           "cli_equivalent": f"fm-search query sequences --scoring esm-codes --fasta-file {fasta} --accelerator gpu "
                             "--devices 1 (defaults: --batch-size 1 --num-workers 0 --min-res-n 0)"}
    t0 = time.time()
    preflight = S._fasta_preflight(fasta, 0, strict=True)
    system = ss.get(ss.DEFAULT_ESM_CODES)
    prepared = prepare_scoring_system(system, residue_output=QUERY_ENCODING, num_nodes=1)
    ec = EmbeddingComputer(embedding_folder=out / "shards")
    enc_path = S.query_encodings_path(ec)
    t1 = time.time()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    ec.compute_from_fasta(fasta_file=fasta, min_res_n=0, batch_size=1, num_workers=0, num_nodes=1,
                          accelerator=Accelerator(a.accelerator), devices=arg_devices(["1"]), strategy=Strategy.auto,
                          preflight=preflight, residue_output=QUERY_ENCODING, scoring_system=prepared)
    t2 = time.time()
    gpu = dict(device=torch.cuda.get_device_name(0), total_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
               peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
               peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30) if cuda else {"accelerator": a.accelerator}
    S._release_pass_models()
    qi = ec.build_query_encodings(enc_path, system)
    t3 = time.time()

    seqs = ec.input_sequences()
    ids = ec.input_ids()
    batches = list(ec.get_embedding_batches(rows=True))
    by_row = {r: e for rows, emb in batches for r, e in zip(rows, emb)}
    emb_by_id = ec.in_input_order(by_row)
    assert list(emb_by_id) == ids == list(seqs), "input order"
    emb = np.stack([emb_by_id[q] for q in ids]).astype(np.float32)
    np.savez(out / "embeddings.npz", ids=np.array(ids), emb=emb)
    C.write_json(seqs, str(out / "sequences.json"))
    bad_len = [q for q in ids if qi.entries[q].length != len(seqs[q])]
    bad_sha = [q for q in ids if bytes(qi.entries[q].seq_sha1) != seq_sha1(seqs[q])]
    mm = qi.memmap()
    finite = bool(np.all(np.isfinite(mm)))
    entries = {q: [int(e.offset), int(e.length), bytes(e.seq_sha1).hex()] for q, e in qi.entries.items()}
    ent = dict(path=str(qi.path), rows=qi.rows, width=qi.width, system_name=qi.system_name,
               sha256=C.sha256_file(str(qi.path)), entries=entries)
    C.write_json(ent, str(out / "entries.json"))
    L = np.array([len(seqs[q]) for q in ids])
    rec.update(
        run_id=ec.run_id, shards=[str(p) for p in ec.shard_files()], encodings=str(qi.path), encodings_sha256=ent["sha256"],
        n=len(ids), residues=int(L.sum()), max_len=int(L.max()), n_len_gt_966=int((L > 966).sum()),
        rows=qi.rows, width=qi.width, system=qi.system_name, gpu=gpu,
        wall_s=dict(resolve=round(t1 - t0, 2), esm3_pass=round(t2 - t1, 2), encodings=round(t3 - t2, 2),
                    total=round(time.time() - t0, 2)),
        residues_per_s=round(float(L.sum()) / (t2 - t1), 1),
        embeddings=dict(file=str(out / "embeddings.npz"), shape=list(emb.shape), sha256=C.array_sha256(emb)),
        checks=dict(lengths_match=not bad_len, seq_sha1_match=not bad_sha, encodings_finite=finite,
                    rows_equal_residues=qi.rows == int(L.sum()), one_encoding_per_input=len(qi) == len(ids),
                    max_abs_q=float(np.max(np.abs(mm.astype(np.float32)))) if qi.rows else 0.0),
        provenance=prov)
    ok = all(v for k, v in rec["checks"].items() if k != "max_abs_q")
    rec["ok"] = ok
    sha = C.write_json(rec, str(out / "qpass.json"))
    C.log("qpass", {k: rec[k] for k in ("n", "residues", "rows", "wall_s", "residues_per_s", "gpu", "checks", "ok")},
          "sha256", sha)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
