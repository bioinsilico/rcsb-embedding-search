#!/usr/bin/env python
"""P8 A4: end-to-end ESM3 parity of FoldMatch ``esm-codes-v1`` through its PRODUCTION path (plan FINAL v1, section 8
P8, A4 row; gates pre-registered in RS/src/scripts/esm_codes/A4_THRESHOLDS.json, sha256 7eabf36e..., D4).

A4 re-runs the P0.5 measurement (esm3_parity.py) with the codes and query encodings computed by FoldMatch cc55c01
itself, exactly as ``fm-search build sequences --residue-codes`` and ``fm-search query sequences --scoring esm-codes``
compute them, and compares them with the RESEARCH encodings as A4_THRESHOLDS.json's protocol describes:

  ESM3 pass   foldmatch.search.embedding_computer.EmbeddingComputer.compute_from_fasta(residue_output=
              'residue_codes' | 'query_encoding', scoring_system=<prepared esm-codes-v1 bundle>) - the call both CLI
              commands make (chain_complete_inference.predict -> ChainCompleteModule with the P2 encoder head of the
              registry bundle, ParquetBatchWriter residue shards, one protein per batch, MPS cache released per batch)
  read back   iter_residue_codes -> ResidueCodeStore.create + validate (against a SequenceStore of the same FASTA and
              its pre-flight; what the build does after FAISS), and QueryEncodingIndex.build (what the query does)
  Stage 2     foldmatch.search.code_alignment.align_candidates_esm_codes (P3 kernel, P4 statistics, the bundle's
              shipped panel for Q90) with N = 8,557,236 (the annotated residues of the 50,000 benchmark targets, the
              protocol's N), max_evalue None (every candidate traced back), panel 'always'

The only addition to the production pass is a passive observer around ChainCompleteModule's per_residue_embedding
(it returns the very tensor ESM3 produced and records, per protein, its shape, finiteness, max |emb| and the
tokenizer's token / <unk> counts) for the A4 invariants.

Subcommands
  inputs        A4/inputs/{codes,queries,q200u}.fasta + inputs.json from the P0.5 inputs (p0/parity/inputs/sets.json
                sha256 fb1e30af..., the five set FASTAs, md5-checked against the thresholds' protocol): codes = the
                unique sequences of panel, enccheck, chains, tsub (target side); queries = q200, chains, enccheck
  embed         one production ESM3 pass over chunk k of n (LPT by length) of codes.fasta / queries.fasta on one
                device -> A4/<dev>/<pass>/cKKofNN/{emb_<run>-0.parquet, payload.npz, observer.npz, done.json}
  assemble      per device: the code store and the query-encoding index from every chunk's shards; 'research': the
                same objects from the research encodings (tsub + chains codes; q200 + chains q)
  analyze       every A4 metric, invariant and gate -> A4/analysis/a4_parity.json and a4_parity.txt
Devices: gpu (CUDA, FoldMatch default precision: fp32 weights, bf16 autocast inside ESM3.logits), cpu_linux (NERSC
EPYC fp32), cpu_mac (M1 Max fp32), mps (M1 Max Apple GPU fp32, D18).
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import platform
import re
import shutil
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

IS_NERSC = os.path.isdir("/pscratch/sd/j/jsegura")
SANDBOX = "/Users/joan/devel/foldmatch-p2p3-sandbox"
P8_ROOT = os.environ.get("P8_ROOT") or ("/pscratch/sd/j/jsegura/embcomp/p8" if IS_NERSC else f"{SANDBOX}/p8")
A4 = os.environ.get("A4_ROOT") or os.path.join(P8_ROOT, "A4")
SYSTEM = "esm-codes-v1"
ALL_DEVS = ("gpu", "cpu_linux", "cpu_mac", "mps")
DEVS = ALL_DEVS
SRCS = ("research",) + DEVS
SETS_CODES = ("panel", "enccheck", "chains", "tsub")          # target side; code bits pooled over these (P0.5)
SETS_QUERY = ("q200", "chains", "enccheck")                   # query side
PASSES = {"codes": ("residue_codes", "codes.fasta"), "queries": ("query_encoding", "queries.fasta")}
# device tag -> (FoldMatch --accelerator, the device type the pass record must show)
ACCEL = {"gpu": ("gpu", "cuda"), "cpu_linux": ("cpu", "cpu"), "cpu_mac": ("cpu", "cpu"), "mps": ("auto", "mps")}
HOST_OF = {"gpu": "nersc", "cpu_linux": "nersc", "cpu_mac": "mac", "mps": "mac"}
SETS_SHA256 = "fb1e30af7a1730e0dd171867c72343678e76544d1a9e0eebf3c035c14fbb43a5"     # A4_THRESHOLDS evidence
PAIRS_SHA256 = "378a1155aac103dd4cefbbea9c65fa5bd2bea58a78544bf864cda390afeac6b3"    # A4_THRESHOLDS P_sub
N_BENCH = 8557236                                                                     # protocol evalue.N
F1_REF = 0.613393                                                                     # protocol dF1 reference
FP16_MAX = 65504.0
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
STAGE2_FIELDS = ("query", "target", "raw", "rawrev", "q90", "rawadj", "evalue", "bits", "qstart", "qend", "tstart",
                 "tend", "alnlen", "cigar", "ecal")
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:8.1f}s] " + " ".join(str(x) for x in a), flush=True)


def _C():
    """acc_common, imported lazily: it sets the BLAS thread variables to 1 (by setdefault) when imported, which the
    ESM3 pass must not inherit (the pass runs with the device's own thread settings)."""
    import acc_common
    assert os.path.realpath(acc_common.P8_ROOT) == os.path.realpath(P8_ROOT), (acc_common.P8_ROOT, P8_ROOT)
    return acc_common


def md5_file(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("ascii")).hexdigest()


def read_fasta(path):
    recs, h, s = [], None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if h is not None:
                    recs.append((h, "".join(s)))
                h, s = line[1:].split()[0], []
            elif line.strip():
                s.append(line.strip())
    if h is not None:
        recs.append((h, "".join(s)))
    return recs


def write_fasta(path, recs):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        for k, s in recs:
            fh.write(f">{k}\n")
            for i in range(0, len(s), 80):
                fh.write(s[i:i + 80] + "\n")
    os.replace(tmp, path)


def write_json(obj, path):
    """Strict JSON, atomic (acc_common.jsonable conversion)."""
    from acc_common import jsonable  # noqa: PLC0415  (light: numpy only)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(jsonable(obj), fh, indent=1, allow_nan=False)
        fh.write("\n")
    os.replace(tmp, path)
    return sha256_file(path)


def read_json(p):
    with open(p) as fh:
        return json.load(fh)


# =====================================================================================================================
# inputs
# =====================================================================================================================
def src_input(name):
    """A P0.5 input file (p0/parity/inputs/<name>): the staged copy on NERSC, A4/inputs/src/<name> on the Mac (copied
    from NERSC, md5-checked by `inputs`)."""
    if IS_NERSC:
        return _C().inp(f"p0/parity/inputs/{name}")
    return os.path.join(A4, "inputs", "src", name)


def load_sets():
    p = src_input("sets.json")
    h = sha256_file(p)
    if h != SETS_SHA256:
        raise RuntimeError(f"{p}: sha256 {h} != {SETS_SHA256} (A4_THRESHOLDS evidence inputs_sets_json)")
    return read_json(p)


def cmd_inputs(a):
    C = _C()
    th = C.a4_thresholds()
    proto = th["thresholds"]["protocol"]["inputs"]
    man = load_sets()
    out = os.path.join(A4, "inputs")
    os.makedirs(out, exist_ok=True)
    seq_of, sets = {}, {}
    for name in ("panel", "enccheck", "chains", "q200", "tsub"):
        S = man["sets"][name]
        fa = src_input(f"{name}.fasta")
        got = md5_file(fa)
        assert got == S["fasta_md5"] == proto[name]["fasta_md5"], (name, got, S["fasta_md5"], proto[name])
        assert S["n_records"] == proto[name]["n_records"] and S["residues"] == proto[name]["residues"], name
        recs = read_fasta(fa)
        for k, s in recs:
            assert sha1_hex(s) == k, (name, k)
            if k in seq_of:
                assert seq_of[k] == s
            seq_of[k] = s
        keys = S["keys"]
        assert set(keys) == {k for k, _ in recs} and len(recs) == S["n_unique"], name
        assert sum(len(seq_of[k]) for k in keys) == S["residues"], name
        sets[name] = dict(n_records=len(keys), n_unique=len(set(keys)), residues=sum(len(seq_of[k]) for k in keys),
                          fasta=fa, fasta_md5=got)
    files = {}
    for pss, members in (("codes", SETS_CODES), ("queries", SETS_QUERY), ("q200u", ("q200",))):
        order = list(dict.fromkeys(k for s in members for k in man["sets"][s]["keys"]))
        path = os.path.join(out, f"{pss}.fasta")
        write_fasta(path, [(k, seq_of[k]) for k in order])
        L = [len(seq_of[k]) for k in order]
        files[pss] = dict(path=path, md5=md5_file(path), sha256=sha256_file(path), n=len(order), residues=sum(L),
                          max_length=max(L), members=list(members))
        log(f"{pss}.fasta: {len(order)} unique sequences, {sum(L):,} residues, max {max(L)} aa, md5 {files[pss]['md5']}")
    # overlaps between the sets (one ESM3 pass per unique sequence and device; each set compares its own records)
    ov = {}
    for i, s1 in enumerate(("panel", "enccheck", "chains", "q200", "tsub")):
        for s2 in ("panel", "enccheck", "chains", "q200", "tsub")[i + 1:]:
            n = len(set(man["sets"][s1]["keys"]) & set(man["sets"][s2]["keys"]))
            if n:
                ov[f"{s1}&{s2}"] = n
    pairs = man["hits"]["pairs"]
    psha = hashlib.sha256(json.dumps(pairs).encode()).hexdigest()
    assert psha == PAIRS_SHA256 and len(pairs) == 831, psha
    rec = dict(what="A4 input FASTAs (production ESM3 passes) built from the P0.5 inputs",
               created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), host=socket.gethostname(),
               sets_json=dict(path=src_input("sets.json"), sha256=SETS_SHA256), sets=sets, overlaps=ov, files=files,
               P_sub=dict(n_pairs=len(pairs), pairs_sha256=psha), a4_thresholds_sha256=th["sha256"])
    h = write_json(rec, os.path.join(out, "inputs.json"))
    log(f"wrote {out}/inputs.json sha256 {h}; overlaps {ov}")


# =====================================================================================================================
# embed: one production ESM3 pass over one chunk
# =====================================================================================================================
def lpt_chunks(lengths, n):
    """Deterministic LPT split by ~cost (L^1.3 + 200, P0.5 lpt_shards); records keep their file order in a chunk."""
    import numpy as np
    order = np.argsort(-np.asarray(lengths), kind="stable")
    load = np.zeros(n)
    shard = np.zeros(len(lengths), np.int64)
    for i in order:
        r = int(np.argmin(load))
        shard[i] = r
        load[r] += lengths[i] ** 1.3 + 200.0
    return shard


def chunk_fasta(dev, pss, k, n):
    src = os.path.join(A4, "inputs", PASSES[pss][1])
    recs = read_fasta(src)
    shard = lpt_chunks([len(s) for _, s in recs], n)
    mine = [recs[i] for i in range(len(recs)) if shard[i] == k]
    d = os.path.join(A4, dev, "chunks")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{pss}.c{k:02d}of{n:02d}.fasta")
    if os.path.exists(path):
        assert read_fasta(path) == mine, f"{path} differs from the chunk definition"
    else:
        write_fasta(path, mine)
    return path, mine, src


class Observer:
    """Passive observer of ChainCompleteModule's ESM3 call (A4 invariants). It wraps the module-level
    ``per_residue_embedding`` that ``ChainCompleteModule.predict_step`` calls, returns ESM3's tensor unchanged (same
    object), and records per protein: sha1 of the sequence, L, the embedding rows / width / dtype / device,
    finiteness, max |emb|, and the ESM3 tokenizer's token and <unk> counts for the sequence."""

    def __init__(self):
        import torch
        import foldmatch.modules.chain_complete_module as CCM
        self.torch, self.CCM = torch, CCM
        self.orig = CCM.per_residue_embedding
        self.rows = []
        CCM.per_residue_embedding = self.observed

    def observed(self, model, esm_prot):
        emb = self.orig(model, esm_prot)
        seq = esm_prot.sequence or ""
        t = self.torch
        with t.no_grad():
            finite = bool(t.isfinite(emb).all().item()) if emb.numel() else True
            maxabs = float(emb.abs().max().item()) if emb.numel() else 0.0
        tok = model.tokenizers.sequence
        ids = tok.encode(seq, add_special_tokens=True)
        self.rows.append((sha1_hex(seq), len(seq), int(emb.shape[0]), int(emb.shape[-1]), finite, maxabs, len(ids),
                          int(sum(1 for x in ids if x == tok.unk_token_id)), str(emb.dtype), str(emb.device)))
        return emb

    def restore(self):
        self.CCM.per_residue_embedding = self.orig

    def summary(self):
        r = self.rows
        return dict(proteins=len(r), all_finite=all(x[4] for x in r), max_abs_emb=max((x[5] for x in r), default=0.0),
                    max_abs_below_fp16_max=all(x[5] < FP16_MAX for x in r), unk_tokens=sum(x[7] for x in r),
                    tokens_eq_L_plus_2=all(x[6] == x[1] + 2 for x in r), rows_eq_L_plus_2=all(x[2] == x[1] + 2 for x in r),
                    width=sorted({x[3] for x in r}), dtypes=sorted({x[8] for x in r}), devices=sorted({x[9] for x in r}))

    def save(self, path):
        import numpy as np
        r = self.rows
        np.savez(path, sha1=np.array([x[0] for x in r]), L=np.array([x[1] for x in r], np.int64),
                 rows=np.array([x[2] for x in r], np.int64), width=np.array([x[3] for x in r], np.int64),
                 finite=np.array([x[4] for x in r], bool), maxabs=np.array([x[5] for x in r], np.float64),
                 tokens=np.array([x[6] for x in r], np.int64), unk=np.array([x[7] for x in r], np.int64),
                 dtype=np.array([x[8] for x in r]), device=np.array([x[9] for x in r]))


def torch_info(torch):
    t = dict(version=torch.__version__, cuda=torch.version.cuda, num_threads=torch.get_num_threads(),
             num_interop_threads=torch.get_num_interop_threads(),
             float32_matmul_precision=torch.get_float32_matmul_precision(),
             cuda_matmul_allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
             cudnn_allow_tf32=bool(torch.backends.cudnn.allow_tf32),
             mps_available=bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()))
    if torch.cuda.is_available():
        t["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        t["cuda_peak_alloc_gib"] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    try:
        t["parallel_info"] = torch.__config__.parallel_info()
    except Exception:  # noqa: BLE001
        pass
    return t


def cmd_embed(a):
    dev, pss, k, n = a.dev, a.pass_, a.chunk, a.nchunks
    if a.threads:                               # explicit thread count (NERSC); else the platform defaults (Mac)
        for v in THREAD_VARS:
            os.environ[v] = str(a.threads)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    out = os.path.join(A4, dev, pss, f"c{k:02d}of{n:02d}")
    if os.path.exists(os.path.join(out, "done.json")):
        log(f"{out}: done already")
        return
    fasta, mine, src = chunk_fasta(dev, pss, k, n)
    if os.path.isdir(out):                      # an attempt that did not finish: clear this chunk's own files only
        for f in glob.glob(os.path.join(out, "*")):
            os.remove(f)
    os.makedirs(out, exist_ok=True)
    residue_output, _ = PASSES[pss]
    accelerator, expect = ACCEL[dev]
    import numpy as np
    import torch
    from pathlib import Path
    import foldmatch
    from foldmatch.inference.chain_complete_inference import prepare_scoring_system
    from foldmatch.search.embedding_computer import EmbeddingComputer
    from foldmatch.search.residue_shards import (QUERY_ENCODING, RESIDUE_CODES, check_shards, shards_pass_record)
    from foldmatch.types.api_types import Accelerator
    from foldmatch.utils.sequence import preflight_fasta
    obs = Observer()
    log(f"{dev}/{pss} chunk {k}/{n}: {len(mine)} sequences, {sum(len(s) for _, s in mine):,} residues; foldmatch "
        f"{foldmatch.__file__}; torch {torch.__version__} threads {torch.get_num_threads()}")
    t0 = time.time()
    prepared = prepare_scoring_system(None if residue_output == RESIDUE_CODES else SYSTEM,
                                      residue_output=residue_output)
    system = prepared.system
    assert system.name == SYSTEM
    preflight = preflight_fasta(Path(fasta), 0)
    preflight.check(strict=True)
    ec = EmbeddingComputer(embedding_folder=Path(out))
    t_prep = time.time() - t0
    t1 = time.time()
    ec.compute_from_fasta(fasta_file=Path(fasta), min_res_n=0, batch_size=1, num_workers=0, num_nodes=1, devices=1,
                          accelerator=Accelerator(accelerator), preflight=preflight, residue_output=residue_output,
                          scoring_system=prepared)
    t_pass = time.time() - t1
    obs.restore()
    shards = ec.shard_files()
    infos = check_shards(shards, residue_output, system)
    record = shards_pass_record(infos)
    assert record["accelerator"] == expect, (record, expect)
    seq_of = dict(mine)
    keys, lengths, sha1s, blobs = [], [], [], []
    if residue_output == RESIDUE_CODES:
        for b in ec.iter_residue_codes(system):
            for i, c, s in zip(b.ids, b.codes, b.seq_sha1s):
                keys.append(i)
                lengths.append(len(c))
                sha1s.append(bytes(s).hex())
                blobs.append(np.asarray(c))
        payload = dict(codes=np.concatenate(blobs) if blobs else np.zeros((0, 8), np.uint8))
    else:
        qi = ec.build_query_encodings(Path(out) / "query_encodings.f16", system)
        mm = qi.memmap()
        for i, e in qi.entries.items():
            keys.append(i)
            lengths.append(e.length)
            sha1s.append(bytes(e.seq_sha1).hex())
            blobs.append(np.asarray(mm[e.offset:e.offset + e.length]))
        payload = dict(q=np.concatenate(blobs) if blobs else np.zeros((0, 64), np.float16))
        del mm
        qi.remove()
    assert sorted(keys) == sorted(seq_of) and len(keys) == len(seq_of), "payload ids differ from the chunk"
    for i, L, s in zip(keys, lengths, sha1s):
        assert L == len(seq_of[i]) and s == i == sha1_hex(seq_of[i]), i
    np.savez(os.path.join(out, "payload.npz"), keys=np.array(keys), lengths=np.array(lengths, np.int64),
             offsets=np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64), seq_sha1=np.array(sha1s), **payload)
    obs.save(os.path.join(out, "observer.npz"))
    osum = obs.summary()
    assert osum["proteins"] == len(mine), osum
    C = _C()
    rec = dict(
        what=f"A4 production ESM3 pass ({residue_output}) of chunk {k}/{n} on {dev}",
        dev=dev, pass_=pss, residue_output=residue_output, chunk=k, nchunks=n, accelerator=accelerator,
        fasta=fasta, fasta_md5=md5_file(fasta), source_fasta=src, source_fasta_md5=md5_file(src),
        sequences=len(mine), residues=sum(len(s) for _, s in mine), run_id=ec.run_id,
        shards=[dict(path=str(p), bytes=os.path.getsize(p), sha256=sha256_file(p)) for p in shards],
        pass_record=record, payload_npz_sha256=sha256_file(os.path.join(out, "payload.npz")),
        observer=osum, timing=dict(prepare_s=round(t_prep, 2), pass_s=round(t_pass, 2),
                                   residues_per_s=round(sum(len(s) for _, s in mine) / max(t_pass, 1e-9), 1)),
        torch=torch_info(torch), threads_arg=a.threads, slurm_job=os.environ.get("A4_SLURM_JOB_ID"),
        prepared=dict(encoders=str(prepared.encoders_path), scoring=str(prepared.scoring_path)),
        provenance=C.provenance(torch_info=True))
    h = write_json(rec, os.path.join(out, "done.json"))
    log(f"{dev}/{pss} chunk {k}/{n}: pass {t_pass:.1f} s ({rec['timing']['residues_per_s']} res/s), observer {osum}, "
        f"done.json sha256 {h}")


# =====================================================================================================================
# assemble
# =====================================================================================================================
def chunk_records(dev, pss):
    recs = [read_json(p) for p in sorted(glob.glob(os.path.join(A4, dev, pss, "c*of*", "done.json")))]
    if not recs:
        raise RuntimeError(f"{dev}/{pss}: no finished chunks")
    n = {r["nchunks"] for r in recs}
    assert len(n) == 1, f"{dev}/{pss}: chunks of different splits {n}"
    n = n.pop()
    got = sorted(r["chunk"] for r in recs)
    if got != list(range(n)):
        raise RuntimeError(f"{dev}/{pss}: chunks {got} of {n} finished")
    return sorted(recs, key=lambda r: r["chunk"])


def _same_pass_records(recs):
    keys = ("accelerator", "world_size", "esm_precision", "encoder_precision", "esm_revision", "foldmatch_version",
            "torch", "esm", "platform", "min_res_n")
    first = {k: recs[0]["pass_record"].get(k) for k in keys}
    diff = sorted({k for r in recs[1:] for k in keys if r["pass_record"].get(k) != first[k]})
    return first, diff


def assemble_device(dev):
    import numpy as np
    from pathlib import Path
    from foldmatch.search import scoring_system as SSY
    from foldmatch.search.query_encodings import QueryEncodingIndex
    from foldmatch.search.residue_code_store import ResidueCodeStore
    from foldmatch.search.residue_shards import RESIDUE_CODES, iter_residue_codes
    from foldmatch.search.sequence_store import SequenceStore
    from foldmatch.utils.sequence import preflight_fasta
    system = SSY.get(SYSTEM)
    inputs = read_json(os.path.join(A4, "inputs", "inputs.json"))
    out = {"dev": dev, "passes": {}}
    for pss in ("codes", "queries"):
        recs = chunk_records(dev, pss)
        rec0, diff = _same_pass_records(recs)
        for r in recs:                                    # the shards are the production outputs: unchanged?
            for s in r["shards"]:
                p = s["path"] if os.path.exists(s["path"]) else os.path.join(A4, dev, pss, f"c{r['chunk']:02d}of"
                                                                                       f"{r['nchunks']:02d}",
                                                                          os.path.basename(s["path"]))
                assert sha256_file(p) == s["sha256"], p
                s["local_path"] = p
        out["passes"][pss] = dict(
            chunks=len(recs), sequences=sum(r["sequences"] for r in recs), residues=sum(r["residues"] for r in recs),
            pass_record=rec0, pass_record_differences=diff, run_ids=[r["run_id"] for r in recs],
            pass_s=sum(r["timing"]["pass_s"] for r in recs),
            residues_per_s_per_process=[r["timing"]["residues_per_s"] for r in recs],
            torch_threads=sorted({r["torch"]["num_threads"] for r in recs}),
            hosts=sorted({r["provenance"]["host"]["hostname"] for r in recs}),
            foldmatch_commit=sorted({r["provenance"]["foldmatch"]["commit"] for r in recs}),
            foldmatch_file=sorted({r["provenance"]["foldmatch"]["file"] for r in recs}),
            kernel=recs[0]["provenance"].get("kernel"),
            observer={k: (all(r["observer"][k] for r in recs) if isinstance(recs[0]["observer"][k], bool) else None)
                      for k in recs[0]["observer"]})
        o = out["passes"][pss]["observer"]
        o.update(proteins=sum(r["observer"]["proteins"] for r in recs),
                 max_abs_emb=max(r["observer"]["max_abs_emb"] for r in recs),
                 unk_tokens=sum(r["observer"]["unk_tokens"] for r in recs),
                 dtypes=sorted({d for r in recs for d in r["observer"]["dtypes"]}),
                 devices=sorted({d for r in recs for d in r["observer"]["devices"]}))
        f = inputs["files"][pss]
        assert out["passes"][pss]["sequences"] == f["n"] and out["passes"][pss]["residues"] == f["residues"], pss
        out["passes"][pss]["recs"] = recs
    # code store: create from every codes chunk's shards (iter_residue_codes = check_shards + the payload reader), then
    # validate against a sequence store of codes.fasta and its pre-flight (checks a-e), as the build does after FAISS
    db = Path(A4, dev, "db")
    db.mkdir(parents=True, exist_ok=True)
    for p in list(db.glob("codes.*")):
        p.unlink()
    codes_fa = Path(inputs["files"]["codes"]["path"]) if os.path.exists(inputs["files"]["codes"]["path"]) else \
        Path(A4, "inputs", "codes.fasta")
    assert md5_file(codes_fa) == inputs["files"]["codes"]["md5"]
    seq = SequenceStore(db, "codes")
    seq.create(fasta_file=codes_fa, min_res_n=0)
    pre = preflight_fasta(codes_fa, 0)
    pre.check(strict=True)
    store = ResidueCodeStore(db, "codes")
    cr = out["passes"]["codes"]
    build = dict(run_id="+".join(cr["run_ids"]), from_shards=True, accelerator=cr["pass_record"]["accelerator"],
                 devices=1, esm_precision=cr["pass_record"]["esm_precision"],
                 encoder_precision=cr["pass_record"]["encoder_precision"])

    def batches():
        for r in cr["recs"]:
            yield from iter_residue_codes([s["local_path"] for s in r["shards"]], system)
    store.create(batches(), system, min_res_n=0, build=build)
    manifest = store.validate(seq, pre.unique_entries, preflight=pre)
    out["code_store"] = dict(db=str(db), index_name="codes", manifest=manifest,
                             bin_sha256=sha256_file(store.bin_path), faiss_ids_count_note="no FAISS index (Stage 2 "
                             "does not need it): validate() got the pre-flight's entry count in its place")
    # query-encoding index over every queries chunk's shards (QueryEncodingIndex.build, as the query pass)
    qd = Path(A4, dev, "qenc")
    qd.mkdir(parents=True, exist_ok=True)
    dst = qd / "query_encodings.f16"
    dst.unlink(missing_ok=True)
    qr = out["passes"]["queries"]
    shards = [s["local_path"] for r in qr["recs"] for s in r["shards"]]
    qi = QueryEncodingIndex.build(shards, dst, system=system, expected_count=inputs["files"]["queries"]["n"])
    ent = {i: [e.offset, e.length, bytes(e.seq_sha1).hex()] for i, e in qi.entries.items()}
    write_json(dict(path=str(dst), rows=qi.rows, width=qi.width, system_name=qi.system_name, entries=ent),
               str(qd / "entries.json"))
    out["query_index"] = dict(path=str(dst), rows=qi.rows, width=qi.width, entries=len(ent), sha256=sha256_file(dst))
    # the store / index hold exactly what the chunks' payload readers returned (done in embed)
    mm = store.open_memmap()
    nbad = 0
    for r in cr["recs"]:
        z = np.load(os.path.join(os.path.dirname(r["shards"][0]["local_path"]), "payload.npz"))
        off, ln = store.lookup_rows(z["keys"].astype(str).tolist())
        for j in range(len(off)):
            nbad += not np.array_equal(mm[off[j]:off[j] + ln[j]], z["codes"][z["offsets"][j]:z["offsets"][j + 1]])
    qm = qi.memmap()
    for r in qr["recs"]:
        z = np.load(os.path.join(os.path.dirname(r["shards"][0]["local_path"]), "payload.npz"))
        for j, key in enumerate(z["keys"].astype(str)):
            e = qi.entries[key]
            nbad += not np.array_equal(qm[e.offset:e.offset + e.length], z["q"][z["offsets"][j]:z["offsets"][j + 1]])
    out["store_index_equal_payloads"] = nbad == 0
    assert nbad == 0
    for pss in ("codes", "queries"):
        out["passes"][pss]["chunk_done_json"] = [dict(chunk=r["chunk"], fasta_md5=r["fasta_md5"], run_id=r["run_id"],
                                                      shards=[s["sha256"] for s in r["shards"]],
                                                      timing=r["timing"]) for r in out["passes"][pss]["recs"]]
        del out["passes"][pss]["recs"]
    out["provenance"] = _C().provenance()
    h = write_json(out, os.path.join(A4, dev, "assemble.json"))
    log(f"assembled {dev}: code store {manifest['entries']} entries / {manifest['total_residues']} residues, query "
        f"index {len(ent)} / {qi.rows} rows; assemble.json sha256 {h}")


def research_sets(man):
    """Research encodings of the A4 sets in RECORD order (esm3_parity.research_enc, staged inputs)."""
    import numpy as np
    C = _C()
    S = man["sets"]

    def load_concat(name):
        z = np.load(C.inp(name))
        return z
    out = {}
    zq = load_concat("embcomp/evalue/encode/out/queries.L2m_B8_s0.npz")
    off = zq["offsets"]
    rows = S["q200"]["rows"]
    ids = zq["ids"].astype(str)
    assert [ids[r] for r in rows] == S["q200"]["ids"]
    out["q200"] = dict(ids=S["q200"]["ids"], q=[np.asarray(zq["q"][off[r]:off[r + 1]]) for r in rows])
    zt = load_concat("embcomp/evalue/encode/out/targets.L2m_B8_s0.npz")
    off = zt["offsets"]
    tids = zt["ids"].astype(str)
    cols = S["tsub"]["cols"]
    assert [tids[c] for c in cols] == S["tsub"]["ids"]
    codes = zt["codes"]
    out["tsub"] = dict(ids=S["tsub"]["ids"], codes=[np.asarray(codes[off[c]:off[c + 1]]) for c in cols])
    zp = np.load(C.inp("embcomp/panel/fit2/results/shipped_panel_codes.npz"))
    off = zp["offsets"]
    assert zp["ids"].astype(str).tolist() == S["panel"]["ids"]
    out["panel"] = dict(ids=S["panel"]["ids"], codes=[zp["codes"][off[i]:off[i + 1]] for i in range(len(off) - 1)],
                        offsets=zp["offsets"], all_codes=zp["codes"])
    zc = np.load(C.inp("embcomp/throughput/bench/data/L2m_B8_s0.chains.npz"))
    off = zc["offsets"]
    assert zc["ids"].astype(str).tolist() == S["chains"]["ids"]
    out["chains"] = dict(ids=S["chains"]["ids"], q=[zc["q"][off[i]:off[i + 1]] for i in range(len(off) - 1)],
                         codes=[zc["codes"][off[i]:off[i + 1]] for i in range(len(off) - 1)])
    ze = np.load(C.inp("embcomp/sw_mlp/enc_check/L2m_B8_s0.npz"))
    qs, cs = [], []
    for i in S["enccheck"]["ids"]:
        g, nme = i.split("/")
        qs.append(ze[f"{g}/q/{nme}"])
        cs.append(ze[f"{g}/code/{nme}"])
    out["enccheck"] = dict(ids=S["enccheck"]["ids"], q=qs, codes=cs)
    return out


def assemble_research():
    """The research encodings as the same production objects: a validated code store of the research codes of the
    tsub + chains sequences (Stage-2 targets), and a query-encoding index of the research q of the q200 records
    (ids 'q200/<id>') and the chain records ('chains/<id>')."""
    import numpy as np
    from pathlib import Path
    from foldmatch.search import scoring_system as SSY
    from foldmatch.search.query_encodings import QueryEncodingIndex
    from foldmatch.search.residue_code_store import ResidueCodeStore
    from foldmatch.search.sequence_store import SequenceStore
    from foldmatch.utils.sequence import preflight_fasta, seq_sha1
    system = SSY.get(SYSTEM)
    man = load_sets()
    R = research_sets(man)
    seq_of = {}
    for name in ("q200", "tsub", "chains"):
        for k, s in read_fasta(src_input(f"{name}.fasta")):
            seq_of[k] = s
    db = Path(A4, "research", "db")
    db.mkdir(parents=True, exist_ok=True)
    for p in list(db.glob("research.*")):
        p.unlink()
    entries, recs = {}, []
    for name in ("tsub", "chains"):
        for key, c in zip(man["sets"][name]["keys"], R[name]["codes"]):
            assert len(c) == len(seq_of[key]), (name, key)
            if key in entries:
                assert np.array_equal(entries[key], c), f"research codes of {key} differ between sets"
                continue
            entries[key] = np.ascontiguousarray(c)
            recs.append((key, seq_of[key]))
    fa = db / "research.fasta"
    write_fasta(str(fa), recs)
    seq = SequenceStore(db, "research")
    seq.create(fasta_file=fa, min_res_n=0)
    pre = preflight_fasta(fa, 0)
    pre.check(strict=True)
    store = ResidueCodeStore(db, "research")
    ids = [k for k, _ in recs]
    store.create([(ids, [entries[k] for k in ids], [seq_sha1(seq_of[k]) for k in ids])], system, min_res_n=0,
                 build=dict(run_id="research", from_shards=False, accelerator="research-encodings", devices=None,
                            esm_precision="research (bf16-weight batched SequenceOnlyESM3, A100)",
                            encoder_precision="research sw_common"))
    manifest = store.validate(seq, pre.unique_entries, preflight=pre)
    qd = Path(A4, "research", "qenc")
    qd.mkdir(parents=True, exist_ok=True)
    dst = qd / "query_encodings.f16"
    ent, rows = {}, 0
    with open(dst, "wb") as fh:
        for name in ("q200", "chains"):
            for rid, key, q in zip(man["sets"][name]["ids"], man["sets"][name]["keys"], R[name]["q"]):
                q = np.ascontiguousarray(q, dtype="<f2")
                assert q.shape == (len(seq_of[key]), 64)
                ent[f"{name}/{rid}"] = [rows, len(q), seq_sha1(seq_of[key]).hex()]
                fh.write(q.tobytes())
                rows += len(q)
    qi = QueryEncodingIndex(dst, rows, 64, {}, SYSTEM)
    write_json(dict(path=str(dst), rows=rows, width=64, system_name=SYSTEM, entries=ent), str(qd / "entries.json"))
    out = dict(dev="research", code_store=dict(db=str(db), index_name="research", manifest=manifest,
                                               bin_sha256=sha256_file(store.bin_path)),
               query_index=dict(path=str(qi.path), rows=rows, entries=len(ent), sha256=sha256_file(dst)),
               provenance=_C().provenance())
    h = write_json(out, os.path.join(A4, "research", "assemble.json"))
    log(f"assembled research: code store {manifest['entries']} entries, query index {len(ent)} entries; sha256 {h}")


def cmd_assemble(a):
    _C()                                        # BLAS threads 1 before numpy (no ESM3 here)
    for dev in a.devs:
        if dev == "research":
            assemble_research()
        else:
            assemble_device(dev)


# =====================================================================================================================
# analyze
# =====================================================================================================================
def load_qindex(src):
    from foldmatch.search.query_encodings import QueryEncodingEntry, QueryEncodingIndex
    d = read_json(os.path.join(A4, src, "qenc", "entries.json"))
    path = os.path.join(A4, src, "qenc", "query_encodings.f16")
    ent = {k: QueryEncodingEntry(int(v[0]), int(v[1]), bytes.fromhex(v[2])) for k, v in d["entries"].items()}
    assert os.path.getsize(path) == d["rows"] * d["width"] * 2
    return QueryEncodingIndex(path, d["rows"], d["width"], ent, d["system_name"])


def code_store(src):
    from foldmatch.search.residue_code_store import ResidueCodeStore
    st = ResidueCodeStore(os.path.join(A4, src, "db"), "research" if src == "research" else "codes")
    assert st.exists()
    return st


def qid_of(src, set_name, rid, key):
    return f"{set_name}/{rid}" if src == "research" else key


def bit_agreement(a_codes, b_codes):
    import numpy as np
    ba, bb = np.unpackbits(a_codes, axis=1), np.unpackbits(b_codes, axis=1)
    eq = ba == bb
    return dict(bits_total=int(eq.size), bits_equal=int(eq.sum()), agreement=float(eq.mean()),
                residues=int(len(eq)), residues_code_identical=float(eq.all(1).mean()))


def per_item_bits(xs, ys):
    import numpy as np
    ag = np.array([float((np.unpackbits(x, axis=1) == np.unpackbits(y, axis=1)).mean()) for x, y in zip(xs, ys)])
    return dict(items=len(ag), items_identical=int((ag == 1.0).sum()), item_min=float(ag.min()),
                item_p5=float(np.percentile(ag, 5)), item_median=float(np.median(ag)))


def q_cosine(qa, qb):
    import numpy as np
    x, y = qa.astype(np.float64), qb.astype(np.float64)
    c = (x * y).sum(1) / (np.linalg.norm(x, axis=1) * np.linalg.norm(y, axis=1))
    return dict(residues=int(len(c)), min=float(c.min()), p0_1=float(np.percentile(c, 0.1)),
                p1=float(np.percentile(c, 1)), median=float(np.median(c)), mean=float(c.mean()),
                frac_ge_0_99=float((c >= 0.99).mean()), frac_ge_0_999=float((c >= 0.999).mean()),
                q_bitwise_equal_rows=float((qa == qb).all(1).mean()))


def dstats(d):
    """esm3_parity.dstats (P0.5): statistics of |d| (np.median, np.percentile linear)."""
    import numpy as np
    d = np.abs(np.asarray(d, np.float64))
    if d.size == 0:
        return dict(n=0)
    return dict(n=int(d.size), median=float(np.median(d)), p90=float(np.percentile(d, 90)),
                p95=float(np.percentile(d, 95)), p99=float(np.percentile(d, 99)), max=float(d.max()),
                mean=float(d.mean()), frac_gt_0_5=float((d > 0.5).mean()), frac_gt_1=float((d > 1).mean()))


def lnE_bins(L_ref, d, edges=(-math.inf, -23.03, -6.91, -2.30, 0.0)):
    out = {}
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (L_ref > lo) & (L_ref <= hi)
        out[f"lnE_ref in ({lo:.2f}, {hi:.2f}]"] = dstats(d[m])
    return out


def cigar_pairs(m):
    """Aligned (query, target) 0-based residue pairs of a hit's forward path, from its CIGAR (M/I/D, the BLOSUM
    convention: I = query residue vs gap, D = gap in the query) and 1-based start bounds; checked against the end
    bounds."""
    import numpy as np
    if not m.cigar:
        return np.zeros((0, 2), np.int32)
    i, j = m.q_start - 1, m.t_start - 1
    out = []
    for cnt, op in re.findall(r"(\d+)([MID])", m.cigar):
        cnt = int(cnt)
        if op == "M":
            out.append(np.stack([np.arange(i, i + cnt), np.arange(j, j + cnt)], 1))
            i += cnt
            j += cnt
        elif op == "I":
            i += cnt
        else:
            j += cnt
    assert (i, j) == (m.q_end, m.t_end), (m.cigar, m.q_start, m.q_end, m.t_start, m.t_end, i, j)
    return np.concatenate(out).astype(np.int32) if out else np.zeros((0, 2), np.int32)


class Ctx:
    """Everything the analysis shares: sets, sequences, research encodings, device encodings, the prepared system."""

    def __init__(self, workers):
        import numpy as np
        C = _C()
        from foldmatch.search import scoring_system as SSY
        self.C, self.np, self.workers = C, np, workers
        self.th = C.a4_thresholds()
        self.man = load_sets()
        self.inputs = read_json(os.path.join(A4, "inputs", "inputs.json"))
        self.seq_of = {}
        for name in ("panel", "enccheck", "chains", "q200", "tsub"):
            for k, s in read_fasta(src_input(f"{name}.fasta")):
                self.seq_of[k] = s
        self.R = research_sets(self.man)
        self.system = SSY.get(SYSTEM)
        self.prepared = SSY.prepare(self.system, local_files_only=True)
        self.stores = {s: code_store(s) for s in SRCS}
        self.qidx = {s: load_qindex(s) for s in SRCS}

    def dev_codes(self, dev, set_name):
        np = self.np
        st = self.stores[dev]
        mm = st.open_memmap()
        keys = self.man["sets"][set_name]["keys"]
        off, ln = st.lookup_rows(keys)
        return [np.asarray(mm[o:o + n]) for o, n in zip(off.tolist(), ln.tolist())]

    def dev_q(self, dev, set_name):
        np = self.np
        qi = self.qidx[dev]
        return [np.asarray(qi.get(k)) for k in self.man["sets"][set_name]["keys"]]

    def stage2(self, qsrc, tsrc, pairs, set_name, panel):
        """Production Stage 2 for (query source, target source) on pairs [(query record id, query key, target key)]
        -> ({(qid, tkey): metrics}, stats)."""
        from foldmatch.search.code_alignment import CodeAlignmentPool, align_candidates_esm_codes
        pre, qseq = {}, {}
        for rid, qkey, tkey in pairs:
            q = qid_of(qsrc, set_name, rid, qkey)
            lst = pre.setdefault(q, [])
            if tkey not in lst:
                lst.append(tkey)
            qseq[q] = self.seq_of[qkey]
        prefilter = {q: (t, [0.0] * len(t)) for q, t in pre.items()}
        st = self.stores[tsrc]
        stats = {}
        t0 = time.time()
        with CodeAlignmentPool(self.workers) as pool:
            res = align_candidates_esm_codes(
                self.qidx[qsrc], prefilter, st, st.sequence_store().fetch, self.prepared, N_BENCH, pool,
                query_sequences=qseq, min_seq_identity=0.0, min_coverage=0.0, max_evalue=None,
                output_needs=list(STAGE2_FIELDS), panel_policy="always" if panel else "never", stats=stats)
        out = {}
        for q, hits in res.items():
            for h in hits:
                out[(q, h.subject_id)] = h.metrics
        stats["wall_s"] = round(time.time() - t0, 2)
        log(f"  stage2 {set_name} {qsrc}__{tsrc}: {len(pairs)} pairs, {len(out)} hits, {stats['wall_s']} s")
        return out, stats


def analyze_bits(X):
    np = X.np
    cb, per_item, xdev, qc, p05 = {}, {}, {}, {}, {}
    codes = {d: {s: X.dev_codes(d, s) for s in SETS_CODES} for d in DEVS}
    qs = {d: {s: X.dev_q(d, s) for s in SETS_QUERY} for d in DEVS}
    for d in DEVS:
        cb[d], per_item[d], qc[d] = {}, {}, {}
        eq = tot = 0
        for s in SETS_CODES:
            r = X.R[s]["codes"]
            b = bit_agreement(np.concatenate(r), np.concatenate(codes[d][s]))
            cb[d][s] = b
            per_item[d][s] = per_item_bits(r, codes[d][s])
            eq += b["bits_equal"]
            tot += b["bits_total"]
        cb[d]["pooled_panel_enccheck_chains_tsub"] = dict(bits_total=tot, bits_equal=eq, agreement=eq / tot)
        for s in SETS_QUERY:
            qc[d][s] = q_cosine(np.concatenate(X.R[s]["q"]), np.concatenate(qs[d][s]))
        log(f"code bits {d} vs research: " + ", ".join(f"{s} {v['agreement']:.6f}" for s, v in cb[d].items()))
    for i, d1 in enumerate(DEVS):
        for d2 in DEVS[i + 1:]:
            k = f"{d1}_vs_{d2}"
            xdev[k] = {}
            eq = tot = 0
            for s in SETS_CODES:
                b = bit_agreement(np.concatenate(codes[d1][s]), np.concatenate(codes[d2][s]))
                xdev[k][s] = dict(code_bits=b, per_item=per_item_bits(codes[d1][s], codes[d2][s]))
                eq += b["bits_equal"]
                tot += b["bits_total"]
            for s in SETS_QUERY:
                xdev[k].setdefault(s, {})["q_cosine"] = q_cosine(np.concatenate(qs[d1][s]), np.concatenate(qs[d2][s]))
            xdev[k]["pooled_code_bits_panel_enccheck_chains_tsub"] = dict(bits_total=tot, bits_equal=eq,
                                                                          agreement=eq / tot)
            log(f"cross-device {k}: pooled code bits {eq / tot:.7f}")
    # the production path vs the P0.5 capture path on the same device class (reported: P0.5 ran FoldMatch's ESM3 at
    # 3c3625c with the research encoders in the aggregator slot; cc55c01 adds G13 and the P2 encoder module)
    C = X.C
    for d in DEVS:
        p05[d] = {}
        for s in sorted(set(SETS_CODES) | set(SETS_QUERY)):
            name, sub = f"p0/parity/enc/{d}.{s}.npz", None
            if not os.path.exists(C.inp(name, must_exist=False)):
                if d == "gpu" and s == "tsub":
                    name, sub = "p0/parity/enc/gpu.a5t.npz", ("cols", X.man["sets"]["tsub"]["cols"])
                elif d == "gpu" and s == "q200":
                    name, sub = "p0/parity/enc/gpu.a5q.npz", ("rows", X.man["sets"]["q200"]["rows"])
                else:
                    continue
            z = np.load(C.inp(name))
            off = z["offsets"]
            idx = sub[1] if sub else range(len(off) - 1)
            if sub is None:
                assert z["ids"].astype(str).tolist() == X.man["sets"][s]["ids"], name
            e = {}
            if s in SETS_CODES:
                zc = z["codes"]
                pc = [np.asarray(zc[off[i]:off[i + 1]]) for i in idx]
                e["code_bits"] = bit_agreement(np.concatenate(pc), np.concatenate(codes[d][s]))
            if s in SETS_QUERY and "q" in z.files:
                zq = z["q"]
                pq = [np.asarray(zq[off[i]:off[i + 1]]) for i in idx]
                e["q_cosine"] = q_cosine(np.concatenate(pq), np.concatenate(qs[d][s]))
            e["source"] = name
            p05[d][s] = e
    return dict(code_bits_vs_research=cb, code_bits_per_item_vs_research=per_item, q_cosine_vs_research=qc,
                cross_device=xdev, production_vs_P0_5_same_device=p05), codes, qs


def analyze_evalues(X):
    """P_sub: every (query source, target source) through production Stage 2; research reference lnE from the stored
    S_D matrices and bench_q90_shipped.npy (the protocol's 'reference')."""
    np, C = X.np, X.C
    man = X.man
    q200 = man["sets"]["q200"]
    tsub = man["sets"]["tsub"]
    rpos = {r: k for k, r in enumerate(q200["rows"])}
    cpos = {c: k for k, c in enumerate(tsub["cols"])}
    P = np.array(man["hits"]["pairs"], np.int64)
    B = C.Bench()
    assert B.N_ann == float(N_BENCH), B.N_ann
    tabs = C.ship_tables()
    tab = tabs["all"]
    f = np.load(C.inp("embcomp/evalue/score/out/S_D_fwd.npy"), mmap_mode="r")
    r = np.load(C.inp("embcomp/evalue/score/out/S_D_rev.npy"), mmap_mode="r")
    Sf_ref = np.array([f[i, j] for i, j in P], np.float32)
    Sr_ref = np.array([r[i, j] for i, j in P], np.float32)
    S_ref = Sf_ref.astype(np.float64) - Sr_ref.astype(np.float64)
    q90s = np.load(C.inp("embcomp/panel/fit2/results/bench_q90_shipped.npy"))
    m = B.qlen[P[:, 0]]
    L_ref = C.ln_evalue(tab, m, S_ref[:, None], q90s[P[:, 0]], B.N_ann)[:, 0]
    pairs = [(q200["ids"][rpos[i]], q200["keys"][rpos[i]], tsub["keys"][cpos[j]]) for i, j in P]
    for k, (rid, qk, tk) in enumerate(pairs):
        assert len(X.seq_of[qk]) == int(B.qlen[P[k, 0]]) and len(X.seq_of[tk]) == int(B.tlen[P[k, 1]])
    allq = [(rid, k, None) for rid, k in zip(q200["ids"], q200["keys"])]
    # every q200 record also gets one extra candidate (the first tsub target), so that the queries without a P_sub
    # pair get their panel pass and Q90 too (only the P_sub pairs' hits are read; S16: a pair's result does not
    # depend on the other candidates)
    run_pairs = pairs + [(rid, k, tsub["keys"][0]) for rid, k, _ in allq]
    res, q90_by_src, raw = {}, {}, {}
    for qs in SRCS:
        for ts in SRCS:
            hits, stats = X.stage2(qs, ts, run_pairs, "q200", panel=True)
            qq = {}
            for rid, k, _ in allq:
                qq[rid] = stats["q90"][qid_of(qs, "q200", rid, k)]
            q90_by_src.setdefault(qs, qq)
            assert all(abs(q90_by_src[qs][x] - qq[x]) == 0.0 for x in qq), "Q90 depends on the target source"
            Sf, Sr, Q, L = (np.full(len(P), np.nan) for _ in range(4))
            missing = np.zeros(len(P), bool)
            for k, (rid, qk, tk) in enumerate(pairs):
                mt = hits.get((qid_of(qs, "q200", rid, qk), tk))
                if mt is None:
                    missing[k] = True
                    continue
                Sf[k], Sr[k], Q[k], L[k] = mt.score, mt.score_rev, mt.q90, mt.log_evalue
            Lchk = C.ln_evalue(tab, m, (Sf - Sr)[:, None], Q, B.N_ann)[:, 0]   # P4 statistics vs bench_validate
            d = L - L_ref
            ok = np.isfinite(d)
            rec = dict(dstats(d[ok]), signed_median=float(np.median(d[ok])), missing_pairs=int(missing.sum()),
                       by_research_lnE=lnE_bins(L_ref[ok], d[ok]),
                       hits_E_le_1_config=int((L <= 0).sum()), hits_E_le_1e3_ref=int((L_ref <= math.log(1e-3)).sum()),
                       hits_E_le_1e3_config=int((L <= math.log(1e-3)).sum()),
                       lnE_vs_bench_validate_formula_max_abs=float(np.nanmax(np.abs(Lchk - L))),
                       stage2=dict((k2, v) for k2, v in stats.items() if k2 not in ("q90", "degenerate_queries")))
            res[f"{qs}__{ts}"] = rec
            raw[f"{qs}__{ts}"] = dict(Sf=Sf, Sr=Sr, Q=Q, L=L)
            log(f"dlnE {qs}__{ts}: median {rec.get('median', float('nan')):.4f} p95 {rec.get('p95', float('nan')):.4f} "
                f"max {rec.get('max', float('nan')):.4f} (n {rec.get('n')}, missing {rec['missing_pairs']})")
    # dQ90 per query source (its q vs the shipped panel codes of the bundle) vs the research stored Q90
    dq = {}
    for qs in SRCS:
        x = np.array([q90_by_src[qs][rid] for rid in q200["ids"]])
        dd = x - q90s[np.asarray(q200["rows"])]
        dq[qs] = dict(dstats(dd), signed_median=float(np.median(dd)))
    # invariant: the research encodings through FoldMatch reproduce the stored Q90 and S (S16 tolerances)
    rr = raw["research__research"]
    inv = dict(
        q90_max_abs=float(np.max(np.abs(np.array([q90_by_src["research"][rid] for rid in q200["ids"]])
                                        - q90s[np.asarray(q200["rows"])]))),
        Sf_max_rel=float(np.nanmax(np.abs(rr["Sf"] - Sf_ref) / np.maximum(np.abs(Sf_ref), 1.0))),
        Sr_max_rel=float(np.nanmax(np.abs(rr["Sr"] - Sr_ref) / np.maximum(np.abs(Sr_ref), 1.0))),
        S_max_abs=float(np.nanmax(np.abs((rr["Sf"] - rr["Sr"]) - S_ref))),
        lnE_max_abs=float(np.nanmax(np.abs(rr["L"] - L_ref))),
        n_pairs=int(np.isfinite(rr["L"]).sum()), n_q200=len(q200["ids"]))
    inv["pass"] = bool(inv["q90_max_abs"] <= 1e-4 and inv["Sf_max_rel"] <= 5e-5 and inv["Sr_max_rel"] <= 5e-5
                       and inv["lnE_max_abs"] <= 1e-4 and inv["n_pairs"] == len(P))
    inv["criterion"] = ("S16 (FoldMatch vs the research code): |dQ90| <= 1e-4 for all 200 q200 records vs "
                        "bench_q90_shipped.npy; S_f / S_r within 5e-5 * max(|ref|, 1) of the stored S_D_fwd / S_D_rev "
                        "on the 831 P_sub pairs; |d lnE| <= 1e-4")
    ref = dict(N=B.N_ann, table="ship_final.json tables['%s']['all']" % C.SHIP_KEY, n_pairs=len(P),
               L_ref=dict(min=float(L_ref.min()), max=float(L_ref.max()), n_le_0=int((L_ref <= 0).sum())))
    return dict(P_sub=res, dQ90=dq, research_reproduces_stored=inv, reference=ref), raw, L_ref


def analyze_f1(X, harness):
    """|dF1| on the 582 similar chain pairs: the production Stage-2 alignment (forward path of S_f, setting D) of every
    chain pair for each (query source, target source), F1 = chain_eval / harness prf vs the US-align primary
    reference, minus the research M8_s0 setting-D mean (M8_s0_D.npz, 0.613393)."""
    np, C = X.np, X.C
    ce_dir = os.path.dirname(C.inp("embcomp/chains/chain_eval.py"))
    if ce_dir not in sys.path:
        sys.path.append(ce_dir)
    import chain_eval as CE
    import harness as H
    S = CE.load()
    sim = np.array([p["kind"] != "unrelated" for p in S.pairs])
    ref = CE.load_result(C.inp("embcomp/evalue/f1_settings/results/chains/M8_s0_D.npz"))
    fref = np.nan_to_num(ref["fields"]["f1"][sim]).mean()
    assert abs(fref - F1_REF) < 5e-7, fref
    ch = X.man["sets"]["chains"]
    key_of = dict(zip(ch["ids"], ch["keys"]))
    pairs = [(p["a"], key_of[p["a"]], key_of[p["b"]]) for p in S.pairs]
    out = {"research_M8_s0_D_mean_f1": float(fref), "n_similar": int(sim.sum()), "n_pairs": len(S.pairs),
           "configs": {}, "harness_configs": {}}
    for qs in SRCS:
        for ts in SRCS:
            hits, stats = X.stage2(qs, ts, pairs, "chains", panel=False)
            f1 = np.full(len(S.pairs), np.nan)
            alns, same, n_empty = [], [], 0
            for k, (p, (a_id, ak, bk)) in enumerate(zip(S.pairs, pairs)):
                mt = hits.get((qid_of(qs, "chains", a_id, ak), bk))
                pr = cigar_pairs(mt) if mt is not None else np.zeros((0, 2), np.int32)
                n_empty += len(pr) == 0
                alns.append(pr)
                rp = p.get("ref_primary")
                if rp is not None and len(rp):
                    f1[k] = H.prf(pr, rp)[2]
                ra = ref["alignments"][k]
                same.append(ra is not None and np.array_equal(ra, pr))
            res = dict(name=f"{qs}__{ts}", fields={"f1": f1}, evaluated=np.ones(len(S.pairs), bool), alignments=alns,
                       meta={})
            fm = float(np.nan_to_num(f1[sim]).mean())
            pdl = CE.paired_delta(res, ref)["ALL similar"]
            pdc = CE.paired_delta(res, ref, cluster=True)["ALL similar"]
            out["configs"][f"{qs}__{ts}"] = dict(
                mean_f1=fm, dF1=fm - F1_REF, dF1_vs_ref_file=fm - float(fref), dF1_ci_pairs=[pdl[1], pdl[2]],
                dF1_ci_cluster=[pdc[1], pdc[2]], alignments_identical_to_research=float(np.mean(same)),
                empty_alignments=n_empty, max_abs_pair_dF1=float(np.abs(np.nan_to_num(f1[sim])
                                                                        - np.nan_to_num(ref["fields"]["f1"][sim])).max()),
                stage2=dict((k2, v) for k2, v in stats.items() if k2 not in ("q90", "degenerate_queries")))
            log(f"dF1 {qs}__{ts}: mean F1 {fm:.6f} dF1 {fm - F1_REF:+.6f}; alignments identical to research "
                f"{np.mean(same):.4f}")
    if harness:
        # cross-check (reported): the P0.5 method, chain_eval.evaluate's harness DP (float64) on the production encodings
        for qs in SRCS:
            for ts in SRCS:
                qe = {c: X.qidx[qs].get(qid_of(qs, "chains", c, key_of[c])) for c in ch["ids"]}
                st = X.stores[ts]
                mm = st.open_memmap()
                off, ln = st.lookup_rows([key_of[c] for c in ch["ids"]])
                te = {c: np.asarray(mm[o:o + n]) for c, o, n in zip(ch["ids"], off.tolist(), ln.tolist())}
                r = CE.evaluate(f"{qs}__{ts}", query_fn=lambda c, e: np.asarray(qe[c], np.float32),
                                target_fn=lambda c, e: np.unpackbits(te[c], axis=1)[:, :64].astype(np.float32) * 2 - 1,
                                sw=(1.0, 5.0, 0.5), workers=X.workers, verbose=False)
                fm = float(np.nan_to_num(r["fields"]["f1"][sim]).mean())
                out["harness_configs"][f"{qs}__{ts}"] = dict(mean_f1=fm, dF1=fm - F1_REF)
                log(f"harness dF1 {qs}__{ts}: {fm - F1_REF:+.6f}")
    return out


def observer_invariants(X):
    out, ok = {}, True
    for d in DEVS:
        a = read_json(os.path.join(A4, d, "assemble.json"))
        e = {}
        for pss in ("codes", "queries"):
            o = a["passes"][pss]["observer"]
            p = a["passes"][pss]
            e[pss] = dict(proteins=o["proteins"], all_finite=o["all_finite"], max_abs_emb=o["max_abs_emb"],
                          max_abs_below_fp16_max=o["max_abs_emb"] < FP16_MAX, unk_tokens=o["unk_tokens"],
                          tokens_eq_L_plus_2=o["tokens_eq_L_plus_2"], rows_eq_L_plus_2=o["rows_eq_L_plus_2"],
                          width=o.get("width"), dtypes=o["dtypes"], devices=o["devices"],
                          pass_record=p["pass_record"], pass_record_differences=p["pass_record_differences"],
                          sequences=p["sequences"], residues=p["residues"])
            good = (o["all_finite"] and o["max_abs_emb"] < FP16_MAX and o["unk_tokens"] == 0 and o["tokens_eq_L_plus_2"]
                    and o["rows_eq_L_plus_2"] and not p["pass_record_differences"]
                    and o["proteins"] == p["sequences"])
            e[pss]["pass"] = bool(good)
            ok &= bool(good)
        e["store_index_equal_payloads"] = a["store_index_equal_payloads"]
        e["code_store_entries"] = a["code_store"]["manifest"]["entries"]
        e["query_index_entries"] = a["query_index"]["entries"]
        ok &= bool(a["store_index_equal_payloads"])
        out[d] = e
    out["pass"] = bool(ok)
    return out


def evaluate_gates(X, bits, ev, f1):
    th = X.th["thresholds"]
    G = {g["id"]: g for g in th["gates"]}
    assert G["A4-bits"]["threshold"] == {"gpu": 0.99, "cpu": 0.99, "mps": 0.99}
    assert G["A4-dlnE"]["threshold"] == {"median": 0.1, "p95": 0.5}
    assert G["A4-dF1"]["threshold"] == 0.01 and G["A4-xcpu"]["threshold"] == 0.999
    res = {}
    g = G["A4-bits"]
    cls = {"gpu": "gpu", "cpu_linux": "cpu", "cpu_mac": "cpu", "mps": "mps"}
    res["A4-bits"] = {d: dict(value=bits["code_bits_vs_research"][d]["pooled_panel_enccheck_chains_tsub"]["agreement"],
                              threshold=g["threshold"][cls[d]], comparator=">=") for d in DEVS}
    for d, v in res["A4-bits"].items():
        v["pass_"] = bool(v["value"] >= v["threshold"])
    g = G["A4-dlnE"]
    res["A4-dlnE"] = {}
    for qs in DEVS:
        for ts in DEVS:
            v = ev["P_sub"][f"{qs}__{ts}"]
            res["A4-dlnE"][f"{qs}__{ts}"] = dict(median=v["median"], p95=v["p95"], n=v["n"], threshold=g["threshold"],
                                                 kind="same-device" if qs == ts else "mixed",
                                                 pass_=bool(v["median"] <= g["threshold"]["median"]
                                                            and v["p95"] <= g["threshold"]["p95"]
                                                            and v["n"] == 831))
    g = G["A4-dF1"]
    res["A4-dF1"] = {}
    for qs in DEVS:
        for ts in DEVS:
            v = f1["configs"][f"{qs}__{ts}"]
            res["A4-dF1"][f"{qs}__{ts}"] = dict(dF1=v["dF1"], mean_f1=v["mean_f1"], threshold=g["threshold"],
                                                comparator="abs <=", kind="same-device" if qs == ts else "mixed",
                                                pass_=bool(abs(v["dF1"]) <= g["threshold"]))
    g = G["A4-xcpu"]
    res["A4-xcpu"] = {}
    if "cpu_linux_vs_cpu_mac" in bits["cross_device"]:
        v = bits["cross_device"]["cpu_linux_vs_cpu_mac"]["pooled_code_bits_panel_enccheck_chains_tsub"]["agreement"]
        res["A4-xcpu"]["cpu_mac_vs_cpu_linux"] = dict(value=v, threshold=g["threshold"], comparator=">=",
                                                      pass_=bool(v >= g["threshold"]))
    verdict = {gid: bool(d) and all(v["pass_"] for v in d.values()) for gid, d in res.items()}
    return res, verdict


CLI_FIELDS = ("query,target,embscore,raw,rawrev,q90,rawadj,evalue,bits,qstart,qend,tstart,tend,alnlen,cigar,"
              "ecal")


def cli_compare(X):
    """GPU job cross-check of the function path against the real CLI on the same node: (1) the code store that
    'fm-search build sequences --residue-codes' wrote for codes.fasta vs the function-path GPU code store, entry by
    entry; (2) the rows 'fm-search query sequences --scoring esm-codes' wrote for the 199 q200 sequences (its own
    query pass) vs production Stage 2 on the same (query, target) pairs with the function-path GPU query encodings
    (same N = the CLI database's residues), written by the same output writer; every column but embscore compared."""
    np = X.np
    d = os.path.join(A4, "gpu", "cli")
    tsv = os.path.join(d, "q200_out.tsv")
    if not os.path.exists(os.path.join(d, "db", "codes.rescodes.json")):
        return {"available": False}
    from foldmatch.search import output
    from foldmatch.search.code_alignment import CodeAlignmentPool, align_candidates_esm_codes
    from foldmatch.search.residue_code_store import ResidueCodeStore
    cli, fn = ResidueCodeStore(os.path.join(d, "db"), "codes"), X.stores["gpu"]
    keys = [k for k, _ in read_fasta(os.path.join(A4, "inputs", "codes.fasta"))]
    (ao, al), (bo, bl) = cli.lookup_rows(keys), fn.lookup_rows(keys)
    am, bm = cli.open_memmap(), fn.open_memmap()
    same = [bool(np.array_equal(am[o:o + n], bm[p:p + m])) for o, n, p, m in zip(ao.tolist(), al.tolist(),
                                                                                    bo.tolist(), bl.tolist())]
    out = {"available": True, "code_store_entries": len(keys), "code_store_entries_identical": int(sum(same)),
           "cli_manifest_build": cli.manifest()["build"], "cli_bin_sha256": sha256_file(cli.bin_path),
           "function_bin_sha256": sha256_file(fn.bin_path)}
    if not os.path.exists(tsv):
        out["query"] = None
        return out
    fields = CLI_FIELDS.split(",")
    rows = [line.rstrip("\n").split("\t") for line in open(tsv) if line.strip()]
    pre = {}
    for r in rows:
        t, sc = pre.setdefault(r[0], ([], []))
        t.append(r[1])
        sc.append(float(r[2]))
    N = int(cli.manifest()["total_residues"])
    with CodeAlignmentPool(X.workers) as pool:
        res = align_candidates_esm_codes(X.qidx["gpu"], pre, fn, fn.sequence_store().fetch, X.prepared, N, pool,
                                         query_sequences={q: X.seq_of[q] for q in pre}, max_evalue=None,
                                         output_needs=fields)
    mine = os.path.join(A4, "analysis", "cli_compare_function_path.tsv")
    os.makedirs(os.path.dirname(mine), exist_ok=True)
    output.write_aligned_results(res, fields, mine)
    rows2 = [line.rstrip("\n").split("\t") for line in open(mine) if line.strip()]
    drop = fields.index("embscore")
    key = lambda r: tuple(x for i, x in enumerate(r) if i != drop)  # noqa: E731
    a_, b_ = sorted(map(key, rows)), sorted(map(key, rows2))
    out["query"] = dict(cli_rows=len(rows), function_rows=len(rows2), queries=len(pre),
                        rows_identical_except_embscore=int(sum(x == y for x, y in zip(a_, b_))),
                        all_identical=a_ == b_, N=N, fields=fields)
    log(f"CLI cross-check: code store {sum(same)}/{len(keys)} entries identical; query rows "
        f"{out['query']['rows_identical_except_embscore']}/{len(rows)} identical (all {a_ == b_})")
    return out


def cmd_analyze(a):
    global DEVS, SRCS
    if a.srcs:                                   # a subset (dry runs); the verdict then says INCOMPLETE
        SRCS = tuple(x for x in ("research",) + ALL_DEVS if x in a.srcs)
        DEVS = tuple(x for x in SRCS if x != "research")
        assert "research" in SRCS
    C = _C()
    np = __import__("numpy")
    X = Ctx(a.workers)
    t0 = time.time()
    out = {"what": "P8 A4 end-to-end ESM3 parity of FoldMatch esm-codes-v1 (production path) vs the research encodings",
           "a4_thresholds": dict(path=X.th["path"], sha256=X.th["sha256"], adopted_sha256=C.A4_THRESHOLDS_SHA256,
                                 sha256_matches=X.th["sha256"] == C.A4_THRESHOLDS_SHA256),
           "inputs": dict(sets_json_sha256=SETS_SHA256, inputs_json_sha256=sha256_file(os.path.join(A4, "inputs",
                                                                                                   "inputs.json")),
                          files=X.inputs["files"], P_sub_pairs_sha256=PAIRS_SHA256),
           "protocol": X.th["thresholds"]["protocol"]}
    # the bundle's panel (FoldMatch Q90) is the shipped research panel
    codes, offsets = X.prepared.for_worker().load_panel()
    zp = X.R["panel"]
    out["panel_check"] = dict(bundle_panel_equals_shipped_panel_codes=bool(np.array_equal(codes, zp["all_codes"])
                                                                           and np.array_equal(offsets, zp["offsets"])),
                              codes_md5=hashlib.md5(np.ascontiguousarray(codes).tobytes()).hexdigest())
    assert out["panel_check"]["bundle_panel_equals_shipped_panel_codes"]
    out["devices"] = {}
    for s in SRCS:
        asm = read_json(os.path.join(A4, s, "assemble.json"))
        out["devices"][s] = {k: asm[k] for k in ("code_store", "query_index") if k in asm}
        if s != "research":
            out["devices"][s]["passes"] = {p: {k: v for k, v in asm["passes"][p].items() if k != "chunk_done_json"}
                                           for p in asm["passes"]}
            out["devices"][s]["chunks"] = {p: asm["passes"][p]["chunk_done_json"] for p in asm["passes"]}
    bits, _codes, _qs = analyze_bits(X)
    out.update(bits)
    ev, raw, L_ref = analyze_evalues(X)
    out["dlnE"] = ev
    out["chain_f1"] = analyze_f1(X, a.harness_f1)
    out["cli_cross_check"] = cli_compare(X)
    inv = dict(observer=observer_invariants(X), research_reproduces_stored=ev["research_reproduces_stored"],
               lnE_statistics_vs_bench_validate_formula=max(v["lnE_vs_bench_validate_formula_max_abs"]
                                                            for v in ev["P_sub"].values()),
               panel=out["panel_check"]["bundle_panel_equals_shipped_panel_codes"])
    inv["lnE_statistics_pass"] = bool(inv["lnE_statistics_vs_bench_validate_formula"] <= 1e-9)
    inv["pass"] = bool(inv["observer"]["pass"] and inv["research_reproduces_stored"]["pass"]
                       and inv["lnE_statistics_pass"] and inv["panel"])
    out["invariants"] = inv
    gates, verdict = evaluate_gates(X, bits, ev, out["chain_f1"])
    out["gates"] = gates
    complete = set(DEVS) == set(ALL_DEVS)
    ok = bool(all(verdict.values()) and inv["pass"])
    out["verdict"] = dict(gates=verdict, invariants=inv["pass"], device_classes=list(DEVS), complete=complete,
                          A4_pass=bool(ok and complete),
                          G8=("INCOMPLETE: not every device class was analysed (dry run)" if not complete else
                              "A4 PASSES: every pre-registered gate and invariant holds" if ok else
                              "A4 FAILS -> GATE G8: stop and report; remedy = v2 refit (never a threshold change)"))
    out["wall_s"] = round(time.time() - t0, 1)
    out["provenance"] = C.provenance(torch_info=False)
    p = os.path.join(A4, "analysis", f"{a.out_name}.json")
    h = write_json(out, p)
    txt = report_text(out)
    with open(os.path.join(A4, "analysis", f"{a.out_name}.txt"), "w") as fh:
        fh.write(txt + f"\na4_parity.json sha256 {h}\n")
    print(txt)
    log(f"wrote {p} sha256 {h}")


def report_text(o):
    L = [f"P8 A4 end-to-end ESM3 parity (FoldMatch {o['provenance']['foldmatch']['commit'][:7]}, production path)",
         f"A4_THRESHOLDS.json sha256 {o['a4_thresholds']['sha256']} (adopted {o['a4_thresholds']['adopted_sha256'][:8]}"
         f"..., match {o['a4_thresholds']['sha256_matches']})", ""]
    g = o["gates"]
    L.append("A4-bits (pooled panel+enccheck+chains+tsub, >= 0.990):")
    for d, v in g["A4-bits"].items():
        L.append(f"  {d:10s} {v['value']:.6f}  {'PASS' if v['pass_'] else 'FAIL'}")
    L.append("A4-xcpu (CPU Mac vs CPU Linux code bits, >= 0.999):")
    for d, v in g["A4-xcpu"].items():
        L.append(f"  {d:22s} {v['value']:.7f}  {'PASS' if v['pass_'] else 'FAIL'}")
    L.append("A4-dlnE (|d ln E| on the 831 P_sub pairs, median <= 0.1 and P95 <= 0.5):")
    for k, v in g["A4-dlnE"].items():
        L.append(f"  {k:22s} median {v['median']:.4f}  P95 {v['p95']:.4f}  (n {v['n']})  "
                 f"{'PASS' if v['pass_'] else 'FAIL'}")
    L.append("A4-dF1 (|mean F1 - 0.613393| on the 582 similar chain pairs, <= 0.01):")
    for k, v in g["A4-dF1"].items():
        L.append(f"  {k:22s} mean F1 {v['mean_f1']:.6f}  dF1 {v['dF1']:+.6f}  {'PASS' if v['pass_'] else 'FAIL'}")
    inv = o["invariants"]
    L.append("Invariants:")
    for d in DEVS:
        for pss in ("codes", "queries"):
            e = inv["observer"][d][pss]
            L.append(f"  {d:10s} {pss:8s} proteins {e['proteins']:5d} finite {e['all_finite']} max|emb| "
                     f"{e['max_abs_emb']:.0f} unk {e['unk_tokens']} tokens=L+2 {e['tokens_eq_L_plus_2']} rows=L+2 "
                     f"{e['rows_eq_L_plus_2']} {e['pass_record']['accelerator']}/{e['pass_record']['esm_precision']}"
                     f"  {'PASS' if e['pass'] else 'FAIL'}")
    r = inv["research_reproduces_stored"]
    L.append(f"  research encodings via FoldMatch: |dQ90| max {r['q90_max_abs']:.2e}, S_f rel {r['Sf_max_rel']:.2e}, "
             f"S_r rel {r['Sr_max_rel']:.2e}, |d lnE| max {r['lnE_max_abs']:.2e}  {'PASS' if r['pass'] else 'FAIL'}")
    L.append(f"  P4 lnE vs the bench_validate formula: max |d| {inv['lnE_statistics_vs_bench_validate_formula']:.1e}; "
             f"bundle panel == shipped panel codes: {inv['panel']}")
    L.append("")
    L.append(f"VERDICT: {o['verdict']['G8']}")
    return "\n".join(L)


# =====================================================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inputs")
    e = sub.add_parser("embed")
    e.add_argument("--dev", required=True, choices=DEVS)
    e.add_argument("--pass", dest="pass_", required=True, choices=tuple(PASSES))
    e.add_argument("--chunk", type=int, required=True)
    e.add_argument("--nchunks", type=int, required=True)
    e.add_argument("--threads", type=int, default=0, help="set the thread variables (0: platform defaults)")
    s = sub.add_parser("assemble")
    s.add_argument("--devs", nargs="+", required=True, choices=SRCS)
    z = sub.add_parser("analyze")
    z.add_argument("--workers", type=int, default=32)
    z.add_argument("--harness-f1", action="store_true", help="also chain_eval.evaluate (P0.5 method) per config")
    z.add_argument("--srcs", nargs="+", choices=SRCS, help="analyse only these sources (dry run)")
    z.add_argument("--out-name", default="a4_parity", help="output basename in A4/analysis/")
    a = ap.parse_args()
    {"inputs": cmd_inputs, "embed": cmd_embed, "assemble": cmd_assemble, "analyze": cmd_analyze}[a.cmd](a)


if __name__ == "__main__":
    main()
