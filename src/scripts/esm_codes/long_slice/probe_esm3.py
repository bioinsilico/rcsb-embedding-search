#!/usr/bin/env python
"""P0.7: ESM3 practical maximum length and memory, one sequence at a time, length ladder until the first OOM.

Sequences: real residues, deterministic: the proteins of long_t.fasta concatenated in file order; length L = the first
L letters (chimeras above 3,000 aa; composition is that of real proteins). Each length is run twice (first run = warm-up
for kernels / allocator); the second run is timed. Peak = torch.cuda.max_memory_allocated / reserved after
reset_peak_memory_stats; CPU runs record ru_maxrss (monotone, lengths increase) and the wall time.

    python probe_esm3.py research  [cuda|cpu]   # esm3 env: SequenceOnlyESM3 (research store path: bf16 weights on GPU,
                                                #   bf16 autocast, geometric attention OFF = exact zero, batched forward,
                                                #   B = 1) AND SequenceOnlyESM3.reference() (ESM SDK forward_and_sample,
                                                #   geometric attention ON)
    python probe_esm3.py foldmatch [cuda|cpu]   # foldmatch env: the FoldMatch path itself, imported read-only:
                                                #   foldmatch.utils.esm.loaders.esm_open(device) (fp32 weights) and
                                                #   model.forward_and_sample(model.encode(ESMProtein(seq)),
                                                #   SamplingConfig(return_per_residue_embeddings=True)).per_residue_embedding
                                                #   exactly as ChainCompleteModule.predict_step (bf16 autocast inside
                                                #   ESM3.logits on CUDA; plain fp32 on CPU)
Env: LENGTHS (comma list), DEADLINE (epoch s). Output: results/probe_<mode>_<device>.<job>.json
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
MODE = sys.argv[1]
DEV = torch.device(sys.argv[2] if len(sys.argv) > 2 else ("cuda:0" if torch.cuda.is_available() else "cpu"))
DEFAULT = "500,1000,1500,2000,2500,3000,3500,4000,5000,6000,8000,10000,12000,16000"
LENGTHS = [int(x) for x in os.environ.get("LENGTHS", DEFAULT).split(",")]
DEADLINE = float(os.environ.get("DEADLINE", "0"))
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def pool_sequence():
    s, parts = None, []
    for line in open(os.path.join(HERE, "long_t.fasta")):
        if not line.startswith(">"):
            parts.append(line.strip())
    s = "".join(parts)
    return s


def sync():
    if DEV.type == "cuda":
        torch.cuda.synchronize(DEV)


def measure(fn, seq):
    rec = {}
    for rep in range(2):
        if DEV.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(DEV)
        sync()
        t = time.time()
        out = fn(seq)
        sync()
        dt = time.time() - t
        assert out.shape == (len(seq), 1536), out.shape
        fin = bool(torch.isfinite(out).all())
        rec = dict(seconds=round(dt, 3), finite=fin, max_abs=float(out.abs().max()))
        if DEV.type == "cuda":
            rec["peak_alloc_gib"] = round(torch.cuda.max_memory_allocated(DEV) / 2 ** 30, 3)
            rec["peak_reserved_gib"] = round(torch.cuda.max_memory_reserved(DEV) / 2 ** 30, 3)
        else:
            rec["maxrss_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 3)
        del out
        if DEV.type == "cpu":
            break                                     # CPU: one run (fp32 is slow); maxrss is process-wide anyway
    return rec


def ladder(name, fn, seqpool, out):
    res = []
    for L in LENGTHS:
        if DEADLINE and time.time() > DEADLINE:
            res.append(dict(L=L, skipped="deadline"))
            break
        seq = seqpool[:L]
        try:
            rec = measure(fn, seq)
            rec["L"] = L
            res.append(rec)
            log(name, rec)
        except torch.cuda.OutOfMemoryError as ex:
            rec = dict(L=L, error="OOM", message=str(ex).split("\n")[0][:400])
            if DEV.type == "cuda":
                rec["peak_alloc_gib_before_oom"] = round(torch.cuda.max_memory_allocated(DEV) / 2 ** 30, 3)
            res.append(rec)
            log(name, rec)
            torch.cuda.empty_cache()
            break
        except RuntimeError as ex:                     # CPU allocation failure or other
            res.append(dict(L=L, error=type(ex).__name__, message=str(ex)[:400]))
            log(name, res[-1])
            break
    out[name] = res


def main():
    seqpool = pool_sequence()
    assert len(seqpool) >= max(LENGTHS)
    out = {"mode": MODE, "device": str(DEV), "torch": torch.__version__, "lengths": LENGTHS,
           "host": os.uname().nodename, "job": os.environ.get("SLURM_JOB_ID")}
    if DEV.type == "cuda":
        p = torch.cuda.get_device_properties(DEV)
        out["gpu"] = dict(name=p.name, total_gib=round(p.total_memory / 2 ** 30, 2))
    else:
        out["cpu_threads"] = torch.get_num_threads()
    import esm
    out["esm"] = esm.__version__
    t = time.time()
    if MODE == "research":
        sys.path.insert(0, "/global/homes/j/jsegura/devel/rcsb-embedding-search/src/scripts")
        sys.path.insert(0, "/global/homes/j/jsegura/devel/rcsb-embedding-search/src")
        import esm3_sequence_embeddings as S3
        emb = S3.SequenceOnlyESM3(DEV, precision="bf16")
        out["load_s"] = round(time.time() - t, 1)
        out["weights_dtype"] = str(next(emb.model.parameters()).dtype)
        ladder("research_batched_geom_off", lambda s: emb([s])[0][0], seqpool, out)
        ladder("research_reference_sdk_geom_on", lambda s: emb.reference(s), seqpool, out)
    elif MODE == "foldmatch":
        import foldmatch
        from esm.sdk.api import ESMProtein, SamplingConfig
        from foldmatch.utils.esm.loaders import esm_open
        out["foldmatch"] = dict(version=getattr(foldmatch, "__version__", None), file=foldmatch.__file__)
        model = esm_open(DEV)
        out["load_s"] = round(time.time() - t, 1)
        out["weights_dtype"] = str(next(model.parameters()).dtype)

        @torch.inference_mode()
        def fm(s):
            return model.forward_and_sample(model.encode(ESMProtein(sequence=s)),
                                            SamplingConfig(return_per_residue_embeddings=True)).per_residue_embedding[1:-1].float()
        ladder("foldmatch_forward_and_sample", fm, seqpool, out)
    else:
        raise SystemExit(__doc__)
    out["wall_s"] = round(time.time() - T0, 1)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    fn = os.path.join(HERE, "results", f"probe_{MODE}_{DEV.type}.{os.environ.get('SLURM_JOB_ID', 'local')}.json")
    json.dump(out, open(fn, "w"), indent=1)
    log("wrote", fn)


if __name__ == "__main__":
    main()
