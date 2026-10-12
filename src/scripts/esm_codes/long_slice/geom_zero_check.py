#!/usr/bin/env python
"""P0.7 memory follow-up (foldmatch env, FoldMatch imported read-only): is ESM3's geometric attention an exact no-op in
FoldMatch's sequence-only path, so that switching it off (as the research store path does) would remove the
O(256 x L^2 x 3) tensor that sets FoldMatch's practical length limit?

FoldMatch path = foldmatch.utils.esm.loaders.esm_open(device) + forward_and_sample(encode(ESMProtein(seq)),
SamplingConfig(return_per_residue_embeddings=True)).per_residue_embedding (ChainCompleteModule.predict_step).
'geom off' = the same call with block.use_geom_attn = False on the blocks that have it (block 0), restored afterwards.
For each length: per_residue_embedding with geom on vs off -> bit-identical?, max |diff|; peak memory of each.
Then the geom-off ladder above the geom-on limit (peak memory, seconds).
    python geom_zero_check.py cuda:0|cpu     -> results/geom_zero_<dev>.<job>.json
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cuda:0")
CMP = [int(x) for x in os.environ.get("CMP", "300,1000,2000").split(",")]
LADDER = [int(x) for x in os.environ.get("LADDER", "3000,4000,6000,8000,12000,16000").split(",")]
DEADLINE = float(os.environ.get("DEADLINE", "0"))


def pool_sequence():
    return "".join(line.strip() for line in open(os.path.join(HERE, "long_t.fasta")) if not line.startswith(">"))


def peak():
    if DEV.type == "cuda":
        return round(torch.cuda.max_memory_allocated(DEV) / 2 ** 30, 3)
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20, 3)


def main():
    import foldmatch
    from esm.sdk.api import ESMProtein, SamplingConfig
    from foldmatch.utils.esm.loaders import esm_open
    model = esm_open(DEV)
    geom = [b for b in model.transformer.blocks if getattr(b, "use_geom_attn", False)]
    out = {"device": str(DEV), "foldmatch": foldmatch.__file__, "torch": torch.__version__, "n_geom_blocks": len(geom),
           "compare": [], "ladder_geom_off": []}
    if DEV.type == "cuda":
        out["gpu"] = torch.cuda.get_device_name(DEV)

    @torch.inference_mode()
    def run(seq, on):
        for b in geom:
            b.use_geom_attn = on
        try:
            if DEV.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(DEV)
                torch.cuda.synchronize(DEV)
            t = time.time()
            e = model.forward_and_sample(model.encode(ESMProtein(sequence=seq)),
                                         SamplingConfig(return_per_residue_embeddings=True)).per_residue_embedding
            if DEV.type == "cuda":
                torch.cuda.synchronize(DEV)
            return e.float().cpu(), round(time.time() - t, 3), peak()
        finally:
            for b in geom:
                b.use_geom_attn = True

    s = pool_sequence()
    for L in CMP:
        run(s[:L], True)                                  # warm-up
        a, ta, pa = run(s[:L], True)
        b, tb, pb = run(s[:L], False)
        rec = dict(L=L, bit_identical=bool(torch.equal(a, b)), max_abs_diff=float((a - b).abs().max()),
                   seconds_on=ta, seconds_off=tb, peak_on_gib=pa, peak_off_gib=pb)
        out["compare"].append(rec)
        print(rec, flush=True)
    for L in LADDER:
        if DEADLINE and time.time() > DEADLINE:
            break
        try:
            run(s[:L], False)
            _, t, p = run(s[:L], False)
            rec = dict(L=L, seconds=t, peak_gib=p)
        except torch.cuda.OutOfMemoryError as ex:
            rec = dict(L=L, error="OOM", message=str(ex).split("\n")[0][:300])
            torch.cuda.empty_cache()
            out["ladder_geom_off"].append(rec)
            print(rec, flush=True)
            break
        out["ladder_geom_off"].append(rec)
        print(rec, flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(out, open(os.path.join(HERE, "results", f"geom_zero_{DEV.type}.{os.environ.get('SLURM_JOB_ID', 'local')}.json"),
                        "w"), indent=1)


if __name__ == "__main__":
    main()
