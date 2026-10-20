#!/usr/bin/env python
"""P8 A8 follow-up: how many exhaustive esm-codes hits survive the Stage-1 prefilter at other thresholds and top_k.

Uses the exhaustive A5 run (every benchmark query against all 50,000 targets, FoldMatch production encodings) and the
exact chain-embedding cosine matrix written next to it. A pair reaches Stage 2 when its exact cosine is >= the threshold
and its exact rank among the query's targets is <= top_k (an exact search: the HNSW misses measured in A8, 63 of
16,435 TPs at E <= 1, are not modelled). For each setting it reports the Stage-2 work, the TPs and FPs kept at the
pipeline's E (shipped table, N = all database residues) and the research protocol's sensitivity at 1e-3 and FP ratio.

    python a8_threshold_curve.py --a5-dir /pscratch/sd/j/jsegura/embcomp/p8/A5/foldmatch --out curve.json
"""
from __future__ import annotations

import argparse
import itertools
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acc_common as AC  # noqa: E402

THRESHOLDS = (0.8, 0.7, 0.6, None)        # None: no threshold
TOP_KS = (100, 10_000, None)              # None: every target


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a5-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nboot", type=int, default=200)
    a = ap.parse_args(argv)
    t0 = time.time()
    B = AC.Bench()
    tabs = AC.ship_tables()
    z = np.load(os.path.join(a.a5_dir, "matrices.npz"))
    assert list(z["qids"].astype(str)) == list(B.qids) and list(z["tids"].astype(str)) == list(B.tids)
    S = z["S_f"].astype(np.float64) - z["S_r"].astype(np.float64)
    q90 = z["q90"]
    L = AC.ln_evalue(tabs["all"], B.qlen, S, q90, B.N_all)          # the pipeline's E
    cos = np.load(os.path.join(a.a5_dir, "cos.npz"))["cos"]
    order = np.argsort(-cos, axis=1, kind="stable")
    rank = np.empty_like(order)
    rank[np.arange(B.nq)[:, None], order] = np.arange(1, B.nt + 1)[None, :]
    del order

    rows = []
    for thr, k in itertools.product(THRESHOLDS, TOP_KS):
        keep = np.ones(S.shape, bool)
        if thr is not None:
            keep &= cos >= thr
        if k is not None:
            keep &= rank <= k
        row = dict(threshold=thr, top_k=k, stage2_pairs=int(keep.sum()),
                   stage2_pairs_per_query=float(keep.sum() / B.nq),
                   queries_without_candidates=int((keep.sum(1) == 0).sum()))
        for x in (1e-3, 1.0):
            hit = (L <= math.log(x)) & keep
            row[f"E<={x:g}"] = dict(tp=int((hit & B.TP).sum()), tp_exhaustive=int(((L <= math.log(x)) & B.TP).sum()),
                                    fp_real=int((hit & B.REAL).sum()), fp_reversed=int((hit & B.REVM).sum()),
                                    queries_with_tp=int((hit & B.TP).any(1).sum()))
            row[f"E<={x:g}"]["tp_kept_fraction"] = row[f"E<={x:g}"]["tp"] / max(row[f"E<={x:g}"]["tp_exhaustive"], 1)
        Sm = np.where(keep, S, -np.inf)
        p = AC.bench_protocol(B, Sm, tabs, q90, nboot=a.nboot)
        row["research_protocol"] = dict(
            sens_1e3={sp: p["sensitivity"][sp]["sens_1e3"] for sp in ("main", "swap")},
            fp_ratio_E1={e: p[e]["1.0"]["ratio"] for e in ("main", "swap", "heldout_union")},
            fp_ratio_E001={e: p[e]["0.01"]["ratio"] for e in ("main", "swap", "heldout_union")})
        rows.append(row)
        r1, r3 = row["E<=1"], row["E<=0.001"]
        print(f"thr {thr!s:>4} top_k {k!s:>6}: pairs/query {row['stage2_pairs_per_query']:9.1f}  no-cand "
              f"{row['queries_without_candidates']:4d}  TP@1 {r1['tp']:6d} ({r1['tp_kept_fraction']:.3f})  "
              f"TP@1e-3 {r3['tp']:5d} ({r3['tp_kept_fraction']:.3f})  FP@1 {r1['fp_real']:4d}/{r1['fp_reversed']:3d}  "
              f"sens@1e-3 {row['research_protocol']['sens_1e3']['main']:.4f}/{row['research_protocol']['sens_1e3']['swap']:.4f}",
              flush=True)
    res = dict(what="A8 follow-up: Stage-1 threshold / top_k curve from the exhaustive A5 run (exact search)",
               a5_dir=os.path.abspath(a.a5_dir), nboot=a.nboot, rows=rows,
               provenance=AC.provenance(kernel=False), seconds=round(time.time() - t0, 1))
    AC.write_json(res, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
