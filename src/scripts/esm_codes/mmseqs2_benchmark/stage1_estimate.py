#!/usr/bin/env python
"""Estimate of the Stage-2 pair count of the FoldMatch benchmark queries (--top-k 10000 --threshold 0.7) from an
earlier FoldMatch run on the same queries and targets (NERSC; job 56276866, 2026-07-21: the BLOSUM62 database
fm-db/db/mmseqs-targetdb-lt2000, --top-k 4096 --threshold 0.7, 17,532,906 rows, default output columns:
query,target,embscore,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits).

  python stage1_estimate.py [--tsv FILE] [--top-k-old 4096] [--top-k-new 10000] [--threshold 0.7] [--out FILE]

Per query the old run holds min(n_q, 4096) candidates with embscore >= 0.7 (n_q = candidates above the threshold).
For a query at the 4096 cap, the embedding score is modelled as linear in ln(rank) between ranks 2048 and 4096 and
extrapolated to the rank where it reaches the threshold (capped at the new top-k). Bounds: no growth (17.5 M) and
every capped query at the new top-k. Assumes the new database (targetdb-lt2000-codes) has the same Stage-1
behaviour (same embedding model, index type ivf_pq and nprobe) as the old one.
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_common as BC  # noqa: E402

OLD = ("/pscratch/sd/j/jsegura/fm-db/results/query-lt2000-vs-mmseqs-targetdb-lt2000-thr-7-top-k-4096-ouputs-2.tsv")
COLS = "query,target,embscore,fident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits".split(",")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tsv", default=OLD)
    ap.add_argument("--top-k-old", type=int, default=4096)
    ap.add_argument("--top-k-new", type=int, default=10000)
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--out", default=os.path.join(BC.EVAL_ROOT, "stage2_estimate.json"))
    args = ap.parse_args()
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pcsv
    tab = pcsv.read_csv(args.tsv, read_options=pcsv.ReadOptions(column_names=COLS, block_size=1 << 26),
                        parse_options=pcsv.ParseOptions(delimiter="\t", quote_char=False),
                        convert_options=pcsv.ConvertOptions(include_columns=["query", "embscore"],
                                                            column_types={"query": pa.string(),
                                                                          "embscore": pa.float64()}))
    d = pc.dictionary_encode(tab["query"]).unify_dictionaries()
    q = np.concatenate([c.indices.to_numpy(zero_copy_only=False) for c in d.chunks]).astype(np.int64)
    s = tab["embscore"].to_numpy()
    nq_seen = int(q.max()) + 1
    o = np.lexsort((-s, q))
    q, s = q[o], s[o]
    starts = np.searchsorted(q, np.arange(nq_seen + 1))
    cnt = np.diff(starts)
    L = BC.Labels.load() if os.path.exists(os.path.join(BC.LABELS_DIR, "labels.npz")) else None
    n_all = L.nq if L is not None else 6294
    capped = np.flatnonzero(cnt >= args.top_k_old)
    est = cnt.astype(np.float64).copy()
    for i in capped:
        a, b = starts[i], starts[i + 1]
        s_half, s_cap = s[a + args.top_k_old // 2 - 1], s[a + args.top_k_old - 1]
        slope = (s_half - s_cap) / math.log(2.0)
        if s_cap < args.threshold:
            r = args.top_k_old
        elif slope <= 0:
            r = args.top_k_new
        else:
            r = args.top_k_old * math.exp((s_cap - args.threshold) / slope)
        est[i] = min(args.top_k_new, max(args.top_k_old, r))
    out = dict(source=args.tsv, rows_old=int(len(q)), queries_with_candidates=int((cnt > 0).sum()),
               queries_total=n_all, queries_without_candidates=n_all - int((cnt > 0).sum()),
               queries_at_old_cap=int(capped.size), threshold=args.threshold, top_k_old=args.top_k_old,
               top_k_new=args.top_k_new,
               pairs_lower_bound=int(cnt.sum()), pairs_upper_bound=int(cnt.sum() + capped.size *
                                                                      (args.top_k_new - args.top_k_old)),
               pairs_estimate=int(round(est.sum())), queries_estimated_at_new_cap=int((est >= args.top_k_new).sum()),
               estimate_quantiles={str(p): float(np.quantile(est, p)) for p in (0.1, 0.5, 0.9)},
               emb_score_at_old_cap_quantiles={str(p): float(np.quantile(s[starts[capped] + args.top_k_old - 1], p))
                                               for p in (0.1, 0.5, 0.9)} if capped.size else None)
    BC.write_json(out, args.out)
    print(out)


if __name__ == "__main__":
    main()
