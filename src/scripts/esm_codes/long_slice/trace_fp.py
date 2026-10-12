#!/usr/bin/env python
"""P0.7 diagnostic (Mac): where do the real-null false positives align on the long protein?

For every real-null FP pair at E <= 1 written by analyze_long.py (fp_pairs_lqxn.npy: long query x A5 annotated target,
union of the clamp and extrap3000 rules; fp_pairs_nqxlt.npy: A5 query x long annotated target, clamp), the forward
setting-D alignment is traced (swfast.swz_ad_batch tb=1, local NEON build of the same swkern.c) and the aligned residues
are compared with the SCOP-annotated ranges in the FASTA headers ('|target ranges' field; for A5 queries in
MMseqs2-benchmark format the same field, SCOP25 domains are fully annotated). A hit whose aligned residues lie >= 50%
inside annotated ranges on BOTH sides is a 'clean' FP (two SCOP domains of different folds); a hit lying mostly in an
UNANNOTATED part of the long protein may be homology the SCOP annotation missed (label noise that grows with length,
because annotation coverage falls from 0.65 (300-600 aa) to 0.16 (2,500-3,000 aa)).
Output: fp_location.json
"""
from __future__ import annotations

import json
import re
import sys

import numpy as np

SP = "/private/tmp/claude-501/-Users-joan-devel-rcsb-embedding-search/d9266e69-77ea-4121-a428-2e6f3bcc4ca2/scratchpad"
HERE = f"{SP}/p0/long"
sys.path.insert(0, f"{SP}/embcomp/throughput/build")
import swfast as F  # noqa: E402

Q_EDGES = [300, 600, 1000, 1500, 2000, 2500, 3001]
QBL = ["300-600", "600-1000", "1000-1500", "1500-2000", "2000-2500", "2500-3000"]
T_EDGES = [1000, 1500, 2000, 2500, 3001]
TBL = ["1000-1500", "1500-2000", "2000-2500", "2500-3000"]


def read_fasta(path):
    recs, h, s = [], None, []
    for line in open(path):
        line = line.rstrip("\n\r")
        if line.startswith(">"):
            if h is not None:
                recs.append((h, "".join(s)))
            h, s = line[1:], []
        elif line:
            s.append(line.strip())
    recs.append((h, "".join(s)))
    return recs


def ann_mask(h, L):
    m = np.zeros(L, bool)
    parts = h.split("|")
    if " |" not in h:                       # SCOP25 domain query: the whole sequence is the domain
        m[:] = True
        return m
    if len(parts) >= 5:
        for t in parts[-1].split():
            g = re.fullmatch(r"(\d+)-(\d+)", t)
            if g:
                m[max(int(g.group(1)), 1) - 1:min(int(g.group(2)), L)] = True
    return m


def trace(qp, codes):
    T = F.decode_unpackbits(codes)
    sim = np.ascontiguousarray(qp @ T.T, dtype=np.float32)
    sc, ends, path, poff = F.swz_ad_batch(sim, np.array([0, len(codes)], np.int64), 1.0, 5.0, 0.5, tb=True)
    p = path[poff[0]:poff[1]]
    p = p[(p[:, 0] >= 0) & (p[:, 1] >= 0)]
    return float(sc[0]), p


def run(pairs, qset, tset, long_side, bins_of, labels):
    Qp, Qoff, Qh = qset
    Tc, Toff, Th = tset
    rows = []
    for i, j in pairs:
        qp = F.query_prep(Qp[Qoff[i]:Qoff[i + 1]].astype(np.float32))
        sc, p = trace(qp, Tc[Toff[j]:Toff[j + 1]])
        qm = ann_mask(Qh[i], Qoff[i + 1] - Qoff[i])
        tm = ann_mask(Th[j], Toff[j + 1] - Toff[j])
        fq = float(qm[p[:, 0]].mean()) if len(p) else 0.0
        ft = float(tm[p[:, 1]].mean()) if len(p) else 0.0
        rows.append(dict(q=int(i), t=int(j), score=sc, n_aligned=int(len(p)), q_in_ann=fq, t_in_ann=ft,
                         q_cov=float(qm.mean()), t_cov=float(tm.mean()), bin=int(bins_of(i, j))))
    out = {}
    for k, g in enumerate(labels):
        rr = [r for r in rows if r["bin"] == k]
        if not rr:
            continue
        side_in = np.array([r["q_in_ann"] if long_side == "q" else r["t_in_ann"] for r in rr])
        side_cov = np.array([r["q_cov"] if long_side == "q" else r["t_cov"] for r in rr])
        both = np.array([(r["q_in_ann"] >= 0.5) and (r["t_in_ann"] >= 0.5) for r in rr])
        out[g] = dict(n_fp=len(rr), long_side_median_in_annotation=float(np.median(side_in)),
                      long_side_frac_ge50_in_annotation=float((side_in >= 0.5).mean()),
                      long_side_mean_coverage=float(side_cov.mean()),
                      clean_both_sides_ge50=float(both.mean()), median_aligned=float(np.median([r["n_aligned"] for r in rr])))
    return out, rows


def main():
    zq = np.load(f"{HERE}/enc/out/long_q.L2m_B8_s0.npz")
    zt = np.load(f"{HERE}/enc/out/long_t.L2m_B8_s0.npz")
    zbt = np.load(f"{HERE}/enc/bench_targets.L2m_B8_s0.npz")
    zbq = np.load(f"{SP}/embcomp/evalue/encode/out/queries.L2m_B8_s0.npz")
    ix = np.load(f"{HERE}/out/index.npz")
    lqh = [h for h, _ in read_fasta(f"{HERE}/long_q.fasta")]
    lth = [h for h, _ in read_fasta(f"{HERE}/long_t.fasta")]
    qh = [h for h, _ in read_fasta(f"{SP}/calib3di/results/queries.fasta")]
    th = [h for h, _ in read_fasta(f"{SP}/calib3di/results/targets.fasta")]
    ben = ix["ben_cols"]
    lq_len = ix["lq_len"]
    long_len = ix["long_len"]
    res = {}
    p1 = np.load(f"{HERE}/fp_pairs_lqxn.npy")
    # columns of lq_ben -> benchmark target rows
    pairs = [(int(i), int(ben[j])) for i, j in p1]
    res["LQxN_real_E1"], rows1 = run(pairs, (zq["q"], zq["offsets"], lqh), (zbt["codes"], zbt["offsets"], th), "q",
                                     lambda i, j: np.digitize(lq_len[i], Q_EDGES[1:-1]), QBL)
    p2 = np.load(f"{HERE}/fp_pairs_nqxlt.npy")
    pairs = [(int(i), int(j)) for i, j in p2]        # long_cols index j < 2000 = long_t row j
    assert all(j < len(lth) for _, j in pairs)
    res["NQxLT_real_E1_clamp"], rows2 = run(pairs, (zbq["q"], zbq["offsets"], qh), (zt["codes"], zt["offsets"], lth),
                                            "t", lambda i, j: np.digitize(long_len[j], T_EDGES[1:-1]), TBL)
    res["definition"] = __doc__.strip()
    json.dump(res, open(f"{HERE}/fp_location.json", "w"), indent=1)
    json.dump({"LQxN": rows1, "NQxLT": rows2}, open(f"{HERE}/fp_location_pairs.json", "w"))
    for k in ("LQxN_real_E1", "NQxLT_real_E1_clamp"):
        print(k)
        for g, v in res[k].items():
            print(f"  {g:10s} {json.dumps(v)}")


if __name__ == "__main__":
    main()
