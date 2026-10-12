#!/usr/bin/env python
"""P0.7 long-sequence calibration slice, step 4 (Mac): false-positive ratios under the shipped E-value system with the
high-m rule CLAMP (shipped: np.interp in ln m, clamped outside the 6 centres) vs EXTRAPOLATE (lambda and lnK continued
linearly in ln m beyond the last centre 594.918 with the slope of the last segment 276.49 -> 594.918, ln m capped at
966 = the calibrated-domain limit, or at 3,000 = the longest length this slice validates). Below 32.584: clamp (S11).

Score and statistics exactly as S9-S11 / bench_validate.py: S = fp64(S_f) - fp64(S_r) (setting D), Q90 = np.quantile(S
over the 1,000 SHIPPED panel members, 0.9) (long queries: computed here from out/lq_pan_*; benchmark queries: the
shipped panel/fit2/results/bench_q90_shipped.npy), lnE = lnK(m) + ln m + ln N - lambda(m) (S - Q90), m = query length,
N = residues of the evaluated column set's null population (A5 'own' protocol: annotated residues for the real null;
decoy residues for the reversed null). FP ratio at x = (# null pairs with E <= x) / (n_queries * x).
Null (A5 = evalue/score/plan.py rules): real null = annotated target, no shared SCOP fold (a.b), pair not excepted
(both b.67-70 propellers, or either side class e); reversed null = every decoy (homology-free by construction).
CIs: 2,000 cluster-bootstrap resamples of queries (percentile 95%); clusters = connected components of queries sharing a
SCOP fold (A5's rule) for the benchmark queries, and SCOP fold-set keys (identical set of folds = same architecture) for
the long queries (their fold components are reported too); plus query bootstrap and exact Poisson (Garwood) intervals.

SLICES
  LQxN   long_q (6 length bins; [300,600) and [600,1000) = population-matched baseline) x the 25,000 A5 annotated targets
  LQxR   long_q x 5,000 A5 reversed decoys (homology-free control of the length effect)
  NQxLT  the 1,465 A5 queries x 2,000 long annotated targets (per target-length bin, N = bin residues)
  NQxLR  the 1,465 A5 queries x 2,000 long reversed decoys (per bin)
  A5     the 1,465 A5 queries x the 25,000 annotated targets (reference; reproduces bench_validate all_insample) and x
         the same 5,000 decoys; m > 594.918 subset (58 queries) under each rule
DIAGNOSTICS  coverage strata (SCOP-annotated fraction of the long protein), censored-Gumbel refit per long-query bin
  (rlib.fit_gumbel, q = 0.999, a = m n; descriptive only, nothing is refitted for shipping), location of the long-side
  alignment of real-null FPs relative to the SCOP-annotated ranges (forward traceback, swfast NEON build).
Output: long_slice.json (and long_slice.txt), this directory.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter

import numpy as np
from scipy.stats import chi2

SP = "/private/tmp/claude-501/-Users-joan-devel-rcsb-embedding-search/d9266e69-77ea-4121-a428-2e6f3bcc4ca2/scratchpad"
EMB = f"{SP}/embcomp"
HERE = f"{SP}/p0/long"
OUTD = f"{HERE}/out"
SHIP = f"{EMB}/panel/fit2/results/ship_final.json"
Q90N = f"{EMB}/panel/fit2/results/bench_q90_shipped.npy"
BSCORE = f"{EMB}/evalue/score"
QFA = f"{SP}/calib3di/results/queries.fasta"
TFA = f"{SP}/calib3di/results/targets.fasta"
KEY = "D_fmr/pnl_uniref50/M3t999"
XS = (1.0, 0.1, 0.01, 0.001)
NB = 2000
Q_EDGES = [300, 600, 1000, 1500, 2000, 2500, 3001]
T_EDGES = [1000, 1500, 2000, 2500, 3001]
QBL = ["300-600", "600-1000", "1000-1500", "1500-2000", "2000-2500", "2500-3000"]
TBL = ["1000-1500", "1500-2000", "2000-2500", "2500-3000"]
RULES = ("clamp", "extrap966", "extrap3000")
CAPS = {"extrap966": 966.0, "extrap3000": 3000.0, "lamonly3000": 3000.0}
sys.path.insert(0, f"{EMB}/evalue/reconcile")
import rlib as R  # noqa: E402  (fit_gumbel only)
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


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


def sccs_of(h):                        # calib3di/auc1/reconcile/auc_final.sccs_of
    toks = h.split()
    if toks[0].startswith("rev_") or toks[0].startswith("lrev_"):
        return []
    if " |" in h:
        out = []
        for t in toks[1:]:
            if t.startswith("|"):
                break
            out.append(t)
        return sorted(set(out))
    return [toks[1]]


def ranges_of(h):
    parts = h.split("|")
    rr = []
    if len(parts) >= 5:
        for t in parts[-1].split():
            m = re.fullmatch(r"(\d+)-(\d+)", t)
            if m:
                rr.append((int(m.group(1)), int(m.group(2))))
    return rr


def lvl(s, n):
    return ".".join(s.split(".")[:n])


def null_mask(qs, ts, t_ann, prop_re=r"^b\.(67|68|69|70)\."):
    """A5 real null for query sccs lists qs x target sccs lists ts (t_ann: annotated targets). prop_re = the propeller
    exception group (A5: b.67-70; the sensitivity analysis adds b.66, the 4-bladed propeller)."""
    qd = [set(lvl(s, 2) for s in l_) for l_ in qs]
    td = [set(lvl(s, 2) for s in l_) for l_ in ts]
    inv = {}
    for j, s in enumerate(td):
        for x in s:
            inv.setdefault(x, []).append(j)
    inv = {k: np.array(v) for k, v in inv.items()}
    FOLD = np.zeros((len(qs), len(ts)), bool)
    for i, s in enumerate(qd):
        for x in s:
            if x in inv:
                FOLD[i, inv[x]] = True
    prop = lambda l_: any(re.match(prop_re, s) for s in l_)
    cle = lambda l_: any(re.match(r"^e\.", s) for s in l_)
    qp = np.array([prop(l_) for l_ in qs]); tp = np.array([prop(l_) for l_ in ts])
    qe = np.array([cle(l_) for l_ in qs]); te = np.array([cle(l_) for l_ in ts])
    EXC = (qp[:, None] & tp[None, :]) | qe[:, None] | te[None, :]
    return (~FOLD & ~EXC) & t_ann[None, :], FOLD


def params(tab, m, rule):
    lc = np.log(np.asarray(tab["center_len"], np.float64))
    lam = np.asarray(tab["lam"], np.float64)
    lnK = np.asarray(tab["lnK"], np.float64)
    lm = np.log(np.asarray(m, np.float64))
    a, k = np.interp(lm, lc, lam), np.interp(lm, lc, lnK)
    if rule != "clamp":
        hi = lm > lc[-1]
        lmc = np.minimum(lm, math.log(CAPS[rule]))
        sl = (lam[-1] - lam[-2]) / (lc[-1] - lc[-2])
        sk = (lnK[-1] - lnK[-2]) / (lc[-1] - lc[-2])
        a = np.where(hi, lam[-1] + sl * (lmc - lc[-1]), a)
        if rule != "lamonly3000":                   # exploratory (post hoc): lambda extrapolated, lnK clamped
            k = np.where(hi, lnK[-1] + sk * (lmc - lc[-1]), k)
    return a, k


def lnE(S, q90, m, N, tab, rule):
    lam, lnK = params(tab, m, rule)
    return (lnK + np.log(m) + math.log(N))[:, None] - lam[:, None] * (S - q90[:, None])


def ci(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v.size else [None, None]


def garwood(k, denom):
    lo = chi2.ppf(0.025, 2 * k) / 2 if k > 0 else 0.0
    hi = chi2.ppf(0.975, 2 * k + 2) / 2
    return [float(lo / denom), float(hi / denom)]


def boot_idx(groups, nb, seed):
    ug = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in ug]
    rng = np.random.default_rng(seed)
    return [np.concatenate([members[k] for k in rng.integers(0, len(ug), len(ug))]) for _ in range(nb)]


def components(folds):
    par = {}

    def find(a):
        par.setdefault(a, a)
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for fs in folds:
        fs = sorted(fs)
        for f in fs[1:]:
            par[find(f)] = find(fs[0])
    roots = [find(sorted(fs)[0]) for fs in folds]
    u = {r: k for k, r in enumerate(sorted(set(roots)))}
    return np.array([u[r] for r in roots])


def summarize(counts, x, bc, bq, bins=None, labels=None):
    """counts: per-query FP counts at threshold x. bins: per-query bin index (optional)."""
    c = np.asarray(counts)
    n = len(c)
    o = dict(nq=int(n), fp=int(c.sum()), ratio=float(c.sum() / (n * x)), fq=float((c > 0).mean()),
             ci_cluster=ci([c[i].sum() / (len(i) * x) for i in bc]),
             ci_query=ci([c[i].sum() / (len(i) * x) for i in bq]), ci_poisson=garwood(int(c.sum()), n * x),
             top1_share=(float(np.sort(c)[-1] / c.sum()) if c.sum() else None))
    if bins is not None:
        o["bins"] = {}
        for k, g in enumerate(labels):
            m = bins == k
            if not m.any():
                continue
            vals = []
            for i in bc:
                ii = i[m[i]]
                vals.append(c[ii].sum() / (max(len(ii), 1) * x))
            o["bins"][g] = dict(nq=int(m.sum()), fp=int(c[m].sum()), ratio=float(c[m].sum() / (m.sum() * x)),
                                fq=float((c[m] > 0).mean()), ci_cluster=ci(vals),
                                ci_poisson=garwood(int(c[m].sum()), m.sum() * x))
    return o


def slice_eval(S, q90, m, N, tab, null, bc, bq, bins=None, labels=None, rules=RULES):
    out, perq = {}, {}
    for rule in rules:
        L = lnE(S, q90, m, N, tab, rule)
        out[rule] = {}
        for x in XS:
            cnt = ((L <= math.log(x)) & null).sum(1)
            perq[(rule, x)] = cnt
            out[rule][str(x)] = summarize(cnt, x, bc, bq, bins, labels)
    return out, perq


def fmt(e, xs=XS):
    return " / ".join(f"{e[str(x)]['ratio']:.2f} [{e[str(x)]['ci_cluster'][0]:.2f},{e[str(x)]['ci_cluster'][1]:.2f}]"
                      f"({e[str(x)]['fp']})" for x in xs)


def main():
    lines = []
    out = {"meta": {}, "slices": {}, "diagnostics": {}}
    ship = json.load(open(SHIP))
    T = ship["tables"][KEY]
    tabs = {"all": T["all"], "main": T["main"], "swap": T["swap"]}
    out["meta"]["table"] = dict(file=SHIP, md5=md5(SHIP), key=KEY, version=ship["meta"]["version"],
                                centers=T["all"]["center_len"], lam=T["all"]["lam"], lnK=T["all"]["lnK"])
    lc = np.log(T["all"]["center_len"])
    sl = (T["all"]["lam"][-1] - T["all"]["lam"][-2]) / (lc[-1] - lc[-2])
    sk = (T["all"]["lnK"][-1] - T["all"]["lnK"][-2]) / (lc[-1] - lc[-2])
    out["meta"]["extrapolation"] = dict(
        rule="for m > 594.918: lambda = lam6 + s_lam (ln min(m, cap) - ln 594.918); lnK = lnK6 + s_K (same); "
             "slopes of the last segment (centres 276.49 -> 594.918) of the table in use",
        s_lam_all=sl, s_lnK_all=sk,
        at={str(m): {r: [float(v[0]) for v in params(T["all"], [m], r)] for r in RULES}
            for m in (594.918, 800, 966, 1000, 1500, 2000, 2500, 3000)})
    ix = np.load(f"{OUTD}/index.npz")
    man = json.load(open(f"{OUTD}/manifest.json"))
    out["meta"]["score_manifest"] = man
    for f, h in man["files"].items():
        assert md5(f"{OUTD}/{f}") == h, f
    lq_len = ix["lq_len"].astype(np.float64)
    nq_len = ix["nq_len"].astype(np.float64)
    ben_cols, ben_len = ix["ben_cols"], ix["ben_len"].astype(np.float64)
    long_len, long_is_rev = ix["long_len"].astype(np.float64), ix["long_is_rev"]

    # ---------------- labels
    lq_recs = read_fasta(f"{HERE}/long_q.fasta")
    lt_recs = read_fasta(f"{HERE}/long_t.fasta")
    lr_recs = read_fasta(f"{HERE}/long_r.fasta")
    assert [h.split()[0] for h, _ in lq_recs] == ix["lq_ids"].astype(str).tolist()
    assert [h.split()[0] for h, _ in lt_recs + lr_recs] == ix["long_ids"].astype(str).tolist()
    qh = [h for h, _ in read_fasta(QFA)]
    th = [h for h, _ in read_fasta(TFA)]
    bix = np.load(f"{BSCORE}/out/index.npz")
    assert [h.split()[0] for h in qh] == bix["query_ids"].astype(str).tolist() == ix["nq_ids"].astype(str).tolist()
    assert [h.split()[0] for h in th] == bix["target_ids"].astype(str).tolist()
    assert np.array_equal(bix["target_len"][ben_cols], ben_len)
    lq_sccs = [sccs_of(h) for h, _ in lq_recs]
    nq_sccs = [sccs_of(h) for h in qh]
    bt_sccs = [sccs_of(h) for h in th]
    lt_sccs = [sccs_of(h) for h, _ in lt_recs] + [[] for _ in lr_recs]
    ben_rev = np.array([th[j].startswith("rev_") for j in ben_cols])
    assert ben_rev.sum() == 5000 and (~ben_rev).sum() == 25000
    lq_cov = np.zeros(len(lq_recs))
    for i, (h, s) in enumerate(lq_recs):
        cv = np.zeros(len(s) + 2, bool)
        for a, b in ranges_of(h):
            cv[max(a, 1):min(b, len(s)) + 1] = True
        lq_cov[i] = cv.sum() / len(s)

    # ---------------- scores
    ld = lambda n: np.load(f"{OUTD}/{n}.npy").astype(np.float64)
    S_pan = ld("lq_pan_fwd") - ld("lq_pan_rev")
    q90_lq = np.quantile(S_pan, 0.9, axis=1)
    del S_pan
    q90_nq = np.load(Q90N)
    S_lb = ld("lq_ben_fwd") - ld("lq_ben_rev")
    S_nl = ld("nq_long_fwd") - ld("nq_long_rev")
    Sb = np.load(f"{BSCORE}/out/S_D_fwd.npy").astype(np.float64) - np.load(f"{BSCORE}/out/S_D_rev.npy").astype(np.float64)
    S_nb = Sb[:, ben_cols]
    del Sb
    out["meta"]["q90"] = dict(long_q_pct={str(p): float(np.percentile(q90_lq, p)) for p in (5, 50, 95)},
                              long_q_median_by_bin=[float(np.median(q90_lq[(lq_len >= Q_EDGES[k]) & (lq_len < Q_EDGES[k + 1])]))
                                                    for k in range(6)],
                              bench_q_median=float(np.median(q90_nq)), bench_q90_md5=md5(Q90N))
    log("scores loaded")

    # ---------------- null masks, N
    NULL_lb, FOLD_lb = null_mask(lq_sccs, [bt_sccs[j] for j in ben_cols], ~ben_rev)
    NULL_nb, FOLD_nb = null_mask(nq_sccs, [bt_sccs[j] for j in ben_cols], ~ben_rev)
    NULL_nl, FOLD_nl = null_mask(nq_sccs, lt_sccs, ~long_is_rev)
    REV_lb = np.broadcast_to(ben_rev[None, :], S_lb.shape)
    REV_nb = np.broadcast_to(ben_rev[None, :], S_nb.shape)
    N_ann = float(ben_len[~ben_rev].sum())
    N_rev = float(ben_len[ben_rev].sum())
    out["meta"]["columns"] = dict(bench_annotated=int((~ben_rev).sum()), bench_annotated_residues=N_ann,
                                  bench_rev_subset=int(ben_rev.sum()), bench_rev_subset_residues=N_rev,
                                  long_t=int((~long_is_rev).sum()), long_t_residues=float(long_len[~long_is_rev].sum()),
                                  long_r=int(long_is_rev.sum()), long_r_residues=float(long_len[long_is_rev].sum()),
                                  null_pairs=dict(LQxN=int(NULL_lb.sum()), NQxN=int(NULL_nb.sum()),
                                                  NQxLT=int(NULL_nl.sum())))
    assert int(NULL_nb.sum()) + 0 > 0

    # ---------------- clusters
    lq_bin = np.digitize(lq_len, Q_EDGES[1:-1])
    lq_folds = [set(lvl(s, 2) for s in l_) for l_ in lq_sccs]
    lq_comp = components(lq_folds)
    lq_key = np.unique(np.array([";".join(sorted(f)) for f in lq_folds]), return_inverse=True)[1]
    nq_folds = [set(lvl(s, 2) for s in l_) for l_ in nq_sccs]
    nq_comp = components(nq_folds)
    out["meta"]["clusters"] = dict(
        long_q_fold_components=int(lq_comp.max() + 1), long_q_largest_component=int(np.bincount(lq_comp).max()),
        long_q_fold_set_keys=int(lq_key.max() + 1), long_q_largest_key=int(np.bincount(lq_key).max()),
        bench_q_fold_components=int(nq_comp.max() + 1), bench_q_largest_component=int(np.bincount(nq_comp).max()))
    big_comp = np.bincount(lq_comp).max() / len(lq_comp)
    lq_groups = lq_key if big_comp > 0.2 else lq_comp
    out["meta"]["clusters"]["long_q_bootstrap_unit"] = "fold_set_key" if big_comp > 0.2 else "fold_component"
    bc_lq = boot_idx(lq_groups, NB, 11)
    rng = np.random.default_rng(12)
    bq_lq = [rng.integers(0, len(lq_len), len(lq_len)) for _ in range(NB)]
    bc_nq = boot_idx(nq_comp, NB, 1)
    rng = np.random.default_rng(2)
    bq_nq = [rng.integers(0, len(nq_len), len(nq_len)) for _ in range(NB)]
    log("clusters", out["meta"]["clusters"])

    # ---------------- A5 reference (normal x normal), shipped table, own protocol (25,000 annotated; N annotated)
    SL = out["slices"]
    nq_bin6 = np.digitize(nq_len, R.EDGES[1:-1])
    SL["A5_NQxN_real"], _ = slice_eval(S_nb, q90_nq, nq_len, N_ann, tabs["all"], NULL_nb, bc_nq, bq_nq, nq_bin6, R.BL)
    lines.append("A5 reference, 1,465 benchmark queries x 25,000 annotated (shipped 'all' table, FP/q/x at "
                 "1/.1/.01/.001 [cluster CI](FPs)):")
    for r in RULES:
        lines.append(f"  {r:10s} {fmt(SL['A5_NQxN_real'][r])}")
    SL["A5_NQxN_rev5000"], _ = slice_eval(S_nb, q90_nq, nq_len, N_rev, tabs["all"], REV_nb, bc_nq, bq_nq, nq_bin6, R.BL)
    lines.append("A5 reversed null (same 5,000 decoys, N = their residues):")
    for r in RULES:
        lines.append(f"  {r:10s} {fmt(SL['A5_NQxN_rev5000'][r])}")
    hi = np.flatnonzero(nq_len > 594.918)
    grp = nq_comp[hi]
    bch = boot_idx(grp, NB, 5)
    rng = np.random.default_rng(6)
    bqh = [rng.integers(0, len(hi), len(hi)) for _ in range(NB)]
    SL["A5_NQxN_real_m_gt_594"], _ = slice_eval(S_nb[hi], q90_nq[hi], nq_len[hi], N_ann, tabs["all"], NULL_nb[hi],
                                                bch, bqh)
    lines.append(f"A5 benchmark queries with m > 594.918 ({len(hi)}, 595-966 aa) x 25,000 annotated:")
    for r in RULES:
        lines.append(f"  {r:10s} {fmt(SL['A5_NQxN_real_m_gt_594'][r])}")
    # held-out tables on the benchmark queries (main fitted on even rows -> odd rows, swap -> even rows)
    idx = np.arange(len(nq_len))

    # ---------------- LQ x N (real null) and LQ x R (reversed null)
    for tname in ("all", "main", "swap"):
        SL[f"LQxN_real_{tname}"], perq = slice_eval(S_lb, q90_lq, lq_len, N_ann, tabs[tname], NULL_lb, bc_lq, bq_lq,
                                                    lq_bin, QBL)
        if tname == "all":
            perq_lqn = perq
    SL["LQxR_rev_all"], perq_lqr = slice_eval(S_lb, q90_lq, lq_len, N_rev, tabs["all"], REV_lb, bc_lq, bq_lq, lq_bin, QBL)
    # pooled long bins (>= 1000) and the two baseline bins
    longm = lq_len >= 1000
    basem = lq_len < 1000
    for nm, msk in (("long_1000_3000", longm), ("base_300_1000", basem)):
        ii = np.flatnonzero(msk)
        bci = boot_idx(lq_groups[ii], NB, 21)
        rng = np.random.default_rng(22)
        bqi = [rng.integers(0, len(ii), len(ii)) for _ in range(NB)]
        for tname in ("all", "main", "swap"):
            SL[f"LQxN_real_{tname}_{nm}"], _ = slice_eval(S_lb[ii], q90_lq[ii], lq_len[ii], N_ann, tabs[tname],
                                                          NULL_lb[ii], bci, bqi)
        SL[f"LQxR_rev_all_{nm}"], _ = slice_eval(S_lb[ii], q90_lq[ii], lq_len[ii], N_rev, tabs["all"], REV_lb[ii],
                                                 bci, bqi)
    lines.append("")
    lines.append("LONG QUERIES x 25,000 A5 annotated targets (real null), shipped table; FP/q/x at 1/.1/.01/.001:")
    for r in RULES:
        lines.append(f"  {r:10s} pooled 1000-3000: {fmt(SL['LQxN_real_all_long_1000_3000'][r])}")
        lines.append(f"  {'':10s} baseline 300-1000: {fmt(SL['LQxN_real_all_base_300_1000'][r])}")
        e = SL["LQxN_real_all"][r]
        for g in QBL:
            lines.append(f"      {g:10s} " + " / ".join(
                f"{e[str(x)]['bins'][g]['ratio']:.2f} [{e[str(x)]['bins'][g]['ci_cluster'][0]:.2f},"
                f"{e[str(x)]['bins'][g]['ci_cluster'][1]:.2f}]({e[str(x)]['bins'][g]['fp']})" for x in XS))
    lines.append("LONG QUERIES x 5,000 A5 reversed decoys (reversed null), shipped table:")
    for r in RULES:
        lines.append(f"  {r:10s} pooled 1000-3000: {fmt(SL['LQxR_rev_all_long_1000_3000'][r])}")
        lines.append(f"  {'':10s} baseline 300-1000: {fmt(SL['LQxR_rev_all_base_300_1000'][r])}")
        e = SL["LQxR_rev_all"][r]
        for g in QBL:
            lines.append(f"      {g:10s} " + " / ".join(
                f"{e[str(x)]['bins'][g]['ratio']:.2f}({e[str(x)]['bins'][g]['fp']})" for x in XS))
    log("LQ slices done")

    # ---------------- NQ x LT (real null) and NQ x LR (reversed), per target bin; shipped table + held-out union
    SL["NQxLT_real"] = {}
    SL["NQxLR_rev"] = {}
    tb = np.digitize(long_len, T_EDGES[1:-1])
    for k, g in enumerate(["all_bins"] + TBL):
        cols_t = (~long_is_rev) if g == "all_bins" else ((~long_is_rev) & (tb == k - 1))
        cols_r = long_is_rev if g == "all_bins" else (long_is_rev & (tb == k - 1))
        for nm, cols, dst, msk in (("real", cols_t, SL["NQxLT_real"], NULL_nl), ("rev", cols_r, SL["NQxLR_rev"], None)):
            N = float(long_len[cols].sum())
            Ssub = S_nl[:, cols]
            null = msk[:, cols] if msk is not None else np.ones_like(Ssub, bool)
            dst[g], _ = slice_eval(Ssub, q90_nq, nq_len, N, tabs["all"], null, bc_nq, bq_nq, nq_bin6, R.BL)
            # held-out union: main table on odd rows, swap table on even rows
            hu = {}
            for rule in RULES:
                hu[rule] = {}
                for x in XS:
                    c = np.zeros(len(nq_len), np.int64)
                    for tname, rows in (("main", idx[idx % 2 == 1]), ("swap", idx[idx % 2 == 0])):
                        L = lnE(Ssub[rows], q90_nq[rows], nq_len[rows], N, tabs[tname], rule)
                        c[rows] = ((L <= math.log(x)) & null[rows]).sum(1)
                    hu[rule][str(x)] = summarize(c, x, bc_nq, bq_nq)
            dst[g + "_heldout_union"] = hu
    # A5 held-out union for the same comparison (normal targets)
    hu = {}
    for rule in RULES:
        hu[rule] = {}
        for x in XS:
            c = np.zeros(len(nq_len), np.int64)
            for tname, rows in (("main", idx[idx % 2 == 1]), ("swap", idx[idx % 2 == 0])):
                L = lnE(S_nb[rows], q90_nq[rows], nq_len[rows], N_ann, tabs[tname], rule)
                c[rows] = ((L <= math.log(x)) & NULL_nb[rows]).sum(1)
            hu[rule][str(x)] = summarize(c, x, bc_nq, bq_nq)
    SL["A5_NQxN_real_heldout_union"] = hu
    lines.append("")
    lines.append("BENCHMARK QUERIES x LONG TARGETS (shipped 'all' table; held-out union in brackets at E<=1):")
    for g in ["all_bins"] + TBL:
        e, h = SL["NQxLT_real"][g]["clamp"], SL["NQxLT_real"][g + "_heldout_union"]["clamp"]
        lines.append(f"  real {g:10s} {fmt(e)}   held-out@1 {h['1.0']['ratio']:.2f}")
    lines.append(f"  A5 ref (normal targets) {fmt(SL['A5_NQxN_real']['clamp'])}   held-out@1 "
                 f"{SL['A5_NQxN_real_heldout_union']['clamp']['1.0']['ratio']:.2f}")
    for g in ["all_bins"] + TBL:
        lines.append(f"  rev  {g:10s} {fmt(SL['NQxLR_rev'][g]['clamp'])}")
    lines.append(f"  A5 rev ref (5,000 normal decoys) {fmt(SL['A5_NQxN_rev5000']['clamp'])}")
    for r in ("extrap966", "extrap3000"):
        lines.append(f"  real all_bins {r}: {fmt(SL['NQxLT_real']['all_bins'][r])}")
    log("NQ slices done")

    # ---------------- diagnostics
    D = out["diagnostics"]
    # FP-ratio curve on a finer x grid (Poisson CIs only). lnE <= ln x depends on N and x only through ln N - ln x, so
    # x on this slice's N is equivalent to E <= x * N_db / N on a database of N_db residues.
    XF = (1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001)
    D["curves"] = {"x": XF, "equivalence": "FP ratio at E <= x with N (this slice) = FP ratio at E <= x N_db / N for a "
                                          "database of N_db residues (same score threshold)"}
    for nm, (Sx, q9, mm, NN, nul) in {
            "LQxN_long_1000_3000": (S_lb[longm], q90_lq[longm], lq_len[longm], N_ann, NULL_lb[longm]),
            "LQxN_base_300_1000": (S_lb[basem], q90_lq[basem], lq_len[basem], N_ann, NULL_lb[basem]),
            "LQxR_long_1000_3000": (S_lb[longm], q90_lq[longm], lq_len[longm], N_rev, REV_lb[longm]),
            "LQxR_base_300_1000": (S_lb[basem], q90_lq[basem], lq_len[basem], N_rev, REV_lb[basem]),
            "A5_NQxN": (S_nb, q90_nq, nq_len, N_ann, NULL_nb),
            "A5_NQxR5000": (S_nb, q90_nq, nq_len, N_rev, REV_nb),
            "NQxLT": (S_nl[:, ~long_is_rev], q90_nq, nq_len, float(long_len[~long_is_rev].sum()), NULL_nl[:, ~long_is_rev]),
            "NQxLR": (S_nl[:, long_is_rev], q90_nq, nq_len, float(long_len[long_is_rev].sum()),
                      np.ones((len(nq_len), int(long_is_rev.sum())), bool))}.items():
        D["curves"][nm] = {"N": NN, "nq": int(len(mm))}
        for rule in RULES:
            L = lnE(Sx, q9, mm, NN, tabs["all"], rule)
            row = []
            for x in XF:
                k = int(((L <= math.log(x)) & nul).sum())
                row.append(dict(x=x, fp=k, ratio=k / (len(mm) * x), ci_poisson=garwood(k, len(mm) * x)))
            D["curves"][nm][rule] = row
    lines.append("")
    lines.append("FP-ratio curves (x = " + ", ".join(str(x) for x in XF) + "):")
    for nm in [k for k in D["curves"] if k not in ("x", "equivalence")]:
        for rule in RULES:
            lines.append(f"  {nm:22s} {rule:10s} " + " ".join(f"{v['ratio']:.2f}({v['fp']})" for v in D["curves"][nm][rule]))
    # coverage strata within the long bins (real null, shipped table)
    cov_edges = [0.0, 0.15, 0.3, 0.5, 1.01]
    cb = np.digitize(lq_cov, cov_edges[1:-1])
    D["LQxN_real_by_coverage"] = {}
    for rule in RULES:
        D["LQxN_real_by_coverage"][rule] = {}
        for nm, msk in (("long_1000_3000", longm), ("base_300_1000", basem)):
            for k in range(4):
                ii = np.flatnonzero(msk & (cb == k))
                if len(ii) < 20:
                    continue
                cnt = perq_lqn[(rule, 1.0)][ii]
                cr = perq_lqr[(rule, 1.0)][ii]
                D["LQxN_real_by_coverage"][rule][f"{nm}/cov{cov_edges[k]:.2f}-{min(cov_edges[k + 1], 1):.2f}"] = dict(
                    nq=int(len(ii)), median_len=float(np.median(lq_len[ii])), real_fp_at1=int(cnt.sum()),
                    real_ratio_at1=float(cnt.sum() / len(ii)), real_ci_poisson=garwood(int(cnt.sum()), len(ii)),
                    rev_ratio_at1=float(cr.sum() / len(ii)))
    lines.append("")
    lines.append("COVERAGE strata (fraction of the long query covered by SCOP annotations), E<=1, clamp: real / rev ratio")
    for k_, v in D["LQxN_real_by_coverage"]["clamp"].items():
        lines.append(f"  {k_:32s} nq {v['nq']:4d} len~{v['median_len']:.0f}: real {v['real_ratio_at1']:.2f} "
                     f"[{v['real_ci_poisson'][0]:.2f},{v['real_ci_poisson'][1]:.2f}]  rev {v['rev_ratio_at1']:.2f}")
    # censored-Gumbel refit per long-query bin (descriptive)
    D["gumbel_refit_by_lq_bin"] = {}
    for k, g in enumerate(QBL):
        ii = np.flatnonzero(lq_bin == k)
        Sp = (S_lb[ii] - q90_lq[ii][:, None])
        msk = NULL_lb[ii]
        A = lq_len[ii][:, None] * ben_len[None, :]
        f = R.fit_gumbel(Sp[msk], np.broadcast_to(A, Sp.shape)[msk], 0.999)
        cen = float(np.exp(np.log(lq_len[ii]).mean()))
        f.update(center_len=cen, n_queries=int(len(ii)),
                 table_clamp=[float(v[0]) for v in params(tabs["all"], [cen], "clamp")],
                 table_extrap3000=[float(v[0]) for v in params(tabs["all"], [cen], "extrap3000")])
        D["gumbel_refit_by_lq_bin"][g] = f
    lines.append("")
    lines.append("Censored-Gumbel refit (q .999) on the long-query real null, per bin: centre lambda lnK | table clamp | "
                 "table extrap:")
    for g, f in D["gumbel_refit_by_lq_bin"].items():
        lines.append(f"  {g:10s} {f['center_len']:7.1f} {f['lam']:.4f} {f['lnK']:8.3f} | {f['table_clamp'][0]:.4f} "
                     f"{f['table_clamp'][1]:8.3f} | {f['table_extrap3000'][0]:.4f} {f['table_extrap3000'][1]:8.3f}  "
                     f"n_unc {f['n_unc']}")
    # sensitivity: propeller exception extended to b.66 (4-bladed beta-propeller), i.e. pairs where both sides carry any
    # of b.66-b.70 are not counted as FP (A5 excepts only b.67-70); and a robust view without the top 1% of queries
    P66 = r"^b\.(66|67|68|69|70)\."
    NULL_lb66, _ = null_mask(lq_sccs, [bt_sccs[j] for j in ben_cols], ~ben_rev, P66)
    NULL_nb66, _ = null_mask(nq_sccs, [bt_sccs[j] for j in ben_cols], ~ben_rev, P66)
    NULL_nl66, _ = null_mask(nq_sccs, lt_sccs, ~long_is_rev, P66)
    D["sensitivity_propeller_b66"] = {"null_pairs": dict(LQxN=int(NULL_lb66.sum()), NQxN=int(NULL_nb66.sum()),
                                                         NQxLT=int(NULL_nl66.sum()))}
    for nm, (Sx, q9, mm, NN, nul, grp) in {
            "LQxN_long_1000_3000": (S_lb[longm], q90_lq[longm], lq_len[longm], N_ann, NULL_lb66[longm], lq_groups[longm]),
            "LQxN_base_300_1000": (S_lb[basem], q90_lq[basem], lq_len[basem], N_ann, NULL_lb66[basem], lq_groups[basem]),
            "A5_NQxN": (S_nb, q90_nq, nq_len, N_ann, NULL_nb66, nq_comp),
            "NQxLT": (S_nl[:, ~long_is_rev], q90_nq, nq_len, float(long_len[~long_is_rev].sum()),
                      NULL_nl66[:, ~long_is_rev], nq_comp)}.items():
        bci = boot_idx(grp, NB, 31)
        rng = np.random.default_rng(32)
        bqi = [rng.integers(0, len(mm), len(mm)) for _ in range(NB)]
        D["sensitivity_propeller_b66"][nm], pq = slice_eval(Sx, q9, mm, NN, tabs["all"], nul, bci, bqi)
        for rule in RULES:
            for x in XS:
                c = np.sort(pq[(rule, x)])
                k1 = max(1, int(round(0.01 * len(c))))
                D["sensitivity_propeller_b66"][nm][rule][str(x)]["ratio_without_top1pct_queries"] = float(
                    c[:-k1].sum() / ((len(c) - k1) * x))
    lines.append("")
    lines.append("SENSITIVITY: propeller exception extended to b.66 (FP/q/x at 1/.1/.01/.001; robust = without top 1% queries):")
    for nm in ("LQxN_long_1000_3000", "LQxN_base_300_1000", "A5_NQxN", "NQxLT"):
        for rule in ("clamp", "extrap3000"):
            e = D["sensitivity_propeller_b66"][nm][rule]
            lines.append(f"  {nm:22s} {rule:10s} {fmt(e)}  robust " + " / ".join(
                f"{e[str(x)]['ratio_without_top1pct_queries']:.2f}" for x in XS))
    # EXPLORATORY (post hoc, not pre-specified): lambda extrapolated in ln m (cap 3,000), lnK clamped
    D["exploratory_lambda_only_extrapolation"] = {"rule": "lambda as extrap3000, lnK clamped at lnK6; chosen after "
                                                          "seeing the per-bin Gumbel refit, so NOT a validated rule"}
    for nm, (Sx, q9, mm, NN, nul, grp) in {
            "LQxN_long_1000_3000": (S_lb[longm], q90_lq[longm], lq_len[longm], N_ann, NULL_lb[longm], lq_groups[longm]),
            "LQxR_long_1000_3000": (S_lb[longm], q90_lq[longm], lq_len[longm], N_rev, REV_lb[longm], lq_groups[longm]),
            "LQxN_600_1000": (S_lb[lq_bin == 1], q90_lq[lq_bin == 1], lq_len[lq_bin == 1], N_ann, NULL_lb[lq_bin == 1],
                              lq_groups[lq_bin == 1])}.items():
        bci = boot_idx(grp, NB, 41)
        rng = np.random.default_rng(42)
        bqi = [rng.integers(0, len(mm), len(mm)) for _ in range(NB)]
        e, _ = slice_eval(Sx, q9, mm, NN, tabs["all"], nul, bci, bqi, rules=("lamonly3000",))
        D["exploratory_lambda_only_extrapolation"][nm] = e["lamonly3000"]
        lines.append(f"EXPLORATORY lambda-only extrapolation {nm:22s} {fmt(e['lamonly3000'])}")
    json.dump(out, open(f"{HERE}/long_slice_partial.json", "w"), indent=1, default=float)

    # FP pairs at E <= 1 for the traceback location diagnostic (written; traced by trace_fp.py)
    pairs = []
    for rule in ("clamp", "extrap3000"):
        L = lnE(S_lb, q90_lq, lq_len, N_ann, tabs["all"], rule)
        qi, tj = np.nonzero((L <= 0.0) & NULL_lb)
        pairs.append(np.stack([qi, tj, np.full(len(qi), RULES.index(rule))], 1))
        del L
    pairs = np.unique(np.concatenate(pairs)[:, :2], axis=0)
    np.save(f"{HERE}/fp_pairs_lqxn.npy", pairs)
    L = lnE(S_nl, q90_nq, nq_len, float(long_len[~long_is_rev].sum()), tabs["all"], "clamp")
    qi, tj = np.nonzero((L <= 0.0) & NULL_nl)
    np.save(f"{HERE}/fp_pairs_nqxlt.npy", np.stack([qi, tj], 1))
    D["fp_pairs_written"] = dict(lqxn=int(len(pairs)), nqxlt=int(len(qi)))
    out["text"] = lines
    json.dump(out, open(f"{HERE}/long_slice_core.json", "w"), indent=1, default=float)
    open(f"{HERE}/long_slice_core.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
