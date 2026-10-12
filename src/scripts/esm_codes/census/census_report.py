#!/usr/bin/env python
"""P0.6 census, step 4 (local Mac): FP check, code-level tables, degenerate S_f - S_r check, benchmark baseline, and
the D14 recommendation -> degenerate_census.json.

Inputs (copied from NERSC /pscratch/sd/j/jsegura/embcomp/p0/census/): seq/seq_census.json, seq/items.tsv,
out/codes.npz, out/targets.npz, out/queries.npz, out/check.json, enc meta (enc_meta.json).
Research files (local mirror SP/embcomp, md5-identical to NERSC): panel/fit2/results/{ship_final.json,
bench_q90_shipped.npy}, evalue/score/out/{S_D_fwd,S_D_rev,index}.npy/npz, evalue/score/results/labels.npz,
calib3di/results/{queries,targets}.fasta.

E-value (shipped system): ln E = lnK(m) + ln m + ln N - lambda(m) (S - Q90(q)), S = S_f - S_r (setting D), table
'D_fmr/pnl_uniref50/M3t999' 'all' (np.interp in ln m, clamped), Q90 over the shipped 1,000-member panel.

FP check (A5 protocol = the benchmark real null, N = annotated benchmark residues N_ann):
  as QUERIES: census item x 25,000 annotated benchmark targets; every hit at E <= x is a false positive unless the pair is
      sequence-detectable homologous (BLOSUM62 local, BLAST 11/1 gaps, Karlin-Altschul lambda 0.267, K 0.041, E_seq
      over m x N_ann <= 1e-3: 'filtered').  Expected under calibration: x per query.  ratio = hits / (n_items x).
      Reversed-target null: hits among the 25,000 reversed targets with N = N_rev.
  as TARGETS: 1,465 benchmark queries x census items; hit = E(N_ann) <= x (same homology filter).  Expected under
      calibration: n_q x N_G / N_ann (the items as a slice of an N_ann-residue database).  ratio = hits / expected.
  CIs: bootstrap over items (2,000 resamples, ratio of sums) and exact Poisson (Garwood) on the hit count.
  Both the research kernel scores and the S7-variant scores (degenerate terms = 0) are evaluated.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.stats import chi2

SP = "/private/tmp/claude-501/-Users-joan-devel-rcsb-embedding-search/d9266e69-77ea-4121-a428-2e6f3bcc4ca2/scratchpad"
EMB = f"{SP}/embcomp"
TRACK = os.environ.get("TRACK", f"{SP}/p0/census")
IN = os.environ.get("CENSUS_IN", f"{TRACK}/nersc")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import census_seq as CS  # noqa: E402  (features(): the sequence-level category definitions)

TKEY = "D_fmr/pnl_uniref50/M3t999"
XS = (1.0, 0.1, 0.01, 1e-3)
NB = 2000
HOM_E = 1e-3
KA_LAMBDA, KA_K = 0.267, 0.041
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def read_fasta(p):
    out, h, s = [], None, []
    for line in open(p):
        line = line.rstrip("\n")
        if line.startswith(">"):
            if h is not None:
                out.append((h, "".join(s)))
            h, s = line[1:], []
        elif line.strip():
            s.append(line.strip())
    if h is not None:
        out.append((h, "".join(s)))
    return out


def table(split="all"):
    t = json.load(open(f"{EMB}/panel/fit2/results/ship_final.json"))["tables"][TKEY][split]
    return np.log(np.asarray(t["center_len"], np.float64)), np.asarray(t["lam"]), np.asarray(t["lnK"])


def lam_lnk(tab, m):
    lnc, lam, lnk = tab
    lm = np.log(np.asarray(m, np.float64))
    return np.interp(lm, lnc, lam), np.interp(lm, lnc, lnk)


def poisson_ci(k, expected):
    lo = chi2.ppf(0.025, 2 * k) / 2 if k > 0 else 0.0
    hi = chi2.ppf(0.975, 2 * (k + 1)) / 2
    return [float(lo / expected), float(hi / expected)] if expected > 0 else [None, None]


# ------------------------------------------------------------------------------------------------ homology filter
_ALN = {}


def _hom_init():
    import biotite.sequence as bs
    import biotite.sequence.align as ba
    _ALN["m"] = ba.SubstitutionMatrix.std_protein_matrix()
    _ALN["bs"], _ALN["ba"] = bs, ba


def _clean(s):
    ok = set("ACDEFGHIKLMNPQRSTVWYBZX")
    return "".join(c if c in ok else "X" for c in s.upper())


def _hom_score(pair):
    a, b = pair
    bs, ba = _ALN["bs"], _ALN["ba"]
    al = ba.align_optimal(bs.ProteinSequence(_clean(a)), bs.ProteinSequence(_clean(b)), _ALN["m"],
                          gap_penalty=(-12, -1), local=True, max_number=1)
    return float(al[0].score) if al else 0.0


def homology_scores(pairs, nproc=4):
    """BLOSUM62 local scores for (a, b) pairs; cached in TRACK/hom_cache.pkl keyed by sha1(a|b)."""
    import pickle
    cp = f"{TRACK}/hom_cache.pkl"
    cache = pickle.load(open(cp, "rb")) if os.path.exists(cp) else {}
    key = lambda p: hashlib.sha1((p[0] + "|" + p[1]).encode()).digest()
    uniq = sorted(set(pairs))
    todo = [p for p in uniq if key(p) not in cache]
    if todo:
        with Pool(nproc, initializer=_hom_init) as pool:
            sc = pool.map(_hom_score, todo, chunksize=16)
        for p, v in zip(todo, sc):
            cache[key(p)] = v
        pickle.dump(cache, open(cp, "wb"))
    return {p: cache[key(p)] for p in uniq}


def seq_evalue(score, m, N):
    return KA_K * m * N * math.exp(-KA_LAMBDA * score)


# ------------------------------------------------------------------------------------------------ baseline
def baseline():
    sys.path.insert(0, f"{EMB}/evalue/reconcile")
    import rlib as R
    B = R.Bench()
    S = B.scores("D", "fmr")
    q90 = np.load(f"{EMB}/panel/fit2/results/bench_q90_shipped.npy")
    ann, rev = ~B.rev, B.rev
    N_ann, N_rev = float(B.tlen[ann].sum()), float(B.tlen[rev].sum())
    idx = np.arange(B.nq)
    ev, od = idx[idx % 2 == 0], idx[idx % 2 == 1]
    out = {"N_ann": N_ann, "N_rev": N_rev, "nq": int(B.nq)}

    def cnt(tab, rows):
        lam, lnk = lam_lnk(tab, B.qlen[rows])
        base = (lnk + np.log(B.qlen[rows]))[:, None] - lam[:, None] * (S[rows] - q90[rows][:, None])
        real = {x: ((base + math.log(N_ann) <= math.log(x)) & B.REAL[rows]).sum(1) for x in XS}
        rv = {x: ((base + math.log(N_rev) <= math.log(x)) & B.REVM[rows]).sum(1) for x in XS}
        return real, rv
    ra, va = cnt(table("all"), idx)
    rm, vm = cnt(table("main"), od)
    rs, vs = cnt(table("swap"), ev)
    for name, (r, v, n) in {"insample_all": (ra, va, B.nq),
                            "heldout_union": ({x: np.concatenate([rm[x], rs[x]]) for x in XS},
                                              {x: np.concatenate([vm[x], vs[x]]) for x in XS}, B.nq)}.items():
        out[name] = {str(x): {"real_fp": int(r[x].sum()), "real_ratio": float(r[x].sum() / (n * x)),
                              "real_poisson_ci": poisson_ci(int(r[x].sum()), n * x),
                              "real_fq": float((r[x] > 0).mean()),
                              "rev_fp": int(v[x].sum()), "rev_ratio": float(v[x].sum() / (n * x))} for x in XS}
    return out


# ------------------------------------------------------------------------------------------------ main
def boot_ratio(num, den, nb=NB, seed=1):
    n = len(num)
    if n == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(nb):
        i = rng.integers(0, n, n)
        d = den[i].sum()
        v.append(num[i].sum() / d if d > 0 else np.nan)
    v = np.array(v)
    v = v[np.isfinite(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    seqc = json.load(open(f"{IN}/seq_census.json"))
    rows = [line.rstrip("\n").split("\t") for line in open(f"{IN}/items.tsv")][1:]
    iids = [r[0] for r in rows]
    iseq = [r[6] for r in rows]
    ilen = np.array([int(r[1]) for r in rows])
    sel = {"pdb": [set(filter(None, r[2].split(","))) for r in rows],
           "uniref50": [set(filter(None, r[3].split(","))) for r in rows]}
    feats = [CS.features(s.encode())[0] for s in iseq]
    has = {c: np.array([(f >> k) & 1 for f in feats], bool) for k, c in enumerate(CS.ALLC)}
    n_items = len(iids)
    cz = np.load(f"{IN}/codes.npz")
    assert list(cz["ids"].astype(str)) == iids
    distinct = cz["distinct_codes"]
    ident_q = cz["identical_q_rows"]
    tz = np.load(f"{IN}/targets.npz")
    qz = np.load(f"{IN}/queries.npz")
    assert list(tz["item_ids"].astype(str)) == iids and list(qz["item_ids"].astype(str)) == iids
    qmeta = json.loads(str(qz["meta_json"]))
    N_ann, N_rev = qmeta["N_ann"], qmeta["N_rev"]
    tab = table("all")

    # ---------------- groups
    G = {}
    for src in ("pdb", "uniref50"):
        for c in CS.CATS + ["random"]:
            m = np.array([c in s for s in sel[src]])
            if m.any():
                G[f"{src}:{c}"] = m
    for c in CS.CATS + ["random"]:
        m = np.array([c in a or c in b for a, b in zip(sel["pdb"], sel["uniref50"])])
        if m.any():
            G[f"both:{c}"] = m
    anycat = np.zeros(n_items, bool)
    for c in CS.CATS:
        anycat |= has[c]
    rnd = G["both:random"]
    G["random_clean(no requested category, >=10 distinct codes)"] = rnd & ~anycat & (distinct >= 10)
    G["code:distinct==1"] = distinct == 1
    G["code:distinct<3"] = distinct < 3
    G["code:distinct<5"] = distinct < 5
    G["code:distinct<10"] = distinct < 10
    G["code:identical_q_rows"] = ident_q
    q_deg = (np.nan_to_num(qz["agg_panel_deg_pairs"]) > 0) | (np.nan_to_num(qz["agg_tgt_deg_pairs"]) > 0)
    G["code:query_any_degenerate_column"] = q_deg
    tdeg_items = np.zeros(n_items, bool)
    tdeg_items[np.unique(tz["deg_t"])] = True
    G["code:target_any_degenerate_row"] = tdeg_items
    G["D14:target_rule(distinct<2 or nonstd>1%)"] = (distinct < 2) | has["nonstd"]
    G["D14:nonstd>1%"] = has["nonstd"]
    G["seq:allX|homopoly|histag (any source, sequence-defined)"] = has["allX"] | has["homopoly"] | has["histag"]
    G["seq:L<21 and not allX/homopoly/histag"] = has["Llt21"] & ~(has["allX"] | has["homopoly"] | has["histag"])
    G["seq:nonstd>1% and not allX"] = has["nonstd"] & ~has["allX"]

    # ---------------- code-level census
    tdeg_q = np.bincount(tz["deg_t"], minlength=n_items)          # (query, target) pairs with degenerate rows
    code = {}
    for g, m in G.items():
        d = distinct[m]
        code[g] = {"n_items": int(m.sum()), "residues": int(ilen[m].sum()),
                   "distinct_codes_eq1": int((d == 1).sum()), "distinct_lt2": int((d < 2).sum()),
                   "distinct_lt3": int((d < 3).sum()), "distinct_lt5": int((d < 5).sum()),
                   "distinct_lt10": int((d < 10).sum()),
                   "median_distinct_over_L": float(np.median(d / ilen[m])) if m.any() else None,
                   "identical_q_rows": int(ident_q[m].sum()), "zero_norm_q_rows": int(cz["zero_norm_q_rows"][m].sum()),
                   "as_query_any_degenerate_column(panel or bench targets)": int(q_deg[m].sum()),
                   "as_target_any_degenerate_row(1465 bench queries)": int(tdeg_items[m].sum()),
                   "as_target_degenerate_row_pairs_fraction": float(tdeg_q[m].sum() / (m.sum() * len(tz["query_ids"])))
                   if m.any() else None,
                   "tau_certificate_ge_1e-6": int((cz["tau_lower_bound"][m] >= 1e-6).sum())}
    # distinct-code histogram for the short items
    code_hist = {int(k): int(v) for k, v in zip(*np.unique(np.minimum(distinct, 20), return_counts=True))}
    # prevalence estimates per source: category fraction x category unique count; remainder from clean random
    prev = {}
    for src in ("pdb", "uniref50"):
        ss = seqc["sources"][src]
        uc = ss["unique"] if src == "pdb" else ss["records_by_category"]
        e = {}
        for c in CS.CATS:
            m = G.get(f"{src}:{c}")
            if m is None or not m.any():
                continue
            f1 = float((distinct[m] < 2).mean())
            e[c] = {"selected": int(m.sum()), "lt2_in_selected": int((distinct[m] < 2).sum()),
                    "category_unique": int(uc[c]), "estimated_lt2_in_category": round(f1 * uc[c], 1)}
        m = G[f"{src}:random"]
        e["random"] = {"selected": int(m.sum()), "lt2_in_selected": int((distinct[m] < 2).sum())}
        prev[src] = e

    # ---------------- as TARGETS
    Sf, Sr = tz["Sf"].astype(np.float64), tz["Sr"].astype(np.float64)
    S = Sf - Sr
    S7 = S.copy()
    S7[tz["deg_q"], tz["deg_t"]] = tz["deg_sf7"].astype(np.float64) - tz["deg_sr7"]
    bq = np.load(f"{EMB}/evalue/encode/out/queries.L2m_B8_s0.npz")
    assert list(bq["ids"].astype(str)) == list(tz["query_ids"].astype(str))
    qlen = bq["lengths"].astype(np.float64)
    bq_seq = bq["seq_concat"].tobytes().decode("ascii")
    bq_off = bq["offsets"]
    q90 = np.load(f"{EMB}/panel/fit2/results/bench_q90_shipped.npy")
    lam, lnk = lam_lnk(tab, qlen)
    base = (lnk + np.log(qlen) + math.log(N_ann))[:, None]
    LE = base - lam[:, None] * (S - q90[:, None])
    LE7 = base - lam[:, None] * (S7 - q90[:, None])
    nq = len(qlen)
    # homology for target-side hits at E <= 1 (either score form)
    hq, ht = np.nonzero((LE <= 0) | (LE7 <= 0))
    pairs_t = [(bq_seq[bq_off[i]:bq_off[i + 1]], iseq[t]) for i, t in zip(hq, ht)]
    log(f"target-side hit pairs at E<=1: {len(pairs_t)}; homology scoring")
    hs = homology_scores(pairs_t)
    homT = np.zeros_like(LE, bool)
    for (i, t), p in zip(zip(hq, ht), pairs_t):
        if seq_evalue(hs[p], qlen[i], N_ann) <= HOM_E:
            homT[i, t] = True
    as_target = {}
    sticky = {}
    for form, L_ in (("research", LE), ("S7", LE7)):
        out = {}
        for g, m in G.items():
            cols = np.flatnonzero(m)
            NG = float(ilen[cols].sum())
            r = {"n_items": int(len(cols)), "residues": int(NG)}
            for x in XS:
                H = (L_[:, cols] <= math.log(x))
                Hf = H & ~homT[:, cols]
                for tag, HH in (("raw", H), ("filtered", Hf)):
                    per_t = HH.sum(0).astype(np.float64)
                    exp_t = nq * x * ilen[cols] / N_ann
                    k = int(per_t.sum())
                    E_ = float(exp_t.sum())
                    r[f"{tag}@{x:g}"] = {"hits": k, "expected": E_, "ratio": k / E_ if E_ > 0 else None,
                                         "boot_ci": boot_ratio(per_t, exp_t), "poisson_ci": poisson_ci(k, E_),
                                         "targets_with_hit": int((per_t > 0).sum()),
                                         "max_hits_one_target": int(per_t.max()) if len(per_t) else 0}
            out[g] = r
        as_target[form] = out
        per = ((L_ <= 0) & ~homT).sum(0)
        top = np.argsort(-per)[:12]
        sticky[form] = [{"item": iids[t], "L": int(ilen[t]), "distinct_codes": int(distinct[t]),
                         "queries_hit_E<=1_filtered": int(per[t]), "pdb_sel": sorted(sel["pdb"][t]),
                         "uniref50_sel": sorted(sel["uniref50"][t]), "seq": iseq[t][:60]} for t in top if per[t] > 0]
    # fully degenerate targets
    d1 = np.flatnonzero(distinct == 1)
    fmr_abs = np.abs(S[:, d1]) if len(d1) else np.zeros((nq, 0))
    fmr7_abs = np.abs(S7[:, d1]) if len(d1) else np.zeros((nq, 0))
    degT = {"n_targets_1_code": int(len(d1)), "pairs": int(fmr_abs.size),
            "max_abs_Sf_minus_Sr_research": float(fmr_abs.max()) if fmr_abs.size else None,
            "max_abs_Sf_minus_Sr_S7": float(fmr7_abs.max()) if fmr7_abs.size else None,
            "fraction_pairs_exactly_0_research": float((fmr_abs == 0).mean()) if fmr_abs.size else None,
            "max_Sf_research": float(Sf[:, d1].max()) if len(d1) else None,
            "min_lnE_Nann_research": float(LE[:, d1].min()) if len(d1) else None,
            "min_lnE_Nann_S7": float(LE7[:, d1].min()) if len(d1) else None,
            "all_rows_degenerate_pairs": None}
    if len(d1):
        dq, dt, dc = tz["deg_q"], tz["deg_t"], tz["deg_counts"]
        mm = np.isin(dt, d1)
        degT["all_rows_degenerate_pairs"] = int((dc[mm, 0] == qlen[dq[mm]]).sum())
    log("target side done")

    # ---------------- as QUERIES
    hi, ht2, hsco = qz["hit_item"], qz["hit_target"], qz["hit_scores"].astype(np.float64)
    trev = qz["target_rev"]
    tl = qz["target_len"]
    q90i, q90i7 = qz["agg_q90"], qz["agg_q90_s7"]
    lam_i, lnk_i = lam_lnk(tab, ilen)
    Sq = hsco[:, 0] - hsco[:, 1]
    Sq7 = hsco[:, 2] - hsco[:, 3]
    Nsel = np.where(trev[ht2], N_rev, N_ann)
    bl = lnk_i[hi] + np.log(ilen[hi]) + np.log(Nsel)
    LQ = bl - lam_i[hi] * (Sq - q90i[hi])
    LQ7 = bl - lam_i[hi] * (Sq7 - q90i7[hi])
    trecs = read_fasta(f"{SP}/calib3di/results/targets.fasta")
    tseq = [s for _, s in trecs]
    assert [h.split()[0] for h, _ in trecs] == list(qz["target_ids"].astype(str))
    w = np.flatnonzero(((LQ <= 0) | (LQ7 <= 0)) & ~trev[ht2])
    pairs_q = [(iseq[hi[k]], tseq[ht2[k]]) for k in w]
    log(f"query-side annotated hit pairs at E<=1: {len(pairs_q)}; homology scoring")
    hs2 = homology_scores(pairs_q)
    homQ = np.zeros(len(hi), bool)
    for k, p in zip(w, pairs_q):
        if seq_evalue(hs2[p], ilen[hi[k]], N_ann) <= HOM_E:
            homQ[k] = True
    as_query = {}
    for form, L_ in (("research", LQ), ("S7", LQ7)):
        out = {}
        for g, m in G.items():
            r = {"n_items": int(m.sum())}
            inm = m[hi]
            for x in XS:
                for tag, extra in (("raw", np.ones(len(hi), bool)), ("filtered", ~homQ)):
                    hit = inm & ~trev[ht2] & (L_ <= math.log(x)) & extra
                    per = np.bincount(hi[hit], minlength=n_items)[m].astype(np.float64)
                    k = int(per.sum())
                    E_ = float(m.sum() * x)
                    r[f"{tag}@{x:g}"] = {"hits": k, "expected": E_, "ratio": k / E_ if E_ > 0 else None,
                                         "boot_ci": boot_ratio(per, np.full(len(per), x)),
                                         "poisson_ci": poisson_ci(k, E_), "items_with_hit": int((per > 0).sum())}
                hitr = inm & trev[ht2] & (L_ <= math.log(x))
                kr = int(hitr.sum())
                r[f"rev@{x:g}"] = {"hits": kr, "ratio": kr / (m.sum() * x), "poisson_ci": poisson_ci(kr, m.sum() * x)}
            out[g] = r
        as_query[form] = out
    degQ = {}
    for g in ("code:identical_q_rows", "code:query_any_degenerate_column"):
        m = G[g]
        degQ[g] = {"n": int(m.sum()),
                   "max_abs_Sf_minus_Sr_targets": float(np.nanmax(qz["agg_max_abs_fmr_tgt"][m])) if m.any() else None,
                   "max_abs_Sf_minus_Sr_panel": float(np.nanmax(qz["agg_max_abs_fmr_panel"][m])) if m.any() else None,
                   "q90": [float(x) for x in np.quantile(q90i[m], [0, .5, 1])] if m.any() else None}
    qagg = {}
    for g, m in G.items():
        if not m.any():
            continue
        qagg[g] = {"q90_quantiles_0_50_100": [float(x) for x in np.quantile(q90i[m], [0, .5, 1])],
                   "max_S_ann_minus_q90_median": float(np.median(qz["agg_max_S_ann"][m] - q90i[m])),
                   "median_Sf_ann_median": float(np.median(qz["agg_med_Sf_ann"][m]))}
    log("query side done")

    # ---------------- as TARGETS, homology-free null: the REVERSED item sequences (ESM3-encoded) as targets
    rev_t = None
    if os.path.exists(f"{IN}/rev/targets.npz"):
        rz = np.load(f"{IN}/rev/targets.npz")
        assert list(rz["item_ids"].astype(str)) == iids and np.array_equal(rz["item_len"], ilen)
        RS = rz["Sf"].astype(np.float64) - rz["Sr"]
        RS7 = RS.copy()
        RS7[rz["deg_q"], rz["deg_t"]] = rz["deg_sf7"].astype(np.float64) - rz["deg_sr7"]
        baseR = (lnk + np.log(qlen) + math.log(N_rev))[:, None]
        rcz = np.load(f"{IN}/rev/codes.npz")
        rev_t = {"N_ref": "N_rev (benchmark reversed-target residues; comparator = the benchmark's own reversed targets)",
                 "reversed_items_distinct_codes_lt2": int((rcz["distinct_codes"] < 2).sum()),
                 "degenerate_pairs": int(len(rz["deg_q"]))}
        for form, SS in (("research", RS), ("S7", RS7)):
            LR = baseR - lam[:, None] * (SS - q90[:, None])
            out = {}
            for g, m in G.items():
                cols = np.flatnonzero(m)
                NG = float(ilen[cols].sum())
                r = {"n_items": int(len(cols)), "residues": int(NG)}
                for x in XS:
                    per_t = (LR[:, cols] <= math.log(x)).sum(0).astype(np.float64)
                    exp_t = nq * x * ilen[cols] / N_rev
                    k = int(per_t.sum())
                    E_ = float(exp_t.sum())
                    r[f"raw@{x:g}"] = {"hits": k, "expected": E_, "ratio": k / E_ if E_ > 0 else None,
                                       "boot_ci": boot_ratio(per_t, exp_t), "poisson_ci": poisson_ci(k, E_),
                                       "targets_with_hit": int((per_t > 0).sum()),
                                       "max_hits_one_target": int(per_t.max()) if len(per_t) else 0}
                out[g] = r
            rev_t[form] = out
        log("reversed-decoy target side done")

    # ---------------- real-ESM3 single-residue items (fully degenerate: m = 1 / n = 1)
    l1 = None
    if os.path.exists(f"{IN}/l1/targets.npz"):
        lz, lc = np.load(f"{IN}/l1/targets.npz"), np.load(f"{IN}/l1/codes.npz")
        lq = np.load(f"{IN}/l1/queries.shard0of1.npz")
        lS = lz["Sf"].astype(np.float64) - lz["Sr"]
        lS7 = lS.copy()
        lS7[lz["deg_q"], lz["deg_t"]] = lz["deg_sf7"].astype(np.float64) - lz["deg_sr7"]
        lL = base - lam[:, None] * (lS - q90[:, None])
        lL7 = base - lam[:, None] * (lS7 - q90[:, None])
        lamq, lnkq = lam_lnk(tab, np.ones(1))
        hsl = lq["hit_scores"].astype(np.float64)
        l1 = {"items": [str(x) for x in lc["ids"]], "distinct_codes": lc["distinct_codes"].tolist(),
              "as_target": {"pairs": int(lS.size), "pairs_all_rows_degenerate": int(
                  (lz["deg_counts"][:, 0] == qlen[lz["deg_q"]]).sum()),
                  "max_abs_Sf_minus_Sr_research": float(np.abs(lS).max()),
                  "fraction_exactly_0_research": float((lS == 0).mean()),
                  "max_abs_Sf_minus_Sr_S7": float(np.abs(lS7).max()), "max_Sf_research": float(lz["Sf"].max()),
                  "max_Sf_S7": float(lz["deg_sf7"].max()),
                  "min_lnE_Nann_research": float(lL.min()), "min_lnE_Nann_S7": float(lL7.min()),
                  "hits_E_le_1_Nann": int((lL <= 0).sum()), "hits_E_le_1_Nann_S7": int((lL7 <= 0).sum())},
              "as_query": {"q90_research": lq["agg_q90"].tolist(), "q90_S7": lq["agg_q90_s7"].tolist(),
                           "max_abs_Sf_minus_Sr_targets": float(np.max(lq["agg_max_abs_fmr_tgt"])),
                           "max_abs_Sf_minus_Sr_panel": float(np.max(lq["agg_max_abs_fmr_panel"])),
                           "pairs_with_all_columns_degenerate": int(np.sum(lq["agg_tgt_deg_pairs"])),
                           "max_Sf_ann_research": float(np.max(lq["agg_max_Sf_ann"])),
                           "max_S_minus_Sr_S7_ann": float(np.max(lq["agg_max_S7_ann"])),
                           "kept_pairs_lnE_Nmin_le_ln100": int(len(hsl)),
                           "kept_pairs_max_abs_S": float(np.abs(hsl[:, 0] - hsl[:, 1]).max()) if len(hsl) else None,
                           "lnE_every_target_at_N_ann": float(lnkq[0] + 0.0 + math.log(N_ann)),
                           "E_every_target_at_N_ann": float(math.exp(lnkq[0] + math.log(N_ann))),
                           "N_below_which_every_target_has_E_le_1": float(math.exp(-lnkq[0])),
                           "note": "S = S_f - S_r = 0 for every target and panel member, Q90 = 0 -> ln E = lnK(1) + ln 1 "
                                   "+ ln N for EVERY target (no information)"}}
        log("L=1 set done")

    # ---------------- baseline
    base_b = baseline()
    log("baseline done")

    res = {"groups_definition": {g: int(m.sum()) for g, m in G.items()},
           "code_level": code, "distinct_code_histogram_all_items(capped at 20)": code_hist,
           "prevalence_lt2_distinct_codes": prev,
           "fully_degenerate": {"targets_1_code": degT, "queries": degQ},
           "fp_as_target": as_target, "fp_as_query": as_query, "sticky_targets": sticky,
           "fp_as_target_reversed_decoys": rev_t, "single_residue_real_esm3": l1,
           "query_side_null_shape": qagg, "baseline_benchmark": base_b,
           "homology_filter": {"target_side_pairs_scored": len(pairs_t), "target_side_homologous": int(homT.sum()),
                               "query_side_pairs_scored": len(pairs_q), "query_side_homologous": int(homQ.sum()),
                               "definition": f"biotite align_optimal local BLOSUM62, gap (-12,-1) = BLAST 11/1; "
                                             f"E = {KA_K} m N_ann exp(-{KA_LAMBDA} S) <= {HOM_E}"},
           "N_ann": N_ann, "N_rev": N_rev, "n_benchmark_queries": nq,
           "inputs_md5": {os.path.basename(p): md5(p) for p in (f"{IN}/codes.npz", f"{IN}/targets.npz",
                                                               f"{IN}/queries.npz", f"{IN}/items.tsv",
                                                               f"{IN}/seq_census.json")}}
    json.dump(res, open(f"{TRACK}/census_report.json", "w"), indent=1)
    log("wrote", f"{TRACK}/census_report.json")


if __name__ == "__main__":
    main()
