#!/usr/bin/env python
"""Self-hit study, step 4: why esm-codes-v1 self hits of short / low-complexity sequences get weak E-values.

Inputs (DATA = the study's data folder): enc/<set>.npz (encode_fm.py), score/<set>.npz (selfscore.py), the P6 check
outputs (E_inf / B_inf of job 60008416, copied to p6_nersc/), census/items.tsv (PDB unique-sequence lengths).
FoldMatch (P6 tree, read-only) supplies the registered esm-codes-v1 table, its E arithmetic and the BLOSUM62 path's
Karlin-Altschul E.

    PYTHONPATH=<wt-P6/src> python analyze.py DATA P6_RUNS OUT_DIR

Writes OUT_DIR/results.json, OUT_DIR/tables.md and OUT_DIR/figs/*.png.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import seqfeat  # noqa: E402

from foldmatch.search import code_statistics as cs  # noqa: E402
from foldmatch.search import karlin_altschul as ka  # noqa: E402
from foldmatch.search import scoring_system as ss  # noqa: E402

SYSTEM = ss.get("esm-codes-v1")
LN1E3 = math.log(1e-3)
SETS = ("u50_q200", "u50_db800", "scop_q", "pdb_r600")
LABEL = {"u50_db800": "UniRef50 random (800)", "u50_q200": "P6 queries (200)", "scop_q": "SCOP calibration queries",
         "pdb_r600": "PDB random unique (600)"}
SANDBOX_PANEL = "/Users/joan/devel/foldmatch-p2p3-sandbox/artefacts/embcomp/panel/fit2/results"
LEN_BINS = [(1, 20), (21, 50), (51, 100), (101, 200), (201, 300), (301, 600), (601, 966), (967, 100000)]


def lam_lnk(m):
    return cs.lam_lnk(np.asarray(m, np.float64), SYSTEM)


def ln_e(s_prime, m, n):
    la, lk = lam_lnk(m)
    return lk + np.log(m) + math.log(n) - la * s_prime


def need_sprime(m, n, e=1e-3):
    """S' a query of length m needs for E <= e at N residues."""
    la, lk = lam_lnk(m)
    return (lk + np.log(m) + math.log(n) - math.log(e)) / la


def load_set(data, name):
    e = np.load(os.path.join(data, "enc", f"{name}.npz"), allow_pickle=False)
    s = np.load(os.path.join(data, "score", f"{name}.npz"), allow_pickle=False)
    assert [str(x) for x in e["ids"]] == [str(x) for x in s["ids"]]
    off = e["offsets"]
    sc = e["seq_concat"].tobytes().decode("ascii")
    hc, ho = e["header_concat"].tobytes().decode("utf-8"), e["header_offsets"]
    seqs = [sc[off[i]:off[i + 1]] for i in range(len(off) - 1)]
    heads = [hc[ho[i]:ho[i + 1]] for i in range(len(ho) - 1)]
    m = e["lengths"].astype(np.float64)
    d = {k: s[k] for k in ("S_f", "S_r", "q90", "z_path_sum", "n_match", "n_diag", "pos_off", "distinct_codes",
                           "db_S_f", "db_S_r")}
    d.update(set=name, ids=[str(x) for x in e["ids"]], seqs=seqs, heads=heads, m=m)
    per = {}
    for k in ("z_ii", "a_ii", "b_ii", "s_ii", "mu", "sd", "beats"):
        v = s[k]
        per[k] = [v[off[i]:off[i + 1]] for i in range(len(off) - 1)]
    d["per"] = per
    S = d["S_f"].astype(np.float64) - d["S_r"].astype(np.float64)
    d["S"] = S
    d["Sp"] = S - d["q90"]
    la, lk = lam_lnk(m)
    d["lam"], d["lnK"] = la, lk
    feats = [seqfeat.features(sq, h) for sq, h in zip(seqs, heads)]
    for k in feats[0]:
        d["f_" + k] = np.array([f[k] for f in feats])
    d["mean_z"] = np.array([x.mean() for x in per["z_ii"]])
    d["mean_s"] = np.array([x.mean() for x in per["s_ii"]])
    d["mean_mu"] = np.array([x.mean() for x in per["mu"]])
    d["mean_sd"] = np.array([x.mean() for x in per["sd"]])
    d["mean_a"] = np.array([x.mean() for x in per["a_ii"]])
    d["mean_b"] = np.array([x.mean() for x in per["b_ii"]])
    d["beaten"] = np.array([(x > 0).mean() for x in per["beats"]])
    d["frac_zlow"] = np.array([(x < SYSTEM.offset).mean() for x in per["z_ii"]])   # diagonal cells scoring <= 0
    d["blosum_self"] = np.array([sum(int(seqfeat._blosum()[seqfeat._IDX[ord(a)], seqfeat._IDX[ord(a)]])
                                     for a in sq) for sq in seqs])
    return d


def q(a, p):
    a = np.asarray(a, np.float64)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, p)) if len(a) else math.nan


def p6_rows(runs, label):
    meta = json.load(open(os.path.join(runs, label, "meta.json")))
    F = meta["fields"]
    return [dict(zip(F, line.rstrip("\n").split("\t"))) for line in open(os.path.join(runs, label, "out.tsv"))]


def pdb_unique_mean_length(data):
    L = []
    with open(os.path.join(data, "census", "items.tsv"), encoding="utf-8") as fh:
        cols = fh.readline().rstrip("\n").split("\t")
        for line in fh:
            r = dict(zip(cols, line.rstrip("\n").split("\t")))
            if "random" in r["pdb_cats"].split(","):
                L.append(int(r["length"]))
    return float(np.mean(L)), len(L)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def fmt_e(x):
    return f"{x:.2g}" if np.isfinite(x) else "nan"


def main():
    data, runs, outdir = sys.argv[1:4]
    os.makedirs(os.path.join(outdir, "figs"), exist_ok=True)
    R = {}
    md = []
    pdb_mean_len, pdb_n = pdb_unique_mean_length(data)
    NS = {"10k sample (P6 check)": 2957247.0,
          "PDB unique seqres": 187064 * pdb_mean_len,
          "PDB all protein chains": 321321772.0,
          "UniRef50 2026_03": 12182780325.0}
    R["N"] = NS
    R["pdb_unique_mean_length"] = [pdb_mean_len, pdb_n]
    N10 = NS["10k sample (P6 check)"]

    # ------------------------------------------------------------------ 1. P6 in-database self hits, decomposed
    rows = p6_rows(runs, "E_inf")
    brows = p6_rows(runs, "B_inf")
    selfh = {r["query"]: r for r in rows if r["query"] == r["target"]}
    bself = {r["query"]: r for r in brows if r["query"] == r["target"]}
    ids = sorted(selfh, key=lambda k: int(selfh[k]["qlen"]))
    m = np.array([float(selfh[k]["qlen"]) for k in ids])
    Sf = np.array([float(selfh[k]["raw"]) for k in ids])
    Sr = np.array([float(selfh[k]["rawrev"]) for k in ids])
    Q = np.array([float(selfh[k]["q90"]) for k in ids])
    Sp = np.array([float(selfh[k]["rawadj"]) for k in ids])
    E = np.array([float(selfh[k]["evalue"]) for k in ids])
    lnE = ln_e(Sp, m, N10)
    assert np.allclose(np.exp(lnE), E, rtol=2e-3, atol=0), "S11 arithmetic does not reproduce the P6 E column"
    la, lk = lam_lnk(m)
    space = lk + np.log(m) + math.log(N10)
    sig, rev, shift = la * Sf, la * Sr, la * Q
    # extrapolated lambda / lnK below the first centre (instead of the clamp), for the short-query question
    c = np.log(np.array(SYSTEM.bin_centres))
    lam_t, lnk_t = np.array(SYSTEM.lam), np.array(SYSTEM.lnK)
    sl_l = (lam_t[1] - lam_t[0]) / (c[1] - c[0])
    sl_k = (lnk_t[1] - lnk_t[0]) / (c[1] - c[0])
    la_x = np.where(m < SYSTEM.bin_centres[0], lam_t[0] + sl_l * (np.log(m) - c[0]), la)
    lk_x = np.where(m < SYSTEM.bin_centres[0], lnk_t[0] + sl_k * (np.log(m) - c[0]), lk)
    lnE_x = lk_x + np.log(m) + math.log(N10) - la_x * Sp
    ok = lnE <= LN1E3
    ok_nor = (space - la * (Sf - Q)) <= LN1E3
    ok_noq = (space - la * (Sf - Sr)) <= LN1E3
    ok_none = (space - la * Sf) <= LN1E3
    ok_x = lnE_x <= LN1E3
    p6 = dict(n=len(ids), e_le_1e3=int(ok.sum()), without_S_r=int(ok_nor.sum()), without_Q90=int(ok_noq.sum()),
              without_both=int(ok_none.sum()), lambda_extrapolated_below_32aa=int(ok_x.sum()))
    tab = []
    for lo, hi in [(21, 100), (101, 300), (301, 966), (967, 5000), (1, 20)]:
        k = (m >= lo) & (m <= hi)
        if not k.any():
            continue
        tab.append([f"{lo}-{hi}" if hi < 5000 else f">{lo - 1}", int(k.sum()), f"{np.median(Sf[k] / m[k]):.2f}",
                    f"{np.median(Sr[k]):.1f}", f"{np.median(Q[k]):.2f}", f"{np.median(la[k]):.4f}",
                    f"{np.median(sig[k]):.1f}", f"{np.median(rev[k]):.2f}", f"{np.median(shift[k]):.2f}",
                    f"{np.median(space[k]):.1f}", f"{np.median(need_sprime(m[k], N10) / m[k]):.2f}",
                    f"{int(ok[k].sum())}/{int(k.sum())}", int(ok_nor[k].sum()), int(ok_noq[k].sum()),
                    int(ok_none[k].sum()), int(ok_x[k].sum())])
        p6[f"bin_{lo}_{hi}"] = dict(n=int(k.sum()), median_lnE=float(np.median(lnE[k])))
    md.append("## T1. P6 in-database self hits (job 60008416 E_inf), ln E decomposed\n\n"
              "ln E = [lnK(m) + ln m + ln N] - lambda(m) S_f + lambda(m) S_r + lambda(m) Q90, N = 2,957,247. "
              "Columns: medians; nats = lambda x score; 'need/res' = S' per residue needed for E <= 1e-3; the last "
              "four columns count self hits that would reach E <= 1e-3 without S_r, without Q90, without both, or "
              "with lambda/lnK extrapolated in ln m below 32.6 aa instead of clamped.\n")
    md.append(md_table(["m", "n", "S_f/m", "S_r", "Q90", "lambda", "nats S_f", "nats S_r", "nats Q90", "nats space",
                        "need/res", "E<=1e-3", "no S_r", "no Q90", "neither", "lam extrap"], tab))
    R["p6_selfhits"] = p6
    # BLOSUM62 in the same run
    bE = np.array([float(bself[k]["evalue"]) if k in bself else np.nan for k in ids])
    R["p6_blosum_self_e_le_1e10"] = int(np.nansum(bE <= 1e-10))
    R["p6_blosum_found"] = int(np.isfinite(bE).sum())

    # ------------------------------------------------------------------ 2. populations
    S = {name: load_set(data, name) for name in SETS if os.path.exists(os.path.join(data, "score", f"{name}.npz"))}
    bg = np.load(os.path.join(data, "probe", "bg.npz"), allow_pickle=False)
    R["panel_code_homogeneity"] = float(bg["panel_h"])
    for name, x in S.items():
        assert [str(v) for v in bg[f"{name}__ids"]] == x["ids"]
        for k in ("h", "mu_p", "zP_ii", "mu_self"):
            x[k] = bg[f"{name}__{k}"]
    # check the local recomputation against P6 (in-database queries)
    d = S["u50_q200"]
    idx = {k: i for i, k in enumerate(d["ids"])}
    rel = [abs(float(selfh[k]["raw"]) - d["S_f"][idx[k]]) / float(selfh[k]["raw"]) for k in ids]
    reldb = [abs(float(selfh[k]["raw"]) - d["db_S_f"][idx[k]]) / float(selfh[k]["raw"]) for k in ids]
    R["recompute_vs_p6"] = dict(S_f_rel_median=float(np.median(rel)), S_f_rel_max=float(np.max(rel)),
                                S_f_rel_median_db_codes=float(np.median(reldb)),
                                q90_abs_median=float(np.median([abs(float(selfh[k]["q90"]) - d["q90"][idx[k]])
                                                                for k in ids])))
    calib = lambda x: (x["m"] >= 21) & (x["m"] <= 966)  # noqa: E731
    pop = []
    for name, x in S.items():
        k = calib(x)
        row = [LABEL[name], int(k.sum()), f"{np.median(x['m'][k]):.0f}", f"{q(x['S_f'][k] / x['m'][k], 50):.2f}",
               f"{q(x['S_f'][k] / x['m'][k], 10):.2f}-{q(x['S_f'][k] / x['m'][k], 90):.2f}",
               f"{q(x['S_r'][k] / x['S_f'][k], 50):.3f}", f"{q(x['q90'][k], 50):.2f}",
               f"{q(x['mean_z'][k], 50):.2f}", f"{q(x['f_lc_frac'][k], 50):.2f}",
               f"{100 * np.mean(x['f_lc_frac'][k] > 0.1):.0f}%"]
        for nn in NS.values():
            row.append(f"{100 * np.mean(ln_e(x['Sp'][k], x['m'][k], nn) <= LN1E3):.0f}%")
        pop.append(row)
        R.setdefault("populations", {})[name] = dict(
            n=int(k.sum()), median_len=float(np.median(x["m"][k])),
            sf_per_res=[q(x["S_f"][k] / x["m"][k], p) for p in (10, 25, 50, 75, 90)],
            frac_e1e3={nk: float(np.mean(ln_e(x["Sp"][k], x["m"][k], nn) <= LN1E3)) for nk, nn in NS.items()},
            frac_e1e10={nk: float(np.mean(ln_e(x["Sp"][k], x["m"][k], nn) <= math.log(1e-10))) for nk, nn in NS.items()},
            lc_gt01=float(np.mean(x["f_lc_frac"][k] > 0.1)))
    md.append("\n\n## T2. Self hits by population (queries of 21-966 aa, the calibrated query domain)\n\n"
              "Local recomputation (FoldMatch ESM3 + head on MPS, FoldMatch kernel, shipped panel). The last four "
              "columns: share of self hits with E <= 1e-3 at N = " + ", ".join(f"{k} ({v:.3g})" for k, v in NS.items())
              + ".\n")
    md.append(md_table(["population", "n", "median m", "S_f/m median", "S_f/m P10-P90", "S_r/S_f", "Q90",
                        "mean z_ii", "LC frac median", "LC > 0.1"] + [f"E<=1e-3 @ {k}" for k in NS], pop))

    # by length bin, per population
    lb = []
    for name, x in S.items():
        for lo, hi in LEN_BINS:
            k = (x["m"] >= lo) & (x["m"] <= hi)
            if k.sum() < 5:
                continue
            lb.append([LABEL[name], f"{lo}-{hi}" if hi < 100000 else f">{lo - 1}", int(k.sum()),
                       f"{q(x['S_f'][k] / x['m'][k], 50):.2f}", f"{q(need_sprime(x['m'][k], N10) / x['m'][k], 50):.2f}",
                       f"{q(need_sprime(x['m'][k], NS['UniRef50 2026_03']) / x['m'][k], 50):.2f}",
                       fmt_e(np.exp(q(ln_e(x['Sp'][k], x['m'][k], N10), 50))),
                       f"{100 * np.mean(ln_e(x['Sp'][k], x['m'][k], N10) <= LN1E3):.0f}%",
                       f"{100 * np.mean(ln_e(x['Sp'][k], x['m'][k], NS['PDB unique seqres']) <= LN1E3):.0f}%",
                       f"{100 * np.mean(ln_e(x['Sp'][k], x['m'][k], NS['UniRef50 2026_03']) <= LN1E3):.0f}%"])
    md.append("\n\n## T3. By query length\n\n'need/res' = S'/m needed for E <= 1e-3 at the 10k N and at UniRef50 N. "
              "(Queries outside 21-966 aa are outside S13 and get ecal = 0; shown for completeness.)\n")
    md.append(md_table(["population", "m", "n", "S_f/m", "need/res 10k", "need/res U50", "median E 10k",
                        "E<=1e-3 10k", "E<=1e-3 PDB", "E<=1e-3 U50"], lb))

    # ------------------------------------------------------------------ 3. what the per-residue self score follows
    from scipy.stats import spearmanr
    allx = {k: np.concatenate([S[n][k] for n in ("u50_db800", "u50_q200") if n in S])
            for k in ("S_f", "m", "f_lc_frac", "f_topidp", "f_comp_entropy", "f_kmer3_redundancy", "f_blosum_repeat",
                      "f_uncharacterized", "f_fragment", "f_n", "mean_s", "mean_mu", "mean_sd", "mean_z", "beaten",
                      "pos_off", "mean_a", "mean_b", "frac_zlow", "S_r", "q90", "h", "mu_p", "zP_ii")}
    k = (allx["m"] >= 21) & (allx["m"] <= 966)
    spr = allx["S_f"][k] / allx["m"][k]
    corr = []
    for f, lab in [("m", "length (ln)"), ("f_lc_frac", "LC fraction"), ("f_topidp", "TOP-IDP (disorder)"),
                   ("f_comp_entropy", "composition entropy"), ("f_kmer3_redundancy", "3-mer redundancy"),
                   ("f_blosum_repeat", "BLOSUM62 internal repeat"), ("f_uncharacterized", "'uncharacterized'"),
                   ("f_fragment", "'fragment'"), ("f_n", "cluster size n"), ("mean_s", "mean s_ii"),
                   ("mean_mu", "mean row mean mu_i"), ("mean_sd", "mean row std sigma_i"),
                   ("beaten", "share of residues out-scored in their row"), ("pos_off", "off-diagonal z>1 share"),
                   ("h", "code homogeneity h = |mean code| / 8"),
                   ("mu_p", "mean similarity to the panel background mu_i^P")]:
        r, p = spearmanr(allx[f][k].astype(np.float64), spr)
        corr.append([lab, f"{r:+.2f}", f"{p:.1e}"])
        R.setdefault("spearman_sf_per_res_u50", {})[lab] = [float(r), float(p)]
    md.append("\n\n## T4. UniRef50 (1,000 sequences, 21-966 aa): Spearman correlation with S_f/m\n")
    md.append(md_table(["feature", "rho", "p"], corr))
    # variance of ln(S_f/m) explained by least squares on feature sets (UniRef50, 21-966 aa)
    y = np.log(np.maximum(spr, 1e-3))
    feats_all = dict(lnm=np.log(allx["m"][k]), h=allx["h"][k], lc=allx["f_lc_frac"][k], idp=allx["f_topidp"][k],
                     unch=allx["f_uncharacterized"][k].astype(float), ent=allx["f_comp_entropy"][k],
                     lnn=np.log(np.maximum(allx["f_n"][k], 1)))
    r2 = []
    for lab, keys in [("length (ln m)", ["lnm"]), ("homogeneity h", ["h"]), ("h + ln m", ["h", "lnm"]),
                      ("sequence features only (ln m, LC, TOP-IDP, entropy, uncharacterized, ln n)",
                       ["lnm", "lc", "idp", "ent", "unch", "lnn"]),
                      ("all of the above + h", ["lnm", "lc", "idp", "ent", "unch", "lnn", "h"])]:
        X = np.column_stack([np.ones(len(y))] + [feats_all[c] for c in keys])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        res = y - X @ beta
        r2.append([lab, f"{1 - res.var() / y.var():.2f}"])
        R.setdefault("r2_ln_sf_per_res_u50", {})[lab] = float(1 - res.var() / y.var())
    md.append("\n\nVariance of ln(S_f/m) explained (least squares, UniRef50 21-966 aa):\n")
    md.append(md_table(["predictors", "R^2"], r2))

    def t5row(lab, x, kk):
        mm = x["m"][kk]
        return [lab, int(kk.sum()), f"{np.median(mm):.0f}", f"{np.median(x['S_f'][kk] / mm):.2f}",
                f"{np.median(x['h'][kk]):.3f}", f"{np.median(x['mean_s'][kk]):.3f}",
                f"{np.median(x['mean_mu'][kk]):.3f}", f"{np.median(x['mu_p'][kk]):.3f}",
                f"{np.median(x['mean_sd'][kk]):.3f}", f"{np.median(x['mean_z'][kk]):.2f}",
                f"{np.median(x['zP_ii'][kk]):.2f}", f"{np.median(x['frac_zlow'][kk]):.2f}",
                f"{np.median(x['beaten'][kk]):.2f}", f"{np.median(x['pos_off'][kk]):.3f}",
                f"{np.median(x['S_r'][kk] / x['S_f'][kk]):.3f}", f"{np.median(x['q90'][kk]):.2f}"]

    grp = []
    defs = [("LC = 0", allx["f_lc_frac"] == 0), ("LC (0, 0.1]", (allx["f_lc_frac"] > 0) & (allx["f_lc_frac"] <= 0.1)),
            ("LC > 0.1", allx["f_lc_frac"] > 0.1), ("LC > 0.3", allx["f_lc_frac"] > 0.3),
            ("TOP-IDP <= 0.0673", allx["f_topidp"] <= 0.0673), ("TOP-IDP > 0.0998", allx["f_topidp"] > 0.0998),
            ("'uncharacterized'", allx["f_uncharacterized"]), ("annotated (not uncharacterized)",
                                                               ~allx["f_uncharacterized"]),
            ("'fragment'", allx["f_fragment"]), ("singleton cluster (n=1)", allx["f_n"] == 1),
            ("cluster n >= 10", allx["f_n"] >= 10)]
    for lab, sel in defs:
        if (k & sel).sum() >= 5:
            grp.append(t5row("UniRef50 " + lab, allx, k & sel))
    for name in ("scop_q", "pdb_r600"):
        if name in S:
            grp.append(t5row(LABEL[name], S[name], calib(S[name])))
    # code homogeneity tertiles over all four populations pooled
    pool = {kk: np.concatenate([S[n][kk] for n in S]) for kk in allx if kk in S["u50_db800"]}
    kp = (pool["m"] >= 21) & (pool["m"] <= 966)
    edges_h = np.quantile(pool["h"][kp], [1 / 3, 2 / 3])
    R["h_tertile_edges_pooled"] = edges_h.tolist()
    for lab, sel in ((f"all populations, h <= {edges_h[0]:.3f}", pool["h"] <= edges_h[0]),
                     (f"all populations, h in ({edges_h[0]:.3f}, {edges_h[1]:.3f}]",
                      (pool["h"] > edges_h[0]) & (pool["h"] <= edges_h[1])),
                     (f"all populations, h > {edges_h[1]:.3f}", pool["h"] > edges_h[1])):
        grp.append(t5row(lab, pool, kp & sel))
    md.append("\n\n## T5. The self matrix, per residue (medians over sequences of per-sequence means; 21-966 aa)\n\n"
              "h = |mean residue code| / 8 (1 = all residues share one code; the panel's 355,626 residues give "
              f"{float(bg['panel_h']):.3f}); s_ii = diagonal similarity cos(q_i, d_i); mu_i, sigma_i = mean and std "
              "of row i of the self matrix (S7); mu_i^P = mean similarity of residue i to the panel's codes; z_ii = "
              "the diagonal rowcol z; z_ii^P = the diagonal standardised against the panel background; 'z_ii<=1' = "
              "share of diagonal cells whose SW cell score z-1 is <= 0; 'beaten' = share of residues i with some "
              "j != i where s_ij > s_ii; 'offdiag+' = share of cells |i-j|>3 with z > 1 (internal repeats).\n")
    md.append(md_table(["group", "n", "median m", "S_f/m", "h", "s_ii", "mu_i", "mu_i^P", "sigma_i", "z_ii",
                        "z_ii^P", "z_ii<=1", "beaten", "offdiag+", "S_r/S_f", "Q90"], grp))

    # weakest and strongest P6 in-database self hits (the P6 E column), with the local diagnostics
    xq = S["u50_q200"]
    pos = {kk: i for i, kk in enumerate(xq["ids"])}
    ex = []
    order = np.argsort([float(selfh[kk]["evalue"]) / 1.0 for kk in ids])
    pick = [ids[i] for i in order if 21 <= m[i] <= 966]
    for kk in pick[-12:][::-1] + pick[:4]:
        i = pos[kk]
        hd = xq["heads"][i].split(" n=")[0].split(" ", 1)
        ex.append([kk, int(xq["m"][i]), (hd[1] if len(hd) > 1 else "")[:42], f"{float(selfh[kk]['evalue']):.2g}",
                   f"{float(selfh[kk]['raw']) / xq['m'][i]:.2f}", f"{xq['h'][i]:.2f}", f"{xq['f_lc_frac'][i]:.2f}",
                   f"{xq['f_topidp'][i]:.2f}", f"{xq['mean_mu'][i]:.2f}", f"{xq['mean_z'][i]:.2f}"])
    md.append("\n\n## T5b. P6 in-database self hits: the 12 weakest (21-966 aa) and the 4 strongest\n")
    md.append(md_table(["id", "m", "name", "E (P6)", "S_f/m", "h", "LC", "TOP-IDP", "mu_i", "z_ii"], ex))

    # ------------------------------------------------------------------ 4. N scaling
    grid = np.logspace(5, 11, 25)
    ns = []
    for name, x in S.items():
        kk = calib(x)
        fr = [float(np.mean(ln_e(x["Sp"][kk], x["m"][kk], nn) <= LN1E3)) for nn in grid]
        R.setdefault("n_scaling", {})[name] = dict(N=grid.tolist(), frac_e1e3=fr)
        for lo, hi in [(21, 100), (101, 300), (301, 966)]:
            k2 = kk & (x["m"] >= lo) & (x["m"] <= hi)
            if k2.sum() < 5:
                continue
            ns.append([LABEL[name], f"{lo}-{hi}", int(k2.sum())] +
                      [f"{100 * np.mean(ln_e(x['Sp'][k2], x['m'][k2], nn) <= LN1E3):.0f}%" for nn in NS.values()])
    md.append("\n\n## T6. Share of self hits at E <= 1e-3 as the database grows (E is proportional to N: "
              "every factor 10 in N costs 2.3 nats)\n")
    md.append(md_table(["population", "m", "n"] + list(NS), ns))

    # BLOSUM62 comparison (same sequences, the BLOSUM62 path's E arithmetic, full-length self alignment)
    bl = []
    params = ka.params_for(11, 1)
    for name, x in S.items():
        bits_esm = (x["lam"] * x["Sp"] - x["lnK"]) / math.log(2)
        bits_b = np.array([ka.bit_score(int(b), params) for b in x["blosum_self"]])
        eb = np.array([ka.evalue(int(b), int(mm), int(NS["UniRef50 2026_03"]), params)
                       for b, mm in zip(x["blosum_self"], x["m"])])
        for lo, hi in [(21, 100), (101, 300), (301, 966)]:
            kk = (x["m"] >= lo) & (x["m"] <= hi)
            if kk.sum() < 5:
                continue
            bl.append([LABEL[name], f"{lo}-{hi}", int(kk.sum()), f"{np.median(bits_esm[kk] / x['m'][kk]):.3f}",
                       f"{np.median(bits_b[kk] / x['m'][kk]):.2f}",
                       f"{100 * np.mean(ln_e(x['Sp'][kk], x['m'][kk], NS['UniRef50 2026_03']) <= LN1E3):.0f}%",
                       f"{100 * np.mean(eb[kk] <= 1e-3):.0f}%"])
    md.append("\n\n## T7. Information per residue of a self hit: esm-codes-v1 bits/m vs BLOSUM62 bits/m "
              "(BLOSUM62: full-length self score, FoldMatch's 11/1 Karlin-Altschul parameters, no composition "
              "correction)\n")
    md.append(md_table(["population", "m", "n", "esm-codes bits/res", "BLOSUM62 bits/res", "esm-codes E<=1e-3 @ U50",
                        "BLOSUM62 E<=1e-3 @ U50"], bl))

    # T8: what the reversed-query subtraction takes from self scores, by low-complexity fraction (all four sets)
    allp = {kk: np.concatenate([S[n][kk] for n in S]) for kk in ("m", "S_f", "S_r", "q90", "f_lc_frac")}
    kp = (allp["m"] >= 21) & (allp["m"] <= 966)
    la_p, lk_p = lam_lnk(allp["m"])
    rat = allp["S_r"] / np.maximum(allp["S_f"], 1e-9)
    t8 = []
    lcv = allp["f_lc_frac"]
    for lab, sel in (("0", lcv == 0), ("(0, 0.1]", (lcv > 0) & (lcv <= 0.1)), ("(0.1, 0.3]", (lcv > 0.1) & (lcv <= 0.3)),
                     ("(0.3, 0.6]", (lcv > 0.3) & (lcv <= 0.6)), ("> 0.6", lcv > 0.6), ("all", lcv >= 0)):
        kk = kp & sel
        lifted = []
        for nn in (N10, NS["UniRef50 2026_03"]):
            base = lk_p + np.log(allp["m"]) + math.log(nn)
            w = base - la_p * (allp["S_f"] - allp["S_r"] - allp["q90"]) <= LN1E3
            wo = base - la_p * (allp["S_f"] - allp["q90"]) <= LN1E3
            lifted.append(int((wo & ~w)[kk].sum()))
        t8.append([lab, int(kk.sum()), f"{np.median(rat[kk]):.3f} ({np.percentile(rat[kk], 90):.3f})",
                   f"{np.median(la_p[kk] * allp['S_r'][kk]):.2f} ({np.percentile(la_p[kk] * allp['S_r'][kk], 90):.2f})",
                   f"{lifted[0]} / {lifted[1]}"])
    md.append("\n\n## T8. Reversed-query cost by low-complexity fraction (all sets, 21-966 aa)\n")
    md.append(md_table(["LC fraction", "n", "S_r/S_f median (P90)", "lambda S_r nats median (P90)",
                        "self hits dropping S_r lifts over 1e-3 (10k / U50)"], t8))
    # T9: code homogeneity of the shipped panel's own strata
    pz = np.load(os.path.join(SANDBOX_PANEL, "shipped_panel_codes.npz"), allow_pickle=False)
    lut = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).astype(np.float32) * 2 - 1
    Dp, po = lut[pz["codes"]].reshape(-1, 64), pz["offsets"]
    hp = {str(i): float(np.linalg.norm(Dp[po[k]:po[k + 1]].mean(0)) / 8) for k, i in enumerate(pz["ids"])}
    with open(os.path.join(SANDBOX_PANEL, "shipped_panel.tsv"), encoding="utf-8") as fh:
        tsv = list(csv.DictReader(fh, delimiter="\t"))
    hh = np.array([hp[r["cluster_id"]] for r in tsv])
    col = {c: np.array([int(r[c]) for r in tsv]) for c in ("lc_bin", "dis_bin", "len_bin")}
    t9 = []
    for lab, sel in (("LC 0", col["lc_bin"] == 0), ("LC (0, 0.1]", col["lc_bin"] == 1), ("LC > 0.1", col["lc_bin"] == 2),
                     ("D0 (ordered)", col["dis_bin"] == 0), ("D1", col["dis_bin"] == 1),
                     ("D2 (disordered)", col["dis_bin"] == 2), ("30-134 aa", col["len_bin"] == 0),
                     (">= 700 aa", col["len_bin"] == 7)):
        t9.append([lab, int(sel.sum()), f"{np.median(hh[sel]):.3f}", f"{100 * np.mean(hh[sel] > 0.409):.0f}%"])
    md.append("\n\n## T9. Code homogeneity of the shipped panel's strata\n")
    md.append(md_table(["panel stratum", "n", "median h", "h > 0.409"], t9))

    with open(os.path.join(outdir, "results.json"), "w") as fh:
        json.dump(R, fh, indent=1, default=float)
    with open(os.path.join(outdir, "tables.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))
    figures(S, NS, outdir)


def figures(S, NS, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"u50_db800": "#1f77b4", "u50_q200": "#6baed6", "scop_q": "#d62728", "pdb_r600": "#2ca02c"}
    mm = np.logspace(math.log10(21), math.log10(966), 200)
    # F1: S_f/m vs length with the E <= 1e-3 requirement
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name in ("scop_q", "pdb_r600", "u50_db800", "u50_q200"):
        if name not in S:
            continue
        x = S[name]
        ax.scatter(x["m"], x["S_f"] / x["m"], s=7, alpha=0.45, color=col[name], label=LABEL[name], lw=0)
    for (lab, nn), ls in zip(NS.items(), ["-", "--", "-.", ":"]):
        ax.plot(mm, need_sprime(mm, nn) / mm, color="k", ls=ls, lw=1.2, label=f"S'/m needed, E=1e-3, N={nn:.2g}")
    ax.set_xscale("log")
    ax.set_xlim(10, 2100)
    ax.set_ylim(0, 6)
    ax.axvspan(21, 966, color="0.93", zorder=-1)
    ax.set_xlabel("sequence length m (calibrated query domain shaded)")
    ax.set_ylabel("self S_f per residue")
    ax.legend(fontsize=7, loc="upper right", frameon=False)
    ax.set_title("esm-codes-v1 self hits: per-residue score vs what E <= 1e-3 needs", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "figs", "f1_selfscore_vs_length.png"), dpi=130)
    plt.close(fig)
    # F2: what the per-residue self score follows
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for name in ("scop_q", "pdb_r600", "u50_db800", "u50_q200"):
        if name not in S:
            continue
        x = S[name]
        k = (x["m"] >= 21) & (x["m"] <= 966)
        kw = dict(s=7, alpha=0.45, color=col[name], lw=0)
        axs[0].scatter(x["h"][k], x["S_f"][k] / x["m"][k], label=LABEL[name], **kw)
        axs[1].scatter(x["f_lc_frac"][k], x["S_f"][k] / x["m"][k], **kw)
        axs[2].scatter(x["mean_mu"][k], x["mean_s"][k], **kw)
    axs[0].set_xlabel("code homogeneity h = |mean residue code| / 8")
    axs[0].set_ylabel("self S_f per residue")
    axs[0].legend(fontsize=7, frameon=False)
    axs[1].set_xlabel("low-complexity fraction (panel SEG-like)")
    axs[1].set_ylabel("self S_f per residue")
    axs[2].set_xlabel("mean row background mu_i of the self matrix")
    axs[2].set_ylabel("mean diagonal similarity s_ii")
    axs[2].plot([0, 0.6], [0, 0.6], color="k", lw=0.8, ls=":")
    for a_ in axs:
        a_.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "figs", "f2_homogeneity.png"), dpi=130)
    plt.close(fig)
    # F3: N scaling
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    grid = np.logspace(5, 11, 25)
    for name in ("scop_q", "pdb_r600", "u50_db800"):
        if name not in S:
            continue
        x = S[name]
        k = (x["m"] >= 21) & (x["m"] <= 966)
        fr = [np.mean(ln_e(x["Sp"][k], x["m"][k], nn) <= LN1E3) for nn in grid]
        ax.plot(grid, fr, color=col[name], label=LABEL[name])
    for lab, nn in NS.items():
        ax.axvline(nn, color="0.6", lw=0.8, ls=":")
        ax.text(nn, 0.02, lab, rotation=90, fontsize=7, va="bottom", ha="right", color="0.4")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("database residues N")
    ax.set_ylabel("share of self hits with E <= 1e-3")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "figs", "f3_n_scaling.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
