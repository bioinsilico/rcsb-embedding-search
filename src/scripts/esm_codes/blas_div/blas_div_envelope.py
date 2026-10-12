#!/usr/bin/env python
"""blas_div_envelope.py -- build-to-build envelope of a set of blas_div.py runs on the SAME work set.

blas_div.py compare measures one run against one reference.  This script answers "how far apart can ANY two of these
builds be?" for a whole set of runs (search: same queries; a1: same pairs):

 1. it groups the runs into bit-identical classes
      search: S_f, S_r (targets) and P_f, P_r (panel) of the common query rows
      a1:     S_f, S_r, traceback score, end cell and every forward path
 2. for every pair of classes it measures the divergence with blas_div.py compare's definitions
      search: S_f / S_r fraction bit-equal, max abs, max rel (|d| / max(|ref|, 1)), max ulp; S = f64 S_f - f64 S_r
              (max abs, log-histogram p99 / p999); Q90 (max abs); lnE (max abs over every pair; |dlnE| over the pairs with
              E <= 1 in EITHER class: n, median, p95, p99, max, exact); E-threshold flips (pairs with (E_a <= t) != (E_b <= t))
              for t in 1e-3, 1e-2, 0.1, 1, 10; queries whose top-10 order differs; queries whose E <= 1e-3 hit set differs
      a1:     S_f / S_r bit-equal fraction and max rel; forward paths identical; aligned residue-pair sets identical;
              F1 against US-align (pairs with a reference): number of pairs whose F1 differs, max |dF1|, per-set mean F1
 3. it reports the envelope: the maximum of each metric over all class pairs and the pair that attains it, twice:
      'blas'  classes whose runs all use a BLAS similarity (backend != fixed_c) with the fixed-order C query_prep
              (--qprep c) -- D2 option (b) as specified (only the GEMM varies);
      'all'   every class (adds fixed_c and the qprep-numpy variant).

Runs are described by meta.json and blas_probe.json (python, numpy, BLAS library / version / core, threads, env).

usage: blas_div_envelope.py --work search|a1 --out OUT.json RUN_DIR [RUN_DIR ...]   (or --glob 'ROOT/runs/x/*/...')
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blas_div as BD  # noqa: E402  (sets the BLAS thread variables to 1 before numpy)
import numpy as np  # noqa: E402

FLIP_T = (1e-3, 1e-2, 0.1, 1.0, 10.0)


def describe(d):
    m = json.load(open(os.path.join(d, "meta.json")))
    out = dict(dir=d, host_tag=m.get("host_tag"), label=m.get("label"), backend=m["cfg"]["backend"],
               qprep=m["cfg"]["qprep"], threads=m["cfg"]["threads"], host=m["host"].get("hostname"),
               cpu=m["host"].get("cpu"), python=m["host"].get("executable"), numpy=m["numpy"].get("version"))
    b = m["numpy"].get("blas") or {}
    out["numpy_blas"] = f"{b.get('name')} {b.get('version')}" if b else None
    tp = m["numpy"].get("threadpoolctl")
    if isinstance(tp, list):
        out["threadpoolctl"] = [f"{x.get('internal_api')} {x.get('version')} {x.get('architecture')} "
                                f"t={x.get('num_threads')}" for x in tp]
    env = m["env"].get("blas_related", {})
    out["env"] = {k: v for k, v in env.items() if k.startswith(("OPENBLAS_CORETYPE", "MKL_CBWR", "MKL_ENABLE",
                                                                "MKL_DEBUG", "LD_LIBRARY_PATH"))}
    if "LD_LIBRARY_PATH" in out["env"]:
        out["env"]["LD_LIBRARY_PATH"] = out["env"]["LD_LIBRARY_PATH"][:120]
    if m.get("torch"):
        t = m["torch"]
        out["torch"] = dict(version=t.get("version"), num_threads=t.get("num_threads"),
                            mkl=[ln.strip() for ln in (t.get("config") or "").splitlines() if "Math Kernel" in ln][:1])
    p = os.path.join(d, "blas_probe.json")
    if os.path.exists(p):
        pr = json.load(open(p))
        out["probe"] = dict(openblas=[dict(corename=o.get("corename"), config=o.get("config"),
                                           num_threads=o.get("num_threads")) for o in pr.get("openblas", [])],
                            mkl=[x.get("version", "").strip()[:90] for x in pr.get("mkl", [])],
                            libs=[os.path.basename(x) for x in pr.get("mapped_libs", [])],
                            numpy_gemm_fingerprint=pr.get("numpy", {}).get("gemm_fingerprint", "")[:16],
                            torch_gemm_fingerprint=(pr.get("torch") or {}).get("gemm_fingerprint", "")[:16]
                            if isinstance(pr.get("torch"), dict) else None)
    out["kind"] = "blas" if (out["backend"] != "fixed_c" and out["qprep"] == "c") else (
        "fixed_c" if out["backend"] == "fixed_c" else "other")
    return out


def _ex(h):
    return h.summary()


# =====================================================================================================================
# search
# =====================================================================================================================
def search_envelope(runs, data, chunk=32):
    R = []
    for d in runs:
        p = os.path.join(d, "search.npz")
        z = np.load(p)
        R.append(dict(dir=d, p=p, rows=z["rows"], nt=int(z["nt"]), panel="Pf" in z.files))
    common = R[0]["rows"]
    for r in R[1:]:
        common = np.intersect1d(common, r["rows"])
    NT = min(r["nt"] for r in R)
    assert all(r["panel"] for r in R), "every run needs the panel (Q90)"
    for r in R:
        r["idx"] = np.searchsorted(r["rows"], common)
        for k in ("Sf", "Sr", "Pf", "Pr"):
            r[k] = BD.npz_member(r["p"], k)
    # classes
    BD.log(f"search: {len(R)} runs, {len(common)} common rows x {NT} targets; hashing")
    hashes = []
    for r in R:
        h = hashlib.sha256()
        for c0 in range(0, len(common), 256):
            ii = r["idx"][c0:c0 + 256]
            for k in ("Sf", "Sr"):
                h.update(np.ascontiguousarray(np.asarray(r[k][ii])[:, :NT]).tobytes())
            for k in ("Pf", "Pr"):
                h.update(np.ascontiguousarray(np.asarray(r[k][ii])).tobytes())
        hashes.append(h.hexdigest())
    cls, reps = {}, []
    for i, hx in enumerate(hashes):
        if hx not in cls:
            cls[hx] = len(reps)
            reps.append(i)
    K = len(reps)
    cid = [cls[hx] for hx in hashes]
    BD.log(f"{K} bit-identical classes")
    Bn = BD.load_bench(data)
    qlen = Bn["qlen"][common]
    tab = BD.ship_table(data)
    q90 = []
    for c in range(K):
        r = R[reps[c]]
        P = np.asarray(r["Pf"][r["idx"]], np.float64) - np.asarray(r["Pr"][r["idx"]], np.float64)
        q90.append(np.quantile(P, 0.9, axis=1))
    pairs = [(a, b) for a in range(K) for b in range(a + 1, K)]
    st = {}
    for a, b in pairs:
        st[(a, b)] = dict(sf_eq=0, sr_eq=0, n=0, sf_max_abs=0.0, sf_max_rel=0.0, sf_max_ulp=0, sr_max_abs=0.0,
                          sr_max_rel=0.0, S_hist=BD.Hist(), S_max_abs=0.0, lnE_max_abs=0.0, dl_hits=[],
                          flips={f"{t:g}": 0 for t in FLIP_T}, hits={f"{t:g}": [0, 0] for t in FLIP_T},
                          top10_order_diff=0, top10_set_diff=0, hit1e3_set_diff=0, hit1e3_queries=0,
                          q90_max_abs=float(np.abs(q90[a] - q90[b]).max()), q90_frac_equal=float((q90[a] == q90[b]).mean()))
    for c0 in range(0, len(common), chunk):
        sl = slice(c0, min(c0 + chunk, len(common)))
        SF, SR, S, L, O = [], [], [], [], []
        for c in range(K):
            r = R[reps[c]]
            sf = np.ascontiguousarray(np.asarray(r["Sf"][r["idx"][sl]])[:, :NT])
            sr = np.ascontiguousarray(np.asarray(r["Sr"][r["idx"][sl]])[:, :NT])
            s = sf.astype(np.float64) - sr.astype(np.float64)
            SF.append(sf)
            SR.append(sr)
            S.append(s)
            L.append(BD.ln_evalue(tab, qlen[sl], s - q90[c][sl, None]))
            O.append([BD._rank_order(s[i]) for i in range(s.shape[0])])
        for a, b in pairs:
            x = st[(a, b)]
            x["n"] += SF[a].size
            for nm, A, B in (("sf", SF[a], SF[b]), ("sr", SR[a], SR[b])):
                eq = BD.bits_equal(A, B)
                x[f"{nm}_eq"] += int(eq.sum())
                d = np.abs(A.astype(np.float64) - B.astype(np.float64))
                x[f"{nm}_max_abs"] = max(x[f"{nm}_max_abs"], float(d.max()))
                x[f"{nm}_max_rel"] = max(x[f"{nm}_max_rel"], float((d / np.maximum(np.abs(A.astype(np.float64)), 1.0)).max()))
                if nm == "sf" and not eq.all():
                    x["sf_max_ulp"] = max(x["sf_max_ulp"], int(BD.ulp_diff(A[~eq], B[~eq]).max()))
            dS = np.abs(S[a] - S[b])
            x["S_hist"].add(dS)
            x["S_max_abs"] = max(x["S_max_abs"], float(dS.max()))
            dl = np.abs(L[a] - L[b])
            x["lnE_max_abs"] = max(x["lnE_max_abs"], float(dl.max()))
            hit = (L[a] <= 0.0) | (L[b] <= 0.0)
            x["dl_hits"].append(dl[hit])
            for t in FLIP_T:
                ha, hb = L[a] <= math.log(t), L[b] <= math.log(t)
                x["flips"][f"{t:g}"] += int((ha != hb).sum())
                x["hits"][f"{t:g}"][0] += int(ha.sum())
                x["hits"][f"{t:g}"][1] += int(hb.sum())
            for i in range(S[a].shape[0]):
                oa, ob = O[a][i], O[b][i]
                if not np.array_equal(oa[:10], ob[:10]):
                    x["top10_order_diff"] += 1
                    if set(oa[:10].tolist()) != set(ob[:10].tolist()):
                        x["top10_set_diff"] += 1
                ta = set(np.flatnonzero(L[a][i] <= math.log(1e-3)).tolist())
                tb = set(np.flatnonzero(L[b][i] <= math.log(1e-3)).tolist())
                if ta or tb:
                    x["hit1e3_queries"] += 1
                    x["hit1e3_set_diff"] += int(ta != tb)
    out_pairs = []
    for a, b in pairs:
        x = st[(a, b)]
        dlh = np.concatenate(x["dl_hits"]) if x["dl_hits"] else np.zeros(0)
        n = x["n"]
        out_pairs.append(dict(
            a=a, b=b, pairs=n, S_f_frac_bit_equal=x["sf_eq"] / n, S_r_frac_bit_equal=x["sr_eq"] / n,
            S_f_max_abs=x["sf_max_abs"], S_f_max_rel_floor1=x["sf_max_rel"], S_f_max_ulp=x["sf_max_ulp"],
            S_r_max_abs=x["sr_max_abs"], S_r_max_rel_floor1=x["sr_max_rel"], S_max_abs=x["S_max_abs"],
            S_abs_p99_hist=x["S_hist"].pct(0.99), S_abs_p999_hist=x["S_hist"].pct(0.999),
            S_frac_equal=x["S_hist"].zeros / x["S_hist"].n, Q90_max_abs=x["q90_max_abs"],
            Q90_frac_equal=x["q90_frac_equal"], lnE_max_abs=x["lnE_max_abs"],
            lnE_hits_E_le_1_either=dict(n=int(dlh.size), median=float(np.median(dlh)) if dlh.size else None,
                                        p95=float(np.percentile(dlh, 95)) if dlh.size else None,
                                        p99=float(np.percentile(dlh, 99)) if dlh.size else None,
                                        max=float(dlh.max()) if dlh.size else None),
            E_flips=x["flips"], E_hits_a_b=x["hits"], top10_order_diff_queries=x["top10_order_diff"],
            top10_set_diff_queries=x["top10_set_diff"], hitlist_E_le_1e3_queries=x["hit1e3_queries"],
            hitlist_E_le_1e3_set_diff_queries=x["hit1e3_set_diff"]))
    return dict(n_runs=len(R), common_rows=len(common), n_targets=NT, pairs_per_class_pair=len(common) * NT,
                rows_sha256=hashlib.sha256(np.asarray(common, np.int64).tobytes()).hexdigest(),
                classes=K, class_of_run=cid, class_hash=[hashes[i][:16] for i in reps], class_pairs=out_pairs,
                E=dict(table=BD.SHIP_KEY, version=tab["version"], N=tab["N"]))


# =====================================================================================================================
# a1
# =====================================================================================================================
def a1_envelope(runs, data):
    A1 = BD.load_a1(data)
    Z = [dict(np.load(os.path.join(d, "a1.npz"))) for d in runs]
    common = Z[0]["pairs"]
    for z in Z[1:]:
        common = np.intersect1d(common, z["pairs"])
    hashes, P = [], []
    for z in Z:
        ii = np.searchsorted(z["pairs"], common)
        paths = BD._paths(z)
        pl = [paths[k] for k in ii]
        h = hashlib.sha256()
        for k in ("Sf", "Sr", "tb"):
            h.update(np.ascontiguousarray(z[k][ii]).tobytes())
        h.update(np.ascontiguousarray(z["ends"][ii]).tobytes())
        for x in pl:
            h.update(np.int64(len(x)).tobytes() + np.ascontiguousarray(x, np.int32).tobytes())
        hashes.append(h.hexdigest())
        P.append(dict(Sf=z["Sf"][ii], Sr=z["Sr"][ii], paths=pl))
    cls, reps = {}, []
    for i, hx in enumerate(hashes):
        if hx not in cls:
            cls[hx] = len(reps)
            reps.append(i)
    K = len(reps)
    has_ref = A1["has_ref"][common].astype(bool)
    sets = A1["set_names"][A1["pair_set"][common]]
    F1 = []
    for c in range(K):
        f = np.full(len(common), np.nan)
        for k, p in enumerate(common):
            if has_ref[k]:
                f[k] = BD.prf(BD.matched(P[reps[c]]["paths"][k]), BD.a1_ref(A1, p))[2]
        F1.append(f)
    per_class = []
    for c in range(K):
        per_class.append(dict(mean_F1_per_set={s: float(np.nanmean(F1[c][(sets == s) & has_ref]))
                                               for s in np.unique(sets[has_ref])}))
    out_pairs = []
    for a in range(K):
        for b in range(a + 1, K):
            pa, pb = P[reps[a]], P[reps[b]]
            same = np.array([np.array_equal(x, y) for x, y in zip(pa["paths"], pb["paths"])])
            mt = np.array([np.array_equal(BD.matched(x), BD.matched(y)) for x, y in zip(pa["paths"], pb["paths"])])
            dF = np.abs(F1[a] - F1[b])[has_ref]
            d = np.abs(pa["Sf"].astype(np.float64) - pb["Sf"])
            out_pairs.append(dict(
                a=a, b=b, pairs=len(common), S_f_frac_bit_equal=float(BD.bits_equal(pa["Sf"], pb["Sf"]).mean()),
                S_r_frac_bit_equal=float(BD.bits_equal(pa["Sr"], pb["Sr"]).mean()), S_f_max_abs=float(d.max()),
                S_f_max_rel_floor1=float((d / np.maximum(np.abs(pa["Sf"].astype(np.float64)), 1.0)).max()),
                paths_identical=int(same.sum()), paths_differing=int((~same).sum()),
                aligned_pairs_identical=int(mt.sum()), f1_pairs=int(has_ref.sum()),
                f1_differing_pairs=int((dF > 0).sum()), max_abs_dF1=float(dF.max()) if dF.size else 0.0,
                mean_F1_diff_per_set={s: per_class[b]["mean_F1_per_set"][s] - per_class[a]["mean_F1_per_set"][s]
                                      for s in per_class[a]["mean_F1_per_set"]},
                differing_pair_ids=[int(p) for p in common[~same][:200]]))
    return dict(n_runs=len(Z), common_pairs=len(common), classes=K, class_of_run=[cls[h] for h in hashes],
                class_hash=[hashes[i][:16] for i in reps], per_class=per_class, class_pairs=out_pairs)


# =====================================================================================================================
def envelope(res, runs_desc, keys_max, keys_min):
    cls_kind = {}
    for i, c in enumerate(res["class_of_run"]):
        cls_kind.setdefault(c, set()).add(runs_desc[i]["kind"])
    out = {}
    for scope in ("blas", "all"):
        ok = {c for c, k in cls_kind.items() if scope == "all" or k == {"blas"}}
        cp = [p for p in res["class_pairs"] if p["a"] in ok and p["b"] in ok]
        e = dict(classes=sorted(ok), n_class_pairs=len(cp))
        for k in keys_max:
            vals = [(_get(p, k), (p["a"], p["b"])) for p in cp if _get(p, k) is not None]
            if vals:
                v, ab = max(vals, key=lambda t: t[0])
                e[k] = dict(max=v, pair=ab)
        for k in keys_min:
            vals = [(_get(p, k), (p["a"], p["b"])) for p in cp if _get(p, k) is not None]
            if vals:
                v, ab = min(vals, key=lambda t: t[0])
                e[k] = dict(min=v, pair=ab)
        out[scope] = e
    return out


def _get(p, k):
    x = p
    for part in k.split("/"):
        if not isinstance(x, dict) or part not in x:
            return None
        x = x[part]
    return x


SEARCH_MAX = ("S_f_max_abs", "S_f_max_rel_floor1", "S_f_max_ulp", "S_r_max_rel_floor1", "S_max_abs", "S_abs_p99_hist",
              "Q90_max_abs", "lnE_max_abs", "lnE_hits_E_le_1_either/median", "lnE_hits_E_le_1_either/p95",
              "lnE_hits_E_le_1_either/p99", "lnE_hits_E_le_1_either/max", "E_flips/0.001", "E_flips/0.01",
              "E_flips/0.1", "E_flips/1", "E_flips/10", "top10_order_diff_queries", "top10_set_diff_queries",
              "hitlist_E_le_1e3_set_diff_queries")
SEARCH_MIN = ("S_f_frac_bit_equal", "S_r_frac_bit_equal", "Q90_frac_equal")
A1_MAX = ("S_f_max_abs", "S_f_max_rel_floor1", "paths_differing", "f1_differing_pairs", "max_abs_dF1")
A1_MIN = ("S_f_frac_bit_equal", "paths_identical")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--glob", action="append", default=[])
    ap.add_argument("--work", choices=("search", "a1"), required=True)
    ap.add_argument("--data", default=BD.DATA_DEFAULT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    fin = "search.npz" if a.work == "search" else "a1.npz"
    runs = list(a.runs)
    for g in a.glob:
        runs += sorted(glob.glob(g))
    runs = [os.path.abspath(r) for r in runs if os.path.exists(os.path.join(r, fin))]
    runs = list(dict.fromkeys(runs))
    if len(runs) < 2:
        raise SystemExit(f"need >= 2 finished runs, got {runs}")
    desc = [describe(d) for d in runs]
    t0 = time.time()
    if a.work == "search":
        res = search_envelope(runs, os.path.abspath(a.data))
        env = envelope(res, desc, SEARCH_MAX, SEARCH_MIN)
    else:
        res = a1_envelope(runs, os.path.abspath(a.data))
        env = envelope(res, desc, A1_MAX, A1_MIN)
    classes = {}
    for i, c in enumerate(res["class_of_run"]):
        classes.setdefault(c, []).append(desc[i]["label"])
    out = dict(what=f"blas_div envelope ({a.work}): bit-identical classes of runs and the divergence between classes",
               created=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=time.time() - t0, runs=desc,
               classes={str(c): v for c, v in sorted(classes.items())}, result=res, envelope=env,
               definitions=dict(rel_floor1="|b - a| / max(|a|, 1)", S="f64 S_f - f64 S_r",
                                lnE="lnK(m) + ln m + ln N - lambda(m) (S - Q90); ship_final 'all'",
                                flips="pairs with (E_a <= t) != (E_b <= t)",
                                S_abs_p99_hist="upper edge of the log-histogram bin (<= 4.7% high)",
                                scopes="blas: classes made only of BLAS-sim runs with qprep c; all: every class"))
    BD.jdump(out, a.out)
    print(json.dumps(dict(classes=out["classes"], envelope=env), indent=1, default=BD._jdefault))
    BD.log(f"-> {a.out}")


if __name__ == "__main__":
    main()
