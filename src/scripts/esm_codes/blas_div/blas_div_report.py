#!/usr/bin/env python
"""blas_div_report.py -- Analyze phase of the BLAS-divergence study (FoldMatch esm-codes-v1, plan decision D2).

Reads every run of the three run phases (ROOT/runs/{epyc,a100,m1}/<backend>/<work>/<label>/, made by blas_div.py sim)
and measures, against the golden fixed_c run AND between every pair of bit-distinct builds:
  - S_f, S_r, S, Q90, S', lnE differences with EXACT percentiles (np.percentile on the full per-pair arrays; the
    blas_div.py compare JSONs use log-histogram bin edges);
  - E-threshold flips (with benchmark label), benchmark TP / FP / real-null counts and AUC1 per build;
  - what a user sees: the FoldMatch 'evalue' column ('%.3E'), the 'bits' integer, the 'raw' value, top-10 / top-100
    ranking (S descending, target index ascending: no embscore here) and the E <= 1e-3 / E <= 1 hit lists;
  - A1: forward-path identity, differing columns, F1 vs US-align, float64-oracle margins of ALL 12,960 pairs;
  - sampled similarity cells vs golden, vs the float64 oracle and between builds;
  - repeat / nondeterminism records (repeat.json) and same-machine pairs (threads, block shape, MKL_CBWR);
  - the ESM3 device-drift context scale from the P0.5 parity archive, on the same E arithmetic.
Only the similarity GEMM (and, as a variant, the fp32 numpy query norm) differs between runs: z, DP, Q90 and E come
from the same fixed-order spike kernel and float64 code (see blas_div.py).

SUBCOMMANDS (outputs -> ROOT/report/<name>.json; ROOT as in blas_div.py, env BLASDIV_ROOT)
  catalog                     run inventory + bit-identity classes (sha256 of the stored arrays)    -> catalog.json
  search --work W [--workers] per run vs golden, class-pair matrix, envelopes, same-machine pairs    -> search_<W>.json
  a1 [--workers]              per run vs golden + float64 oracle on all pairs                        -> a1.json
  sim                         sampled similarity blocks vs golden / float64 oracle / between builds  -> sim.json
  repeat                      every repeat.json                                                      -> repeat.json
  context                     ESM3 device drift (P0.5 archive)                                       -> context.json
  extra                       spread vs score magnitude / E; golden pairs near each E threshold      -> extra.json
  summary [--report-dir --out] compact study summary from the JSONs above               -> blas_div_summary.json
  all [--workers]             everything (search for search-q150-T50000 and search-all-T50000)
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

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blas_div as BD  # noqa: E402

ROOT = BD.ROOT
DATA = BD.DATA_DEFAULT
OUT = os.path.join(ROOT, "report")
HOSTS = ("epyc", "a100", "m1")
GOLD_LABEL = "fixed_c-qpc-t1"
THRESH = (1e-3, 1e-2, 0.1, 1.0, 10.0)
LN10 = math.log(10.0)
PARITY = "/global/homes/j/jsegura/esm_codes_archive/p0/parity"
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:8.1f}s] " + " ".join(str(x) for x in a), flush=True)


def clean(o):
    """JSON-safe: numpy scalars / arrays, non-finite floats -> strings."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else ("inf" if f > 0 else ("-inf" if f < 0 else "nan"))
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def dump(obj, name):
    p = os.path.join(OUT, name)
    BD.jdump(clean(obj), p)
    log("->", p)
    return p


def pstats(d):
    """exact stats of |d| (float64): median, P99, P99.9, P99.99, max (np.percentile, linear), mean, frac zero."""
    d = np.abs(np.asarray(d, np.float64)).ravel()
    if not d.size:
        return dict(n=0)
    p = np.percentile(d, [50, 99, 99.9, 99.99, 100])
    return dict(n=int(d.size), frac_zero=float((d == 0).mean()), median=float(p[0]), p99=float(p[1]),
                p999=float(p[2]), p9999=float(p[3]), max=float(p[4]), mean=float(d.mean()))


def sha(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str(a.dtype).encode() + str(a.shape).encode())
        h.update(a.view(np.uint8).ravel().data if a.size else b"")
    return h.hexdigest()


# =====================================================================================================================
# catalog
# =====================================================================================================================
def describe(m, host, label):
    cfg = m["cfg"]
    b = cfg["backend"]
    env = (m.get("env") or {}).get("blas_related") or {}
    npi = m.get("numpy") or {}
    tpc = npi.get("threadpoolctl")
    tp = [x for x in (tpc if isinstance(tpc, list) else []) if isinstance(x, dict) and x.get("user_api") == "blas"]
    tor = m.get("torch") or {}
    nv = npi.get("version")
    if b == "fixed_c":
        lib = "fixed-order C (spike fmsw_sim_codes)"
    elif b in ("numpy", "numpy_rowwise"):
        if "libsci" in label:
            lib = f"numpy {nv} + Cray LibSci 26.03"
        elif tp:
            x = tp[0]
            lib = f"numpy {nv} + {x.get('internal_api')} {x.get('version')}"
            if x.get("architecture"):
                lib += f" [{x.get('architecture')} kernel]"
            if x.get("threading_layer"):
                lib += f" ({x.get('threading_layer')})"
        elif (npi.get("blas") or {}).get("name") == "accelerate":
            lib = f"numpy {nv} + Apple Accelerate"
        else:
            lib = f"numpy {nv} + {(npi.get('blas') or {}).get('name')} {(npi.get('blas') or {}).get('version')}"
        if b == "numpy_rowwise":
            lib += ", one GEMM per target"
    elif b == "torch_cpu":
        lib = f"torch {tor.get('version')} CPU" + (" (MKL 2024.2)" if host == "epyc" else " (macOS arm64)")
    elif b == "torch_cuda":
        lib = f"torch {tor.get('version')} cuBLAS FP32 ({(m.get('backend_info') or {}).get('name')})"
    elif b == "torch_cuda_tf32":
        lib = f"torch {tor.get('version')} cuBLAS TF32 ({(m.get('backend_info') or {}).get('name')})"
    elif b == "torch_mps":
        lib = f"torch {tor.get('version')} MPS fp32"
    else:
        lib = b
    for k in ("OPENBLAS_CORETYPE", "MKL_CBWR"):
        if env.get(k):
            lib += f", {k}={env[k]}"
    threads = "default" if "tdef" in label else int(cfg.get("threads") or 1)
    return lib, threads


def categorize(r):
    """kind: golden | blas (FP32 GEMM, qprep c) | qpnumpy (research fp32 query norm) | tf32;  flags for the scopes."""
    b, lab = r["backend"], r["label"]
    if r["qprep"] == "numpy":
        kind = "qpnumpy"
    elif b == "torch_cuda_tf32":
        kind = "tf32"
    elif b == "fixed_c":
        kind = "golden"
    else:
        kind = "blas"
    flags = []
    if r["threads"] == "default" or (isinstance(r["threads"], int) and r["threads"] > 1):
        flags.append("threaded")
    if "OPENBLAS_CORETYPE" in r["env"]:
        flags.append("forced_core")
    if "MKL_CBWR" in r["env"]:
        flags.append("cbwr")
    if b == "numpy_rowwise":
        flags.append("rowwise")
    if r["block_rule"] != "prod":
        flags.append("block_rule_" + r["block_rule"])
    return kind, flags


def find_runs(work_prefix):
    out = []
    for h in HOSTS:
        for mp in sorted(glob.glob(f"{ROOT}/runs/{h}/*/{work_prefix}*/*/meta.json")):
            d = os.path.dirname(mp)
            m = json.load(open(mp))
            label = os.path.basename(d)
            work = os.path.basename(os.path.dirname(d))
            lib, threads = describe(m, h, label)
            env = {k: v for k, v in ((m.get("env") or {}).get("blas_related") or {}).items()
                   if k in ("OPENBLAS_CORETYPE", "MKL_CBWR", "MKL_DEBUG_CPU_TYPE")}
            r = dict(id=f"{h}:{label}", host=h, label=label, work=work, dir=d, backend=m["cfg"]["backend"],
                     qprep=m["cfg"]["qprep"], threads=threads, block_rule=m["cfg"]["block_rule"],
                     block_floats=m["cfg"]["block_floats"], lib=lib, env=env, cpu=(m.get("host") or {}).get("cpu"),
                     hostname=(m.get("host") or {}).get("hostname"), status=m.get("status"))
            r["kind"], r["flags"] = categorize(r)
            r["default_t1"] = r["kind"] in ("blas", "golden") and not r["flags"]
            out.append(r)
    return out


def _hash_run(args):
    d, work = args
    if work.startswith("search"):
        z = np.load(os.path.join(d, "search.npz"))
        h = dict(scores=sha(z["Sf"], z["Sr"], z["Pf"], z["Pr"]), rows=sha(z["rows"]), nt=int(z["nt"]))
    else:
        z = np.load(os.path.join(d, "a1.npz"))
        h = dict(scores=sha(z["Sf"], z["Sr"], z["tb"]), paths=sha(z["ends"], z["path_flat"], z["path_off"]),
                 pairs=sha(z["pairs"]))
    sp = os.path.join(d, "simblocks.npz")
    if os.path.exists(sp):
        z = np.load(sp)
        h["sim"] = sha(z["row"], z["k0"], z["k1"], z["sim"])
        h["sim_keys"] = sha(z["row"], z["k0"], z["k1"])
    return h


def cmd_catalog(a, pool=None):
    cat = {}
    for work in ("search-q150-T50000", "search-all-T50000", "a1-all"):
        runs = find_runs(work)
        runs = [r for r in runs if r["work"] == work]
        args = [(r["dir"], work) for r in runs]
        hs = pool.map(_hash_run, args) if pool else [_hash_run(x) for x in args]
        classes, cls_of = {}, {}
        key = "scores" if work.startswith("search") else "scores"
        for r, h in zip(runs, hs):
            r["hash"] = h
            k = h[key] + (h.get("paths", "") if not work.startswith("search") else "")
            if k not in classes:
                classes[k] = len(classes)
            r["class"] = classes[k]
        gold = [r for r in runs if r["host"] == "epyc" and r["label"] == GOLD_LABEL]
        assert len(gold) == 1, (work, gold)
        g = gold[0]["class"]
        # renumber classes: golden first, then by first appearance
        order = [g] + [c for c in dict.fromkeys(r["class"] for r in runs) if c != g]
        ren = {c: i for i, c in enumerate(order)}
        for r in runs:
            r["class"] = ren[r["class"]]
        cat[work] = dict(n_runs=len(runs), n_classes=len(order), runs=runs,
                         classes={str(c): [r["id"] for r in runs if r["class"] == c] for c in range(len(order))})
        log(f"catalog {work}: {len(runs)} runs, {len(order)} classes")
    cat["meta"] = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), root=ROOT,
                       definition="class = identical sha256 of the stored float32 score arrays (search: S_f, S_r, panel "
                                  "P_f, P_r; a1: S_f, S_r, traceback score + ends + forward paths); class 0 = golden "
                                  "epyc fixed_c-qpc-t1")
    dump(cat, "catalog.json")
    return cat


def load_catalog():
    return json.load(open(os.path.join(OUT, "catalog.json")))


# =====================================================================================================================
# search
# =====================================================================================================================
_CTX: dict = {}


def _search_ctx(rows):
    key = ("search", tuple(rows.tolist()))
    if key in _CTX:
        return _CTX[key]
    Bn = BD.load_bench(DATA)
    tab = BD.ship_table(DATA)
    qlen = Bn["qlen"][rows].astype(np.int64)
    lm = np.log(qlen.astype(np.float64))
    lam = np.interp(lm, tab["lnc"], tab["lam"])
    lnK = np.interp(lm, tab["lnc"], tab["lnK"])
    L = np.load(os.path.join(DATA, "bench_labels.npz"))
    nt = 50000
    lab = {k: np.unpackbits(L[k][rows], axis=1)[:, :nt].astype(bool) for k in ("TP", "FP", "FPA")}
    ctx = dict(tab=tab, qlen=qlen, lam=lam, lnK=lnK, lab=lab, qids=Bn["qids"][rows])
    _CTX.clear()
    _CTX[key] = ctx
    return ctx


def load_search(d):
    z = np.load(os.path.join(d, "search.npz"))
    R = dict(rows=z["rows"], Sf=z["Sf"], Sr=z["Sr"], Pf=z["Pf"], Pr=z["Pr"], nt=int(z["nt"]))
    assert R["nt"] == 50000 and R["Sf"].shape == (len(R["rows"]), 50000)
    return R


def derive(R, ctx):
    S = R["Sf"].astype(np.float64)
    S -= R["Sr"]
    P = R["Pf"].astype(np.float64) - R["Pr"]
    q90 = np.quantile(P, 0.9, axis=1)
    lnE = BD.ln_evalue(ctx["tab"], ctx["qlen"], S - q90[:, None])
    return S, q90, lnE


def render_E(lnE):
    """FoldMatch 'evalue' column: f'{E:.3E}' (output.py); below exp(-700) rendered from log10 (never underflows)."""
    out = []
    for v in np.asarray(lnE, np.float64).tolist():
        if v > -700.0:
            out.append(f"{math.exp(v):.3E}")
        else:
            l10 = v / LN10
            e = math.floor(l10)
            s = f"{10.0 ** (l10 - e):.3f}"
            if s == "10.000":
                s, e = "1.000", e + 1
            out.append(f"{s}E{e:+03d}")
    return np.array(out)


def bits_int(Sp, lam, lnK):
    return np.floor((lam * Sp - lnK) / math.log(2.0) + 0.5)


def bench_counts(S, lnE, ctx):
    """benchmark numbers of one build: TP / FP (all benchmark FPs incl. reversed targets) / FPA (real null) hits at
    each E threshold; real-null FP per query per x (research 'FP/q/x'); AUC1 (TPs above the best FP, ties 0.5) mean
    over queries with a TP, FP = all FPs and FP = real null (research summarize.py auc1)."""
    lab = ctx["lab"]
    nq = S.shape[0]
    out = {}
    for t in THRESH:
        h = lnE <= math.log(t)
        out[f"{t:g}"] = dict(hits=int(h.sum()), TP=int((h & lab["TP"]).sum()), FP=int((h & lab["FP"]).sum()),
                             FPA=int((h & lab["FPA"]).sum()), FPA_per_q_per_x=float((h & lab["FPA"]).sum() / (nq * t)))
    for nm, fpm in (("auc1_fp_all", lab["FP"]), ("auc1_fp_real", lab["FPA"])):
        best = np.where(fpm, S, -np.inf).max(1)
        gt = ((S > best[:, None]) & lab["TP"]).sum(1)
        eq = ((S == best[:, None]) & lab["TP"]).sum(1)
        n = lab["TP"].sum(1)
        ok = n > 0
        out[nm] = float(np.mean((gt[ok] + 0.5 * eq[ok]) / n[ok]))
    out["n_queries"] = nq
    out["queries_with_TP"] = int((lab["TP"].sum(1) > 0).sum())
    return out


def _rank(S_row):
    return np.lexsort((np.arange(len(S_row)), -S_row))


def pair_search(dA, dB, full=True):
    """B vs A (A = reference). Exact stats; flips; user-visible fields; ranking; hit lists."""
    A, B = load_search(dA), load_search(dB)
    assert np.array_equal(A["rows"], B["rows"]), (dA, dB)
    ctx = _search_ctx(A["rows"])
    SA, qA, lA = derive(A, ctx)
    SB, qB, lB = derive(B, ctx)
    o = dict(n_queries=len(A["rows"]), n_targets=A["nt"], pairs=int(SA.size))
    for nm in ("Sf", "Sr"):
        a, b = A[nm], B[nm]
        eq = BD.bits_equal(a, b)
        d = np.abs(b.astype(np.float64) - a)
        rel = d / np.maximum(np.abs(a.astype(np.float64)), 1.0)
        u = BD.ulp_diff(a, b) if full else None
        k_abs, k_rel = np.unravel_index(int(np.argmax(d)), d.shape), np.unravel_index(int(np.argmax(rel)), d.shape)
        o[nm] = dict(frac_bit_equal=float(eq.mean()), n_diff=int((~eq).sum()), abs=pstats(d), rel_floor1=pstats(rel),
                     at_max_abs=dict(row=int(A["rows"][k_abs[0]]), target=int(k_abs[1]), ref=float(a[k_abs])),
                     at_max_rel=dict(row=int(A["rows"][k_rel[0]]), target=int(k_rel[1]), ref=float(a[k_rel])))
        if full:
            pu = np.percentile(u, [99.9, 100])
            o[nm]["ulp"] = dict(p999=float(pu[0]), max=int(pu[1]), frac_le1=float((u <= 1).mean()))
        del d, rel, u
    dS = SB - SA
    o["S"] = dict(frac_equal=float((dS == 0).mean()), abs=pstats(dS),
                  rel_floor1=pstats(np.abs(dS) / np.maximum(np.abs(SA), 1.0)))
    dq = qB - qA
    o["Q90"] = dict(n=len(dq), frac_equal=float((dq == 0).mean()), median=float(np.median(np.abs(dq))),
                    p99=float(np.percentile(np.abs(dq), 99)), max=float(np.abs(dq).max()),
                    panel_Sf_frac_bit_equal=float(BD.bits_equal(A["Pf"], B["Pf"]).mean()),
                    panel_S_max_abs=float(np.abs((B["Pf"].astype(np.float64) - B["Pr"])
                                                 - (A["Pf"].astype(np.float64) - A["Pr"])).max()))
    dSp = dS - dq[:, None]
    SpA = SA - qA[:, None]
    o["S_prime"] = dict(frac_equal=float((dSp == 0).mean()), abs=pstats(dSp),
                        rel_floor1=pstats(np.abs(dSp) / np.maximum(np.abs(SpA), 1.0)))
    del dSp
    dl = np.abs(lB - lA)
    m10 = np.minimum(lA, lB) <= math.log(10.0)
    m3 = np.minimum(lA, lB) <= math.log(1e-3)
    o["lnE"] = dict(all=pstats(dl), E_le_10=pstats(dl[m10]), E_le_1e_3=pstats(dl[m3]))
    k = np.unravel_index(int(np.argmax(dl)), dl.shape)
    o["lnE"]["at_max"] = dict(row=int(A["rows"][k[0]]), target=int(k[1]), lnE_ref=float(lA[k]), lnE_test=float(lB[k]),
                              m=int(ctx["qlen"][k[0]]), lam=float(ctx["lam"][k[0]]))
    del dl
    lab = ctx["lab"]
    flips = {}
    for t in THRESH:
        ha, hb = lA <= math.log(t), lB <= math.log(t)
        x = ha != hb
        f = dict(ref_hits=int(ha.sum()), test_hits=int(hb.sum()), flips=int(x.sum()),
                 frac_of_pairs=float(x.mean()), frac_of_ref_hits=float(x.sum() / max(int(ha.sum()), 1)),
                 ref_only=int((ha & ~hb).sum()), test_only=int((hb & ~ha).sum()),
                 TP=int((x & lab["TP"]).sum()), FPA=int((x & lab["FPA"]).sum()), FP=int((x & lab["FP"]).sum()))
        if x.any():
            ii, jj = np.nonzero(x)
            f["examples"] = [dict(row=int(A["rows"][i]), qid=str(ctx["qids"][i]), m=int(ctx["qlen"][i]), target=int(j),
                                  lnE_ref=float(lA[i, j]), lnE_test=float(lB[i, j]), E_ref=math.exp(lA[i, j]),
                                  E_test=math.exp(lB[i, j]), TP=bool(lab["TP"][i, j]), FPA=bool(lab["FPA"][i, j]),
                                  FP=bool(lab["FP"][i, j]))
                             for i, j in list(zip(ii.tolist(), jj.tolist()))[:20]]
        flips[f"{t:g}"] = f
    o["E_flips"] = flips
    # what a user sees (FoldMatch output.py): evalue '%.3E', bits int(x + 0.5) of floor form, raw = S_f
    vis = {}
    for nm, msk in (("E_le_10", m10), ("E_le_1e_3", m3)):
        ii, jj = np.nonzero(msk)
        if not len(ii):
            vis[nm] = dict(n=0)
            continue
        eA, eB = render_E(lA[ii, jj]), render_E(lB[ii, jj])
        lam, lnK = ctx["lam"][ii], ctx["lnK"][ii]
        bA = bits_int(SA[ii, jj] - qA[ii], lam, lnK)
        bB = bits_int(SB[ii, jj] - qB[ii], lam, lnK)
        vis[nm] = dict(n=int(len(ii)), evalue_str_diff=int((eA != eB).sum()), bits_diff=int((bA != bB).sum()),
                       raw_diff=int((~BD.bits_equal(A["Sf"][ii, jj], B["Sf"][ii, jj])).sum()),
                       evalue_examples=[(str(x), str(y)) for x, y in list(zip(eA[eA != eB], eB[eA != eB]))[:5]])
        vis[nm]["frac_evalue_str_diff"] = vis[nm]["evalue_str_diff"] / vis[nm]["n"]
        vis[nm]["frac_bits_diff"] = vis[nm]["bits_diff"] / vis[nm]["n"]
        vis[nm]["frac_raw_diff"] = vis[nm]["raw_diff"] / vis[nm]["n"]
    o["user_visible"] = vis
    rank = {f"top{k}": dict(order_diff=0, set_diff=0, max_displacement=0) for k in (10, 100)}
    hl = {f"{t:g}": dict(queries_with_hits=0, set_diff=0, order_diff=0) for t in (1e-3, 1.0)}
    for r in range(SA.shape[0]):
        oa, ob = _rank(SA[r]), _rank(SB[r])
        for kk in (10, 100):
            R = rank[f"top{kk}"]
            if not np.array_equal(oa[:kk], ob[:kk]):
                R["order_diff"] += 1
                if set(oa[:kk].tolist()) != set(ob[:kk].tolist()):
                    R["set_diff"] += 1
                pos = np.empty(len(ob), np.int64)
                pos[ob] = np.arange(len(ob))
                R["max_displacement"] = max(R["max_displacement"], int(np.abs(pos[oa[:kk]] - np.arange(kk)).max()))
        for t in (1e-3, 1.0):
            H = hl[f"{t:g}"]
            ha = oa[lA[r][oa] <= math.log(t)]
            hb = ob[lB[r][ob] <= math.log(t)]
            if len(ha) or len(hb):
                H["queries_with_hits"] += 1
                H["set_diff"] += int(set(ha.tolist()) != set(hb.tolist()))
                H["order_diff"] += int(not np.array_equal(ha, hb))
    o["ranking"] = rank
    o["hit_lists"] = hl
    o["bench_test"] = bench_counts(SB, lB, ctx)
    return o


def _task_pair(args):
    tag, dA, dB, full = args
    t = time.time()
    try:
        o = pair_search(dA, dB, full)
    except Exception as e:  # noqa: BLE001
        import traceback
        return tag, dict(error=repr(e), tb=traceback.format_exc())
    o["seconds"] = time.time() - t
    return tag, o


def _task_bench(d):
    R = load_search(d)
    ctx = _search_ctx(R["rows"])
    S, q90, lnE = derive(R, ctx)
    return d, bench_counts(S, lnE, ctx)


def envelope(dirs, rows_ref=None):
    """per pair spread over a set of builds (one per class): max - min of S_f, S, lnE, Q90; hit-decision disagreement
    (not unanimous) at each threshold; top-10 lists not unanimous."""
    run = None
    for i, d in enumerate(dirs):
        R = load_search(d)
        ctx = _search_ctx(R["rows"])
        S, q90, lnE = derive(R, ctx)
        sf = R["Sf"].astype(np.float64)
        if run is None:
            run = dict(sf_min=sf.copy(), sf_max=sf.copy(), s_min=S.copy(), s_max=S.copy(), l_min=lnE.copy(),
                       l_max=lnE.copy(), q_min=q90.copy(), q_max=q90.copy(),
                       hits={t: (lnE <= math.log(t)).astype(np.int16) for t in THRESH},
                       top10=[{tuple(_rank(S[r])[:10].tolist())} for r in range(S.shape[0])])
        else:
            for a, b, x in (("sf_min", "sf_max", sf), ("s_min", "s_max", S), ("l_min", "l_max", lnE),
                            ("q_min", "q_max", q90)):
                np.minimum(run[a], x, out=run[a])
                np.maximum(run[b], x, out=run[b])
            for t in THRESH:
                run["hits"][t] += (lnE <= math.log(t))
            for r in range(S.shape[0]):
                run["top10"][r].add(tuple(_rank(S[r])[:10].tolist()))
        del R, S, lnE, sf
    K = len(dirs)
    sfr = run["sf_max"] - run["sf_min"]
    o = dict(n_builds=K, S_f=dict(abs=pstats(sfr), rel_floor1=pstats(sfr / np.maximum(np.abs(run["sf_min"]), 1.0)),
                                  frac_all_identical=float((sfr == 0).mean())),
             S=dict(abs=pstats(run["s_max"] - run["s_min"])),
             Q90=dict(max=float((run["q_max"] - run["q_min"]).max()),
                      frac_identical=float(((run["q_max"] - run["q_min"]) == 0).mean())))
    lr = run["l_max"] - run["l_min"]
    m10 = run["l_min"] <= math.log(10.0)
    o["lnE"] = dict(all=pstats(lr), E_le_10=pstats(lr[m10]))
    o["E_disagreement"] = {f"{t:g}": dict(pairs_not_unanimous=int(((run["hits"][t] > 0) & (run["hits"][t] < K)).sum()),
                                          pairs_hit_by_all=int((run["hits"][t] == K).sum()),
                                          pairs_hit_by_any=int((run["hits"][t] > 0).sum())) for t in THRESH}
    o["top10_not_unanimous_queries"] = int(sum(len(s) > 1 for s in run["top10"]))
    return o


def cmd_search(a, pool):
    cat = load_catalog()
    W = a.work
    runs = cat[W]["runs"]
    gold = [r for r in runs if r["host"] == "epyc" and r["label"] == GOLD_LABEL][0]
    reps = {}
    for r in runs:
        reps.setdefault(r["class"], r)
    # prefer an epyc, default-t1 representative where the class has one
    for c in list(reps):
        mem = [r for r in runs if r["class"] == c]
        mem.sort(key=lambda r: (not r["default_t1"], r["host"] != "epyc", r["label"]))
        reps[c] = mem[0]
    tasks = []
    for r in runs:
        if r["id"] != gold["id"]:
            tasks.append((("vs_golden", r["id"]), gold["dir"], r["dir"], True))
    cl = sorted(reps)
    for i in range(len(cl)):
        for j in range(i + 1, len(cl)):
            if cl[i] == 0:
                continue  # class 0 vs X == golden vs a member of X (already in vs_golden)
            tasks.append((("class_pair", f"{cl[i]}|{cl[j]}"), reps[cl[i]]["dir"], reps[cl[j]]["dir"], False))
    # same-machine pairs (threads, block shape, library setting) by label
    same = SAME_MACHINE.get(W, [])
    byid = {r["id"]: r for r in runs}
    for x, y, what in same:
        if x in byid and y in byid:
            tasks.append((("same_machine", f"{x}|{y}|{what}"), byid[x]["dir"], byid[y]["dir"], True))
    log(f"search {W}: {len(tasks)} pair tasks over {len(runs)} runs / {len(cl)} classes")
    res = pool.map(_task_pair, tasks, chunksize=1) if pool else [_task_pair(t) for t in tasks]
    out = dict(work=W, golden=gold["id"], classes={str(c): dict(rep=reps[c]["id"],
                                                               members=[r["id"] for r in runs if r["class"] == c])
                                                   for c in cl},
               vs_golden={}, class_pairs={}, same_machine={})
    for (kind, key), o in res:
        if kind == "vs_golden":
            out["vs_golden"][key] = o
        elif kind == "class_pair":
            out["class_pairs"][key] = o
        else:
            out["same_machine"][key] = o
    out["bench_golden"] = _task_bench(gold["dir"])[1]
    # envelopes over scopes (one representative per class present in the scope)
    scopes = {
        "blas_fp32_default_t1": lambda r: r["kind"] in ("golden", "blas") and r["default_t1"],
        "blas_fp32_all": lambda r: r["kind"] in ("golden", "blas"),
        "blas_fp32_all_plus_qpnumpy": lambda r: r["kind"] in ("golden", "blas", "qpnumpy"),
    }
    env_tasks = []
    for nm, f in scopes.items():
        members = [r for r in runs if f(r)]
        cls = sorted({r["class"] for r in members})
        dirs = [next(r for r in members if r["class"] == c)["dir"] for c in cls]
        env_tasks.append((nm, dirs, cls, [r["id"] for r in members]))
    env_res = pool.map(_task_env, env_tasks, chunksize=1) if pool else [_task_env(t) for t in env_tasks]
    out["envelopes"] = {nm: o for nm, o in env_res}
    out["meta"] = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=time.time() - T0, thresholds=THRESH,
                       definitions=DEFS)
    dump(out, f"search_{W}.json")


def _task_env(args):
    nm, dirs, cls, members = args
    t = time.time()
    o = envelope(dirs)
    o.update(classes=cls, members=members, seconds=time.time() - t)
    return nm, o


DEFS = dict(
    abs="|test - ref|", rel_floor1="|test - ref| / max(|ref|, 1)", S="f64 S_f - f64 S_r", Q90="np.quantile(panel S, .9)",
    S_prime="S - Q90", lnE="lnK(m) + ln m + ln N - lambda(m) S' (ship_final 'all', N = 8,557,236)",
    E_le_10="pairs with E <= 10 in ref OR test", flips="pairs with (E_ref <= t) != (E_test <= t)",
    user_visible=("FoldMatch output.py: evalue f'{E:.3E}', bits floor(x + 0.5) of (lambda S' - lnK)/ln 2, raw = S_f "
                  "(any float32 bit change is visible)"),
    ranking="per query: S descending, target index ascending (no embscore); top-k order / set",
    percentiles="exact np.percentile (linear) over every pair",
    bench="TP = shared SCOP family & annotated; FP = benchmark FP (incl. reversed targets); FPA = FP & annotated "
          "(real null); FPA_per_q_per_x = research 'FP/q/x'; AUC1 as research summarize.py")

# same machine / same library, one factor changed (search sets); ids are host:label
SAME_MACHINE = {
    "search-q150-T50000": [
        ("epyc:numpy-qpc-t1-rep-ob0.3.31-default", "epyc:numpy-qpc-t8-rep-ob0.3.31-default", "threads 1 -> 8"),
        ("epyc:numpy-qpc-t1-rep-ob0.3.31-default", "epyc:numpy-qpc-t32-rep-ob0.3.31-default", "threads 1 -> 32"),
        ("epyc:numpy-qpc-t1-rep-ob0.3.31-default", "epyc:numpy-qpc-t64-rep-ob0.3.31-default", "threads 1 -> 64"),
        ("epyc:numpy-qpc-t32-rep-ob0.3.31-default", "epyc:numpy-qpc-t64-rep-ob0.3.31-default", "threads 32 -> 64"),
        ("epyc:numpy-qpc-t1-ob0.3.23-3dino", "epyc:numpy-qpc-t32-rep-ob0.3.23-3dino", "threads 1 -> 32"),
        ("epyc:numpy-qpc-t1-ob0.3.34omp-nersc26.8", "epyc:numpy-qpc-t32-rep-ob0.3.34omp-nersc26.8", "threads 1 -> 32"),
        ("epyc:numpy-qpc-t1-mkl2023.2-nersc24.1", "epyc:numpy-qpc-t32-rep-mkl2023.2-nersc24.1", "threads 1 -> 32"),
        ("epyc:torch_cpu-qpc-t1-rep-mkl2024.2", "epyc:torch_cpu-qpc-t32-rep-mkl2024.2", "threads 1 -> 32"),
        ("epyc:numpy-qpc-t1-mkl2023.2-nersc24.1", "epyc:numpy-qpc-t1-mkl2023.2-nersc24.1-cbwrCOMPAT",
         "MKL_CBWR=COMPATIBLE"),
        ("epyc:numpy-qpc-t1-rep-ob0.3.31-default", "epyc:numpy-qpc-t1-ob0.3.31-Sandybridge",
         "OPENBLAS_CORETYPE=Sandybridge"),
        ("epyc:numpy-qpc-t1-rep-ob0.3.31-default", "epyc:numpy-qpc-t1-ob0.3.31-Prescott", "OPENBLAS_CORETYPE=Prescott"),
        ("epyc:numpy-qpc-t1-rep-ob0.3.31-default", "epyc:numpy_rowwise-qpc-t1-ob0.3.31-default",
         "one GEMM per block -> one GEMM per target"),
        ("epyc:numpy-qpc-t1-rep-ob0.3.31-default", "epyc:numpy-qpnumpy-t1-ob0.3.31-default",
         "qprep C float64 -> numpy fp32 norm"),
        ("a100:torch_cuda-qpc-t1-rep", "a100:torch_cuda-qpc-t1-research1048576", "blocks 2^17 -> research 2^20 chunks"),
        ("a100:torch_cuda-qpc-t1-rep", "a100:torch_cuda_tf32-qpc-t1-rep", "FP32 -> TF32"),
        ("m1:numpy-qpc-t1-rep", "m1:numpy-qpc-tdef-rep", "threads 1 -> default"),
        ("m1:torch_cpu-qpc-t1", "m1:torch_cpu-qpc-tdef-rep", "threads 1 -> default"),
        ("m1:numpy-qpc-t1-rep", "m1:numpy_rowwise-qpc-t1", "one GEMM per block -> one GEMM per target"),
        ("m1:numpy-qpc-t1-ob0.3.21", "m1:numpy-qpc-t1-ob0.3.30", "OpenBLAS 0.3.21 -> 0.3.30"),
        ("m1:fixed_c-qpc-t1-rep", "m1:fixed_c-qpnumpy-t1", "qprep C float64 -> numpy fp32 norm"),
        ("epyc:numpy-qpnumpy-t1-ob0.3.31-default", "m1:numpy-qpnumpy-t1",
         "qprep numpy: EPYC OpenBLAS vs M1 Accelerate"),
    ],
    "search-all-T50000": [
        ("epyc:numpy-qpc-t1-ob0.3.31-default", "epyc:numpy-qpc-t1-mkl2023.2-nersc24.1", "OpenBLAS -> MKL (numpy)"),
        ("epyc:numpy-qpc-t1-ob0.3.31-default", "epyc:torch_cpu-qpc-t1-mkl2024.2", "OpenBLAS -> MKL (torch)"),
    ],
}


# =====================================================================================================================
# a1
# =====================================================================================================================
def load_a1run(d):
    z = np.load(os.path.join(d, "a1.npz"))
    R = {k: z[k] for k in z.files}
    off, flat = R["path_off"], R["path_flat"]
    R["paths"] = [flat[off[k]:off[k + 1]] for k in range(len(off) - 1)]
    return R


_A1: dict = {}


def _a1_init():
    if "A" not in _A1:
        _A1["A"] = BD.load_a1(DATA)


def _oracle_one(args):
    p, gold_path, gold_sf = args
    _a1_init()
    A = _A1["A"]
    q16, codes = BD.a1_pair(A, p)
    s = BD.oracle_sim(q16, codes)
    z = BD.oracle_z(s)
    o = BD.oracle_dp(z - BD.SETTING_D[0])
    sd, tau = s.std(1), s.std(0)
    okr, okc = sd >= 1e-6, tau >= 1e-6
    return dict(pair=int(p), n=int(s.shape[0]), m=int(s.shape[1]), margin_best=o["margin_best"],
                margin_tb=o["margin_tb"], margin=min(o["margin_best"], o["margin_tb"]), oracle_score=o["score"],
                gold_Sf=float(gold_sf), d_score=abs(o["score"] - float(gold_sf)),
                oracle_eq_gold=bool(np.array_equal(o["path"], gold_path)), path_len=int(len(o["path"])),
                path=o["path"],
                n_match=int(((o["path"][:, 0] >= 0) & (o["path"][:, 1] >= 0)).sum()) if len(o["path"]) else 0,
                sd_min=float(sd[okr].min()) if okr.any() else None, sd_med=float(np.median(sd)),
                tau_min=float(tau[okc].min()) if okc.any() else None, tau_med=float(np.median(tau)),
                degenerate_rows=int((~okr).sum()), degenerate_cols=int((~okc).sum()))


def cmd_a1(a, pool):
    cat = load_catalog()
    runs = cat["a1-all"]["runs"]
    _a1_init()
    A1 = _A1["A"]
    gold = [r for r in runs if r["host"] == "epyc" and r["label"] == GOLD_LABEL][0]
    G = load_a1run(gold["dir"])
    P = len(G["pairs"])
    assert np.array_equal(G["pairs"], np.arange(P))
    sets = A1["set_names"][A1["pair_set"]]
    # golden F1 per pair vs US-align and identity vs the stored setting-D alignments
    f1g = np.full(P, np.nan)
    eq_stored = 0
    for p in range(P):
        mG = BD.matched(G["paths"][p])
        eq_stored += int(np.array_equal(mG, BD.a1_stored(A1, p)))
        if A1["has_ref"][p]:
            f1g[p] = BD.prf(mG, BD.a1_ref(A1, p))[2]
    f1_sets = {s: float(np.nanmean(f1g[sets == s])) for s in np.unique(sets) if np.isfinite(f1g[sets == s]).any()}
    out = dict(golden=gold["id"], n_pairs=P, golden_eq_stored=eq_stored, golden_f1_per_set=f1_sets,
               research_f1_rounded=BD.RESEARCH_F1, golden_f1_all=float(np.nanmean(f1g)), runs={})
    log("a1: oracle on all pairs ...")
    tasks = [(p, G["paths"][p], G["Sf"][p]) for p in range(P)]
    # biggest first for load balance
    sizes = np.array([(A1["q_off"][A1["pair_q"][p] + 1] - A1["q_off"][A1["pair_q"][p]])
                      * (A1["c_off"][A1["pair_t"][p] + 1] - A1["c_off"][A1["pair_t"][p]]) for p in range(P)])
    order = np.argsort(-sizes)
    res = pool.map(_oracle_one, [tasks[i] for i in order], chunksize=4) if pool else [_oracle_one(tasks[i]) for i in order]
    orc = sorted(res, key=lambda d: d["pair"])
    opaths = {d["pair"]: d.pop("path") for d in orc}
    marg = np.array([d["margin"] for d in orc])
    rel_marg = marg / np.maximum(np.array([abs(d["oracle_score"]) for d in orc]), 1.0)
    out["oracle"] = dict(
        n=P, oracle_eq_golden=int(sum(d["oracle_eq_gold"] for d in orc)),
        golden_ne_oracle=[d for d in orc if not d["oracle_eq_gold"]][:50],
        d_score=pstats([d["d_score"] for d in orc]),
        margin=dict(pstats(marg[np.isfinite(marg)]), **{f"n_lt_{t:g}": int((marg < t).sum())
                                                       for t in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)},
                    min5=sorted(marg.tolist())[:5]),
        margin_rel_floor1={f"n_lt_{t:g}": int((rel_marg < t).sum()) for t in (1e-6, 1e-5, 1e-4)},
        smallest_margins=sorted(orc, key=lambda d: d["margin"])[:20],
        sd_min=pstats([d["sd_min"] for d in orc if d["sd_min"] is not None]),
        sd_med=pstats([d["sd_med"] for d in orc]), tau_min=pstats([d["tau_min"] for d in orc if d["tau_min"] is not None]),
        tau_med=pstats([d["tau_med"] for d in orc]), n_match=pstats([d["n_match"] for d in orc]),
        path_len=pstats([d["path_len"] for d in orc]),
        pairs_with_degenerate_rows=int(sum(d["degenerate_rows"] > 0 for d in orc)),
        pairs_with_degenerate_cols=int(sum(d["degenerate_cols"] > 0 for d in orc)))
    margin_of = {d["pair"]: dict(d, path=opaths[d["pair"]]) for d in orc}
    np.savez_compressed(os.path.join(OUT, "a1_oracle.npz"), pair=np.array([d["pair"] for d in orc]), margin=marg,
                        margin_best=np.array([d["margin_best"] for d in orc]),
                        margin_tb=np.array([d["margin_tb"] for d in orc]),
                        oracle_score=np.array([d["oracle_score"] for d in orc]),
                        oracle_eq_gold=np.array([d["oracle_eq_gold"] for d in orc]),
                        sd_min=np.array([np.nan if d["sd_min"] is None else d["sd_min"] for d in orc]),
                        sd_med=np.array([d["sd_med"] for d in orc]), n_match=np.array([d["n_match"] for d in orc]))
    for r in runs:
        if r["id"] == gold["id"]:
            continue
        R = load_a1run(r["dir"])
        assert np.array_equal(R["pairs"], G["pairs"])
        o = {}
        for nm in ("Sf", "Sr"):
            eq = BD.bits_equal(G[nm], R[nm])
            d = np.abs(R[nm].astype(np.float64) - G[nm])
            o[nm] = dict(frac_bit_equal=float(eq.mean()), abs=pstats(d),
                         rel_floor1=pstats(d / np.maximum(np.abs(G[nm].astype(np.float64)), 1.0)))
        dS = (R["Sf"].astype(np.float64) - R["Sr"]) - (G["Sf"].astype(np.float64) - G["Sr"])
        o["S"] = dict(abs=pstats(dS))
        diff = []
        f1r = f1g.copy()
        for p in range(P):
            if np.array_equal(R["paths"][p], G["paths"][p]):
                continue
            ca = {tuple(x) for x in G["paths"][p].tolist()}
            cb = {tuple(x) for x in R["paths"][p].tolist()}
            mR = BD.matched(R["paths"][p])
            f1 = BD.prf(mR, BD.a1_ref(A1, p))[2] if A1["has_ref"][p] else None
            if f1 is not None:
                f1r[p] = f1
            om = margin_of[p]
            diff.append(dict(pair=p, set=str(sets[p]), n=om["n"], m=om["m"], Sf_gold=float(G["Sf"][p]),
                             Sf_test=float(R["Sf"][p]), n_cols_diff=len(ca ^ cb), f1_gold=None if np.isnan(f1g[p])
                             else float(f1g[p]), f1_test=f1, dF1=None if f1 is None else f1 - float(f1g[p]),
                             margin=om["margin"], margin_best=om["margin_best"], margin_tb=om["margin_tb"],
                             oracle_eq_gold=om["oracle_eq_gold"],
                             oracle_eq_test=bool(np.array_equal(om["path"], R["paths"][p]))))
        o["paths"] = dict(identical=P - len(diff), frac_identical=(P - len(diff)) / P, n_diff=len(diff), diffs=diff[:50],
                          all_margins_lt_1e_4=all(d["margin"] < 1e-4 for d in diff) if diff else None)
        o["f1_per_set"] = {s: float(np.nanmean(f1r[sets == s])) for s in f1_sets}
        o["tb_eq_Sf"] = int(BD.bits_equal(R["tb"], R["Sf"]).sum())
        out["runs"][r["id"]] = dict(class_=r["class"], kind=r["kind"], flags=r["flags"], lib=r["lib"], **o)
        log("a1", r["id"], "paths diff", len(diff))
    # class-pair maxima (paths + S_f) between all a1 classes
    reps = {}
    for r in runs:
        reps.setdefault(r["class"], r)
    cl = sorted(reps)
    loaded = {c: load_a1run(reps[c]["dir"]) for c in cl}
    cp = {}
    for i in range(len(cl)):
        for j in range(i + 1, len(cl)):
            X, Y = loaded[cl[i]], loaded[cl[j]]
            d = np.abs(Y["Sf"].astype(np.float64) - X["Sf"])
            cp[f"{cl[i]}|{cl[j]}"] = dict(
                a=reps[cl[i]]["id"], b=reps[cl[j]]["id"],
                paths_diff=int(sum(not np.array_equal(X["paths"][p], Y["paths"][p]) for p in range(P))),
                Sf_frac_bit_equal=float(BD.bits_equal(X["Sf"], Y["Sf"]).mean()), Sf_max_abs=float(d.max()),
                Sf_max_rel_floor1=float((d / np.maximum(np.abs(X["Sf"].astype(np.float64)), 1.0)).max()))
    out["class_pairs"] = cp
    out["classes"] = cat["a1-all"]["classes"]
    out["meta"] = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), seconds=time.time() - T0,
                       margin=("float64 oracle (sim64 -> S7 z -> spike-recurrence DP, blas_div.oracle_dp): "
                               "min(best-cell margin, traceback decision margin)"))
    dump(out, "a1.json")


# =====================================================================================================================
# sim
# =====================================================================================================================
def load_sim(d):
    p = os.path.join(d, "simblocks.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p)
    keys = list(zip(z["row"].tolist(), z["k0"].tolist(), z["k1"].tolist()))
    return dict(keys=keys, off=z["off"], sim=z["sim"])


def _sim_task(args):
    gdir, d, s64_path = args
    G, R = load_sim(gdir), load_sim(d)
    if R is None or R["keys"] != G["keys"]:
        return d, dict(skipped="different sampled blocks (block rule) or no simblocks")
    s64 = np.load(s64_path)["s64"]
    a, b = G["sim"], R["sim"]
    eq = BD.bits_equal(a, b)
    dd = np.abs(b.astype(np.float64) - a)
    big = np.abs(a) >= 1e-3
    u = BD.ulp_diff(a[big], b[big])
    pu = np.percentile(u, [99.9, 100])
    ea, eb = np.abs(a - s64), np.abs(b - s64)
    cr = s64.astype(np.float32)
    return d, dict(cells=int(a.size), frac_bit_equal=float(eq.mean()), abs=pstats(dd),
                   ulp_abs_ge_1e_3=dict(n=int(big.sum()), p999=float(pu[0]), max=int(pu[1]),
                                        frac_le1=float((u <= 1).mean())),
                   oracle=dict(test_max_abs_err=float(eb.max()), golden_max_abs_err=float(ea.max()),
                               test_frac_correctly_rounded=float(BD.bits_equal(b, cr).mean()),
                               golden_frac_correctly_rounded=float(BD.bits_equal(a, cr).mean()),
                               test_closer=float((eb < ea).mean()), golden_closer=float((ea < eb).mean())))


def cmd_sim(a, pool):
    cat = load_catalog()
    runs = cat["search-q150-T50000"]["runs"]
    gold = [r for r in runs if r["host"] == "epyc" and r["label"] == GOLD_LABEL][0]
    G = load_sim(gold["dir"])
    Bn = BD.load_bench(DATA)
    s64 = []
    for (r, k0, k1) in G["keys"]:
        q16 = Bn["q"][Bn["qoff"][r]:Bn["qoff"][r + 1]]
        s64.append(BD.oracle_sim(q16, Bn["codes"][Bn["toff"][k0]:Bn["toff"][k1]]).ravel())
    s64 = np.concatenate(s64)
    assert s64.size == G["sim"].size
    s64p = os.path.join(OUT, "sim_oracle64.npz")
    np.savez(s64p, s64=s64)
    gerr = np.abs(G["sim"] - s64)
    out = dict(golden=gold["id"], blocks=len(G["keys"]), cells=int(G["sim"].size),
               golden_vs_oracle=dict(abs_err=pstats(gerr),
                                     frac_correctly_rounded=float(BD.bits_equal(G["sim"],
                                                                                s64.astype(np.float32)).mean()),
                                     max_abs_sim=float(np.abs(s64).max())),
               runs={})
    tasks = [(gold["dir"], r["dir"], s64p) for r in runs if r["id"] != gold["id"]]
    res = pool.map(_sim_task, tasks) if pool else [_sim_task(t) for t in tasks]
    idof = {r["dir"]: r for r in runs}
    for d, o in res:
        r = idof[d]
        out["runs"][r["id"]] = dict(kind=r["kind"], flags=r["flags"], lib=r["lib"], **o)
    # class pairs by sim hash
    simcls = {}
    for r in runs:
        h = r["hash"].get("sim")
        if h and r["hash"].get("sim_keys") == gold["hash"].get("sim_keys"):
            simcls.setdefault(h, r)
    reps = list(simcls.values())
    loaded = {r["id"]: load_sim(r["dir"])["sim"] for r in reps}
    cp = {}
    for i in range(len(reps)):
        for j in range(i + 1, len(reps)):
            x, y = loaded[reps[i]["id"]], loaded[reps[j]["id"]]
            d = np.abs(y.astype(np.float64) - x)
            cp[f"{reps[i]['id']}|{reps[j]['id']}"] = dict(frac_bit_equal=float(BD.bits_equal(x, y).mean()),
                                                           max_abs=float(d.max()), p999=float(np.percentile(d, 99.9)))
    out["sim_classes"] = {h[:12]: [r["id"] for r in runs if r["hash"].get("sim") == h] for h in simcls}
    out["class_pairs"] = cp
    out["meta"] = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"),
                       blocks="16 length-spread q150 queries x {first, middle} production block (2^17 floats)")
    dump(out, "sim.json")


# =====================================================================================================================
# repeat, context
# =====================================================================================================================
def cmd_repeat(a, pool=None):
    out = {}
    for h in HOSTS:
        for p in sorted(glob.glob(f"{ROOT}/runs/{h}/*/*/*/repeat.json")):
            d = os.path.dirname(p)
            r = json.load(open(p))
            key = f"{h}:{os.path.basename(os.path.dirname(d))}:{os.path.basename(d)}"
            out[key] = dict(rows_or_pairs=r.get("rows_or_pairs"), backend=r.get("backend"),
                            stats={v: {k: s.get(k) for k in ("blocks", "blocks_diff", "cand", "sf_diff", "sr_diff",
                                                            "max_abs", "max_abs_sf", "max_abs_sr", "paths_diff")}
                                   for v, s in (r.get("stats") or {}).items()})
    dump(dict(runs=out, meta=dict(definition="blas_div.py --repeat: every block recomputed with the variant layout "
                                             "(same call, +4/+8/+16-byte offsets, padded leading dimension, "
                                             "materialised D^T) and compared bit for bit")), "repeat.json")


def cmd_context(a, pool=None):
    d = json.load(open(os.path.join(PARITY, "esm3_parity.json")))
    out = dict(source=os.path.join(PARITY, "esm3_parity.json"),
               provisional_threshold_check=d["provisional_threshold_check"],
               P_sub={k: {x: v[x] for x in ("n", "median", "p95", "p99", "max")} for k, v in d["dlnE"]["P_sub"].items()
                      if isinstance(v, dict) and "median" in v},
               full_q200_research_hits={k: {x: v["research_hits"][x] for x in ("n", "median", "p95", "p99", "max")}
                                        for k, v in d["dlnE"]["full_q200"].items()
                                        if isinstance(v, dict) and "research_hits" in v},
               full_a5_gpu=d["dlnE"]["full_a5"]["gpu__gpu"]["research_hits"],
               code_bits_cpu_mac_vs_cpu_linux=d["cross_device"]["cpu_mac_vs_cpu_linux"]
               ["pooled_code_bits_panel_enccheck_chains_tsub"])
    # device vs device on the 831 P_sub pairs, same arithmetic as here (Q90 = device q x shipped panel)
    sc = os.path.join(PARITY, "scores")
    Bn = BD.load_bench(DATA)
    tab = BD.ship_table(DATA)
    pos = {q: i for i, q in enumerate(Bn["qids"].tolist())}

    def cfg(q, t):
        z = np.load(f"{sc}/pairs.{q}.q200__{t}.tsub.npz")
        qp = f"{sc}/q90.{q}.q200__research.panel.npz"
        if not os.path.exists(qp):
            qp = f"{sc}/q90.{q}.a5q__research.panel.npz"
        qz = np.load(qp)
        q90all = np.quantile(qz["Sf"].astype(np.float64) - qz["Sr"], 0.9, axis=1)
        qpos = {x: i for i, x in enumerate(qz["q_ids"].tolist())}
        q90 = q90all[[qpos[x] for x in z["q_ids"].tolist()]]
        qi = z["pairs"][:, 0]
        m = np.array([Bn["qlen"][pos[x]] for x in z["q_ids"]])[qi]
        S = z["Sf"].astype(np.float64) - z["Sr"]
        lm = np.log(m.astype(np.float64))
        lam, lnK = np.interp(lm, tab["lnc"], tab["lam"]), np.interp(lm, tab["lnc"], tab["lnK"])
        return lnK + lm + math.log(tab["N"]) - lam * (S - q90[qi]), z["pairs"]

    dd = {}
    for (q1, t1), (q2, t2) in ((("cpu_mac", "cpu_mac"), ("cpu_linux", "cpu_linux")),
                               (("gpu", "gpu"), ("cpu_linux", "cpu_linux")),
                               (("mps", "mps"), ("cpu_mac", "cpu_mac")),
                               (("gpu", "gpu"), ("cpu_mac", "cpu_mac")),
                               (("cpu_linux", "cpu_linux"), ("research", "research")),
                               (("gpu", "gpu"), ("research", "research"))):
        try:
            l1, p1 = cfg(q1, t1)
            l2, p2 = cfg(q2, t2)
        except FileNotFoundError as e:
            dd[f"{q1}__{t1} vs {q2}__{t2}"] = dict(error=str(e))
            continue
        assert np.array_equal(p1, p2)
        x = np.abs(l1 - l2)
        p = np.percentile(x, [50, 95, 99, 100])
        dd[f"{q1}__{t1} vs {q2}__{t2}"] = dict(n=int(x.size), median=float(p[0]), p95=float(p[1]), p99=float(p[2]),
                                               max=float(p[3]), frac_zero=float((x == 0).mean()))
    out["device_vs_device_P_sub"] = dd
    out["definition"] = ("|d lnE| between two ESM3 encoding sources (query q and target codes from the same device) on the "
                         "831 sampled research E<=1 hit pairs (P_sub), setting D research kernel, Q90 = that device's q "
                         "x the shipped panel codes, shipped table 'all', N = 8,557,236")
    dump(out, "context.json")


SF_BINS = (0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, math.inf)
LNE_BINS = (-math.inf, math.log(1e-10), math.log(1e-3), 0.0, math.log(10.0), math.inf)


def _extra_work(W, cat):
    """per-pair spread over every FP32 / qprep-c build class (and + qprep numpy) as a function of the golden score
    magnitude and E; golden pairs within delta of each E threshold (exposure to hit-decision flips)."""
    runs = cat[W]["runs"]
    out = {}
    gold = [r for r in runs if r["host"] == "epyc" and r["label"] == GOLD_LABEL][0]
    G = load_search(gold["dir"])
    ctx = _search_ctx(G["rows"])
    Sg, qg, lg = derive(G, ctx)
    sfg = np.abs(G["Sf"].astype(np.float64))
    exp = {}
    for t in THRESH:
        d = np.abs(lg - math.log(t))
        exp[f"{t:g}"] = {f"{x:g}": int((d < x).sum()) for x in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.17)}
    out["golden_pairs_within_delta_of_ln_t"] = exp
    out["pairs"] = int(lg.size)
    for scope, kinds in (("blas_fp32_all", ("golden", "blas")), ("blas_fp32_all_plus_qpnumpy", ("golden", "blas", "qpnumpy"))):
        cls = sorted({r["class"] for r in runs if r["kind"] in kinds})
        dirs = [next(r for r in runs if r["class"] == c and r["kind"] in kinds)["dir"] for c in cls]
        mn = mx = None
        for d in dirs:
            R = load_search(d)
            S, q90, lnE = derive(R, ctx)
            arrs = (R["Sf"].astype(np.float64), R["Sr"].astype(np.float64), lnE)
            if mn is None:
                mn = [a.copy() for a in arrs]
                mx = [a.copy() for a in arrs]
            else:
                for i, a in enumerate(arrs):
                    np.minimum(mn[i], a, out=mn[i])
                    np.maximum(mx[i], a, out=mx[i])
            del R, S, lnE, arrs
        rf, rr, rl = (mx[i] - mn[i] for i in range(3))
        bins = []
        for lo, hi in zip(SF_BINS[:-1], SF_BINS[1:]):
            m = (sfg >= lo) & (sfg < hi)
            if not m.any():
                continue
            x = rf[m]
            bins.append(dict(Sf_lo=lo, Sf_hi=hi, n=int(m.sum()), max_abs=float(x.max()),
                             p9999_abs=float(np.percentile(x, 99.99)),
                             max_rel_floor1=float((x / np.maximum(sfg[m], 1.0)).max()),
                             max_rel=float((x[sfg[m] > 0] / sfg[m][sfg[m] > 0]).max()) if (sfg[m] > 0).any() else None))
        # smallest (a, b) of |dS_f| <= a + b |S_f| over a grid of a: b = max((range - a)/|S_f|)
        fits = []
        for a_ in (1e-5, 2e-5, 3e-5, 5e-5, 1e-4):
            b_ = float(np.max((np.maximum(rf - a_, 0.0)) / np.maximum(sfg, 1e-30)))
            fits.append(dict(a=a_, b=b_))
        lb = []
        for lo, hi in zip(LNE_BINS[:-1], LNE_BINS[1:]):
            m = (lg > lo) & (lg <= hi)
            if not m.any():
                continue
            lb.append(dict(lnE_lo=lo, lnE_hi=hi, n=int(m.sum()), max_abs=float(rl[m].max()),
                           p99=float(np.percentile(rl[m], 99))))
        out[scope] = dict(classes=cls, Sf_range_by_golden_Sf=bins, Sf_bound_fits=fits,
                          Sr_range_max=float(rr.max()), lnE_range_by_golden_lnE=lb)
        del mn, mx, rf, rr, rl
    return out


def cmd_extra(a, pool=None):
    cat = load_catalog()
    out = {W: _extra_work(W, cat) for W in ("search-q150-T50000", "search-all-T50000")}
    out["meta"] = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"),
                       definition=("range = max - min over one build per bit-identity class in the scope, per pair; "
                                   "binned by the golden |S_f| or golden lnE; exposure = golden pairs with "
                                   "|lnE - ln t| < delta (a build change of |d lnE| < delta can only flip these)"))
    dump(out, "extra.json")


# =====================================================================================================================
# summary (runs anywhere on a copy of ROOT/report)
# =====================================================================================================================
CLASS_NAMES = {  # (work, a member id) -> name; first match wins
    "search": [
        ("epyc:fixed_c-qpc-t1", "golden: fixed-order C (EPYC, A100 node, M1) = M1 Accelerate numpy/torch (1 or default "
                                "threads) = M1 OpenBLAS 0.3.21/0.3.23/0.3.30 (armv8, neoversen1) = EPYC OpenBLAS "
                                "forced Sandybridge"),
        ("epyc:numpy-qpc-t1-rep-ob0.3.31-default", "EPYC OpenBLAS default kernel (0.3.23 ... 0.3.34, Haswell/Zen, "
                                                   "pthreads/OpenMP) = Cray LibSci 26.03, 1 thread"),
        ("epyc:numpy-qpc-t1-mkl2023.2-nersc24.1", "EPYC MKL 1 thread (numpy MKL 2023.2, torch 2.9/2.11 MKL 2024.2, "
                                                  "MKL_CBWR=AVX2)"),
        ("epyc:numpy-qpc-t1-mkl2023.2-nersc24.1-cbwrCOMPAT", "EPYC MKL_CBWR=COMPATIBLE (numpy, torch)"),
        ("epyc:numpy-qpc-t1-ob0.3.31-Prescott", "EPYC OpenBLAS forced Prescott"),
        ("epyc:numpy-qpc-t32-rep-mkl2023.2-nersc24.1", "EPYC MKL 32 threads (numpy, torch)"),
        ("epyc:numpy-qpc-t32-rep-ob0.3.23-3dino", "EPYC OpenBLAS 0.3.23 32 threads"),
        ("epyc:numpy-qpc-t32-rep-ob0.3.31-default", "EPYC OpenBLAS 0.3.31 / 0.3.34 32 threads"),
        ("epyc:numpy-qpc-t64-rep-ob0.3.31-default", "EPYC OpenBLAS 0.3.31 64 threads"),
        ("epyc:numpy-qpc-t8-rep-ob0.3.31-default", "EPYC OpenBLAS 0.3.31 8 threads"),
        ("epyc:numpy-qpnumpy-t1-ob0.3.31-default", "EPYC OpenBLAS + numpy fp32 query norm"),
        ("epyc:numpy_rowwise-qpc-t1-ob0.3.31-default", "EPYC OpenBLAS one GEMM per target"),
        ("a100:torch_cuda-qpc-t1-research1048576", "A100 cuBLAS FP32, research 2^20-float chunks"),
        ("a100:torch_cuda-qpc-t1-rep", "A100 cuBLAS FP32"),
        ("a100:torch_cuda-qpnumpy-t1", "A100 cuBLAS FP32 + numpy fp32 query norm"),
        ("a100:torch_cuda_tf32-qpc-t1-rep", "A100 TF32 (hazard)"),
        ("m1:numpy-qpnumpy-t1", "M1 numpy fp32 query norm (fixed-order C, Accelerate, OpenBLAS)"),
        ("m1:numpy_rowwise-qpc-t1", "M1 Accelerate one GEMM per target"),
        ("m1:torch_mps-qpc-rep", "M1 MPS fp32"),
        ("epyc:numpy-qpc-t1-ob0.3.31-default", "EPYC OpenBLAS 0.3.31 default (Haswell kernel) = forced Haswell / Zen"),
        ("epyc:numpy-qpc-t1-ob0.3.31-Sandybridge", "golden: fixed-order C = EPYC OpenBLAS forced Sandybridge"),
    ],
    "search-all": [
        ("epyc:fixed_c-qpc-t1", "golden: fixed-order C = EPYC OpenBLAS forced Sandybridge"),
        ("epyc:numpy-qpc-t1-mkl2023.2-nersc24.1", "EPYC MKL 1 thread (numpy MKL 2023.2, torch 2.11 MKL 2024.2)"),
        ("epyc:numpy-qpc-t1-ob0.3.31-default", "EPYC OpenBLAS 0.3.31 default (Haswell kernel) = forced Haswell / Zen"),
        ("epyc:numpy-qpc-t1-ob0.3.31-Prescott", "EPYC OpenBLAS forced Prescott"),
    ],
    "a1": [
        ("epyc:fixed_c-qpc-t1", "golden: fixed-order C (EPYC, A100 node, M1) = M1 OpenBLAS 0.3.21/0.3.23/0.3.30 = EPYC "
                                "OpenBLAS forced Sandybridge"),
        ("epyc:numpy-qpc-t1-rep-ob0.3.31-default", "EPYC OpenBLAS default kernel (0.3.23 ... 0.3.34) = Cray LibSci, 1 "
                                                   "thread"),
        ("epyc:numpy-qpc-t1-mkl2023.2-nersc24.1", "EPYC MKL 1 thread (numpy, torch 2.9/2.11, MKL_CBWR=AVX2)"),
        ("epyc:numpy-qpc-t1-mkl2023.2-nersc24.1-cbwrCOMPAT", "EPYC MKL_CBWR=COMPATIBLE"),
        ("epyc:numpy-qpc-t1-ob0.3.31-Prescott", "EPYC OpenBLAS forced Prescott"),
        ("epyc:numpy-qpc-t32-rep-ob0.3.31-default", "EPYC OpenBLAS 0.3.31 32 threads"),
        ("epyc:numpy-qpnumpy-t1-ob0.3.31-default", "EPYC OpenBLAS + numpy fp32 query norm"),
        ("epyc:torch_cpu-qpc-t32-rep-mkl2024.2", "EPYC torch MKL 32 threads"),
        ("a100:torch_cuda-qpc-t1-rep", "A100 cuBLAS FP32"),
        ("a100:torch_cuda-qpnumpy-t1", "A100 cuBLAS FP32 + numpy fp32 query norm"),
        ("a100:torch_cuda_tf32-qpc-t1-rep", "A100 TF32 (hazard)"),
        ("m1:fixed_c-qpnumpy-t1", "M1 numpy fp32 query norm (fixed-order C, OpenBLAS 0.3.21)"),
        ("m1:numpy-qpc-t1-rep", "M1 Accelerate numpy/torch (1 or default threads, also one GEMM per target)"),
        ("m1:numpy-qpnumpy-t1", "M1 Accelerate + numpy fp32 query norm"),
        ("m1:torch_mps-qpc-rep", "M1 MPS fp32"),
    ],
}


def class_name(work, members):
    lst = "a1" if work.startswith("a1") else ("search-all" if work.startswith("search-all") else "search")
    for key, nm in CLASS_NAMES[lst]:
        if key in members:
            return nm
    return " = ".join(members)


def _ps(d, keys=("median", "p99", "p9999", "max")):
    return {k: d.get(k) for k in keys} if isinstance(d, dict) else d


def _pair_row(v):
    """compact view of one pair_search result."""
    fl = {t: v["E_flips"][t]["flips"] for t in ("0.001", "0.01", "0.1", "1", "10")}
    uv = v["user_visible"]
    return dict(
        S_f=dict(frac_bit_identical=v["Sf"]["frac_bit_equal"], abs=_ps(v["Sf"]["abs"]),
                 rel_floor1=_ps(v["Sf"]["rel_floor1"]), ulp_p999=(v["Sf"].get("ulp") or {}).get("p999"),
                 ulp_max=(v["Sf"].get("ulp") or {}).get("max"), at_max_abs_Sf=v["Sf"]["at_max_abs"]["ref"],
                 at_max_rel_Sf=v["Sf"]["at_max_rel"]["ref"]),
        S_r=dict(frac_bit_identical=v["Sr"]["frac_bit_equal"], abs=_ps(v["Sr"]["abs"]),
                 rel_floor1=_ps(v["Sr"]["rel_floor1"])),
        S=dict(frac_identical=v["S"]["frac_equal"], abs=_ps(v["S"]["abs"]), rel_floor1=_ps(v["S"]["rel_floor1"])),
        Q90=dict(frac_identical=v["Q90"]["frac_equal"], median=v["Q90"]["median"], p99=v["Q90"]["p99"],
                 max=v["Q90"]["max"]),
        S_prime=dict(frac_identical=v["S_prime"]["frac_equal"], abs=_ps(v["S_prime"]["abs"]),
                     rel_floor1=_ps(v["S_prime"]["rel_floor1"])),
        lnE=dict(all=_ps(v["lnE"]["all"]), E_le_10=dict(_ps(v["lnE"]["E_le_10"]), n=v["lnE"]["E_le_10"]["n"]),
                 at_max=v["lnE"]["at_max"]),
        E_flips=fl, E_flip_examples=[e for t in fl for e in v["E_flips"][t].get("examples", [])][:3],
        user_visible_E_le_10=dict(n=uv["E_le_10"]["n"], evalue_string_changed=uv["E_le_10"]["evalue_str_diff"],
                                  bits_changed=uv["E_le_10"]["bits_diff"], raw_changed=uv["E_le_10"]["raw_diff"],
                                  examples=uv["E_le_10"]["evalue_examples"][:3]),
        user_visible_E_le_1e_3=dict(n=uv["E_le_1e_3"]["n"], evalue_string_changed=uv["E_le_1e_3"]["evalue_str_diff"],
                                    bits_changed=uv["E_le_1e_3"]["bits_diff"], raw_changed=uv["E_le_1e_3"]["raw_diff"]),
        ranking=v["ranking"], hit_lists=v["hit_lists"])


def _bench_delta(b, g):
    d = {}
    for t in ("0.001", "0.01", "0.1", "1", "10"):
        x = {k: b[t][k] - g[t][k] for k in ("hits", "TP", "FP", "FPA") if b[t][k] != g[t][k]}
        if x:
            d[t] = x
    for k in ("auc1_fp_all", "auc1_fp_real"):
        if b[k] != g[k]:
            d[k] = b[k] - g[k]
    return d


def cmd_summary(a, pool=None):
    R = a.report_dir or OUT
    J = {nm: json.load(open(os.path.join(R, f"{nm}.json"))) for nm in
         ("catalog", "search_search-q150-T50000", "search_search-all-T50000", "a1", "sim", "repeat", "context", "extra")}
    src = {nm: f"report/{nm}.json" for nm in J}
    cat = J["catalog"]
    out = dict(what="BLAS-divergence study (FoldMatch esm-codes-v1, plan D2 option b): how far scores, E-values and "
                    "alignments move between similarity-GEMM builds; everything else is the fixed-order spike kernel "
                    "and float64 E code",
               created=time.strftime("%Y-%m-%d %H:%M:%S"), sources=src,
               nersc_report_dir="/pscratch/sd/j/jsegura/embcomp/blasdiv/report", script="blas_div_report.py")
    out["workload"] = dict(
        q150=dict(queries=150, targets=50000, pairs=7_500_000, panel=1000, source=src["search_search-q150-T50000"]),
        all=dict(queries=1465, targets=50000, pairs=73_250_000, panel=1000, source=src["search_search-all-T50000"]),
        a1=dict(pairs=12960, domain=12228, chain=732, source=src["a1"]),
        sim_cells=dict(blocks=J["sim"]["blocks"], cells=J["sim"]["cells"], source=src["sim"]))
    cls = {}
    for w in ("search-q150-T50000", "search-all-T50000", "a1-all"):
        cls[w] = {c: dict(name=class_name(w, m), members=m) for c, m in cat[w]["classes"].items()}
        cls[w]["n_runs"] = cat[w]["n_runs"]
    out["bit_identity_classes"] = dict(cls, source=src["catalog"])
    # ---------------------------------------------------------------- 1. similarity cells
    S = J["sim"]
    sim_rows = {}
    for rid, v in S["runs"].items():
        if "skipped" in v:
            continue
        sim_rows[rid] = dict(frac_bit_identical=v["frac_bit_equal"], max_abs=v["abs"]["max"], p999_abs=v["abs"]["p999"],
                             ulp_p999_abs_ge_1e_3=v["ulp_abs_ge_1e_3"]["p999"], ulp_max_abs_ge_1e_3=v["ulp_abs_ge_1e_3"]["max"],
                             max_abs_err_vs_float64=v["oracle"]["test_max_abs_err"],
                             frac_correctly_rounded=v["oracle"]["test_frac_correctly_rounded"])
    fp32 = [(v["max_abs"], k, v) for k, v in S["class_pairs"].items() if "tf32" not in k]
    fp32.sort(reverse=True)
    tf = [(v["max_abs"], k, v) for k, v in S["class_pairs"].items() if "tf32" in k]
    tf.sort(reverse=True)
    out["sim_cells"] = dict(source=src["sim"], golden_vs_float64=S["golden_vs_oracle"], per_run_vs_golden=sim_rows,
                            worst_fp32_build_pair=dict(pair=fp32[0][1], **fp32[0][2]) if fp32 else None,
                            min_frac_bit_identical_fp32_pair=min(v["frac_bit_equal"] for _, _, v in fp32),
                            worst_tf32_pair=dict(pair=tf[0][1], **tf[0][2]) if tf else None)
    # ---------------------------------------------------------------- 2/3. scores and E (q150, all)
    for w, key in (("search-q150-T50000", "q150"), ("search-all-T50000", "all")):
        Sj = J[f"search_{w}"]
        per_class = {}
        for c, info in Sj["classes"].items():
            rep = info["rep"]
            if rep == Sj["golden"]:
                continue
            v = Sj["vs_golden"][rep]
            per_class[c] = dict(name=class_name(w, info["members"]), rep=rep, **_pair_row(v),
                                bench_delta_vs_golden=_bench_delta(v["bench_test"], Sj["bench_golden"]))
        pairs = {}
        for k, v in Sj["class_pairs"].items():
            a_, b_ = k.split("|")
            pairs[k] = dict(a=class_name(w, Sj["classes"][a_]["members"]), b=class_name(w, Sj["classes"][b_]["members"]),
                            S_f_frac_bit_identical=v["Sf"]["frac_bit_equal"], S_f_max_rel_floor1=v["Sf"]["rel_floor1"]["max"],
                            S_f_max_abs=v["Sf"]["abs"]["max"], S_max_abs=v["S"]["abs"]["max"], Q90_max=v["Q90"]["max"],
                            lnE_max=v["lnE"]["all"]["max"], lnE_E_le_10_p99=v["lnE"]["E_le_10"]["p99"],
                            E_flips={t: v["E_flips"][t]["flips"] for t in ("0.001", "0.01", "0.1", "1", "10")},
                            evalue_string_changed=v["user_visible"]["E_le_10"]["evalue_str_diff"],
                            top10_order_diff=v["ranking"]["top10"]["order_diff"],
                            top100_order_diff=v["ranking"]["top100"]["order_diff"],
                            top100_set_diff=v["ranking"]["top100"]["set_diff"])
        same = {k: dict(change=k.split("|")[2], a=k.split("|")[0], b=k.split("|")[1], **_pair_row(v))
                for k, v in Sj["same_machine"].items()}
        bench = {"golden": Sj["bench_golden"]}
        bench["all_runs_identical_TP_FP_FPA_AUC1"] = all(not {t: x for t, x in _bench_delta(v["bench_test"], Sj["bench_golden"]).items()
                                                               if t.startswith("auc") or any(kk != "hits" for kk in x)}
                                                          for v in Sj["vs_golden"].values())
        bench["runs_with_any_count_change"] = {rid: _bench_delta(v["bench_test"], Sj["bench_golden"])
                                               for rid, v in Sj["vs_golden"].items()
                                               if _bench_delta(v["bench_test"], Sj["bench_golden"])}
        out[f"scores_E_{key}"] = dict(source=src[f"search_{w}"], golden=Sj["golden"], per_class_vs_golden=per_class,
                                      class_pairs=pairs, envelopes=Sj["envelopes"], same_machine=same,
                                      benchmark=bench)
    # ---------------------------------------------------------------- 4. A1
    A = J["a1"]
    a1c = {}
    for c, mem in cat["a1-all"]["classes"].items():
        rep = mem[0]
        if rep == A["golden"]:
            continue
        v = A["runs"][rep]
        a1c[c] = dict(name=class_name("a1", mem), rep=rep, S_f_frac_bit_identical=v["Sf"]["frac_bit_equal"],
                      S_f_max_abs=v["Sf"]["abs"]["max"], S_f_max_rel_floor1=v["Sf"]["rel_floor1"]["max"],
                      S_r_max_rel_floor1=v["Sr"]["rel_floor1"]["max"], S_max_abs=v["S"]["abs"]["max"],
                      paths_identical=v["paths"]["identical"], frac_paths_identical=v["paths"]["frac_identical"],
                      differing=v["paths"]["diffs"], f1_per_set=v["f1_per_set"])
    pmax = [(v["Sf_max_rel_floor1"], k, v) for k, v in A["class_pairs"].items() if "tf32" not in v["a"] + v["b"]]
    pmax.sort(reverse=True)
    out["a1"] = dict(source=src["a1"], golden=A["golden"], n_pairs=A["n_pairs"], golden_eq_stored=A["golden_eq_stored"],
                     golden_f1_per_set=A["golden_f1_per_set"], research_f1_rounded=A["research_f1_rounded"],
                     per_class_vs_golden=a1c,
                     fp32_class_pairs_paths_differing=sum(v["paths_diff"] for v in A["class_pairs"].values()
                                                          if "tf32" not in v["a"] + v["b"]),
                     fp32_class_pairs_max_S_f_rel_floor1=pmax[0][0], fp32_class_pairs_max_S_f_abs=max(
                         v["Sf_max_abs"] for v in A["class_pairs"].values() if "tf32" not in v["a"] + v["b"]),
                     oracle={k: A["oracle"][k] for k in ("n", "oracle_eq_golden", "d_score", "margin", "margin_rel_floor1",
                                                         "sd_min", "sd_med", "tau_min", "tau_med", "n_match")},
                     oracle_smallest_margins=[{k: d[k] for k in ("pair", "n", "m", "margin", "margin_best", "margin_tb",
                                                                 "oracle_score", "gold_Sf")}
                                              for d in A["oracle"]["smallest_margins"][:12]],
                     oracle_source="report/a1.json (per-pair arrays in report/a1_oracle.npz on NERSC)")
    # ---------------------------------------------------------------- repeat / nondeterminism
    rp = {}
    for k, v in J["repeat"]["runs"].items():
        ch = {var: {kk: x[kk] for kk in ("blocks_diff", "blocks", "sf_diff", "max_abs_sf", "max_abs_sr", "paths_diff")}
              for var, x in v["stats"].items() if x.get("blocks_diff")}
        rp[k] = ch or "all variants bit-identical (same, off4, off8, off16, ld, dT)"
    out["repeat"] = dict(source=src["repeat"], runs=rp)
    out["extra"] = dict(source=src["extra"], **{k: v for k, v in J["extra"].items() if k != "meta"})
    out["context_esm3"] = dict(source=src["context"], device_vs_device_P_sub=J["context"]["device_vs_device_P_sub"],
                               vs_research=J["context"]["provisional_threshold_check"],
                               code_bits_cpu_mac_vs_cpu_linux=J["context"]["code_bits_cpu_mac_vs_cpu_linux"],
                               original=J["context"]["source"])
    # ---------------------------------------------------------------- theory vs observed
    u = 2.0 ** -24
    orc = A["oracle"]
    q1 = out["scores_E_q150"]
    env = q1["envelopes"]["blas_fp32_all"]
    envq = q1["envelopes"]["blas_fp32_all_plus_qpnumpy"]
    a100 = q1["per_class_vs_golden"][next(c for c, v in q1["per_class_vs_golden"].items()
                                          if v["rep"] == "a100:torch_cuda-qpc-t1-rep")]
    try:
        tab = BD.ship_table(DATA)
        lam_rng = (float(tab["lam"].min()), float(tab["lam"].max()))
    except Exception:  # noqa: BLE001
        lam_rng = (0.041619283794126286, 0.18775162858061797)
    ds_obs = out["sim_cells"]["worst_fp32_build_pair"]["max_abs"]
    gain_med = 0.5 * (1.0 / orc["sd_med"]["median"] + 1.0 / orc["tau_med"]["median"])
    gain_small = 0.5 * (1.0 / orc["sd_min"]["median"] + 1.0 / orc["tau_min"]["median"])
    sd_order = math.sqrt(63) * 2.0 ** -26 / math.sqrt(12)
    sd_two = math.sqrt(2) * sd_order
    nmed, n99, nmax = orc["n_match"]["median"], orc["n_match"]["p99"], orc["n_match"]["max"]
    dsp_max = env["S"]["abs"]["max"] + env["Q90"]["max"]
    out["theory"] = dict(
        source="arithmetic on the numbers named here (report/a1.json oracle, report/sim.json, report/search_*.json)",
        unit_roundoff_u=u,
        sim=dict(
            l1="sum_k |qp_k| <= sqrt(64) ||qp||_2 = 8 x 1/8 = 1: |s| <= 1, every partial sum < 1; the products "
               "+-qp_k are exact, so FMA vs mul+add is irrelevant; only the summation order / tree differs",
            worst_one_order=63 * 2.0 ** -25, higham_gamma63=63 * u / (1 - 63 * u), worst_two_orders=2 * 63 * 2.0 ** -25,
            random_rounding_sd_one_order=sd_order, random_rounding_sd_two_orders=sd_two,
            random_rounding_extreme_over_cells=sd_two * math.sqrt(2 * math.log(J["sim"]["cells"])),
            observed_max_abs_fp32_build_pair=ds_obs,
            observed_golden_vs_float64=J["sim"]["golden_vs_oracle"]["abs_err"]["max"],
            note="random-rounding model: partial sums ~0.125-0.25 (ulp 2^-26), 63 additions, uniform rounding error"),
        z=dict(identity="dz_ij ~ 0.5 (1/sigma_i + 1/tau_j) ds_ij + (row / column mean and std shifts, averages of ds)",
               gain_at_median_sigma_tau=gain_med, gain_at_median_per_pair_min_sigma_tau=gain_small,
               sigma_median=orc["sd_med"]["median"], tau_median=orc["tau_med"]["median"],
               sigma_min_median=orc["sd_min"]["median"], tau_min_median=orc["tau_min"]["median"]),
        S_f=dict(
            identity="S_f = max over paths of sum(z - 1) - gaps is 1-Lipschitz in the cell scores: |dS_f| <= sum over the "
                     "matched cells of the optimal path(s) of |dz|, also when the path changes; then rounded to float32",
            n_match_A1=dict(median=nmed, p99=n99, max=nmax),
            worst_case_with_observed_max_ds=dict(median_path=nmed * 2 * gain_small * ds_obs,
                                                 longest_path=nmax * 2 * gain_small * ds_obs),
            random_walk_estimate=dict(median_path=math.sqrt(nmed) * gain_med * sd_two,
                                      p99_path=math.sqrt(n99) * gain_med * sd_two),
            float32_ulp_at_1538=2.0 ** (math.floor(math.log2(1538)) - 23),
            observed_A100_vs_golden=dict(median=a100["S_f"]["abs"]["median"], p99=a100["S_f"]["abs"]["p99"],
                                         max=a100["S_f"]["abs"]["max"], S_f_at_max=a100["S_f"]["at_max_abs_Sf"]),
            observed_spread_by_Sf=J["extra"]["search-q150-T50000"]["blas_fp32_all"]["Sf_range_by_golden_Sf"]),
        lnE=dict(identity="lambda(m), lnK(m) depend on the query length only: |d lnE| = lambda(m) |dS'|, "
                          "|dS'| <= |dS| + |dQ90| (Q90 is 1-Lipschitz in the panel scores)",
                 lambda_range=lam_rng, bound_from_observed_max_dS_dQ90=lam_rng[1] * dsp_max,
                 observed_max=env["lnE"]["all"]["max"], observed_at_max=a100["lnE"]["at_max"],
                 E_relative_change_max=math.expm1(env["lnE"]["all"]["max"])))
    # ---------------------------------------------------------------- tolerance contract
    fit = {f["a"]: f["b"] for f in J["extra"]["search-q150-T50000"]["blas_fp32_all_plus_qpnumpy"]["Sf_bound_fits"]}
    out["tolerance_contract"] = dict(
        source="report/search_search-q150-T50000.json envelopes, report/extra.json, report/a1.json; research "
               "agreement from the run phases' verify.json",
        observed=dict(
            S_f_rel_floor1_max_all_fp32_builds=env["S_f"]["rel_floor1"]["max"],
            S_f_rel_floor1_max_default_single_thread=q1["envelopes"]["blas_fp32_default_t1"]["S_f"]["rel_floor1"]["max"],
            S_f_rel_floor1_max_incl_qprep_numpy=envq["S_f"]["rel_floor1"]["max"],
            S_r_range_max=J["extra"]["search-q150-T50000"]["blas_fp32_all_plus_qpnumpy"]["Sr_range_max"],
            S_f_two_term_fits_incl_qprep_numpy=fit,
            Q90_max=env["Q90"]["max"], Q90_max_incl_qprep_numpy=envq["Q90"]["max"],
            lnE_max=env["lnE"]["all"]["max"], lnE_max_incl_qprep_numpy=envq["lnE"]["all"]["max"],
            vs_research_stored_rel_max=dict(
                fixed_c=7.152557373046875e-06, a100_fp32=1.0479693688080492e-05, epyc_openblas_t32=1.0159374166296358e-05,
                source="runs/{epyc,a100}/.../verify.json vs_research_stored_scores (run phases)"),
            A1_fp32_path_changes=out["a1"]["fp32_class_pairs_paths_differing"],
            A1_pairs_margin_lt_1e_4=orc["margin"]["n_lt_0.0001"], A1_min_margin=orc["margin"]["min5"][0],
            tf32_path_change_margin=A["runs"]["a100:torch_cuda_tf32-qpc-t1-rep"]["paths"]["diffs"][0]["margin"]
            if A["runs"]["a100:torch_cuda_tf32-qpc-t1-rep"]["paths"]["diffs"] else None,
            golden_pairs_within_1e_4_of_a_threshold_all=sum(
                v["0.0001"] for v in J["extra"]["search-all-T50000"]["golden_pairs_within_delta_of_ln_t"].values())),
        recommended=dict(
            S_f_and_S_r="|delta| <= 5e-5 * max(|ref|, 1)  (tighter equivalent: |delta| <= 3e-5 + 1e-6 |ref|)",
            Q90="|delta| <= 1e-4",
            lnE="|delta lnE| <= 1e-4 (E within 0.01 %)",
            E_decisions="identical, except pairs with |lnE_ref - ln t| < 1e-4",
            paths="identical, except where the float64-oracle margin < 1e-4 + 2e-6 |S_f|",
            precision="FP32 GEMM only: TF32, bf16 and autocast must be off (TF32 exceeds these bounds 10-50x)",
            not_part_of_contract="bits: no build changes them; raw S_f bits and the 4th digit of '%.3E' E-values do "
                                 "change"),
        plan_values=dict(rel_1e_5="too tight: 1.18e-5 observed between FP32 builds (9.7e-6 for single-thread "
                                  "defaults only; A100 vs research 1.05e-5)",
                         margin_1e_4="consistent with every observation (no FP32 path change; the one TF32 change has "
                                     "margin 9.0e-5)"))
    # ---------------------------------------------------------------- context ratios
    ctxd = J["context"]["device_vs_device_P_sub"]
    blas_pair = a100["lnE"]["E_le_10"]
    blas_env = env["lnE"]["E_le_10"]
    rat = {}
    for nm in ("cpu_mac__cpu_mac vs cpu_linux__cpu_linux", "gpu__gpu vs cpu_linux__cpu_linux",
               "gpu__gpu vs research__research", "mps__mps vs cpu_mac__cpu_mac"):
        e = ctxd[nm]
        rat[nm] = dict(esm3=dict(median=e["median"], p95=e["p95"], max=e["max"]),
                       ratio_median_vs_A100_pair_median=e["median"] / blas_pair["median"],
                       ratio_median_vs_envelope_median=e["median"] / blas_env["median"],
                       ratio_p95_vs_A100_pair_p99=e["p95"] / blas_pair["p99"],
                       ratio_max_vs_envelope_max=e["max"] / blas_env["max"])
    out["context_ratios"] = dict(
        source="report/context.json (ESM3, 831 P_sub research E<=1 hit pairs) vs report/search_search-q150-T50000.json "
               "(BLAS, pairs with E <= 10)",
        blas_A100_vs_golden_E_le_10=dict(median=blas_pair["median"], p99=blas_pair["p99"], max=blas_pair["max"]),
        blas_envelope_all_fp32_E_le_10=dict(median=blas_env["median"], p99=blas_env["p99"], max=blas_env["max"]),
        esm3=rat)
    p = a.out or os.path.join(R, "blas_div_summary.json")
    BD.jdump(clean(out), p)
    log("->", p)


# =====================================================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("catalog", "search", "a1", "sim", "repeat", "context", "extra", "summary", "all"))
    ap.add_argument("--work", default="search-q150-T50000")
    ap.add_argument("--report-dir", default=None, help="summary: directory holding the report JSONs (default ROOT/report)")
    ap.add_argument("--out", default=None, help="summary: output path (default <report-dir>/blas_div_summary.json)")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    if a.cmd != "summary":
        os.makedirs(OUT, exist_ok=True)
    pool = None
    if a.workers > 1:
        import multiprocessing as mp
        pool = mp.get_context("fork").Pool(a.workers)
    if a.cmd in ("catalog", "all"):
        cmd_catalog(a, pool)
    if a.cmd in ("repeat", "all"):
        cmd_repeat(a)
    if a.cmd in ("context", "all"):
        cmd_context(a)
    if a.cmd in ("sim", "all"):
        cmd_sim(a, pool)
    if a.cmd in ("a1", "all"):
        cmd_a1(a, pool)
    if a.cmd == "search":
        cmd_search(a, pool)
    if a.cmd in ("extra", "all"):
        cmd_extra(a)
    if a.cmd in ("summary", "all"):
        cmd_summary(a)
    if a.cmd == "all":
        for w in ("search-q150-T50000", "search-all-T50000"):
            a.work = w
            cmd_search(a, pool)
    if pool:
        pool.close()
        pool.join()
    log("done")


if __name__ == "__main__":
    main()
