#!/usr/bin/env python
"""P0.6 census, step 3: code-level census and the FP-check scores (research kernel, setting D, S = S_f - S_r,
Q90 over the SHIPPED panel codes, shipped M3t999 'all' table).

Kernel: census_swk.c (swkern.c #included verbatim, md5 438d5a98...) -> libcensus.{so,dylib}; per pair S_f, S_r
(bit-identical calls to the evalue/score driver), the degenerate row / column counts (std < 1e-6) of the forward block,
and the S7-variant scores (degenerate terms contribute exactly 0).  Similarity = query_prep(q) @ decode(codes)^T in
float32 GEMM blocks of <= CELLS floats (as evalue/score/score.py), reversed query = row-reversed copy of the block.

STEPS
  check    (a) 40 benchmark queries x 300 benchmark targets vs evalue/score/out/S_D_{fwd,rev}.npy (rel diff);
           (b) 40 benchmark queries x shipped panel -> Q90 vs panel/fit2/results/bench_q90_shipped.npy;
           (c) synthetic degenerate pairs vs a float64 numpy reference of rowcol z (+ S7 rule) and harness SW.
  codes    per item: distinct target codes, identical q rows, zero-norm q rows, min pairwise q-row distance,
           covariance certificate for degenerate columns (tau_min >= 8 sqrt(lambda_min(cov qp))) -> out/codes.npz
  targets  1,465 benchmark queries x all items as TARGETS -> out/targets.npz (dense S_f, S_r float32 [1465, n_items];
           sparse S7 scores / degenerate counts where any row or column is degenerate)
  queries  items as QUERIES x shipped panel (Q90, Q90_S7) and x the 50,000 benchmark targets -> out/queries.npz
           (per item: Q90s, aggregates; sparse pairs with ln E(N_min) <= ln 100 under research or S7 scores)
Env: NWORK (default 128), CELLS (default 2^20), ITEMS (items npz), OUT (dir).
"""
from __future__ import annotations

import ctypes as C
import hashlib
import json
import math
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EMB = os.environ.get("EMBCOMP_ROOT", "/pscratch/sd/j/jsegura/embcomp")
ENC = os.environ.get("ENC_DIR", f"{EMB}/evalue/encode/out")
SCORE_OUT = f"{EMB}/evalue/score/out"
FIT2 = f"{EMB}/panel/fit2/results"
ITEMS = os.environ.get("ITEMS", f"{HERE}/enc/items_enc.npz")
OUT = os.environ.get("OUT", f"{HERE}/out")
NWORK = int(os.environ.get("NWORK", "128"))
CELLS = int(os.environ.get("CELLS", str(1 << 20)))
OFFSET, GO, GE = 1.0, 5.0, 0.5           # setting D
TKEY = "D_fmr/pnl_uniref50/M3t999"
KEEP_LNE = math.log(100.0)
T0 = time.time()
G: dict = {}

_LIB = C.CDLL(os.path.join(HERE, "libcensus.dylib" if sys.platform == "darwin" else "libcensus.so"))
_LIB.census_multi_f32.restype = None
_LIB.census_multi_f32.argtypes = [C.c_void_p, C.c_void_p, C.c_int, C.c_int, C.c_void_p, C.c_int, C.c_float, C.c_float,
                                  C.c_float, C.c_void_p, C.c_void_p, C.c_void_p]
_LIB.sw_ad_ws_floats.restype = C.c_size_t
_LIB.sw_ad_ws_floats.argtypes = [C.c_int, C.c_int]


def log(*a):
    print(f"[{time.time() - T0:8.1f}s]", *a, flush=True)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def query_prep(q):
    """swfast.query_prep (float32 rows / (8 |q|)); a zero-norm row becomes 0 (S5) and is counted by the caller."""
    q = np.asarray(q, dtype=np.float32)
    nrm = np.linalg.norm(q, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = q / (8.0 * nrm)
    out[(nrm[:, 0] == 0)] = 0.0
    return out


def decode(codes):
    """swfast.decode_unpackbits: uint8 [L, 8] -> float32 +-1 [L, 64]."""
    return np.unpackbits(codes, axis=1)[:, :64].astype(np.float32) * 2 - 1


def score_set(qp, codes, toff, tsel=None, dec=None):
    """one query (qp [n, 64]) x targets (codes / offsets; tsel = target indices or None = all) ->
    out [K, 4] float32 (S_f, S_r, S_f S7, S_r S7), deg [K, 2] int32 (degenerate rows, columns).
    dec = decode(codes) computed once (optional; same values and block layout -> identical results)."""
    n = len(qp)
    tidx = np.arange(len(toff) - 1) if tsel is None else np.asarray(tsel)
    L = (toff[tidx + 1] - toff[tidx]).astype(np.int64)
    cs = np.concatenate([[0], np.cumsum(L)])
    K = len(tidx)
    out = np.empty((K, 4), np.float32)
    deg = np.empty((K, 2), np.int32)
    budget = max(CELLS // max(n, 1), 1)
    contiguous = tsel is None or bool(np.all(np.diff(tidx) == 1))
    j = 0
    while j < K:
        k = int(np.searchsorted(cs, cs[j] + budget, side="right")) - 1
        k = min(max(k, j + 1), K)
        if contiguous and dec is not None:
            T = dec[toff[tidx[j]]:toff[tidx[k - 1] + 1]]
        else:
            if contiguous:
                rows = codes[toff[tidx[j]]:toff[tidx[k - 1] + 1]]
            else:
                rows = np.concatenate([codes[toff[t]:toff[t + 1]] for t in tidx[j:k]])
            T = decode(rows)
        sim = np.ascontiguousarray(qp @ T.T)
        simr = np.ascontiguousarray(sim[::-1])
        col = np.ascontiguousarray(cs[j:k + 1] - cs[j], np.int64)
        mmax = int(L[j:k].max())
        ws = np.empty(2 * (n + mmax) + int(_LIB.sw_ad_ws_floats(n, mmax)), np.float32)
        o = np.empty((k - j, 4), np.float32)
        d = np.empty((k - j, 2), np.int32)
        _LIB.census_multi_f32(sim.ctypes.data, simr.ctypes.data, n, sim.shape[1], col.ctypes.data, k - j,
                              OFFSET, GO, GE, o.ctypes.data, d.ctypes.data, ws.ctypes.data)
        out[j:k] = o
        deg[j:k] = d
        j = k
    return out, deg


def table():
    t = json.load(open(f"{FIT2}/ship_final.json"))["tables"][TKEY]["all"]
    return np.log(np.asarray(t["center_len"], np.float64)), np.asarray(t["lam"], np.float64), \
        np.asarray(t["lnK"], np.float64)


def lam_lnk(m):
    lnc, lam, lnk = G["table"]
    lm = np.log(np.asarray(m, np.float64))
    return np.interp(lm, lnc, lam), np.interp(lm, lnc, lnk)


def load_npz(path, keys):
    z = np.load(path)
    return {k: z[k] for k in keys if k in z.files}


def load_common():
    G["table"] = table()
    zp = load_npz(f"{FIT2}/shipped_panel_codes.npz", ["codes", "offsets", "lengths", "ids"])
    G["pc"], G["poff"] = np.ascontiguousarray(zp["codes"]), zp["offsets"].astype(np.int64)
    assert G["pc"].shape == (G["poff"][-1], 8) == (355626, 8)


# ------------------------------------------------------------------------------------------------ check
def ref_pair(qp, codes, s7):
    """float64 numpy reference: cosine (qp @ d^T in float64), rowcol z (+ S7 rule), SW (harness) on z - offset."""
    sys.path.insert(0, f"{EMB}/harness")
    import harness as H
    d = decode(codes).astype(np.float64)
    sim = qp.astype(np.float64) @ d.T
    out = []
    for s in (sim, sim[::-1]):
        mr, sr = s.mean(1, keepdims=True), s.std(1, keepdims=True)
        mc, sc = s.mean(0, keepdims=True), s.std(0, keepdims=True)
        zr = (s - mr) / (sr + 1e-8)
        zc = (s - mc) / (sc + 1e-8)
        if s7:
            zr = np.where(sr < 1e-6, 0.0, zr)
            zc = np.where(sc < 1e-6, 0.0, zc)
        z = 0.5 * (zr + zc)
        out.append(H.smith_waterman(z - OFFSET, GO, GE).score)
    return out


def step_check():
    res = {}
    zq = load_npz(f"{ENC}/queries.L2m_B8_s0.npz", ["q", "offsets", "ids"])
    qoff = zq["offsets"]
    rng = np.random.default_rng(7)
    qsel = np.sort(rng.choice(len(qoff) - 1, 40, replace=False))
    # (a) vs the stored evalue/score matrices (NERSC only: the targets npz is not mirrored locally)
    if os.path.exists(f"{ENC}/targets.L2m_B8_s0.npz"):
        zt = load_npz(f"{ENC}/targets.L2m_B8_s0.npz", ["codes", "offsets"])
        tsel = np.sort(rng.choice(len(zt["offsets"]) - 1, 300, replace=False))
        Rf = np.load(f"{SCORE_OUT}/S_D_fwd.npy", mmap_mode="r")
        Rr = np.load(f"{SCORE_OUT}/S_D_rev.npy", mmap_mode="r")
        rel_f, rel_r, eq = [], [], 0
        for i in qsel:
            qp = query_prep(zq["q"][qoff[i]:qoff[i + 1]])
            o, _ = score_set(qp, zt["codes"], zt["offsets"], tsel)
            rf, rr = np.asarray(Rf[i, tsel], np.float64), np.asarray(Rr[i, tsel], np.float64)
            rel_f.append(np.abs(o[:, 0] - rf) / np.maximum(np.abs(rf), 1))
            rel_r.append(np.abs(o[:, 1] - rr) / np.maximum(np.abs(rr), 1))
            eq += int((o[:, 0] == rf).sum() + (o[:, 1] == rr).sum())
        res["vs_evalue_score"] = {"pairs": int(len(qsel) * len(tsel)), "max_rel_fwd": float(np.concatenate(rel_f).max()),
                                  "max_rel_rev": float(np.concatenate(rel_r).max()),
                                  "bit_equal_fraction": eq / (2 * len(qsel) * len(tsel))}
        log("check a", json.dumps(res["vs_evalue_score"]))
    # (b) Q90 over the shipped panel vs the fit2 Q90
    q90ref = np.load(f"{FIT2}/bench_q90_shipped.npy")
    d = []
    for i in qsel:
        qp = query_prep(zq["q"][qoff[i]:qoff[i + 1]])
        o, dg = score_set(qp, G["pc"], G["poff"])
        q90 = float(np.quantile(o[:, 0].astype(np.float64) - o[:, 1].astype(np.float64), 0.9))
        d.append(abs(q90 - q90ref[i]))
        assert (dg == 0).all()
    res["q90_vs_fit2"] = {"queries": int(len(qsel)), "max_abs": float(max(d))}
    log("check b", json.dumps(res["q90_vs_fit2"]))
    # (c) synthetic degenerate pairs vs the float64 reference
    rng = np.random.default_rng(11)
    cases = []
    base_q = zq["q"][qoff[qsel[0]]:qoff[qsel[0] + 1]]
    pcodes = G["pc"][:400]
    one = np.repeat(pcodes[:1], 50, axis=0)
    cases.append(("target_1_code_m50", base_q[:60], one))
    cases.append(("target_m1", base_q[:60], pcodes[:1]))
    cases.append(("query_n1", base_q[:1], pcodes[:120]))
    cases.append(("query_identical_rows", np.repeat(base_q[:1], 40, axis=0), pcodes[:150]))
    cases.append(("target_2_codes", base_q[:80], pcodes[np.array([0, 1] * 30)]))
    cases.append(("short_both", base_q[:3], pcodes[:4]))
    cases.append(("normal", base_q[:70], pcodes[:90]))
    out = []
    for name, q, c in cases:
        qp = query_prep(q)
        o, dg = score_set(qp, c, np.array([0, len(c)], np.int64))
        r_res = ref_pair(qp, c, False)
        r_s7 = ref_pair(qp, c, True)
        out.append({"case": name, "n": int(len(q)), "m": int(len(c)), "deg_rows": int(dg[0, 0]),
                    "deg_cols": int(dg[0, 1]), "kernel": [float(x) for x in o[0]],
                    "ref_research": [float(x) for x in r_res], "ref_s7": [float(x) for x in r_s7],
                    "s7_abs_diff": float(max(abs(o[0, 2] - r_s7[0]), abs(o[0, 3] - r_s7[1])))})
        log("check c", json.dumps(out[-1]))
    res["synthetic"] = out
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(f"{OUT}/check.json", "w"), indent=1)


# ------------------------------------------------------------------------------------------------ items
def load_items():
    z = np.load(ITEMS)
    G["iq"], G["ic"] = z["q"], np.ascontiguousarray(z["codes"])
    G["ioff"], G["ilen"], G["iids"] = z["offsets"].astype(np.int64), z["lengths"], z["ids"].astype(str)
    G["n_items"] = len(G["iids"])
    log(f"items: {G['n_items']} ({G['ioff'][-1]:,} residues)")


def step_codes():
    load_items()
    n = G["n_items"]
    distinct = np.zeros(n, np.int64)
    ident_q = np.zeros(n, bool)
    zero_rows = np.zeros(n, np.int64)
    distinct_q = np.zeros(n, np.int64)
    lam_min = np.zeros(n, np.float64)
    tau_lb = np.zeros(n, np.float64)
    for k in range(n):
        a, b = G["ioff"][k], G["ioff"][k + 1]
        c = G["ic"][a:b]
        distinct[k] = len(np.unique(c.view(np.uint64)))
        q = G["iq"][a:b]
        distinct_q[k] = len(np.unique(q.view(np.uint64).reshape(len(q), -1), axis=0))
        ident_q[k] = distinct_q[k] == 1
        zero_rows[k] = int((np.abs(q.astype(np.float32)).sum(1) == 0).sum())
        qp = query_prep(q).astype(np.float64)
        cov = np.cov(qp.T, bias=True) if len(qp) > 1 else np.zeros((64, 64))
        lam_min[k] = max(float(np.linalg.eigvalsh(cov)[0]), 0.0)
        tau_lb[k] = 8.0 * math.sqrt(lam_min[k])
    os.makedirs(OUT, exist_ok=True)
    np.savez(f"{OUT}/codes.npz", ids=G["iids"], lengths=G["ilen"], distinct_codes=distinct, distinct_q_rows=distinct_q,
             identical_q_rows=ident_q, zero_norm_q_rows=zero_rows, cov_lambda_min=lam_min, tau_lower_bound=tau_lb)
    log("codes: 1 distinct code", int((distinct == 1).sum()), "; identical q rows", int(ident_q.sum()),
        "; zero-norm rows", int(zero_rows.sum()), "; tau certificate >= 1e-6:", int((tau_lb >= 1e-6).sum()))


# ------------------------------------------------------------------------------------------------ targets
def _t_worker(i):
    qp = query_prep(G["bq"][G["bqoff"][i]:G["bqoff"][i + 1]])
    o, d = score_set(qp, G["ic"], G["ioff"])
    return i, o, d


def step_targets():
    from multiprocessing import get_context
    load_items()
    zq = load_npz(f"{ENC}/queries.L2m_B8_s0.npz", ["q", "offsets", "ids", "lengths"])
    G["bq"], G["bqoff"] = zq["q"], zq["offsets"]
    nq, nt = len(zq["ids"]), G["n_items"]
    Sf = np.empty((nq, nt), np.float32)
    Sr = np.empty((nq, nt), np.float32)
    t = time.time()
    order = np.argsort(-np.diff(G["bqoff"]))
    with get_context("fork").Pool(NWORK) as pool:
        for c, (i, o, d) in enumerate(pool.imap_unordered(_t_worker, order.tolist(), chunksize=1)):
            Sf[i], Sr[i] = o[:, 0], o[:, 1]
            w = np.flatnonzero(d.any(1))
            if len(w):
                G.setdefault("s7", []).append((i, w, o[w, 2].copy(), o[w, 3].copy(), d[w].copy()))
            if c % 200 == 0:
                log(f"targets: {c + 1}/{nq} queries, {time.time() - t:.0f}s")
    s7 = G.get("s7", [])
    pq = np.concatenate([np.full(len(w), i) for i, w, *_ in s7]) if s7 else np.zeros(0, np.int64)
    pt = np.concatenate([w for _, w, *_ in s7]) if s7 else np.zeros(0, np.int64)
    pf7 = np.concatenate([a for _, _, a, _, _ in s7]) if s7 else np.zeros(0, np.float32)
    pr7 = np.concatenate([b for _, _, _, b, _ in s7]) if s7 else np.zeros(0, np.float32)
    pdg = np.concatenate([dd for *_, dd in s7]) if s7 else np.zeros((0, 2), np.int32)
    os.makedirs(OUT, exist_ok=True)
    np.savez(f"{OUT}/targets.npz", Sf=Sf, Sr=Sr, query_ids=zq["ids"], query_len=zq["lengths"], item_ids=G["iids"],
             item_len=G["ilen"], deg_q=pq, deg_t=pt, deg_sf7=pf7, deg_sr7=pr7, deg_counts=pdg,
             meta_json=np.array(json.dumps({"items": ITEMS, "items_md5": md5(ITEMS), "setting": [OFFSET, GO, GE],
                                            "wall_s": round(time.time() - t, 1), "nwork": NWORK, "cells": CELLS})))
    log(f"targets done: {nq} x {nt} in {time.time() - t:.0f}s; degenerate pairs {len(pq):,}")


# ------------------------------------------------------------------------------------------------ queries
def _q_worker(k):
    a, b = G["ioff"][k], G["ioff"][k + 1]
    qp = query_prep(G["iq"][a:b])
    m = b - a
    op, dp = score_set(qp, G["pc"], G["poff"], dec=G.get("pdec"))
    Sp = op[:, 0].astype(np.float64) - op[:, 1]
    Sp7 = op[:, 2].astype(np.float64) - op[:, 3]
    q90, q907 = float(np.quantile(Sp, 0.9)), float(np.quantile(Sp7, 0.9))
    ot, dt = score_set(qp, G["tc"], G["toff"], dec=G.get("tdec"))
    S = ot[:, 0].astype(np.float64) - ot[:, 1]
    S7 = ot[:, 2].astype(np.float64) - ot[:, 3]
    lam, lnk = lam_lnk(m)
    base = lnk + math.log(m) + G["lnNmin"]
    le = base - lam * (S - q90)
    le7 = base - lam * (S7 - q907)
    keep = np.flatnonzero((le <= KEEP_LNE) | (le7 <= KEEP_LNE))
    ann = ~G["trev"]
    agg = dict(q90=q90, q90_s7=q907, panel_deg_pairs=int(dp.any(1).sum()), panel_deg_cols=int(dp[:, 1].sum()),
               panel_deg_rows=int(dp[:, 0].sum()), tgt_deg_pairs=int(dt.any(1).sum()),
               tgt_deg_cols=int(dt[:, 1].sum()), tgt_deg_rows=int(dt[:, 0].sum()),
               max_abs_fmr_panel=float(np.abs(Sp).max()), max_abs_fmr_tgt=float(np.abs(S).max()),
               max_S_ann=float(S[ann].max()), q50_S_ann=float(np.quantile(S[ann], 0.5)),
               q99_S_ann=float(np.quantile(S[ann], 0.99)), q999_S_ann=float(np.quantile(S[ann], 0.999)),
               max_S_rev=float(S[~ann].max()), max_S7_ann=float(S7[ann].max()),
               max_Sf_ann=float(ot[ann, 0].max()), med_Sf_ann=float(np.median(ot[ann, 0])),
               med_Sr_ann=float(np.median(ot[ann, 1])), panel_q90_fwd=float(np.quantile(op[:, 0], 0.9)))
    return k, agg, keep, ot[keep], dt[keep]


def step_queries():
    from multiprocessing import get_context
    load_items()
    zt = load_npz(f"{ENC}/targets.L2m_B8_s0.npz", ["codes", "offsets", "ids", "lengths"])
    G["tc"], G["toff"] = np.ascontiguousarray(zt["codes"]), zt["offsets"].astype(np.int64)
    tids = zt["ids"].astype(str)
    G["trev"] = np.array([t.startswith("rev_") for t in tids])
    tl = zt["lengths"].astype(np.float64)
    n_ann, n_rev = float(tl[~G["trev"]].sum()), float(tl[G["trev"]].sum())
    G["lnNmin"] = math.log(min(n_ann, n_rev))
    if os.environ.get("PREDECODE", "0") == "1":        # decode once in the parent; forked workers share it (COW)
        t = time.time()
        G["tdec"] = decode(G["tc"])
        G["pdec"] = decode(G["pc"])
        log(f"pre-decoded targets {G['tdec'].shape} + panel {G['pdec'].shape} in {time.time() - t:.1f}s")
    n = G["n_items"]
    aggs = [None] * n
    rows = []
    t = time.time()
    order = np.argsort(-G["ilen"], kind="stable").tolist()
    sh, nsh = (int(x) for x in os.environ.get("SHARD", "0/1").split("/"))
    order = order[sh::nsh]
    mine = np.zeros(n, bool)
    mine[order] = True
    with get_context("fork").Pool(NWORK) as pool:
        for c, (k, agg, keep, o, d) in enumerate(pool.imap_unordered(_q_worker, order, chunksize=1)):
            aggs[k] = agg
            if len(keep):
                rows.append((np.full(len(keep), k), keep, o, d))
            if c % 1000 == 0:
                log(f"queries: {c + 1}/{n} items, {time.time() - t:.0f}s")
    keys = list(aggs[order[0]].keys())
    A = {k: np.array([a[k] if a is not None else np.nan for a in aggs]) for k in keys}
    hq = np.concatenate([r[0] for r in rows]) if rows else np.zeros(0, np.int64)
    ht = np.concatenate([r[1] for r in rows]) if rows else np.zeros(0, np.int64)
    ho = np.concatenate([r[2] for r in rows]) if rows else np.zeros((0, 4), np.float32)
    hd = np.concatenate([r[3] for r in rows]) if rows else np.zeros((0, 2), np.int32)
    os.makedirs(OUT, exist_ok=True)
    np.savez(f"{OUT}/queries.shard{sh}of{nsh}.npz", item_ids=G["iids"], item_len=G["ilen"], shard_items=mine,
             hit_item=hq, hit_target=ht, hit_scores=ho,
             hit_deg=hd, target_ids=tids, target_len=zt["lengths"], target_rev=G["trev"],
             **{f"agg_{k}": v for k, v in A.items()},
             meta_json=np.array(json.dumps({"items": ITEMS, "items_md5": md5(ITEMS), "setting": [OFFSET, GO, GE],
                                            "N_ann": n_ann, "N_rev": n_rev, "keep_lnE_Nmin": KEEP_LNE,
                                            "wall_s": round(time.time() - t, 1), "nwork": NWORK, "cells": CELLS})))
    log(f"queries shard {sh}/{nsh} done: {len(order)} items in {time.time() - t:.0f}s; kept pairs {len(hq):,}")


def step_qmerge():
    import glob
    fs = sorted(glob.glob(f"{OUT}/queries.shard*of*.npz"))
    zs = [np.load(f) for f in fs]
    mine = np.stack([z["shard_items"] for z in zs])
    assert (mine.sum(0) == 1).all(), "the shard files must partition the items (every item in exactly one shard)"
    out = {k: zs[0][k] for k in ("item_ids", "item_len", "target_ids", "target_len", "target_rev")}
    for k in ("hit_item", "hit_target", "hit_scores", "hit_deg"):
        out[k] = np.concatenate([z[k] for z in zs])
    for k in [k for k in zs[0].files if k.startswith("agg_")]:
        v = np.array(zs[0][k], np.float64)
        for z, m in zip(zs[1:], mine[1:]):
            v[m] = z[k][m]
        out[k] = v
    metas = [json.loads(str(z["meta_json"])) for z in zs]
    meta = dict(metas[0])
    # reduce the kept pairs to ln E <= ln 10 with the pair's own N (annotated: N_ann, reversed: N_rev), either form
    n_before = len(out["hit_item"])
    hi, ht, ho = out["hit_item"], out["hit_target"], out["hit_scores"].astype(np.float64)
    m = out["item_len"][hi].astype(np.float64)
    lam, lnk = lam_lnk(m)
    lnN = np.where(out["target_rev"][ht], math.log(meta["N_rev"]), math.log(meta["N_ann"]))
    le = lnk + np.log(m) + lnN - lam * ((ho[:, 0] - ho[:, 1]) - out["agg_q90"][hi])
    le7 = lnk + np.log(m) + lnN - lam * ((ho[:, 2] - ho[:, 3]) - out["agg_q90_s7"][hi])
    keep = (le <= math.log(10.0)) | (le7 <= math.log(10.0))
    for k in ("hit_item", "hit_target", "hit_scores", "hit_deg"):
        out[k] = out[k][keep]
    meta["merge_keep"] = {"rule": "ln E (own N) <= ln 10, research or S7", "pairs_before": int(n_before),
                          "pairs_after": int(keep.sum())}
    meta["shards"] = [{"file": os.path.basename(f), "wall_s": m["wall_s"]} for f, m in zip(fs, metas)]
    out["meta_json"] = np.array(json.dumps(meta))
    np.savez(f"{OUT}/queries.npz", **out)
    log(f"merged {len(fs)} shards -> queries.npz; kept pairs {len(out['hit_item']):,}")


if __name__ == "__main__":
    load_common()
    for st in sys.argv[1:]:
        log("step", st)
        {"check": step_check, "codes": step_codes, "targets": step_targets, "queries": step_queries,
         "qmerge": step_qmerge}[st]()
