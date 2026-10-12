#!/usr/bin/env python
"""P0.7 long slice, step 3 (NERSC CPU): setting-D scores S_f, S_r (score only) with the SAME kernel and driver
arithmetic as the benchmark matrices and the shipped panel scores (panel/score2/score_panel.py, evalue/score/score.py):

    sim = swfast.query_prep(q) @ decode(codes)^T (float32 GEMM, target chunks of <= 2^20 similarity floats)
    rowcol z (population std, eps 1e-8); SW local affine on (Z - 1.0), gap open 5, extend 0.5 = setting D
    S_f forward query, S_r = row-reversed copy of the forward block (z recomputed); kernel swmulti.c swz_ad_multi_f32,
    R = 2 (throughput/build swkern.c #included verbatim), built here with gcc -O3 -march=znver3 -ffp-contract=off.

WORK ITEMS (one query row x one target set; parts/<task>/q<row>.npy float32 [2, n_cols], atomic, resumable)
  lq_pan   long_q (3,000: 300-3,000 aa) x the 1,000 SHIPPED panel members (panel/fit2/results/shipped_panel_codes.npz)
  lq_ben   long_q x the A5 benchmark targets: all 25,000 annotated + REV_SUB (5,000) reversed decoys drawn with
           default_rng(20260927) from the 25,000 (targets.L2m_B8_s0.npz order; column list in out/index.npz)
  nq_long  the 1,465 A5 benchmark queries x (long_t 2,000 annotated + long_r 2,000 reversed decoys, 1,000-3,000 aa)
Multi-node: items are LPT-assigned to SLURM_NNODES nodes (or SPLIT_N independent 1-node jobs, SPLIT_K = 0..SPLIT_N-1) by
cells (query length x target residues); on every node a fork Pool of NWORK processes runs its items longest first.

STEPS (argv[1:]; default "selftest score"; after all nodes finished: "merge bench check" on one node)
  selftest  multi kernel (R = 2) == 2 x swfast.swz_ad_batch bit for bit (3 long queries x 300 benchmark targets)
  score     the work items of this node
  merge     -> out/lq_pan_{fwd,rev}.npy [3000, 1000], out/lq_ben_{fwd,rev}.npy [3000, 30000],
               out/nq_long_{fwd,rev}.npy [1465, 4000], out/index.npz, out/manifest.json (md5s)
  bench     this driver on 20 benchmark queries x the first 500 benchmark targets (benchmark column order) vs
            evalue/score/out/S_D_{fwd,rev}.npy -> results/bench.<tag>.json
  check     numpy harness float64 reference (harness.normalize_scores rowcol + harness.smith_waterman) on 36 pairs
            (long queries of every bin x benchmark targets; benchmark queries x long targets / decoys; length extremes),
            fwd and rev -> results/check.<tag>.json
"""
from __future__ import annotations

import ctypes as _C
import hashlib
import json
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import swfast as F  # noqa: E402

_M = _C.CDLL(os.path.join(HERE, "libswmulti.so"))
_M.swz_ad_multi_f32.restype = None
_M.swz_ad_multi_f32.argtypes = [_C.c_void_p, _C.c_void_p, _C.c_int, _C.c_int, _C.c_void_p, _C.c_int, _C.c_int,
                                _C.c_void_p, _C.c_void_p, _C.c_void_p, _C.c_void_p, _C.c_void_p, _C.c_void_p]
_M.sw_ad_ws_floats.restype = _C.c_size_t
_M.sw_ad_ws_floats.argtypes = [_C.c_int, _C.c_int]

EMB = "/pscratch/sd/j/jsegura/embcomp"
RUN = "L2m_B8_s0"
ENC_NQ = f"{EMB}/evalue/encode/out/queries.{RUN}.npz"
ENC_BT = f"{EMB}/evalue/encode/out/targets.{RUN}.npz"
PANEL = f"{EMB}/panel/fit2/results/shipped_panel_codes.npz"
PANEL_CODES_MD5 = "885dfc88590e0b270addcd4b61c3a7a7"
BENCH = f"{EMB}/evalue/score/out"
ENC_L = {k: os.path.join(HERE, "enc", "out", f"{k}.{RUN}.npz") for k in ("long_q", "long_t", "long_r")}
SETTING = ("D", 1.0, 5.0, 0.5)
NRUN = 2
R_OFF = np.array([SETTING[1]] * 2, np.float32)
R_GO = np.array([SETTING[2]] * 2, np.float32)
R_GE = np.array([SETTING[3]] * 2, np.float32)
R_REV = np.array([0, 1], np.int32)
REV_SUB = int(os.environ.get("REV_SUB", "5000"))
SEED = 20260927
PARTS, OUT, RES = (os.path.join(HERE, x) for x in ("parts", "out", "results"))
CELLS = int(os.environ.get("CELLS_PER_CHUNK", str(1 << 20)))
NWORK = int(os.environ.get("NWORK", "128"))
NNODES = int(os.environ.get("SPLIT_N", os.environ.get("SLURM_NNODES", "1")))       # SPLIT_N / SPLIT_K: the same LPT split
NODE = int(os.environ.get("SPLIT_K", os.environ.get("SLURM_PROCID", os.environ.get("SLURM_NODEID", "0"))))  # as 1-node jobs
TAG = os.environ.get("TAG", os.environ.get("SLURM_JOB_ID", "local")) + f"_n{NODE}"
TASKS = ("lq_pan", "lq_ben", "nq_long")
T0 = time.time()
G: dict = {}


def log(*a):
    print(f"[{time.time() - T0:8.1f}s n{NODE}] " + " ".join(str(x) for x in a), flush=True)


def md5(path, bs=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def cat_codes(parts):
    """[(codes, offsets, rows)] -> concatenated codes, offsets for the selected rows (in the given order)."""
    cs, lens = [], []
    for codes, off, rows in parts:
        for r in rows:
            cs.append(codes[off[r]:off[r + 1]])
            lens.append(int(off[r + 1] - off[r]))
    return np.ascontiguousarray(np.concatenate(cs)), np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)


def load():
    t = time.time()
    zl = {k: np.load(v) for k, v in ENC_L.items()}
    for k, z in zl.items():
        meta = json.loads(str(z["meta_json"]))
        G[f"{k}_meta"] = meta
        G[f"{k}_ids"] = z["ids"].astype(str)
        G[f"{k}_len"] = z["lengths"]
    znq = np.load(ENC_NQ)
    G["Q"] = {"lq": (F.query_prep(zl["long_q"]["q"]), zl["long_q"]["offsets"], zl["long_q"]["lengths"]),
              "nq": (F.query_prep(znq["q"]), znq["offsets"], znq["lengths"])}
    G["q16"] = {"lq": zl["long_q"]["q"], "nq": znq["q"]}
    G["nq_ids"] = znq["ids"].astype(str)
    bix = np.load(f"{BENCH}/index.npz")
    assert np.array_equal(bix["query_ids"].astype(str), G["nq_ids"]), "benchmark query order"
    assert json.loads(str(znq["meta_json"]))["model_md5"] == G["long_q_meta"]["model_md5"]
    zp = np.load(PANEL)
    assert hashlib.md5(np.ascontiguousarray(zp["codes"]).tobytes()).hexdigest() == PANEL_CODES_MD5
    zt = np.load(ENC_BT)
    tids = zt["ids"].astype(str)
    assert np.array_equal(bix["target_ids"].astype(str), tids), "benchmark target order"
    rev = np.array([t.startswith("rev_") for t in tids])
    ann_rows = np.flatnonzero(~rev)
    rev_rows = np.sort(np.random.default_rng(SEED).choice(np.flatnonzero(rev), REV_SUB, replace=False))
    G["ben_cols"] = np.concatenate([ann_rows, rev_rows])           # indices into the benchmark target order
    G["bt_codes"], G["bt_off"] = np.ascontiguousarray(zt["codes"]), zt["offsets"]
    lt, lr = zl["long_t"], zl["long_r"]
    G["T"] = {
        "pan": (np.ascontiguousarray(zp["codes"]), zp["offsets"].astype(np.int64)),
        "ben": cat_codes([(zt["codes"], zt["offsets"], G["ben_cols"])]),
        "long": cat_codes([(lt["codes"], lt["offsets"], range(len(lt["lengths"]))),
                           (lr["codes"], lr["offsets"], range(len(lr["lengths"])))]),
    }
    G["pan_ids"] = zp["ids"].astype(str)
    G["long_cols"] = np.concatenate([G["long_t_ids"], G["long_r_ids"]])
    for k, (c, o) in G["T"].items():
        assert c.shape == (o[-1], 8), k
    log(f"loaded in {time.time() - t:.1f}s: lq {len(G['Q']['lq'][2])} q ({G['Q']['lq'][1][-1]:,} res), nq "
        f"{len(G['Q']['nq'][2])}; targets pan {len(G['T']['pan'][1]) - 1} ({G['T']['pan'][1][-1]:,}), ben "
        f"{len(G['T']['ben'][1]) - 1} ({G['T']['ben'][1][-1]:,}), long {len(G['T']['long'][1]) - 1} "
        f"({G['T']['long'][1][-1]:,}); kernel lanes {F.VECTOR_LANES}")


TASKDEF = {"lq_pan": ("lq", "pan"), "lq_ben": ("lq", "ben"), "nq_long": ("nq", "long")}


def chunks(n, toff, NT):
    budget = max(CELLS // n, 1)
    j = 0
    while j < NT:
        k = int(np.searchsorted(toff, toff[j] + budget, side="right")) - 1
        k = min(max(k, j + 1), NT)
        yield np.arange(j, k), slice(toff[j], toff[k]), toff[j:k + 1] - toff[j]
        j = k


_WS: dict = {}


def score_block(sim, simr, col, out, impl="multi"):
    n, ld = sim.shape
    K = len(col) - 1
    if impl == "batch":
        for r in range(NRUN):
            out[r] = F.swz_ad_batch(simr if R_REV[r] else sim, col, SETTING[1], SETTING[2], SETTING[3], tb=False)[0]
        return
    col = np.ascontiguousarray(col, np.int64)
    mmax = int(np.max(np.diff(col)))
    need = 2 * (n + mmax) + int(_M.sw_ad_ws_floats(n, mmax))
    ws = _WS.get("ws")
    if ws is None or len(ws) < need:
        ws = _WS["ws"] = np.empty(max(need, 2 * (n + 2048) + int(_M.sw_ad_ws_floats(n, 2048))), np.float32)
    sc = np.empty((NRUN, K), np.float32)
    _M.swz_ad_multi_f32(sim.ctypes.data, simr.ctypes.data, n, ld, col.ctypes.data, K, NRUN, R_OFF.ctypes.data,
                        R_GO.ctypes.data, R_GE.ctypes.data, R_REV.ctypes.data, sc.ctypes.data, ws.ctypes.data)
    out[:] = sc


def query_scores(qp, codes, toff, NT, impl="multi"):
    t0 = time.perf_counter()
    n = len(qp)
    out = np.empty((NRUN, NT), np.float32)
    tdec = tgemm = tsw = 0.0
    for tt, rs, col in chunks(n, toff, NT):
        a = time.perf_counter()
        T = F.decode_unpackbits(codes[rs])
        b = time.perf_counter()
        sim = qp @ T.T
        simr = np.ascontiguousarray(sim[::-1])
        c = time.perf_counter()
        score_block(sim, simr, col, out[:, tt[0]:tt[-1] + 1], impl)
        d = time.perf_counter()
        tdec += b - a; tgemm += c - b; tsw += d - c
    return out, dict(n=n, wall=time.perf_counter() - t0, dec=tdec, gemm=tgemm, sw=tsw)


def qrows(qs, i):
    QP, off, _ = G["Q"][qs]
    return QP[off[i]:off[i + 1]]


def items():
    it = []
    for task in TASKS:
        qs, ts = TASKDEF[task]
        L = G["Q"][qs][2]
        tres = float(G["T"][ts][1][-1])
        it += [(task, int(i), float(L[i]) * tres) for i in range(len(L))]
    it.sort(key=lambda x: -x[2])
    load_ = np.zeros(NNODES)
    mine = []
    for task, i, c in it:
        k = int(np.argmin(load_))
        load_[k] += c
        if k == NODE:
            mine.append((task, i, c))
    return it, mine, load_


def part_path(task, i):
    return os.path.join(PARTS, task, f"q{i:04d}.npy")


def run_item(x):
    task, i, _ = x
    p = part_path(task, i)
    if os.path.exists(p):
        return x, None
    qs, ts = TASKDEF[task]
    codes, toff = G["T"][ts]
    out, st = query_scores(qrows(qs, i), codes, toff, len(toff) - 1)
    assert np.isfinite(out).all()
    tmp = p[:-4] + f".tmp{os.getpid()}.npy"
    np.save(tmp, out)
    os.replace(tmp, p)
    return x, st


def step_selftest():
    L = G["Q"]["lq"][2]
    res = {}
    codes, toff = G["T"]["ben"]
    for i in (int(np.argmax(L)), int(np.argsort(L)[len(L) // 2]), int(np.argmin(L))):
        a, sa = query_scores(qrows("lq", i), codes, toff, 300, "multi")
        b, sb = query_scores(qrows("lq", i), codes, toff, 300, "batch")
        eq = bool(np.array_equal(a, b))
        res[str(i)] = dict(Lq=int(L[i]), identical=eq, max_abs=float(np.abs(a - b).max()), multi_s=sa["wall"],
                           batch_s=sb["wall"])
        log("selftest", res[str(i)])
        assert eq
    json.dump(res, open(os.path.join(RES, f"selftest.{TAG}.json"), "w"), indent=1)


def step_score():
    from multiprocessing import get_context
    for t in TASKS:
        os.makedirs(os.path.join(PARTS, t), exist_ok=True)
    it, mine, load_ = items()
    todo = [x for x in mine if not os.path.exists(part_path(x[0], x[1]))]
    log(f"score: {len(it)} items over {NNODES} nodes (cells per node {[f'{v:.3g}' for v in load_]}); this node "
        f"{len(mine)} items, {len(todo)} to do, {NWORK} workers")
    tot_cells = sum(x[2] for x in todo)
    done = 0.0
    stats = []
    t = time.time()
    with get_context("fork").Pool(NWORK) as pool:
        for k, (x, st) in enumerate(pool.imap_unordered(run_item, todo, chunksize=1)):
            if st is not None:
                stats.append(dict(task=x[0], row=x[1], **st))
                done += x[2]
            if (k + 1) % 200 == 0 or k + 1 == len(todo):
                el = time.time() - t
                log(f"  {k + 1}/{len(todo)} items, {done / max(tot_cells, 1):.3f} of cells, {el:.0f}s, "
                    f"{done * NRUN / max(el, 1e-9) / 1e9:.2f} Gcell-runs/s")
    wall = time.time() - t
    ws = {k: float(sum(s[k] for s in stats)) for k in ("wall", "dec", "gemm", "sw")}
    rec = dict(tag=TAG, host=os.uname().nodename, nwork=NWORK, n_items=len(stats), wall_s=wall, worker_sum_s=ws,
               cells=done, ns_per_cell_run_worker=ws["wall"] / max(done * NRUN, 1) * 1e9,
               longest_item_s=max((s["wall"] for s in stats), default=0.0))
    json.dump(rec, open(os.path.join(RES, f"score_timing.{TAG}.json"), "w"), indent=1)
    log("score done", json.dumps(rec))


def step_merge():
    files = {}
    for task in TASKS:
        qs, ts = TASKDEF[task]
        nq, nt = len(G["Q"][qs][2]), len(G["T"][ts][1]) - 1
        F_ = np.empty((nq, nt), np.float32)
        R_ = np.empty((nq, nt), np.float32)
        for i in range(nq):
            a = np.load(part_path(task, i))
            assert a.shape == (2, nt)
            F_[i], R_[i] = a[0], a[1]
        for d, M in (("fwd", F_), ("rev", R_)):
            fn = os.path.join(OUT, f"{task}_{d}.npy")
            np.save(fn, M)
            files[os.path.basename(fn)] = md5(fn)
    np.savez(os.path.join(OUT, "index.npz"), lq_ids=G["long_q_ids"], lq_len=G["long_q_len"], nq_ids=G["nq_ids"],
             nq_len=G["Q"]["nq"][2], pan_ids=G["pan_ids"], pan_len=np.diff(G["T"]["pan"][1]),
             ben_cols=G["ben_cols"], ben_len=np.diff(G["T"]["ben"][1]), long_ids=G["long_cols"],
             long_len=np.diff(G["T"]["long"][1]), long_is_rev=np.array([x.startswith("lrev_") for x in G["long_cols"]]))
    files["index.npz"] = md5(os.path.join(OUT, "index.npz"))
    man = dict(setting=SETTING, runs=["D_fwd", "D_rev"], rev_sub=REV_SUB, seed=SEED, cells_per_chunk=CELLS,
               kernel_lanes=F.VECTOR_LANES, inputs={k: [v, md5(v)] for k, v in ENC_L.items()} |
               {"nq": [ENC_NQ, md5(ENC_NQ)], "bench_targets": [ENC_BT, md5(ENC_BT)], "panel": [PANEL, md5(PANEL)]},
               sources={f: md5(os.path.join(HERE, f)) for f in ("score_long.py", "swfast.py", "swkern.c", "swmulti.c")},
               files=files)
    json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    log("merge", json.dumps(files))


def _bench_query(i):
    qp = qrows("nq", i)
    out, _ = query_scores(qp, G["bt_codes"], G["bt_off"], 500)
    return i, out


def step_bench():
    from multiprocessing import get_context
    rows = np.sort(np.random.default_rng(3).choice(len(G["nq_ids"]), 20, replace=False))
    with get_context("fork").Pool(min(NWORK, 20)) as pool:
        got = dict(pool.map(_bench_query, rows.tolist()))
    Sf = np.load(f"{BENCH}/S_D_fwd.npy", mmap_mode="r")
    Sr = np.load(f"{BENCH}/S_D_rev.npy", mmap_mode="r")
    eqf = eqr = n = 0
    mrf = mrr = 0.0
    for i in rows:
        rf, rr = np.asarray(Sf[i, :500]), np.asarray(Sr[i, :500])
        eqf += int((got[i][0] == rf).sum()); eqr += int((got[i][1] == rr).sum()); n += 500
        big = np.abs(rf) > 1
        mrf = max(mrf, float((np.abs(got[i][0] - rf)[big] / np.abs(rf[big])).max()) if big.any() else 0.0)
        big = np.abs(rr) > 1
        mrr = max(mrr, float((np.abs(got[i][1] - rr)[big] / np.abs(rr[big])).max()) if big.any() else 0.0)
    rec = dict(rows=rows.tolist(), pairs=n, fwd_bit_equal=eqf, rev_bit_equal=eqr, fwd_max_rel=mrf, rev_max_rel=mrr)
    json.dump(rec, open(os.path.join(RES, f"bench.{TAG}.json"), "w"), indent=1)
    log("bench", json.dumps(rec))


def _ref_pair(args):
    qs, i, ts, j, d = args
    sys.path.insert(0, os.path.join(EMB, "harness"))
    import harness as H
    QP, off, _ = G["Q"][qs]
    q = G["q16"][qs][off[i]:off[i + 1]].astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    if d:
        q = q[::-1]
    codes, toff = G["T"][ts]
    T = np.unpackbits(codes[toff[j]:toff[j + 1]], axis=1)[:, :64].astype(np.float64) * 2 - 1
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    Z = H.normalize_scores(q @ T.T, "rowcol")
    a = H.smith_waterman(Z - SETTING[1], SETTING[2], SETTING[3])
    return float(a.score)


def step_check():
    from multiprocessing import get_context
    os.environ.setdefault("EMBCOMP_DATA", EMB)
    rng = np.random.default_rng(20260930)
    Lq = G["Q"]["lq"][2]
    pairs = []
    for lo, hi in ((300, 600), (600, 1000), (1000, 1500), (1500, 2000), (2000, 2500), (2500, 3001)):
        idx = np.flatnonzero((Lq >= lo) & (Lq < hi))
        for i in rng.choice(idx, 3, replace=False):
            pairs.append(("lq_ben", int(i), int(rng.integers(0, len(G["T"]["ben"][1]) - 1))))
    pairs.append(("lq_ben", int(np.argmax(Lq)), int(np.argmax(np.diff(G["T"]["ben"][1])))))
    pairs.append(("lq_pan", int(np.argmax(Lq)), int(np.argmax(np.diff(G["T"]["pan"][1])))))
    nlong = len(G["T"]["long"][1]) - 1
    for i, j in zip(rng.integers(0, len(G["Q"]["nq"][2]), 14), rng.integers(0, nlong, 14)):
        pairs.append(("nq_long", int(i), int(j)))
    pairs.append(("nq_long", int(np.argmin(G["Q"]["nq"][2])), int(np.argmax(np.diff(G["T"]["long"][1])))))
    pairs.append(("nq_long", int(np.argmax(G["Q"]["nq"][2])), int(np.argmax(np.diff(G["T"]["long"][1])))))
    jobs = [(TASKDEF[t][0], i, TASKDEF[t][1], j, d) for t, i, j in pairs for d in (0, 1)]
    t0 = time.time()
    with get_context("fork").Pool(min(NWORK, len(jobs))) as pool:
        refs = pool.map(_ref_pair, jobs, chunksize=1)
    rows = []
    for k, (t, i, j) in enumerate(pairs):
        f = np.load(os.path.join(OUT, f"{t}_fwd.npy"), mmap_mode="r")[i, j]
        r = np.load(os.path.join(OUT, f"{t}_rev.npy"), mmap_mode="r")[i, j]
        rows.append(dict(task=t, q=i, t=j, Lq=int(G["Q"][TASKDEF[t][0]][2][i]),
                         Lt=int(np.diff(G["T"][TASKDEF[t][1]][1])[j]), got_f=float(f), ref_f=refs[2 * k],
                         got_r=float(r), ref_r=refs[2 * k + 1]))
    df = np.array([abs(x["got_f"] - x["ref_f"]) for x in rows])
    dr = np.array([abs(x["got_r"] - x["ref_r"]) for x in rows])
    rf = np.array([abs(x["ref_f"]) for x in rows])
    rr = np.array([abs(x["ref_r"]) for x in rows])
    rec = dict(n_pairs=len(rows), ref_wall_s=round(time.time() - t0, 1), fwd_max_abs=float(df.max()),
               rev_max_abs=float(dr.max()), fwd_max_rel=float((df / np.maximum(rf, 1)).max()),
               rev_max_rel=float((dr / np.maximum(rr, 1)).max()), pairs=rows)
    json.dump(rec, open(os.path.join(RES, f"check.{TAG}.json"), "w"), indent=1)
    log("check", json.dumps({k: v for k, v in rec.items() if k != "pairs"}))


if __name__ == "__main__":
    steps = sys.argv[1:] or ["selftest", "score"]
    for d in (PARTS, OUT, RES):
        os.makedirs(d, exist_ok=True)
    load()
    for s in steps:
        t = time.time()
        {"selftest": step_selftest, "score": step_score, "merge": step_merge, "bench": step_bench,
         "check": step_check}[s]()
        log(f"step {s} {time.time() - t:.1f}s")
