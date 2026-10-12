#!/usr/bin/env python
"""P0.4 spike throughput: fixed-order C similarity + reversed-view fwd/rev scoring vs the research path.

Workload (bench_data.npz, from the research throughput workload): real query encodings (q16) at Lq 100 / 300 / 1000
against the first K = 1,000 real chain-set candidates (229,955 residues, mean Lt 230). Setting D (1.0; 5 / 0.5).
Every timing is one process, BLAS / OpenMP pinned to one thread (pin the core externally with taskset on Linux);
per variant: one warm-up round, then ROUNDS rounds over all K candidates; the MEDIAN round is reported.

modes
  st     (1) similarity alone: new sim_codes vs research decode (np.unpackbits) + SGEMM (qp @ T.T), and each part;
         query_prep C vs numpy; (2) full fwd+rev pair (query_prep + similarity + S_f + S_r):
           new.py     Python loop over blocks, one ctypes call per block (sim_codes + score_fwd_rev on the view)
           new.c      whole candidate list in one C call (fmsw_list_fwd_rev)
           research   numpy query_prep; per block decode + SGEMM + reversed copy + swz_ad_multi_f32 (R = 2)
                      = evalue/score/score.py's production path
           research.multi_reuse  the same work with every buffer reused (LUT decode into a buffer, SGEMM out=,
                      copyto for the reversed copy): the research path without per-block allocation / page faults
           research.batch2   the same with two swz_ad_batch_f32 calls (sim, simr) instead of swz_ad_multi_f32
           research.fwd      forward only (decode + SGEMM + swz_ad_batch_f32 tb = 0): the Appendix D 'score only' shape
           new.dp / research.dp   the fwd+rev DP alone on precomputed similarity blocks
         at the default blocks (new 2^17 floats, research 2^20 floats = score.py) and at equal blocks
  sweep  block_floats 2^15 .. 2^21 for new.py and research, single process
  pool   the same sweep with N concurrent processes (each runs the full workload; aggregate pairs/s over the common
         wall time; Linux: worker w pinned to core CORES[w])
  tiles  similarity alone for several builds (--libs), to choose the similarity tile
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import spike_lib as SL  # noqa: E402

NEW_BF, RES_BF = 1 << 17, 1 << 20


def host_info():
    info = dict(host=socket.gethostname(), platform=platform.platform(), machine=platform.machine(),
                python=platform.python_version(), numpy=np.__version__, cpus=os.cpu_count())
    try:
        if sys.platform == "darwin":
            import subprocess
            info["cpu"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        else:
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    info["cpu"] = line.split(":", 1)[1].strip()
                    break
            info["affinity"] = sorted(os.sched_getaffinity(0))[:8]
    except Exception:
        pass
    try:
        import threadpoolctl
        info["blas"] = [{k: d.get(k) for k in ("internal_api", "version", "architecture", "num_threads")}
                        for d in threadpoolctl.threadpool_info()]
    except Exception:
        pass
    return info


class Work:
    def __init__(self, inputs, K=1000):
        b = np.load(os.path.join(inputs, "bench_data.npz"))
        self.q16 = {Lq: b[f"q16_{Lq}"] for Lq in (100, 300, 600, 1000)}
        self.codes = np.ascontiguousarray(b[f"codes_K{K}"])
        self.off = np.ascontiguousarray(b[f"off_K{K}"])
        self.K = len(self.off) - 1
        self.lens = np.diff(self.off)
        self.res = int(self.off[-1])


def median_rounds(f, rounds):
    f()
    ts = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)), [round(x, 6) for x in ts]


# ---------------------------------------------------------------------------------------------- variants
def v_new_py(F, W, Lq, bf):
    q16 = W.q16[Lq]
    blk = SL.blocks(W.lens, Lq, bf)
    sf, sr = np.empty(W.K, np.float32), np.empty(W.K, np.float32)

    def f():
        qp = F.query_prep(q16)
        for k0, k1 in blk:
            F.block_fwd_rev(qp, W.codes, W.off[k0:k1 + 1], sf[k0:k1], sr[k0:k1])
    return f


def v_new_c(F, W, Lq, bf):
    q16 = W.q16[Lq]
    sf, sr = np.empty(W.K, np.float32), np.empty(W.K, np.float32)

    def f():
        qp = F.query_prep(q16)
        F.list_fwd_rev(qp, W.codes, W.off, bf, sf, sr)
    return f


def v_research(R, W, Lq, bf, impl="multi"):
    q16 = W.q16[Lq]
    blk = SL.blocks(W.lens, Lq, bf)
    out = np.empty((2, W.K), np.float32)

    bufs = {}
    o2s = {(k0, k1): np.empty((2, k1 - k0), np.float32) for k0, k1 in blk}

    def f():
        qp = R.query_prep(q16.astype(np.float32))
        for k0, k1 in blk:
            co = W.off[k0:k1 + 1]
            if impl == "multi_reuse":
                o2 = o2s[(k0, k1)]
                R.block_fwd_rev_reuse(qp, W.codes, co, o2, bufs)
                out[:, k0:k1] = o2
            elif impl == "multi":
                o2 = np.empty((2, k1 - k0), np.float32)
                R.block_fwd_rev(qp, W.codes, co, o2)
                out[:, k0:k1] = o2
            else:
                j0, j1 = int(co[0]), int(co[-1])
                sim = qp @ R.decode(W.codes[j0:j1]).T
                col = co - j0
                if impl == "batch2":
                    out[0, k0:k1] = R.batch(sim, col, False)[0]
                    out[1, k0:k1] = R.batch(np.ascontiguousarray(sim[::-1]), col, False)[0]
                else:                                   # fwd only
                    out[0, k0:k1] = R.batch(sim, col, False)[0]
    return f


def st_mode(a, F, R, W):
    out = {}
    for Lq in a.lq:
        cells = float(Lq * W.res)
        r = {}

        def rec(name, f, rounds=a.rounds):
            m, ts = median_rounds(f, rounds)
            r[name] = dict(s_med=m, s_rounds=ts, pairs_per_s=W.K / m, ns_per_cell=1e9 * m / cells)
            print(f"Lq={Lq:5d} {name:34s} {W.K / m:10.1f} pairs/s {1e9 * m / cells:7.3f} ns/cell", flush=True)

        q16 = W.q16[Lq]
        qp = F.query_prep(q16)
        rqp = R.query_prep(q16.astype(np.float32))
        rec("query_prep.c", lambda: F.query_prep(q16), 20)
        rec("query_prep.numpy", lambda: R.query_prep(q16.astype(np.float32)), 20)
        for bf in sorted({NEW_BF, RES_BF}):
            blk = SL.blocks(W.lens, Lq, bf)
            buf = np.empty(max(bf, Lq * int(W.lens.max())), np.float32)
            decoded = [R.decode(W.codes[W.off[k0]:W.off[k1]]) for k0, k1 in blk]

            def sim_new():
                for k0, k1 in blk:
                    w = int(W.off[k1] - W.off[k0])
                    F.sim_codes(qp, W.codes, W.off[k0:k1 + 1], out=buf[:Lq * w].reshape(Lq, w))

            def sim_res():
                for k0, k1 in blk:
                    _ = rqp @ R.decode(W.codes[W.off[k0]:W.off[k1]]).T

            def gemm_only():
                for T in decoded:
                    _ = rqp @ T.T

            def decode_only():
                for k0, k1 in blk:
                    _ = R.decode(W.codes[W.off[k0]:W.off[k1]])

            rec(f"sim.new.bf{bf}", sim_new)
            rec(f"sim.research_decode_gemm.bf{bf}", sim_res)
            rec(f"sim.research_gemm_only.bf{bf}", gemm_only)
            rec(f"sim.research_decode_only.bf{bf}", decode_only)
            sims = [np.ascontiguousarray(rqp @ T.T) for T in decoded]
            simrs = [np.ascontiguousarray(s[::-1]) for s in sims]
            cols = [W.off[k0:k1 + 1] - W.off[k0] for k0, k1 in blk]

            def dp_new():
                for s, c in zip(sims, cols):
                    F.score_fwd_rev(s, c)

            def dp_res():
                for s, sr_, c in zip(sims, simrs, cols):
                    R.multi_fwd_rev(s, sr_, c, np.empty((2, len(c) - 1), np.float32))

            rec(f"dp_fwd_rev.new_view.bf{bf}", dp_new)
            rec(f"dp_fwd_rev.research_multi_copy.bf{bf}", dp_res)
            del sims, simrs, decoded
            rec(f"pair.new.py.bf{bf}", v_new_py(F, W, Lq, bf))
            rec(f"pair.new.c.bf{bf}", v_new_c(F, W, Lq, bf))
            rec(f"pair.research.multi.bf{bf}", v_research(R, W, Lq, bf, "multi"))
            rec(f"pair.research.multi_reuse.bf{bf}", v_research(R, W, Lq, bf, "multi_reuse"))
            rec(f"pair.research.batch2.bf{bf}", v_research(R, W, Lq, bf, "batch2"))
            rec(f"pair.research.fwd_only.bf{bf}", v_research(R, W, Lq, bf, "fwd"))
        r["ratio.sim_new_over_research.bf131072_vs_bf1048576"] = (
            r[f"sim.research_decode_gemm.bf{RES_BF}"]["s_med"] / r[f"sim.new.bf{NEW_BF}"]["s_med"])
        r["ratio.pair_new_py_bf131072_over_research_multi_bf1048576"] = (
            r[f"pair.new.py.bf{NEW_BF}"]["pairs_per_s"] / r[f"pair.research.multi.bf{RES_BF}"]["pairs_per_s"])
        best_new = max(r[f"pair.new.py.bf{bf}"]["pairs_per_s"] for bf in (NEW_BF, RES_BF))
        best_res = max(r[f"pair.research.{v}.bf{bf}"]["pairs_per_s"] for bf in (NEW_BF, RES_BF)
                       for v in ("multi", "multi_reuse", "batch2"))
        r["ratio.pair_best_new_py_over_best_research"] = best_new / best_res
        print(f"Lq={Lq} ratios", {k: round(v, 3) for k, v in r.items() if k.startswith("ratio")}, flush=True)
        out[str(Lq)] = r
    return out


def sweep_mode(a, F, R, W):
    out = {}
    for Lq in a.lq:
        cells = float(Lq * W.res)
        r = {}
        for e in a.blocks:
            bf = 1 << e
            for name, f in (("new.py", v_new_py(F, W, Lq, bf)), ("new.c", v_new_c(F, W, Lq, bf)),
                            ("research.multi", v_research(R, W, Lq, bf, "multi")),
                            ("research.multi_reuse", v_research(R, W, Lq, bf, "multi_reuse"))):
                if name == "new.c" and not a.with_c:
                    continue
                m, ts = median_rounds(f, a.rounds)
                r[f"{name}.2^{e}"] = dict(s_med=m, s_rounds=ts, pairs_per_s=W.K / m, ns_per_cell=1e9 * m / cells)
                print(f"Lq={Lq:5d} bf=2^{e:2d} {name:16s} {W.K / m:10.1f} pairs/s", flush=True)
        out[str(Lq)] = r
    return out


# ---------------------------------------------------------------------------------------------- pool
def _worker(wid, core, lib, research, inputs, conn, barrier):
    if core is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {core})
    F, R, W = SL.Fmsw(lib), SL.Research(research), Work(inputs)
    while True:
        cfg = conn.recv()
        if cfg is None:
            break
        path, Lq, bf, rounds = cfg
        if path == "new.py":
            f = v_new_py(F, W, Lq, bf)
        elif path == "new.c":
            f = v_new_c(F, W, Lq, bf)
        else:                                           # research | research.multi_reuse
            f = v_research(R, W, Lq, bf, "multi_reuse" if path.endswith("reuse") else "multi")
        f()                                             # warm-up (not timed)
        barrier.wait()
        t0 = time.perf_counter()
        ts = []
        for _ in range(rounds):
            t1 = time.perf_counter()
            f()
            ts.append(time.perf_counter() - t1)
        t2 = time.perf_counter()
        conn.send((t0, t2, ts))


def pool_mode(a, lib, research, W):
    import multiprocessing as mp
    ctx = mp.get_context("fork" if sys.platform.startswith("linux") else "spawn")
    N = a.workers
    cores = None
    if sys.platform.startswith("linux") and a.cores:
        if "," in a.cores:
            cores = [int(c) for c in a.cores.split(",")][:N]
        else:
            st, step = (int(x) for x in a.cores.split(":"))
            cores = [st + step * w for w in range(N)]
    barrier = ctx.Barrier(N)
    procs, conns = [], []
    for w in range(N):
        p_c, c_c = ctx.Pipe()
        p = ctx.Process(target=_worker, args=(w, cores[w] if cores else None, lib, research, a.inputs, c_c, barrier),
                        daemon=True)
        p.start()
        procs.append(p)
        conns.append(p_c)
    out = dict(workers=N, cores=cores, results={})
    try:
        for Lq in a.lq:
            for e in a.blocks:
                bf = 1 << e
                for path in a.paths:
                    for c in conns:
                        c.send((path, Lq, bf, a.rounds))
                    rs = [c.recv() for c in conns]
                    t0 = min(x[0] for x in rs)
                    t1 = max(x[1] for x in rs)
                    total_pairs = N * a.rounds * W.K
                    per_w_med = [float(np.median(x[2])) for x in rs]
                    agg = total_pairs / (t1 - t0)
                    key = f"Lq{Lq}.{path}.2^{e}"
                    out["results"][key] = dict(wall_s=t1 - t0, pairs_per_s_aggregate=agg,
                                               pairs_per_s_sum_worker_medians=float(sum(W.K / m for m in per_w_med)),
                                               per_worker_median_s=[round(x, 5) for x in per_w_med],
                                               cells_per_s=agg * Lq * W.res / W.K)
                    print(f"P={N} Lq={Lq:5d} bf=2^{e:2d} {path:15s} {agg:11.1f} pairs/s aggregate "
                          f"(worker median {np.median(per_w_med):.4f} s, max {max(per_w_med):.4f} s)", flush=True)
    finally:
        for c in conns:
            try:
                c.send(None)
            except Exception:
                pass
        for p in procs:
            p.join(timeout=10)
    return out


def tiles_mode(a, W):
    out = {}
    for lib in a.libs:
        F = SL.Fmsw(lib)
        r = {}
        for Lq in a.lq:
            qp = F.query_prep(W.q16[Lq])
            blk = SL.blocks(W.lens, Lq, NEW_BF)
            buf = np.empty(max(NEW_BF, Lq * int(W.lens.max())), np.float32)

            def sim_new():
                for k0, k1 in blk:
                    w = int(W.off[k1] - W.off[k0])
                    F.sim_codes(qp, W.codes, W.off[k0:k1 + 1], out=buf[:Lq * w].reshape(Lq, w))
            m, ts = median_rounds(sim_new, a.rounds)
            r[str(Lq)] = dict(s_med=m, ns_per_cell=1e9 * m / (Lq * W.res), s_rounds=ts)
            print(f"{os.path.basename(lib):32s} Lq={Lq:5d} sim {1e9 * m / (Lq * W.res):7.3f} ns/cell", flush=True)
        out[os.path.basename(lib)] = dict(info=F.info(), sim=r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["st", "sweep", "pool", "tiles"])
    ap.add_argument("--lib")
    ap.add_argument("--libs", nargs="*")
    ap.add_argument("--research")
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lq", nargs="+", type=int, default=[100, 300, 1000])
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--blocks", nargs="+", type=int, default=list(range(15, 22)))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--cores", default=None, help="linux pinning: a,b,c or start:step")
    ap.add_argument("--paths", nargs="+", default=["new.py", "research"])
    ap.add_argument("--with-c", action="store_true")
    a = ap.parse_args()
    W = Work(a.inputs)
    res = dict(mode=a.mode, host=host_info(), args=vars(a), K=W.K, residues=W.res, setting="D (1.0; 5/0.5)")
    t0 = time.time()
    if a.mode == "tiles":
        res["tiles"] = tiles_mode(a, W)
    elif a.mode == "pool":
        res["build"] = SL.Fmsw(a.lib).info()
        res["pool"] = pool_mode(a, a.lib, a.research, W)
    else:
        F, R = SL.Fmsw(a.lib), SL.Research(a.research)
        res["build"] = F.info()
        res["research"] = R.info()
        res[a.mode] = st_mode(a, F, R, W) if a.mode == "st" else sweep_mode(a, F, R, W)
    res["seconds"] = round(time.time() - t0, 1)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
