#!/usr/bin/env python
"""P0.4 spike correctness: one fmsw_spike build vs the research path on F1, reversed-view identities, invariances and
the cross-build golden hash (computed from q fp16).

  verify_spike.py --lib BUILD/libfmsw_<cc>_<isa>.<ext> --research BUILD/research --inputs INPUTS_DIR --out OUT.json

INPUTS_DIR: fixture_candidate.npz (F1 candidate, md5 d724b769...), sims.bin (the float32 similarity matrices the
research probe fed to its kernel on the Mac: numpy fp32 query_prep + Accelerate SGEMM), golden_inputs.npz, bench_data.npz.
Checks
  F1.same_sim   the research similarity (sims.bin) through the new score_fwd_rev / align: S_f, S_r, paths vs the stored
                research values (bit-identical expected: the new rowcol order is the research 4-lane order); also vs this
                host's research library (swz_ad_batch_f32 on sim and on the reversed COPY)
  F1.full_path  q fp16 -> C query_prep -> C sim -> S_f, S_r, path vs the research values (tolerance + path identity),
                plus similarity / qp bit-agreement with the research numpy+BLAS values
  view          S_r(view) == score on the materialised reversed copy == S_f(qp[::-1]) == align(rev=1) score, bit for bit
  golden        sha256 over (n, m, S_f, S_r, ends_f, ends_r, forward path) for the 50 F1 pairs + 1,000 golden pairs;
                component hashes: qp, sim; score-only == traceback score; no NaN
  invariance    bench Lq=300 query x 1,000 real candidates: list_fwd_rev with block_floats 2^12 / 2^17 / 2^20 / per-pair
                blocks and the Python block driver give identical S_f / S_r
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import struct
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spike_lib as SL  # noqa: E402


def read_sims(path):
    b = open(path, "rb").read()
    npairs = struct.unpack("i", b[:4])[0]
    off, out = 4, []
    for _ in range(npairs):
        n, m = struct.unpack("ii", b[off:off + 8])
        off += 8
        out.append(np.frombuffer(b, np.float32, n * m, off).reshape(n, m).copy())
        off += 4 * n * m
    return out


def f1_items(z):
    qi = {x: k for k, x in enumerate(z["q_ids"].astype(str))}
    ti = {x: k for k, x in enumerate(z["t_ids"].astype(str))}
    qo, to = z["q_offsets"], z["t_offsets"]
    for p, (a, b) in enumerate(z["pairs"].astype(str)):
        ka, kb = qi[a], ti[b]
        yield p, z["q"][qo[ka]:qo[ka + 1]], z["codes"][to[kb]:to[kb + 1]]


def ulp_diff(a, b):
    ia = np.asarray(a, np.float32).view(np.int32).astype(np.int64)
    ib = np.asarray(b, np.float32).view(np.int32).astype(np.int64)
    ia = np.where(ia < 0, np.int64(-2**31) - ia, ia)
    ib = np.where(ib < 0, np.int64(-2**31) - ib, ib)
    return np.abs(ia - ib)


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1.0))) if len(a) else 0.0


def pair_run(F, qp, codes):
    """New path for one pair from qp: (sf, sr, ends_f, ends_r, sf_tb, path)."""
    sim = F.sim_codes(qp, codes)
    col = np.array([0, len(codes)], np.int64)
    sf, sr, ef, er = F.score_fwd_rev(sim, col, ends=True)
    stb, etb, pflat, poff = F.align(sim, col)
    return sim, float(sf[0]), float(sr[0]), ef[0], er[0], float(stb[0]), etb[0], pflat[poff[0]:poff[1]].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", required=True)
    ap.add_argument("--research", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()
    F = SL.Fmsw(a.lib)
    R = SL.Research(a.research)
    res = dict(build=F.info(), research=R.info(), host=socket.gethostname(), platform=platform.platform(),
               machine=platform.machine(), numpy=np.__version__)
    try:
        import threadpoolctl
        res["blas"] = [{k: d.get(k) for k in ("internal_api", "version", "architecture", "num_threads")}
                       for d in threadpoolctl.threadpool_info()]
    except Exception:
        pass
    z = np.load(os.path.join(a.inputs, "fixture_candidate.npz"))
    sims = read_sims(os.path.join(a.inputs, "sims.bin"))
    fx_sf, fx_sr = z["S_f"].astype(np.float32), z["S_r"].astype(np.float32)
    fx_paths = [z["path_flat"][z["path_off"][p]:z["path_off"][p + 1]] for p in range(len(z["pairs"]))]

    # ------------------------------------------------------------------ F1 with the research similarity
    same = dict(sf_bit_eq=0, sr_bit_eq=0, sf_tb_bit_eq=0, path_eq=0, rhost_sf_bit_eq=0, rhost_sr_bit_eq=0,
                rhost_path_eq=0, view_eq_copy=0, n=len(sims))
    for p, sim in enumerate(sims):
        n, m = sim.shape
        col = np.array([0, m], np.int64)
        sf, sr = F.score_fwd_rev(sim, col)
        stb, _e, pf, po = F.align(sim, col)
        path = pf[po[0]:po[1]]
        same["sf_bit_eq"] += int(sf[0] == fx_sf[p])
        same["sr_bit_eq"] += int(sr[0] == fx_sr[p])
        same["sf_tb_bit_eq"] += int(stb[0] == fx_sf[p])
        same["path_eq"] += int(np.array_equal(path, fx_paths[p]))
        rs, _re, rpf, rpo = R.batch(np.ascontiguousarray(sim), col, True)
        simr = np.ascontiguousarray(sim[::-1])
        rr, _ = R.batch(simr, col, False)
        same["rhost_sf_bit_eq"] += int(rs[0] == sf[0])
        same["rhost_sr_bit_eq"] += int(rr[0] == sr[0])
        same["rhost_path_eq"] += int(np.array_equal(rpf[rpo[0]:rpo[1]], path))
        sfc, _src = F.score_fwd_rev(simr, col)
        same["view_eq_copy"] += int(sfc[0] == sr[0])
    res["F1_same_sim"] = same

    # ------------------------------------------------------------------ F1 full new path from q fp16
    full = dict(n=0, path_eq=0, sf_bit_eq=0, sr_bit_eq=0, max_rel_sf=0.0, max_rel_sr=0.0, sim_cells=0,
                sim_bit_eq_cells=0, sim_max_abs=0.0, sim_max_ulp=0, qp_rows=0, qp_max_ulp=0, qp_bit_eq=0.0,
                score_eq_tb=0, rev_view_eq_qrev=0, rev_view_eq_alignrev=0)
    f1_out = []
    sfs, srs = [], []
    for p, q16, codes in f1_items(z):
        qp = F.query_prep(q16)
        rqp = R.query_prep(q16.astype(np.float32))
        u = ulp_diff(qp, rqp)
        full["qp_rows"] += len(qp)
        full["qp_max_ulp"] = max(full["qp_max_ulp"], int(u.max()))
        full["qp_bit_eq"] += float((u == 0).sum())
        sim, sf, sr, ef, er, stb, etb, path = pair_run(F, qp, codes)
        d = np.abs(sim.astype(np.float64) - sims[p].astype(np.float64))
        full["sim_cells"] += int(sim.size)
        full["sim_bit_eq_cells"] += int((sim.view(np.int32) == sims[p].view(np.int32)).sum())
        full["sim_max_abs"] = max(full["sim_max_abs"], float(d.max()))
        full["sim_max_ulp"] = max(full["sim_max_ulp"], int(ulp_diff(sim, sims[p]).max()))
        full["n"] += 1
        full["path_eq"] += int(np.array_equal(path, fx_paths[p]))
        full["sf_bit_eq"] += int(np.float32(sf) == fx_sf[p])
        full["sr_bit_eq"] += int(np.float32(sr) == fx_sr[p])
        full["score_eq_tb"] += int(np.float32(sf) == np.float32(stb))
        sfs.append(sf)
        srs.append(sr)
        # reversed view == forward score of the reversed query, and == align on the reversed view
        simq = F.sim_codes(np.ascontiguousarray(qp[::-1]), codes)
        sfq, _ = F.score_fwd_rev(simq, np.array([0, len(codes)], np.int64))
        full["rev_view_eq_qrev"] += int(sfq[0] == np.float32(sr))
        sra, _, _, _ = F.align(sim, np.array([0, len(codes)], np.int64), rev=True)
        full["rev_view_eq_alignrev"] += int(sra[0] == np.float32(sr))
        f1_out.append((len(q16), len(codes), sf, sr, ef, er, path))
    full["qp_bit_eq"] = full["qp_bit_eq"] / (full["qp_rows"] * 64)
    full["max_rel_sf"] = rel(sfs, z["S_f"])
    full["max_rel_sr"] = rel(srs, z["S_r"])
    full["max_abs_sf"] = float(np.max(np.abs(np.array(sfs) - z["S_f"])))
    full["max_abs_sr"] = float(np.max(np.abs(np.array(srs) - z["S_r"])))
    res["F1_full_path"] = full

    # ------------------------------------------------------------------ research path on this host (non-identity)
    rsf = []
    rpeq = 0
    for p, q16, codes in f1_items(z):
        rqp = R.query_prep(q16.astype(np.float32))
        sim = np.ascontiguousarray(rqp @ R.decode(codes).T)
        col = np.array([0, len(codes)], np.int64)
        s, _e, pf, po = R.batch(sim, col, True)
        rsf.append(float(s[0]))
        rpeq += int(np.array_equal(pf[po[0]:po[1]], fx_paths[p]))
    res["F1_research_this_host"] = dict(sf_bit_eq_vs_fixture=int(np.sum(np.float32(rsf) == fx_sf)),
                                        max_rel_sf_vs_fixture=rel(rsf, z["S_f"]), path_eq=rpeq,
                                        vector_lanes=R.info()["vector_lanes"])

    # ------------------------------------------------------------------ golden hash
    g = np.load(os.path.join(a.inputs, "golden_inputs.npz"))
    H, Hq, Hs = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    gold = dict(pairs=0, nan=0, score_eq_tb=0, empty=0, kinds={})
    gsf, gsr, glen = [], [], []

    def feed(n, m, sf, sr, ef, er, path):
        H.update(struct.pack("<ii", n, m))
        H.update(np.float32(sf).tobytes() + np.float32(sr).tobytes())
        H.update(np.asarray(ef, "<i4").tobytes() + np.asarray(er, "<i4").tobytes())
        H.update(struct.pack("<i", len(path)))
        H.update(np.asarray(path, "<i4").tobytes())

    for rec in f1_out:
        feed(*rec)
    for p in range(len(g["q_off"]) - 1):
        q16 = g["q"][g["q_off"][p]:g["q_off"][p + 1]]
        codes = g["codes"][g["c_off"][p]:g["c_off"][p + 1]]
        qp = F.query_prep(q16)
        Hq.update(qp.tobytes())
        sim, sf, sr, ef, er, stb, etb, path = pair_run(F, qp, codes)
        Hs.update(sim.tobytes())
        feed(len(q16), len(codes), sf, sr, ef, er, path)
        gold["pairs"] += 1
        gold["nan"] += int(not (np.isfinite(sf) and np.isfinite(sr)))
        gold["score_eq_tb"] += int(np.float32(sf) == np.float32(stb) and np.array_equal(ef, etb))
        gold["empty"] += int(len(path) == 0)
        k = str(g["kind"][p])
        gold["kinds"][k] = gold["kinds"].get(k, 0) + 1
        gsf.append(sf)
        gsr.append(sr)
        glen.append(len(path))
    gold["sha256"] = H.hexdigest()
    gold["sha256_qp"] = Hq.hexdigest()
    gold["sha256_sim"] = Hs.hexdigest()
    res["golden"] = gold

    # ------------------------------------------------------------------ block-size / driver invariance
    b = np.load(os.path.join(a.inputs, "bench_data.npz"))
    qp = F.query_prep(b["q16_300"])
    codes, off = b["codes_K1000"], b["off_K1000"]
    K = len(off) - 1
    ref = None
    inv = {}
    for bf in (1 << 12, 1 << 17, 1 << 20):
        sf, sr = np.empty(K, np.float32), np.empty(K, np.float32)
        F.list_fwd_rev(qp, codes, off, bf, sf, sr)
        cur = np.concatenate([sf, sr])
        if ref is None:
            ref = cur
        inv[f"list_bf{bf}"] = bool(np.array_equal(cur.view(np.int32), ref.view(np.int32)))
    sf, sr = np.empty(K, np.float32), np.empty(K, np.float32)
    for k0, k1 in SL.blocks(np.diff(off), len(qp), 1 << 17):
        F.block_fwd_rev(qp, codes, off[k0:k1 + 1], sf[k0:k1], sr[k0:k1])
    inv["py_blocks_bf131072"] = bool(np.array_equal(np.concatenate([sf, sr]).view(np.int32), ref.view(np.int32)))
    Hb = hashlib.sha256(ref.tobytes()).hexdigest()
    inv["sha256_bench_lq300_k1000"] = Hb
    res["invariance"] = inv
    res["seconds"] = round(time.time() - t0, 2)
    np.savez(a.out.replace(".json", ".npz"), f1_sf=np.float32([r[2] for r in f1_out]),
             f1_sr=np.float32([r[3] for r in f1_out]), g_sf=np.float32(gsf), g_sr=np.float32(gsr),
             g_pathlen=np.int32(glen), bench_sf_sr=ref)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("build", "F1_same_sim", "F1_full_path", "F1_research_this_host")}, indent=1))
    print("golden", gold["sha256"], "qp", gold["sha256_qp"][:16], "sim", gold["sha256_sim"][:16], "inv", inv)


if __name__ == "__main__":
    main()
