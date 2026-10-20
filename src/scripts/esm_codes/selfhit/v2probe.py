#!/usr/bin/env python
"""Self-hit study, step 6 (FUTURE WORK probe, not esm-codes-v1): would another cell score make self hits of
homogeneous-code proteins significant without breaking the false-positive control?

Cell scores (all then Smith-Waterman setting D: offset 1, gaps 5 / 0.5, S = S_f - S_r with the row-reversed query,
S' = S - Q90 over the shipped 1,000-member panel):
  V1    esm-codes-v1: per-pair rowcol z (S7), recomputed in float64 here (checked against FoldMatch's kernel)
  C1    background z: 0.5 ((s_ij - mu_i^P) / sigma_i^P + (s_ij - nu_j^B) / tau_j^B) - offset, the row term against
        the fixed panel code background (bgprobe.py), the column term against a fixed background of query
        encodings (the 800 random UniRef50 sequences): pair-independent, so a protein's own homogeneity is not
        subtracted
  C2k   V1 + k * BLOSUM62(a_i, b_j) (half-bits), k = 0.1 and 0.2: an amino-acid term next to the embedding term, as
        Foldseek's 3Di+AA

Every query is scored against itself, the panel (Q90) and 300 random 10k-database entries (the null; none of them
is a query). A per-candidate exponential tail is then fitted to the pooled null S' in the system's six length bins
(the mini-recalibration a v2 would need), and analyze_v2() reports, per candidate, the null false-positive ratio
by group (code homogeneity, population) and the self-hit E-values at several database sizes.

    PYTHONPATH=<wt-P6/src> HF_HUB_OFFLINE=1 HF_HUB_CACHE=<cache> python v2probe.py DATA OUT.npz [--workers 8]
    python v2probe.py --analyze OUT.npz DATA  (tables to stdout)

The kernel: rawsw.c, compiled once into DATA/probe/librawsw.dylib.
"""
from __future__ import annotations

import argparse
import ctypes as C
import json
import math
import os
import random
import sqlite3
import subprocess
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 20261019
CANDS = ("V1", "C1", "C2_0.1", "C2_0.2")
SUBSAMPLE = {"u50_db800": 400, "scop_q": 300, "pdb_r600": 300}
N_NULL = 300
BLOCK_COLS = 32768
_W = {}


# --------------------------------------------------------------------------------------------------- setup
def build_lib(data):
    out = os.path.join(data, "probe", "librawsw.dylib")
    src = os.path.join(HERE, "rawsw.c")
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(src):
        subprocess.check_call(["cc", "-O3", "-mcpu=native", "-shared", "-fPIC", "-o", out, src])
    return out


def blosum_idx():
    sys.path.insert(0, HERE)
    import seqfeat
    B = seqfeat._blosum().astype(np.float32)
    return B, seqfeat._IDX


def read_fasta(path):
    out, name, buf = {}, None, []
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith(">"):
            if name is not None:
                out[name] = "".join(buf)
            name, buf = line[1:].split()[0], []
        else:
            buf.append(line.strip())
    if name is not None:
        out[name] = "".join(buf)
    return out


def make_tasks(data):
    rng = random.Random(SEED)
    tasks = []
    for name, k in SUBSAMPLE.items():
        z = np.load(os.path.join(data, "enc", f"{name}.npz"), allow_pickle=False)
        off, ids = z["offsets"], [str(x) for x in z["ids"]]
        sc = z["seq_concat"].tobytes().decode("ascii")
        ok = [i for i in range(len(ids)) if 21 <= off[i + 1] - off[i] <= 966]
        pick = sorted(rng.sample(ok, min(k, len(ok))))
        for i in pick:
            tasks.append(dict(set=name, id=ids[i], q16=np.ascontiguousarray(z["q"][off[i]:off[i + 1]]),
                              codes=np.ascontiguousarray(z["codes"][off[i]:off[i + 1]]),
                              seq=sc[off[i]:off[i + 1]]))
    return tasks


def make_null(data, exclude):
    seqs = read_fasta(os.path.join(data, "u10k", "uniref50_10k.fasta"))
    con = sqlite3.connect(f"file:{os.path.join(data, 'db_u10k', 'P.rescodes.idx')}?mode=ro", uri=True)
    rows = con.execute("SELECT id, offset, length FROM entries ORDER BY id").fetchall()
    B = np.memmap(os.path.join(data, "db_u10k", "P.rescodes.bin"), dtype=np.uint8, mode="r").reshape(-1, 8)
    pool = [r for r in rows if r[0] not in exclude and r[2] <= 3000]
    pick = sorted(random.Random(SEED + 1).sample(range(len(pool)), N_NULL))
    ids = [pool[i][0] for i in pick]
    codes = np.concatenate([np.asarray(B[pool[i][1]:pool[i][1] + pool[i][2]]) for i in pick])
    lens = np.array([pool[i][2] for i in pick], np.int64)
    s = "".join(seqs[x] for x in ids)
    assert len(s) == len(codes)
    return ids, codes, lens, s


def setup(data):
    from foldmatch.search import scoring_system as ss
    from foldmatch.search import swkern
    system = ss.get("esm-codes-v1")
    spec = ss.prepare(system, local_files_only=True).for_worker()
    pc, po = spec.load_panel()
    pz = np.load("/Users/joan/devel/foldmatch-p2p3-sandbox/artefacts/embcomp/panel/fit2/results/"
                 "shipped_panel_codes.npz", allow_pickle=False)
    assert np.array_equal(pz["codes"], pc)
    psq = pz["sequences"]
    psq = "".join(str(x) for x in psq) if psq.ndim else str(psq)
    assert len(psq) == len(pc), (len(psq), len(pc))
    excl = set()
    for name in ("u50_db800", "u50_q200"):
        excl |= {str(x) for x in np.load(os.path.join(data, "enc", f"{name}.npz"))["ids"]}
    nid, nc, nl, ns = make_null(data, excl)
    DP = swkern.decode_codes(np.ascontiguousarray(pc)).astype(np.float64)
    bg = np.load(os.path.join(data, "enc", "u50_db800.npz"))
    QB = swkern.query_prep(np.ascontiguousarray(bg["q"])).astype(np.float64)
    shared = dict(panel_codes=np.ascontiguousarray(pc), panel_len=np.diff(po).astype(np.int64), panel_seq=psq,
                  null_codes=np.ascontiguousarray(nc), null_len=nl, null_seq=ns, null_ids=nid,
                  dbar_p=DP.mean(0), C_p=DP.T @ DP / len(DP), qbar_b=QB.mean(0), Cq_b=QB.T @ QB / len(QB),
                  offset=float(system.offset), go=float(system.gap_open), ge=float(system.gap_extend),
                  lib=build_lib(data))
    return shared


def _init(shared):
    from foldmatch.search import swkern
    _W.update(shared)
    lib = C.CDLL(shared["lib"])
    lib.sw_batch.restype = C.c_int
    lib.sw_batch.argtypes = [C.c_void_p, C.c_int, C.c_int64, C.c_void_p, C.c_void_p, C.c_int, C.c_float, C.c_float,
                             C.c_void_p]
    B, idx = blosum_idx()
    _W.update(swkern=swkern, clib=lib, B=B, IDX=idx)
    D = swkern.decode_codes(shared["panel_codes"])
    Dn = swkern.decode_codes(shared["null_codes"])
    # column background of every panel / null residue: nu_j^B = d_j . qbar_B, tau_j^B
    for key, DD in (("panel", D), ("null", Dn)):
        d64 = DD.astype(np.float64)
        nu = d64 @ shared["qbar_b"]
        tau = np.sqrt(np.maximum(np.einsum("jk,kl,jl->j", d64, shared["Cq_b"], d64) - nu ** 2, 0.0))
        _W[key + "_D"], _W[key + "_nu"], _W[key + "_tau"] = DD, nu.astype(np.float32), tau.astype(np.float32)
    _W["panel_aa"] = idx[np.frombuffer(shared["panel_seq"].encode("ascii"), np.uint8)]
    _W["null_aa"] = idx[np.frombuffer(shared["null_seq"].encode("ascii"), np.uint8)]


# --------------------------------------------------------------------------------------------------- scoring
def _sw(X, col0, m):
    X = np.ascontiguousarray(X, dtype=np.float32)
    out = np.zeros(len(m), np.float32)
    st = _W["clib"].sw_batch(X.ctypes.data, X.shape[0], X.shape[1], col0.ctypes.data, m.ctypes.data, len(m),
                             _W["go"], _W["ge"], out.ctypes.data)
    assert st == 0
    return out


def _blocks(lens):
    blocks, s, start = [], 0, 0
    for k, L in enumerate(lens):
        if s and s + L > BLOCK_COLS:
            blocks.append((start, k))
            start, s = k, 0
        s += L
    blocks.append((start, len(lens)))
    return blocks


def _group(qp, qa, muP, sdP, D, nu, tau, aa, lens):
    """S_f, S_r [K] for every candidate against K targets laid side by side in D."""
    off = _W["offset"]
    n = qp.shape[0]
    starts = np.zeros(len(lens) + 1, np.int64)
    np.cumsum(lens, out=starts[1:])
    res = {c: (np.zeros(len(lens), np.float32), np.zeros(len(lens), np.float32)) for c in CANDS}
    for k0, k1 in _blocks(lens.tolist()):
        c0, c1 = starts[k0], starts[k1]
        m = lens[k0:k1].astype(np.int32)
        col0 = (starts[k0:k1] - c0).astype(np.int64)
        sim = qp @ D[c0:c1].T                                          # float32 [n, M]
        seg = np.repeat(np.arange(k1 - k0), m)
        rs = np.add.reduceat(sim, col0, axis=1, dtype=np.float64)
        rq = np.add.reduceat(sim.astype(np.float64) ** 2, col0, axis=1)
        mu = rs / m
        sd = np.sqrt(np.maximum(rq / m - mu ** 2, 0.0))
        a = (sim - mu[:, seg].astype(np.float32)) / (sd[:, seg].astype(np.float32) + 1e-8)
        a[(sd < 1e-6)[:, seg]] = 0.0
        cm = sim.mean(0, dtype=np.float64)
        cs = sim.std(0, dtype=np.float64)
        b = (sim - cm.astype(np.float32)) / (cs.astype(np.float32) + 1e-8)
        b[:, cs < 1e-6] = 0.0
        X1 = 0.5 * (a + b) - off
        del a, b
        XC1 = 0.5 * ((sim - muP[:, None]) / sdP[:, None] + (sim - nu[None, c0:c1]) / tau[None, c0:c1]) - off
        AA = _W["B"][qa[:, None], aa[None, c0:c1]]
        for cand, X in (("V1", X1), ("C1", XC1), ("C2_0.1", X1 + 0.1 * AA), ("C2_0.2", X1 + 0.2 * AA)):
            res[cand][0][k0:k1] = _sw(X, col0, m)
            res[cand][1][k0:k1] = _sw(X[::-1], col0, m)
    return res


def _query(task):
    sk = _W["swkern"]
    qp = sk.query_prep(task["q16"])
    qp64 = qp.astype(np.float64)
    muP = (qp64 @ _W["dbar_p"]).astype(np.float32)
    sdP = np.sqrt(np.maximum(np.einsum("ik,kl,il->i", qp64, _W["C_p"], qp64) - (qp64 @ _W["dbar_p"]) ** 2,
                             0.0)).astype(np.float32)
    qa = _W["IDX"][np.frombuffer(task["seq"].encode("ascii"), np.uint8)]
    Ds = sk.decode_codes(task["codes"]).astype(np.float32)
    d64 = Ds.astype(np.float64)
    nus = (d64 @ _W["qbar_b"]).astype(np.float32)
    taus = np.sqrt(np.maximum(np.einsum("jk,kl,jl->j", d64, _W["Cq_b"], d64) - (d64 @ _W["qbar_b"]) ** 2,
                              0.0)).astype(np.float32)
    out = dict(set=task["set"], id=task["id"], L=len(qp))
    for g, args in (("self", (Ds, nus, taus, qa, np.array([len(qp)], np.int64))),
                    ("panel", (_W["panel_D"], _W["panel_nu"], _W["panel_tau"], _W["panel_aa"], _W["panel_len"])),
                    ("null", (_W["null_D"], _W["null_nu"], _W["null_tau"], _W["null_aa"], _W["null_len"]))):
        r = _group(qp, qa, muP, sdP, *args)
        for c in CANDS:
            out[f"{g}_f_{c}"], out[f"{g}_r_{c}"] = r[c]
    return out


def run(data, out, workers):
    t0 = time.time()
    shared = setup(data)
    tasks = make_tasks(data)
    print(f"{len(tasks)} queries, {N_NULL} null targets ({int(shared['null_len'].sum())} residues), setup "
          f"{time.time() - t0:.0f}s", flush=True)
    order = sorted(range(len(tasks)), key=lambda i: -len(tasks[i]["q16"]))
    res = [None] * len(tasks)
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(shared,)) as pool:
        for k, (i, r) in enumerate(pool.imap_unordered(_indexed, [(i, tasks[i]) for i in order], chunksize=1)):
            res[i] = r
            if (k + 1) % 100 == 0:
                print(f"{k + 1}/{len(tasks)} in {time.time() - t0:.0f}s", flush=True)
    d = {k: np.array([r[k] for r in res]) for k in ("set", "id", "L")}
    for g in ("self", "panel", "null"):
        for c in CANDS:
            for s in ("f", "r"):
                d[f"{g}_{s}_{c}"] = np.stack([r[f"{g}_{s}_{c}"] for r in res])
    d["null_len"] = shared["null_len"]
    d["null_ids"] = np.array(shared["null_ids"])
    d["meta_json"] = json.dumps(dict(cands=CANDS, subsample=SUBSAMPLE, n_null=N_NULL, seed=SEED,
                                     seconds=time.time() - t0, workers=workers))
    np.savez(out, **d)
    print(f"wrote {out} in {time.time() - t0:.0f}s", flush=True)


def _indexed(it):
    i, task = it
    return i, _query(task)


# --------------------------------------------------------------------------------------------------- analysis
def analyze(path, data, exclude=None):
    """Mini-recalibration and false-positive ratios. ``exclude``: npz with ``flagged`` [queries, nulls] (blosum_null.py);
    flagged null pairs are dropped and each query's null search space shrinks by their residues."""
    from foldmatch.search import code_statistics as cs
    from foldmatch.search import scoring_system as ss
    system = ss.get("esm-codes-v1")
    z = np.load(path, allow_pickle=False)
    sets, ids, L = z["set"], z["id"], z["L"].astype(np.float64)
    nl = z["null_len"].astype(np.float64)
    keep = np.ones((len(L), len(nl)), bool)
    if exclude:
        keep = ~np.load(exclude)["flagged"]
        print(f"excluding {int((~keep).sum())} BLOSUM62-detectable null pairs ({int((~keep).any(1).sum())} queries)")
    nres_q = keep.astype(np.float64) @ nl                     # each query's null search space
    n_null_res = float(nl.sum())
    sys.path.insert(0, HERE)
    import seqfeat
    seq_of = {}
    for s_ in np.unique(sets):
        e = np.load(os.path.join(data, "enc", f"{s_}.npz"), allow_pickle=False)
        sc_, off_ = e["seq_concat"].tobytes().decode("ascii"), e["offsets"]
        for k_, x_ in enumerate(e["ids"]):
            seq_of[(str(s_), str(x_))] = sc_[off_[k_]:off_[k_ + 1]]
    lc = np.array([seqfeat.lc_frac(seq_of[(str(s_), str(i_))]) for s_, i_ in zip(sets, ids)])
    # homogeneity h of each query (bgprobe.py)
    bg = np.load(os.path.join(data, "probe", "bg.npz"), allow_pickle=False)
    h = np.array([dict(zip([str(x) for x in bg[f"{s}__ids"]], bg[f"{s}__h"]))[str(i)] for s, i in zip(sets, ids)])
    # V1 check against FoldMatch's kernel (score/<set>.npz)
    chk_f, chk_q = [], []
    for s in np.unique(sets):
        sc = np.load(os.path.join(data, "score", f"{s}.npz"), allow_pickle=False)
        pos = {str(x): k for k, x in enumerate(sc["ids"])}
        for k in np.flatnonzero(sets == s):
            j = pos[str(ids[k])]
            chk_f.append(abs(z["self_f_V1"][k, 0] - sc["S_f"][j]) / max(sc["S_f"][j], 1))
            pS = z["panel_f_V1"][k].astype(np.float64) - z["panel_r_V1"][k]
            chk_q.append(abs(np.quantile(pS, 0.9) - sc["q90"][j]))
    print(f"V1 vs FoldMatch kernel: self S_f rel diff median {np.median(chk_f):.2e} max {np.max(chk_f):.2e}; "
          f"Q90 abs diff median {np.median(chk_q):.2e} max {np.max(chk_q):.2e}")
    edges = np.sqrt(np.array(system.bin_centres[:-1]) * np.array(system.bin_centres[1:]))
    binof = np.searchsorted(edges, L)
    NS = {"10k": 2957247.0, "PDB unique": 4.75e7, "UniRef50": 1.218e10}
    hq = np.quantile(h, [1 / 3, 2 / 3])
    hgrp = np.digitize(h, hq)
    def groups():
        return (("all", np.ones(len(L), bool)), ("h low", hgrp == 0), ("h mid", hgrp == 1), ("h high", hgrp == 2),
                ("LC > 0.1", lc > 0.1), ("LC > 0.3", lc > 0.3), ("UniRef50", sets == "u50_db800"),
                ("SCOP", sets == "scop_q"), ("PDB", sets == "pdb_r600"))

    summary = {}
    for c in CANDS:
        q90 = np.quantile(z[f"panel_f_{c}"].astype(np.float64) - z[f"panel_r_{c}"], 0.9, axis=1)
        sp_null = (z[f"null_f_{c}"].astype(np.float64) - z[f"null_r_{c}"]) - q90[:, None]
        sp_self = (z[f"self_f_{c}"][:, 0].astype(np.float64) - z[f"self_r_{c}"][:, 0]) - q90
        lam = np.zeros(len(edges) + 1)
        lnk = np.zeros(len(edges) + 1)
        for b in range(len(edges) + 1):
            k = binof == b
            if k.sum() < 10:
                lam[b], lnk[b] = np.nan, np.nan
                continue
            x = sp_null[k][keep[k]]
            u = np.quantile(x, 0.99)                      # exponential tail above the pooled 99th percentile
            y = x[x > u] - u
            lam[b] = 1.0 / y.mean()
            mN = (L[k] * nres_q[k]).sum()
            lnk[b] = math.log(len(y)) - math.log(mN) + lam[b] * u
        la, lk = lam[binof], lnk[binof]
        lnE_null = lk[:, None] + np.log(L)[:, None] + np.log(nres_q)[:, None] - la[:, None] * sp_null
        lnE_null[~keep] = np.inf
        rows = {}
        for lab, sel in groups():
            nq = int(sel.sum())
            rows[lab] = {t: float((lnE_null[sel] <= math.log(t)).sum() / (t * nq)) for t in (1.0, 0.1, 0.01)}
            rows[lab]["n"] = nq
        selfE = {}
        for nk, nn in NS.items():
            lnE = lk + np.log(L) + math.log(nn) - la * sp_self
            selfE[nk] = {lab: float(np.mean(lnE[sel] <= math.log(1e-3))) for lab, sel in groups()}
        # the shipped table for V1, as a check of the mini-recalibration
        if c == "V1":
            la0, lk0 = cs.lam_lnk(L, system)
            lnE0 = lk0[:, None] + np.log(L)[:, None] + np.log(nres_q)[:, None] - la0[:, None] * sp_null
            lnE0[~keep] = np.inf
            # the shipped table's own self-hit shares on these queries, for comparison with the refit
            for nk, nn in NS.items():
                lnE_s = lk0 + np.log(L) + math.log(nn) - la0 * sp_self
                selfE[nk + ", shipped v1 table"] = {lab: float(np.mean(lnE_s[sel] <= math.log(1e-3)))
                                                    for lab, sel in groups()}
            for lab, sel in groups():
                rows[f"{lab}, shipped v1 table"] = {t: float((lnE0[sel] <= math.log(t)).sum() / (t * sel.sum()))
                                                    for t in (1.0, 0.1, 0.01)}
        summary[c] = dict(lam=lam.tolist(), lnK=lnk.tolist(), fp=rows, self_e1e3=selfE,
                          self_sp_per_res_median={lab: float(np.median(sp_self[sel] / L[sel])) for lab, sel in
                                                  (("h low", hgrp == 0), ("h high", hgrp == 2))},
                          q90_median={lab: float(np.median(q90[sel])) for lab, sel in
                                      (("h low", hgrp == 0), ("h high", hgrp == 2))})
    summary["h_tertile_edges"] = hq.tolist()
    summary["bin_edges"] = edges.tolist()
    summary["shipped_lambda"] = list(system.lam)
    print(json.dumps(summary, indent=1))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--exclude", default=None, help="blosum_null.py output: drop its flagged null pairs")
    args = ap.parse_args()
    if args.analyze:
        res = analyze(args.a, args.b, args.exclude)
        tag = "_summary_filtered.json" if args.exclude else "_summary.json"
        with open(os.path.splitext(args.a)[0] + tag, "w") as fh:
            json.dump(res, fh, indent=1)
    else:
        run(args.a, args.b, args.workers)


if __name__ == "__main__":
    main()
