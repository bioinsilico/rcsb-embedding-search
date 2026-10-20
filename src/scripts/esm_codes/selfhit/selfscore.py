#!/usr/bin/env python
"""Self-hit study, step 3: score every encoded sequence against ITSELF and against the 1,000-member panel with
FoldMatch's own esm-codes-v1 kernel (read-only, P6 tree on PYTHONPATH), plus float64 diagnostics of the self matrix.

Per item (one sequence of an encode_fm.py npz):
  * self pair, kernel:   S_f, S_r (swkern.score_fwd_rev), the S_f path (swkern.align)
  * panel, kernel:       S_f, S_r of the item as a query vs every panel member -> S = fp64(S_f) - fp64(S_r),
                         Q90 = numpy.quantile(S, 0.9) (S10); the panel S_f / S_r are kept for later analyses
  * self matrix, numpy float64 (S7 written out; diagnostics only, the kernel's fp32 z is what scores):
      diagonal s_ii, row mean / std, column mean / std, z_ii and its row and column halves, the number of target
      residues j with s_ij > s_ii (how many residues beat the true partner), z summed along the kernel path (a
      consistency check against S_f), path coverage and its share on the main diagonal, and the fraction of cells
      at |i - j| > 3 with z > offset (positive cell score) as an internal-repeat measure.
  * optionally (--db DIR) the same self pair with the database's stored target codes (CUDA build) when the item id
    is in the database, to compare with the P6 check outputs.

    PYTHONPATH=<wt-P6/src> HF_HUB_OFFLINE=1 HF_HUB_CACHE=<cache> python selfscore.py ENC.npz OUT.npz [--workers 8]
        [--db DATA/db_u10k]
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

_W = {}


def _init(db_dir):
    from foldmatch.search import scoring_system as ss
    from foldmatch.search import swkern
    system = ss.get("esm-codes-v1")
    spec = ss.prepare(system, local_files_only=True).for_worker()
    codes, offsets = spec.load_panel()
    _W.update(swkern=swkern, params=spec.dp_params(), panel=(np.ascontiguousarray(codes), np.asarray(offsets)),
              offset=float(system.offset), q=float(system.panel_quantile), db=None)
    if db_dir:
        con = sqlite3.connect(f"file:{os.path.join(db_dir, 'P.rescodes.idx')}?mode=ro", uri=True)
        _W["db"] = (con, np.memmap(os.path.join(db_dir, "P.rescodes.bin"), dtype=np.uint8, mode="r").reshape(-1, 8))


def _rowcol(sim):
    mu, sd = sim.mean(1), sim.std(1)
    nu, tau = sim.mean(0), sim.std(0)
    a = (sim - mu[:, None]) / (sd[:, None] + 1e-8)
    b = (sim - nu[None, :]) / (tau[None, :] + 1e-8)
    a[sd < 1e-6, :] = 0.0
    b[:, tau < 1e-6] = 0.0
    return 0.5 * (a + b), a, b, mu, sd, nu, tau


def _item(task):
    idx, item_id, q16, codes = task
    sk, params = _W["swkern"], _W["params"]
    L = len(q16)
    qp = sk.query_prep(np.ascontiguousarray(q16))
    one = np.array([0, len(codes)], np.int64)
    sc = sk.score_fwd_rev(qp, codes, one, params)
    al = sk.align(qp, codes, one, params, traceback=True)
    path = al.path_of(0)
    pc, po = _W["panel"]
    pr = sk.score_fwd_rev(qp, pc, po, params)
    s_panel = pr.s_fwd.astype(np.float64) - pr.s_rev.astype(np.float64)
    q90 = float(np.quantile(s_panel, _W["q"], method="linear"))

    D = sk.decode_codes(codes)
    sim = np.matmul(qp, D.T).astype(np.float64)
    z, a, b, mu, sd, nu, tau = _rowcol(sim)
    d = np.arange(L)
    s_ii = sim[d, d]
    beats = (sim > s_ii[:, None]).sum(1)
    pm = (path[:, 0] >= 0) & (path[:, 1] >= 0)
    pi, pj = path[pm, 0], path[pm, 1]
    z_path_sum = float((z[pi, pj] - _W["offset"]).sum()) if pm.any() else 0.0
    ii, jj = np.meshgrid(d, d, indexing="ij")
    off = np.abs(ii - jj) > 3
    pos_off = float(((z > _W["offset"]) & off).sum() / max(off.sum(), 1))

    out = dict(idx=idx, L=L, S_f=float(sc.s_fwd[0]), S_r=float(sc.s_rev[0]), S_f_tb=float(al.scores[0]), q90=q90,
               panel_sf=pr.s_fwd.astype(np.float32), panel_sr=pr.s_rev.astype(np.float32),
               z_ii=z[d, d].astype(np.float32), a_ii=a[d, d].astype(np.float32), b_ii=b[d, d].astype(np.float32),
               s_ii=s_ii.astype(np.float32), mu=mu.astype(np.float32), sd=sd.astype(np.float32),
               beats=beats.astype(np.int32), z_path_sum=z_path_sum, n_match=int(pm.sum()),
               n_diag=int((pi == pj).sum()), path_len=int(len(path)),
               qstart=int(pi.min()) if pm.any() else -1, qend=int(pi.max()) if pm.any() else -1,
               pos_off=pos_off, distinct_codes=int(len(np.unique(codes.view(np.uint64)))))
    if _W["db"] is not None:
        con, B = _W["db"]
        row = con.execute("SELECT offset, length FROM entries WHERE id = ?", (item_id,)).fetchone()
        if row is not None and row[1] == L:
            dbc = np.ascontiguousarray(B[row[0]:row[0] + row[1]])
            s2 = sk.score_fwd_rev(qp, dbc, one, params)
            out.update(db_S_f=float(s2.s_fwd[0]), db_S_r=float(s2.s_rev[0]),
                       db_bits_equal=float((np.unpackbits(dbc, axis=1) == np.unpackbits(codes, axis=1)).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("enc")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--db", default=None)
    args = ap.parse_args()
    z = np.load(args.enc, allow_pickle=False)
    off, ids = z["offsets"], [str(x) for x in z["ids"]]
    q, codes = z["q"], z["codes"]
    tasks = [(i, ids[i], q[off[i]:off[i + 1]], np.ascontiguousarray(codes[off[i]:off[i + 1]]))
             for i in range(len(ids))]
    order = sorted(range(len(tasks)), key=lambda i: -len(tasks[i][2]))     # longest first
    t0 = time.time()
    res = [None] * len(tasks)
    if args.workers <= 1:
        _init(args.db)
        for i in order:
            res[i] = _item(tasks[i])
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(args.db,)) as pool:
            for k, r in enumerate(pool.imap_unordered(_item, [tasks[i] for i in order], chunksize=1)):
                res[r["idx"]] = r
                if (k + 1) % 200 == 0:
                    print(f"{k + 1}/{len(tasks)} in {time.time() - t0:.0f}s", flush=True)
    scal = [k for k, v in res[0].items() if not isinstance(v, np.ndarray)]
    for r in res:
        for k in ("db_S_f", "db_S_r", "db_bits_equal"):
            r.setdefault(k, np.nan)
    scal = sorted(set(scal) | {"db_S_f", "db_S_r", "db_bits_equal"})
    outd = {k: np.array([r[k] for r in res]) for k in scal}
    for k in ("z_ii", "a_ii", "b_ii", "s_ii", "mu", "sd", "beats"):
        outd[k] = np.concatenate([r[k] for r in res])
    outd["panel_sf"] = np.stack([r["panel_sf"] for r in res])
    outd["panel_sr"] = np.stack([r["panel_sr"] for r in res])
    outd["offsets"] = off
    outd["ids"] = np.array(ids)
    outd["meta_json"] = json.dumps(dict(enc=args.enc, db=args.db, seconds=time.time() - t0, workers=args.workers))
    np.savez(args.out, **outd)
    print(f"wrote {args.out}: {len(res)} items in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
