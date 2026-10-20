#!/usr/bin/env python
"""Self-hit study, step 6b: flag the v2probe "null" pairs that BLOSUM62 already calls homologous, so the probe's
false-positive ratios are not counting real (sequence-detectable) homologs among random UniRef50 targets.

For each probe query and each of the 300 null targets: BLOSUM62 Smith-Waterman score with gaps 11/1 (a gap of length L
costs 11 + L, rawsw.c with go = 12, ge = 1) and FoldMatch's BLOSUM62 11/1 Karlin-Altschul E-value at N = the null set's
residues. Pairs with E <= 0.01 are flagged (about 0.01 chance flags per query). Remote homologs that BLOSUM62 cannot see stay in the null.

    PYTHONPATH=<wt-P6/src> python blosum_null.py DATA V2PROBE.npz OUT.npz [--workers 8]
"""
from __future__ import annotations

import argparse
import ctypes as C
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
_W = {}


def _init(lib, null_seq, null_len):
    sys.path.insert(0, HERE)
    import seqfeat
    from foldmatch.search import karlin_altschul as ka
    L = C.CDLL(lib)
    L.sw_batch.restype = C.c_int
    L.sw_batch.argtypes = [C.c_void_p, C.c_int, C.c_int64, C.c_void_p, C.c_void_p, C.c_int, C.c_float, C.c_float,
                           C.c_void_p]
    _W.update(lib=L, B=seqfeat._blosum().astype(np.float32), IDX=seqfeat._IDX, ka=ka, params=ka.params_for(11, 1),
              taa=seqfeat._IDX[np.frombuffer(null_seq.encode("ascii"), np.uint8)], tlen=null_len)


def _one(it):
    i, seq = it
    qa = _W["IDX"][np.frombuffer(seq.encode("ascii"), np.uint8)]
    X = np.ascontiguousarray(_W["B"][qa[:, None], _W["taa"][None, :]])
    m = _W["tlen"].astype(np.int32)
    col0 = np.zeros(len(m), np.int64)
    np.cumsum(m[:-1], out=col0[1:])
    out = np.zeros(len(m), np.float32)
    assert _W["lib"].sw_batch(X.ctypes.data, X.shape[0], X.shape[1], col0.ctypes.data, m.ctypes.data, len(m),
                              12.0, 1.0, out.ctypes.data) == 0
    N = int(_W["tlen"].sum())
    e = np.array([_W["ka"].evalue(int(round(s)), len(seq), N, _W["params"]) for s in out])
    return i, out, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("probe")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    z = np.load(args.probe, allow_pickle=False)
    sys.path.insert(0, HERE)
    import v2probe
    nid = [str(x) for x in z["null_ids"]]
    seqs = v2probe.read_fasta(os.path.join(args.data, "u10k", "uniref50_10k.fasta"))
    null_seq = "".join(seqs[x] for x in nid)
    null_len = np.array([len(seqs[x]) for x in nid], np.int64)
    assert np.array_equal(null_len, z["null_len"])
    qseq = {}
    for s in np.unique(z["set"]):
        e = np.load(os.path.join(args.data, "enc", f"{s}.npz"), allow_pickle=False)
        sc, off = e["seq_concat"].tobytes().decode("ascii"), e["offsets"]
        for k, x in enumerate(e["ids"]):
            qseq[(str(s), str(x))] = sc[off[k]:off[k + 1]]
    tasks = [(i, qseq[(str(s), str(x))]) for i, (s, x) in enumerate(zip(z["set"], z["id"]))]
    lib = os.path.join(args.data, "probe", "librawsw.dylib")
    score = np.zeros((len(tasks), len(nid)), np.float32)
    ev = np.zeros((len(tasks), len(nid)), np.float64)
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(args.workers, initializer=_init, initargs=(lib, null_seq, null_len)) as pool:
        for i, s, e in pool.imap_unordered(_one, tasks, chunksize=4):
            score[i], ev[i] = s, e
    np.savez(args.out, blosum_score=score, blosum_evalue=ev, flagged=ev <= 0.01, n_null_residues=int(null_len.sum()))
    print(f"flagged {int((ev <= 0.01).sum())} of {ev.size} null pairs (BLOSUM62 E <= 0.01 at N = {int(null_len.sum())}); "
          f"queries with >= 1 flagged: {int((ev <= 0.01).any(1).sum())}")


if __name__ == "__main__":
    main()
