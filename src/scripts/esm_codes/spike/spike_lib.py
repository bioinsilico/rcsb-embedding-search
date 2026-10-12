"""P0.4 kernel spike: ctypes wrapper of fmsw_spike.c and of the RESEARCH path (swkern.c / swmulti.c + numpy BLAS).

New path (fixed order, BLAS-free):  q fp16 -> fp32 -> fmsw_query_prep -> fmsw_sim_codes -> fmsw_score_fwd_rev
                                     (S_r on the row-reversed view of the same block) / fmsw_align (forward traceback)
Research path (embcomp/evalue/score/score.py and inventory_probe/probe.py):
                                     qp = q / (8 |q|) (numpy fp32) ; T = unpackbits(codes) * 2 - 1 ; sim = qp @ T.T
                                     (BLAS SGEMM) ; simr = ascontiguousarray(sim[::-1]) ; swz_ad_multi_f32 (R = 2:
                                     fwd on sim, rev on simr) or swz_ad_batch_f32 (tb = 1 for the forward path)
Setting D: offset 1.0, gap open 5.0, gap extend 0.5.
"""
from __future__ import annotations

import ctypes as C
import os

import numpy as np

OFF, GO, GE = 1.0, 5.0, 0.5
_vp = C.c_void_p


def _a(x):
    return None if x is None else x.ctypes.data


class Fmsw:
    """One build of fmsw_spike.c."""

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        L = self.lib = C.CDLL(self.path)
        L.fmsw_isa.restype = C.c_char_p
        L.fmsw_compiler.restype = C.c_char_p
        L.fmsw_sim_mode.restype = C.c_char_p
        L.fmsw_vector_lanes.restype = C.c_int
        L.fmsw_sim_tile.restype = C.c_int
        L.fmsw_query_prep.restype = C.c_int
        L.fmsw_query_prep.argtypes = [_vp, C.c_int, _vp]
        L.fmsw_sim_codes.restype = C.c_int
        L.fmsw_sim_codes.argtypes = [_vp, C.c_int, _vp, _vp, C.c_int, _vp, C.c_int64]
        L.fmsw_ws_floats.restype = C.c_size_t
        L.fmsw_ws_floats.argtypes = [C.c_int, C.c_int64]
        L.fmsw_flag_bytes.restype = C.c_size_t
        L.fmsw_flag_bytes.argtypes = [C.c_int, C.c_int64]
        L.fmsw_score_fwd_rev.restype = C.c_int
        L.fmsw_score_fwd_rev.argtypes = [_vp, C.c_int, C.c_int64, _vp, C.c_int, C.c_float, C.c_float, C.c_float,
                                         _vp, _vp, _vp, _vp, _vp, C.c_size_t]
        L.fmsw_align.restype = C.c_int
        L.fmsw_align.argtypes = [_vp, C.c_int, C.c_int64, _vp, C.c_int, C.c_float, C.c_float, C.c_float, C.c_int,
                                 _vp, _vp, _vp, C.c_size_t, _vp, _vp, C.c_size_t, _vp, C.c_size_t]
        L.fmsw_block_fwd_rev.restype = C.c_int
        L.fmsw_block_fwd_rev.argtypes = [_vp, C.c_int, _vp, _vp, C.c_int, C.c_float, C.c_float, C.c_float, _vp, _vp,
                                         _vp, C.c_size_t, _vp, C.c_size_t, _vp]
        L.fmsw_list_fwd_rev.restype = C.c_int
        L.fmsw_list_fwd_rev.argtypes = [_vp, C.c_int, _vp, _vp, C.c_int, C.c_float, C.c_float, C.c_float,
                                        C.c_int64, _vp, _vp, _vp, C.c_size_t, _vp, C.c_size_t, _vp]
        self.ws = np.empty(0, np.float32)
        self.fl = np.empty(0, np.uint8)
        self.simbuf = np.empty(0, np.float32)
        self.co = np.empty(0, np.int64)

    def info(self):
        L = self.lib
        t = L.fmsw_sim_tile()
        return dict(path=self.path, isa=L.fmsw_isa().decode(), compiler=L.fmsw_compiler().decode(),
                    sim_mode=L.fmsw_sim_mode().decode(), vector_lanes=L.fmsw_vector_lanes(),
                    sim_tile_rows=t // 1000, sim_tile_cols=t % 1000)

    # ------------------------------------------------------------------ buffers
    def _ws(self, n, mmax):
        need = int(self.lib.fmsw_ws_floats(n, mmax))
        if self.ws.size < need:
            self.ws = np.empty(need, np.float32)
        return self.ws

    def _fl(self, n, mmax):
        need = max(1, int(self.lib.fmsw_flag_bytes(n, mmax)))
        if self.fl.size < need:
            self.fl = np.empty(need, np.uint8)
        return self.fl

    def _sim(self, floats):
        if self.simbuf.size < floats:
            self.simbuf = np.empty(int(floats), np.float32)
        return self.simbuf

    def _co(self, k):
        if self.co.size < k + 1:
            self.co = np.empty(k + 1, np.int64)
        return self.co

    # ------------------------------------------------------------------ primitives
    def query_prep(self, q) -> np.ndarray:
        q = np.ascontiguousarray(np.asarray(q).astype(np.float32))       # fp16 -> fp32 exact
        qp = np.empty_like(q)
        st = self.lib.fmsw_query_prep(_a(q), len(q), _a(qp))
        if st:
            raise ValueError(f"fmsw_query_prep status {st}")
        return qp

    def sim_codes(self, qp, codes, col_off=None, out=None) -> np.ndarray:
        codes = np.ascontiguousarray(codes, dtype=np.uint8)
        col_off = np.array([0, len(codes)], np.int64) if col_off is None else np.ascontiguousarray(col_off, np.int64)
        K = len(col_off) - 1
        w = int(col_off[-1] - col_off[0])
        n = len(qp)
        if out is None:
            out = np.empty((n, w), np.float32)
        st = self.lib.fmsw_sim_codes(_a(qp), n, _a(codes), _a(col_off), K, _a(out), out.strides[0] // 4)
        if st:
            raise ValueError(f"fmsw_sim_codes status {st}")
        return out

    def score_fwd_rev(self, sim, col_off, ends=False):
        assert sim.dtype == np.float32 and sim.strides[1] == 4
        col_off = np.ascontiguousarray(col_off, np.int64)
        K = len(col_off) - 1
        n = sim.shape[0]
        mmax = int(np.max(np.diff(col_off))) if K else 1
        ws = self._ws(n, mmax)
        sf, sr = np.empty(K, np.float32), np.empty(K, np.float32)
        ef = er = None
        if ends:
            ef, er = np.empty((K, 2), np.int32), np.empty((K, 2), np.int32)
        st = self.lib.fmsw_score_fwd_rev(_a(sim), n, sim.strides[0] // 4, _a(col_off), K, OFF, GO, GE, _a(sf), _a(sr),
                                         _a(ef), _a(er), _a(ws), ws.size)
        if st:
            raise ValueError(f"fmsw_score_fwd_rev status {st}")
        return (sf, sr, ef, er) if ends else (sf, sr)

    def align(self, sim, col_off, rev=False):
        """Forward (or reversed-view) DP with traceback: scores [K], ends [K, 2], path_flat [P, 2], path_off [K + 1]."""
        assert sim.dtype == np.float32 and sim.strides[1] == 4
        col_off = np.ascontiguousarray(col_off, np.int64)
        K = len(col_off) - 1
        n = sim.shape[0]
        mmax = int(np.max(np.diff(col_off))) if K else 1
        ws, fl = self._ws(n, mmax), self._fl(n, mmax)
        cap = K * n + int(col_off[-1] - col_off[0]) + 2
        pbuf = np.empty(2 * cap, np.int32)
        poff = np.empty(K + 1, np.int64)
        sc, ends = np.empty(K, np.float32), np.empty((K, 2), np.int32)
        st = self.lib.fmsw_align(_a(sim), n, sim.strides[0] // 4, _a(col_off), K, OFF, GO, GE, int(rev), _a(sc),
                                 _a(ends), _a(pbuf), cap, _a(poff), _a(ws), ws.size, _a(fl), fl.size)
        if st:
            raise ValueError(f"fmsw_align status {st}")
        return sc, ends, pbuf[:2 * poff[-1]].reshape(-1, 2), poff

    # ------------------------------------------------------------------ drivers (benchmarks)
    def block_fwd_rev(self, qp, codes, col_off_abs, sf_out, sr_out):
        """One block (col_off absolute into codes): sim into the reusable buffer + S_f, S_r (one ctypes call)."""
        K = len(col_off_abs) - 1
        n = len(qp)
        w = int(col_off_abs[-1] - col_off_abs[0])
        mmax = int(np.max(np.diff(col_off_abs)))
        ws = self._ws(n, mmax)
        sb = self._sim(n * w)
        co = self._co(K)
        st = self.lib.fmsw_block_fwd_rev(_a(qp), n, _a(codes), _a(col_off_abs), K, OFF, GO, GE, _a(sf_out),
                                         _a(sr_out), _a(sb), sb.size, _a(ws), ws.size, _a(co))
        if st:
            raise ValueError(f"fmsw_block_fwd_rev status {st}")

    def list_fwd_rev(self, qp, codes, col_off, block_floats, sf_out, sr_out):
        K = len(col_off) - 1
        n = len(qp)
        lens = np.diff(col_off)
        mmax = int(lens.max())
        ws = self._ws(n, mmax)
        sb = self._sim(max(int(block_floats), n * mmax))
        co = self._co(K)
        st = self.lib.fmsw_list_fwd_rev(_a(qp), n, _a(codes), _a(col_off), K, OFF, GO, GE, int(block_floats),
                                        _a(sf_out), _a(sr_out), _a(sb), sb.size, _a(ws), ws.size, _a(co))
        if st:
            raise ValueError(f"fmsw_list_fwd_rev status {st}")


class Research:
    """The research path: numpy query_prep / decode / BLAS SGEMM + libswkern (swz_ad_batch_f32) + libswmulti
    (swz_ad_multi_f32), built from the unmodified research sources with the research flags."""

    def __init__(self, libdir: str):
        ext = "dylib" if os.uname().sysname == "Darwin" else "so"
        self.k = C.CDLL(os.path.join(libdir, f"libswkern.{ext}"))
        self.m = C.CDLL(os.path.join(libdir, f"libswmulti.{ext}"))
        self.k.swz_ad_batch_f32.restype = None
        self.k.swz_ad_batch_f32.argtypes = [_vp, C.c_int, C.c_int, _vp, C.c_int, C.c_float, C.c_float, C.c_float,
                                            C.c_int, _vp, _vp, _vp, _vp, _vp, _vp]
        self.k.sw_ad_ws_floats.restype = C.c_size_t
        self.k.sw_ad_ws_floats.argtypes = [C.c_int, C.c_int]
        self.k.sw_ad_flag_bytes.restype = C.c_size_t
        self.k.sw_ad_flag_bytes.argtypes = [C.c_int, C.c_int]
        self.k.swkern_vector_lanes.restype = C.c_int
        self.m.swz_ad_multi_f32.restype = None
        self.m.swz_ad_multi_f32.argtypes = [_vp, _vp, C.c_int, C.c_int, _vp, C.c_int, C.c_int, _vp, _vp, _vp, _vp,
                                            _vp, _vp]
        self.R_OFF = np.array([OFF, OFF], np.float32)
        self.R_GO = np.array([GO, GO], np.float32)
        self.R_GE = np.array([GE, GE], np.float32)
        self.R_REV = np.array([0, 1], np.int32)
        self.ws = np.empty(0, np.float32)
        self.fl = np.empty(0, np.uint8)
        self.libdir = libdir

    def info(self):
        return dict(libdir=self.libdir, vector_lanes=int(self.k.swkern_vector_lanes()))

    def _ws(self, n, mmax):
        need = 2 * (n + mmax) + int(self.k.sw_ad_ws_floats(n, mmax))
        if self.ws.size < need:
            self.ws = np.empty(need, np.float32)
        return self.ws

    @staticmethod
    def query_prep(q):
        q = np.asarray(q, dtype=np.float32)
        return q / (8.0 * np.linalg.norm(q, axis=1, keepdims=True))

    @staticmethod
    def decode(codes):
        return np.unpackbits(codes, axis=1)[:, :64].astype(np.float32) * 2 - 1

    def batch(self, sim, col_off, tb):
        """swfast.swz_ad_batch (research): scores, ends[, path_flat, path_off]."""
        assert sim.dtype == np.float32 and sim.flags.c_contiguous
        n, ld = sim.shape
        col_off = np.ascontiguousarray(col_off, np.int64)
        K = len(col_off) - 1
        mmax = int(np.max(np.diff(col_off)))
        ws = self._ws(n, mmax)
        scores, ends = np.empty(K, np.float32), np.empty((K, 2), np.int32)
        if tb:
            need = max(1, int(self.k.sw_ad_flag_bytes(n, mmax)))
            if self.fl.size < need:
                self.fl = np.empty(need, np.uint8)
            pbuf = np.empty(2 * (K * n + int(col_off[-1] - col_off[0]) + 2), np.int32)
            poff = np.empty(K + 1, np.int64)
        else:
            pbuf = poff = None
        self.k.swz_ad_batch_f32(_a(sim), n, ld, _a(col_off), K, OFF, GO, GE, int(tb), _a(scores), _a(ends), _a(pbuf),
                                _a(poff), _a(ws), _a(self.fl) if tb else None)
        if tb:
            return scores, ends, pbuf[:2 * poff[-1]].reshape(-1, 2), poff
        return scores, ends

    def multi_fwd_rev(self, sim, simr, col_off, out2):
        """swz_ad_multi_f32 with R = 2 (fwd on sim, rev on simr): out2 float32 [2, K]."""
        n, ld = sim.shape
        K = len(col_off) - 1
        mmax = int(np.max(np.diff(col_off)))
        ws = self._ws(n, mmax)
        self.m.swz_ad_multi_f32(_a(sim), _a(simr), n, ld, _a(col_off), K, 2, _a(self.R_OFF), _a(self.R_GO),
                                _a(self.R_GE), _a(self.R_REV), _a(out2), _a(ws))

    _LUT = (np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).astype(np.float32) * 2 - 1)

    def block_fwd_rev_reuse(self, qp, codes, col_off_abs, out2, bufs):
        """The same work with every buffer reused (no per-block allocation): LUT decode into a buffer (identical
        values to unpackbits), SGEMM with out=, reversed copy with copyto. bufs: dict with 'T', 'sim', 'simr'."""
        j0, j1 = int(col_off_abs[0]), int(col_off_abs[-1])
        w, n = j1 - j0, len(qp)
        for key, size in (("T", w * 64), ("sim", n * w), ("simr", n * w)):
            if bufs.get(key) is None or bufs[key].size < size:
                bufs[key] = np.empty(size, np.float32)
        T = bufs["T"][:w * 64].reshape(w, 8, 8)
        np.take(self._LUT, codes[j0:j1], axis=0, out=T)
        sim = bufs["sim"][:n * w].reshape(n, w)
        np.matmul(qp, T.reshape(w, 64).T, out=sim)
        simr = bufs["simr"][:n * w].reshape(n, w)
        np.copyto(simr, sim[::-1])
        self.multi_fwd_rev(sim, simr, col_off_abs - j0, out2)

    def block_fwd_rev(self, qp, codes, col_off_abs, out2):
        """score.py's per-block work: decode + GEMM + reversed copy + swz_ad_multi_f32 (R = 2)."""
        j0, j1 = int(col_off_abs[0]), int(col_off_abs[-1])
        T = self.decode(codes[j0:j1])
        sim = qp @ T.T
        simr = np.ascontiguousarray(sim[::-1])
        self.multi_fwd_rev(sim, simr, col_off_abs - j0, out2)


def blocks(lengths, n, block_floats):
    """research codepath._blocks rule: a candidate opens a new block when the current block is non-empty and
    would exceed block_floats similarity floats. Returns a list of (k0, k1)."""
    out, k0, acc = [], 0, 0
    for k, m in enumerate(lengths):
        if k > k0 and acc + n * m > block_floats:
            out.append((k0, k))
            k0, acc = k, 0
        acc += n * int(m)
    out.append((k0, len(lengths)))
    return out
