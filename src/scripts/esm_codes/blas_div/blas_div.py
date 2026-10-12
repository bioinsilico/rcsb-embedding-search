#!/usr/bin/env python
"""blas_div.py -- measurement harness for the BLAS-divergence study (FoldMatch esm-codes-v1, plan decision D2).

QUESTION. If stage 2 computes the similarity  sim = query_prep(q) . decode(codes)^T  with a BLAS SGEMM (D2 option b,
the research path) instead of the fixed-order C loop (option a, spike fmsw_sim_codes), how far can S_f, S_r, S, Q90,
S', lnE and the reported forward alignment diverge between builds (BLAS library / version / core type / threads /
device)?  Only the similarity (and, as a variant, the fp32 numpy query norm) changes between backends: rowcol z, the
setting-D Smith-Waterman (S_f forward with traceback, S_r on the row-reversed view), Q90 and E always run in the SAME
fixed-order spike kernel (spike/fmsw_spike.c through spike/spike_lib.py) and in float64 numpy.
Reference = backend fixed_c with --qprep c (spike fmsw_query_prep + fmsw_sim_codes: bit-identical on every build,
golden sha256 5204a97a...).  Oracle = float64 similarity / z / DP (compare: margins of differing paths, sim accuracy).

SUBCOMMANDS
  info      host, CPU, env, numpy BLAS (np.show_config / threadpoolctl), torch, spike library -> JSON on stdout
  prep      (NERSC, once) build DATA from the research files: verified copies, q150.json, a1.npz (A1 pair bundle),
            research_SD_q150.npz (research S_D_fwd / S_D_rev rows), MANIFEST.json (md5 + size of every file)
  sim       one backend on one work set -> RUN dir (meta.json, search.npz | a1.npz, simblocks.npz, repeat.json)
  compare   A B -> JSON (+ NPZ): divergence of run B against run A (A = reference)
  verify    golden  : spike golden hash through this harness's fixed_c path (any --backend: per-pair diffs vs fixed_c)
            run RUN : fixed_c plumbing (== spike's own C list driver / align), any backend vs the research stored
                      scores (S_D_fwd / S_D_rev, Q90) and vs the stored setting-D alignments (A1)

BACKENDS (sim --backend NAME; unavailable -> clear error, exit 2)
  fixed_c          spike fmsw_sim_codes (golden reference)
  numpy            np.matmul fp32: qp [n,64] @ D.T, D = np.unpackbits(codes)[:, :64].astype(float32) * 2 - 1, one GEMM
                   per block (exactly research swfast.decode_unpackbits + score.py 'sim = qp @ T.T')
  numpy_rowwise    the same decode, one GEMM per target (shape dependence)
  torch_cpu        torch.matmul fp32 on CPU (torch.set_num_threads = --threads)
  torch_cuda       fp32, torch.backends.cuda.matmul.allow_tf32 = False, float32_matmul_precision('highest')
  torch_cuda_tf32  TF32 allowed (hazard demonstration only)
  torch_mps        fp32 on Apple MPS
  --qprep c        spike fixed-order float64 query_prep (S5; primary)
  --qprep numpy    research fp32 q / (8 np.linalg.norm(q)) (swfast.query_prep)

WORK SETS
  search   benchmark queries x targets: --queries q150 (default; DATA/q150.json, the SAME on every host) | all |
           q150:K (K length-spread q150 members) | first:N | rows:i,j,..;  --tmax N = first N of the 50,000 targets;
           plus the 1,000-member shipped panel (Q90) unless --no-panel.  Per query, targets are concatenated into
           blocks of --block-floats (default 2^17) similarity floats with the production / spike rule (a candidate opens
           a new block when the current one is non-empty and would overflow); --block-rule research = evalue/score/
           score.py chunks (CELLS // n residues, use --block-floats 1048576 to reproduce the stored research run).
           Stored: S_f, S_r float32 [nq, NT] (+ panel P_f, P_r [nq, 1000]); full similarity blocks only for a sample
           (--sample-queries length-spread queries x {first, middle} block) -> simblocks.npz.
  a1       the 12,228 CATH domain pairs (val related / remote / unrelated, test remote / unrelated) + 732 chain pairs
           (DATA/a1.npz), one GEMM per pair; stored: S_f, S_r, traceback score, end cell and the FORWARD PATH
           ([L, 2] (i, j), -1 = gap, forward order).  --a1-limit N = stratified first-N slice.
  --repeat VARIANTS recomputes every block with a deliberately different layout and compares bit for bit:
           same (identical call again), off4 / off8 / off16 (inputs and output copied to buffers at +4/+8/+16 bytes from
           a 64-byte boundary), ld (padded leading dimensions), dT (materialised D^T: no transpose flag).  Differing
           blocks are rescored (and for a1 realigned) -> repeat.json.
  Parallelism: spawn pool of --workers processes over queries (longest first) / A1 chunks; each worker uses --threads
  BLAS / torch threads (default 1).  OPENBLAS_CORETYPE, MKL_CBWR, MKL_DEBUG_CPU_TYPE ... are the caller's: they are
  recorded (meta.json env) with the BLAS / torch / device information.

LAYOUT (ROOT = /pscratch/sd/j/jsegura/embcomp/blasdiv on NERSC, <scratchpad>/blasdiv on the Mac; env BLASDIV_ROOT)
  ROOT/data                 inputs (prep); ROOT/spike/build  spike libraries (spike/build.sh; env BLASDIV_LIB_DIR)
  ROOT/runs/<host-tag>/<work>/<label>   sim outputs;  ROOT/compare/<name>.json   compare outputs
  spike sources: <this dir>/../spike, else ROOT/spike (env BLASDIV_SPIKE)

E-VALUES (compare): ship_final.json table D_fmr/pnl_uniref50/M3t999 'all'; Q90 = np.quantile(f64 P_f - f64 P_r, 0.9);
  S = f64 S_f - f64 S_r; S' = S - Q90; lnE = lnK(m) + ln m + ln N - lambda(m) S' (np.interp in ln m, clamped), m = query
  length, N = 8,557,236 = annotated benchmark target residues (ship.py N_ANN on all 50,000 columns = ship_final meta
  N_ann_fit_db).  Ranking: S descending, target index ascending (no embscore here).
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import socket
import struct
import subprocess
import sys
import time
import zipfile

ENV_RE = re.compile(r"^(OMP_|OPENBLAS|MKL_|VECLIB|KMP_|GOTO|BLIS|NUMEXPR|NPY_|CUBLAS|CUDA_VISIBLE|NVIDIA_TF32|TORCH_|"
                    r"PYTORCH_|BLASDIV_|ACCELERATE|GOMP_|SLURM_JOB_ID|SLURM_CPUS|SLURM_NODELIST|SLURMD_NODENAME)")
ENV_AT_START = {k: v for k, v in sorted(os.environ.items()) if ENV_RE.match(k)}
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
               "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _set_threads() -> int:
    """--threads T (or the parent's BLASDIV_THREADS in spawned workers; default 1) -> every BLAS thread variable,
    BEFORE numpy is imported."""
    t = None
    for i, x in enumerate(sys.argv):
        if x == "--threads" and i + 1 < len(sys.argv):
            t = sys.argv[i + 1]
        elif x.startswith("--threads="):
            t = x.split("=", 1)[1]
    t = str(int(t or os.environ.get("BLASDIV_THREADS") or "1"))
    os.environ["BLASDIV_THREADS"] = t
    for v in THREAD_VARS:
        os.environ[v] = t
    return int(t)


THREADS = _set_threads()
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SP_LOCAL = ("/private/tmp/claude-501/-Users-joan-devel-rcsb-embedding-search/"
            "d9266e69-77ea-4121-a428-2e6f3bcc4ca2/scratchpad")
IS_NERSC = os.path.isdir("/pscratch/sd/j/jsegura")
ROOT = os.environ.get("BLASDIV_ROOT") or ("/pscratch/sd/j/jsegura/embcomp/blasdiv" if IS_NERSC
                                          else f"{SP_LOCAL}/blasdiv")
DATA_DEFAULT = os.path.join(ROOT, "data")
EMB_NERSC = "/pscratch/sd/j/jsegura/embcomp"
SETTING_D = (1.0, 5.0, 0.5)                      # offset, gap open, gap extend
SHIP_KEY = "D_fmr/pnl_uniref50/M3t999"
EDGES = [0, 40, 60, 100, 200, 400, 10 ** 9]      # M3 query-length bins (rlib; left-closed)
BLOCK_FLOATS = 1 << 17
GOLDEN_SHA = "5204a97a94a8d3713219e1c5bf1271fb910a63cd938e96cf1d95e417608720c9"
Q150_SEED, Q150_PER_BIN = 20261012, 25
BACKENDS = ("fixed_c", "numpy", "numpy_rowwise", "torch_cpu", "torch_cuda", "torch_cuda_tf32", "torch_mps")
ALL_VARIANTS = ("same", "off4", "off8", "off16", "ld", "dT")
E_THRESH = (1e-10, 1e-6, 1e-3, 1e-2, 0.1, 1.0, 10.0)
A1_SETS = ("val/related", "val/remote", "val/unrelated", "test/remote", "test/unrelated", "chains")
RESEARCH_F1 = {"val/related": 0.6396, "val/remote": 0.2664, "test/remote": 0.2620, "chains": 0.6134}
# research inputs (paths relative to EMB_NERSC) and their md5 (plan Appendix A / research manifests)
SOURCES = {
    "queries.L2m_B8_s0.npz": ("evalue/encode/out/queries.L2m_B8_s0.npz", "e821497f59dda06a3899c18dce37286c"),
    "targets.L2m_B8_s0.npz": ("evalue/encode/out/targets.L2m_B8_s0.npz", "cfee06cd6f7c46080c4dc9082ad97eb8"),
    "shipped_panel_codes.npz": ("panel/fit2/results/shipped_panel_codes.npz", "15cc1c7b2d1dd488a5f9b742365ccb13"),
    "ship_final.json": ("panel/fit2/results/ship_final.json", "c51735f962c04bd8c441b9625c4cf30a"),
    "bench_labels.npz": ("evalue/score/results/labels.npz", "a8b36754e22753879b6bd5ed22f500f8"),
    "research_q90.npy": ("panel/fit2/results/bench_q90_shipped.npy", None),
    "spike_inputs/golden_inputs.npz": ("p0/spike/inputs/golden_inputs.npz", "b52c3da4de41768b2e32c4dbc20d3d18"),
    "spike_inputs/fixture_candidate.npz": ("p0/spike/inputs/fixture_candidate.npz", "d724b769bcf26178cc20a3bf157a64bf"),
}
READ_ONLY = {
    "S_D_fwd": ("evalue/score/out/S_D_fwd.npy", "1d015c63cb870524664442bd3784384e"),
    "S_D_rev": ("evalue/score/out/S_D_rev.npy", "70f3a0f75bf58243a9697da73d449cbd"),
    "score_index": ("evalue/score/out/index.npz", "ff0f64fc242625b6c7dfce5082a57abb"),
    "score_manifest": ("evalue/score/out/manifest.json", None),
    "enc_val": ("throughput/build/data/L2m_B8_s0.val.npz", "5aebc14942665748313f22ca737ca26d"),
    "enc_test": ("throughput/build/data/L2m_B8_s0.test.npz", "e570bc26e1a142183aaf85e9a7a57df7"),
    "pairs": ("throughput/build/data/pairs_L2m_B8_s0.npz", "4ffd144ad35917167bc0d9252dd181d5"),
    "enc_chains": ("sw_mlp/enc/L2m_B8_s0.chains.npz", "5519866044ab0f348c74b9319125acd2"),
    "f1_val": ("evalue/f1_settings/results/dom/M8_s0.val.npz", "67a6ee1d9d9825978c8cd7a25869c8a9"),
    "f1_test": ("evalue/f1_settings/results/dom/M8_s0.test.npz", "f4225dd075ef82288748749dcbb016ef"),
    "f1_chains_D": ("evalue/f1_settings/results/chains/M8_s0_D.npz", "9931e3c82bcde55dc0bde12fd75fcad5"),
    "chain_pairs": ("chains/data/pairs.json", "40686c82791126f570f00fbad071e846"),
}
T0 = time.time()


class Unavailable(Exception):
    pass


# =====================================================================================================================
# utilities
# =====================================================================================================================
def log(*a):
    print(f"[{time.time() - T0:8.1f}s] " + " ".join(str(x) for x in a), flush=True)


def md5(path, bs=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def _jdefault(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def jdump(obj, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_jdefault)
    os.replace(tmp, path)


def npz_member(path, name):
    """Memory-map one STORED (uncompressed) member of an .npz; falls back to np.load for compressed members."""
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(name + ".npy")
        stored = info.compress_type == zipfile.ZIP_STORED
    if not stored:
        return np.load(path)[name]
    with open(path, "rb") as fh:
        fh.seek(info.header_offset)
        loc = fh.read(30)
        fh.seek(info.header_offset + 30 + int.from_bytes(loc[26:28], "little") + int.from_bytes(loc[28:30], "little"))
        ver = np.lib.format.read_magic(fh)
        rd = np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = rd(fh)
        off = fh.tell()
    if int(np.prod(shape)) == 0:
        return np.zeros(shape, dtype)
    return np.memmap(path, dtype=dtype, mode="r", shape=shape, offset=off, order="F" if fortran else "C")


def ulp_diff(a, b):
    """|ordinal(a) - ordinal(b)| of float32 values (+0 == -0)."""
    ia = np.asarray(a, np.float32).view(np.int32).astype(np.int64)
    ib = np.asarray(b, np.float32).view(np.int32).astype(np.int64)
    ia = np.where(ia < 0, np.int64(-2 ** 31) - ia, ia)
    ib = np.where(ib < 0, np.int64(-2 ** 31) - ib, ib)
    return np.abs(ia - ib)


def bits_equal(a, b):
    """Elementwise float32 bit identity (-0 != +0: the stored bits are compared)."""
    return np.asarray(a, np.float32).view(np.int32) == np.asarray(b, np.float32).view(np.int32)


def spread(values, k):
    """Indices of k members spread over the sorted order of values (always includes min and max)."""
    o = np.argsort(values, kind="stable")
    if k >= len(o):
        return np.sort(o)
    return np.sort(o[np.unique(np.round(np.linspace(0, len(o) - 1, k)).astype(int))])


def cpu_model():
    try:
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True).stdout.strip()
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor()


def host_info():
    return dict(hostname=socket.gethostname(), platform=platform.platform(), machine=platform.machine(),
                cpu=cpu_model(), ncpu=os.cpu_count(), python=sys.version.split()[0], executable=sys.executable,
                nersc=IS_NERSC)


def env_info():
    return dict(at_start=ENV_AT_START, effective={k: os.environ.get(k) for k in THREAD_VARS},
                threads=THREADS, blas_related={k: v for k, v in sorted(os.environ.items()) if ENV_RE.match(k)})


def numpy_info():
    d = dict(version=np.__version__, file=np.__file__)
    try:
        cfg = np.show_config(mode="dicts")
        bd = cfg.get("Build Dependencies", {})
        d["blas"], d["lapack"] = bd.get("blas"), bd.get("lapack")
        d["simd"] = cfg.get("SIMD Extensions")
        d["compilers"] = {k: v.get("version") for k, v in cfg.get("Compilers", {}).items() if isinstance(v, dict)}
    except Exception:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            np.show_config()
        d["show_config_text"] = buf.getvalue()[:6000]
    try:
        import threadpoolctl
        d["threadpoolctl"] = threadpoolctl.threadpool_info()
    except Exception as e:  # noqa: BLE001
        d["threadpoolctl"] = f"unavailable: {e}"
    return d


def torch_info(torch=None):
    try:
        if torch is None:
            import torch
    except Exception as e:  # noqa: BLE001
        return dict(available=False, error=str(e))
    d = dict(available=True, version=torch.__version__, file=torch.__file__, num_threads=torch.get_num_threads(),
             mkl=bool(torch.backends.mkl.is_available()), mkldnn=bool(torch.backends.mkldnn.is_available()),
             cuda_version=torch.version.cuda, cuda_available=bool(torch.cuda.is_available()),
             mps_available=bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
             float32_matmul_precision=torch.get_float32_matmul_precision())
    try:
        d["config"] = torch.__config__.show()
        d["parallel_info"] = torch.__config__.parallel_info()
    except Exception:
        pass
    try:
        d["cuda_matmul_allow_tf32"] = bool(torch.backends.cuda.matmul.allow_tf32)
        d["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
        d["cuda_matmul_allow_fp16_reduced"] = bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction)
    except Exception as e:  # noqa: BLE001
        d["cuda_flags_error"] = str(e)
    if d["cuda_available"]:
        d["devices"] = [dict(name=torch.cuda.get_device_name(i), capability=list(torch.cuda.get_device_capability(i)))
                        for i in range(torch.cuda.device_count())]
        try:
            d["cublas_version"] = torch._C._cuda_getCompiledVersion()
        except Exception:
            pass
    return d


# =====================================================================================================================
# spike kernel (fixed-order rowcol z + SW; fixed-order query_prep / sim_codes)
# =====================================================================================================================
def spike_src_dir():
    for d in (os.environ.get("BLASDIV_SPIKE"), os.path.join(HERE, "..", "spike"), os.path.join(ROOT, "spike")):
        if d and os.path.isfile(os.path.join(d, "spike_lib.py")) and os.path.isfile(os.path.join(d, "fmsw_spike.c")):
            return os.path.abspath(d)
    raise SystemExit("spike sources (spike_lib.py, fmsw_spike.c) not found: set BLASDIV_SPIKE")


def default_lib():
    bd = os.environ.get("BLASDIV_LIB_DIR") or os.path.join(ROOT, "spike", "build")
    ext = "dylib" if sys.platform == "darwin" else "so"
    names = (["gcc_avx2hyb", "gcc_avx2fma", "gcc_sse2"] if platform.machine() in ("x86_64", "AMD64")
             else ["clang_neon", "gcc_neon", "gcc_generic"])
    for nm in names:
        p = os.path.join(bd, f"libfmsw_{nm}.{ext}")
        if os.path.isfile(p):
            return p
    raise SystemExit(f"no spike library in {bd} (build it with spike/build.sh; or pass --lib / BLASDIV_LIB_DIR)")


_SPIKE: dict = {}


def spike(lib=None):
    lib = os.path.abspath(lib or default_lib())
    if _SPIKE.get("lib") != lib:
        d = spike_src_dir()
        if d not in sys.path:
            sys.path.insert(0, d)
        import spike_lib as SL
        assert (SL.OFF, SL.GO, SL.GE) == SETTING_D, "spike_lib setting is not D"
        _SPIKE.update(lib=lib, SL=SL, F=SL.Fmsw(lib))
    return _SPIKE["SL"], _SPIKE["F"]


def spike_info(lib=None):
    SL, F = spike(lib)
    d = spike_src_dir()
    return dict(F.info(), lib_md5=md5(F.path), src_dir=d, fmsw_spike_c_md5=md5(os.path.join(d, "fmsw_spike.c")),
                spike_lib_py_md5=md5(os.path.join(d, "spike_lib.py")))


def qprep_numpy(q16):
    """research swfast.query_prep: fp32 q / (8 * np.linalg.norm(q))."""
    q = np.asarray(q16, dtype=np.float32)
    return q / (8.0 * np.linalg.norm(q, axis=1, keepdims=True))


def decode(codes):
    """research swfast.decode_unpackbits: uint8 [L, 8] -> float32 +-1 [L, 64]."""
    return np.unpackbits(np.asarray(codes), axis=1)[:, :64].astype(np.float32) * 2 - 1


# =====================================================================================================================
# layout helpers for --repeat
# =====================================================================================================================
def offset_empty(shape, dtype, off):
    dtype = np.dtype(dtype)
    nb = int(np.prod(shape)) * dtype.itemsize
    buf = np.empty(nb + 128 + off, np.uint8)
    base = (-buf.ctypes.data) % 64
    return buf[base + off: base + off + nb].view(dtype).reshape(shape)


def offset_copy(x, off):
    x = np.ascontiguousarray(x)
    v = offset_empty(x.shape, x.dtype, off)
    v[...] = x
    return v


def padded(x, pad=8):
    x = np.ascontiguousarray(x)
    big = np.zeros((x.shape[0], x.shape[1] + pad), x.dtype)
    big[:, :x.shape[1]] = x
    return big[:, :x.shape[1]]


# =====================================================================================================================
# backends: sim(qh, codes, variant, col) -> float32 [n, w] C-contiguous
# =====================================================================================================================
class FixedC:
    name, variants = "fixed_c", ("same", "off4", "off8", "off16", "ld")

    def __init__(self, cfg):
        self.SL, self.F = spike(cfg.get("lib"))

    def prepare(self, qp):
        return np.ascontiguousarray(qp, np.float32)

    def sim(self, qh, codes, variant=None, col=None):
        F = self.F
        n, w = len(qh), len(codes)
        if variant in (None, "same"):
            return F.sim_codes(qh, codes)
        if variant.startswith("off"):
            k = int(variant[3:])
            out = offset_empty((n, w), np.float32, k)
            F.sim_codes(offset_copy(qh, k), offset_copy(np.asarray(codes), k), out=out)
            return np.array(out)
        if variant == "ld":
            out = np.empty((n, w + 7), np.float32)[:, :w]
            F.sim_codes(qh, codes, out=out)
            return np.ascontiguousarray(out)
        raise ValueError(variant)

    def info(self):
        return dict(kind="fixed-order C (spike fmsw_sim_codes)")


class NumpyGemm:
    name, variants = "numpy", ALL_VARIANTS

    def __init__(self, cfg):
        pass

    def prepare(self, qp):
        return np.ascontiguousarray(qp, np.float32)

    def sim(self, qp, codes, variant=None, col=None):
        D = decode(codes)
        if variant in (None, "same"):
            return qp @ D.T                                   # research score.py: sim = qp @ T.T
        if variant.startswith("off"):
            k = int(variant[3:])
            out = offset_empty((len(qp), len(D)), np.float32, k)
            np.matmul(offset_copy(qp, k), offset_copy(D, k).T, out=out)
            return np.array(out)
        if variant == "ld":
            return np.ascontiguousarray(padded(qp) @ padded(D).T)
        if variant == "dT":
            return np.ascontiguousarray(qp @ np.ascontiguousarray(D.T))
        raise ValueError(variant)

    def info(self):
        return dict(kind="numpy matmul fp32 (one GEMM per block)")


class NumpyRowwise(NumpyGemm):
    name, variants = "numpy_rowwise", ("same",)

    def sim(self, qp, codes, variant=None, col=None):
        D = decode(codes)
        out = np.empty((len(qp), len(D)), np.float32)
        col = np.array([0, len(D)]) if col is None else col
        for k in range(len(col) - 1):
            a, b = int(col[k]), int(col[k + 1])
            out[:, a:b] = qp @ D[a:b].T
        return out

    def info(self):
        return dict(kind="numpy matmul fp32 (one GEMM per target)")


class TorchGemm:
    variants = ALL_VARIANTS

    def __init__(self, cfg, kind):
        try:
            import torch
        except Exception as e:  # noqa: BLE001
            raise Unavailable(f"torch not importable: {e}")
        self.torch, self.kind, self.name = torch, kind, f"torch_{kind}"
        if kind == "cpu":
            torch.set_num_threads(THREADS)
            self.dev = torch.device("cpu")
        elif kind in ("cuda", "cuda_tf32"):
            if not torch.cuda.is_available():
                raise Unavailable("torch.cuda.is_available() is False")
            idx = int(cfg.get("device_index", 0)) % torch.cuda.device_count()
            self.dev = torch.device(f"cuda:{idx}")
            torch.cuda.set_device(self.dev)
            tf32 = kind == "cuda_tf32"
            torch.backends.cuda.matmul.allow_tf32 = tf32
            torch.backends.cudnn.allow_tf32 = tf32
            torch.set_float32_matmul_precision("high" if tf32 else "highest")
            assert torch.backends.cuda.matmul.allow_tf32 == tf32
        elif kind == "mps":
            if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
                raise Unavailable("torch.backends.mps.is_available() is False")
            self.dev = torch.device("mps")
        else:
            raise ValueError(kind)

    def prepare(self, qp):
        return self.torch.from_numpy(np.ascontiguousarray(qp, np.float32)).to(self.dev)

    def _off(self, t, k):
        flat = self.torch.empty(t.numel() + 16, dtype=t.dtype, device=t.device)
        v = flat[k:k + t.numel()].view(t.shape)
        v.copy_(t)
        return v

    def _pad(self, t, pad=8):
        big = self.torch.zeros((t.shape[0], t.shape[1] + pad), dtype=t.dtype, device=t.device)
        big[:, :t.shape[1]] = t
        return big[:, :t.shape[1]]

    def sim(self, qt, codes, variant=None, col=None):
        torch = self.torch
        D = torch.from_numpy(decode(codes)).to(self.dev)
        if variant in (None, "same"):
            S = torch.matmul(qt, D.T)
        elif variant.startswith("off"):
            k = int(variant[3:]) // 4
            out = self.torch.empty(qt.shape[0] * D.shape[0] + 16, dtype=qt.dtype, device=self.dev)
            out = out[k:k + qt.shape[0] * D.shape[0]].view(qt.shape[0], D.shape[0])
            torch.matmul(self._off(qt, k), self._off(D, k).T, out=out)
            S = out
        elif variant == "ld":
            S = torch.matmul(self._pad(qt), self._pad(D).T)
        elif variant == "dT":
            S = torch.matmul(qt, D.T.contiguous())
        else:
            raise ValueError(variant)
        return np.ascontiguousarray(S.detach().to("cpu").numpy(), dtype=np.float32)

    def info(self):
        d = dict(kind=f"torch.matmul fp32 on {self.dev}", device=str(self.dev))
        if self.dev.type == "cuda":
            d.update(name=self.torch.cuda.get_device_name(self.dev),
                     allow_tf32=bool(self.torch.backends.cuda.matmul.allow_tf32),
                     float32_matmul_precision=self.torch.get_float32_matmul_precision())
        return d


def make_backend(cfg):
    b = cfg["backend"]
    if b == "fixed_c":
        return FixedC(cfg)
    if b == "numpy":
        return NumpyGemm(cfg)
    if b == "numpy_rowwise":
        return NumpyRowwise(cfg)
    if b.startswith("torch_"):
        return TorchGemm(cfg, b[len("torch_"):])
    raise SystemExit(f"unknown backend {b} (choose from {', '.join(BACKENDS)})")


# =====================================================================================================================
# data
# =====================================================================================================================
_DATA: dict = {}


def load_bench(data):
    key = ("bench", data)
    if key not in _DATA:
        qz, tz, pz = (os.path.join(data, f) for f in ("queries.L2m_B8_s0.npz", "targets.L2m_B8_s0.npz",
                                                      "shipped_panel_codes.npz"))
        Q, T, P = np.load(qz), np.load(tz), np.load(pz)
        B = dict(q=npz_member(qz, "q"), qoff=Q["offsets"], qlen=Q["lengths"], qids=Q["ids"].astype(str),
                 codes=npz_member(tz, "codes"), toff=T["offsets"], tlen=T["lengths"],
                 pcodes=npz_member(pz, "codes"), poff=P["offsets"], plen=P["lengths"])
        assert np.array_equal(np.diff(B["qoff"]), B["qlen"]) and np.array_equal(np.diff(B["toff"]), B["tlen"])
        assert np.array_equal(np.diff(B["poff"]), B["plen"]) and len(B["plen"]) == 1000
        _DATA[key] = B
    return _DATA[key]


def load_a1(data):
    key = ("a1", data)
    if key not in _DATA:
        p = os.path.join(data, "a1.npz")
        z = np.load(p)
        A = {k: z[k] for k in z.files if k not in ("q_pool", "c_pool")}
        A["q_pool"], A["c_pool"] = npz_member(p, "q_pool"), npz_member(p, "c_pool")
        A["set_names"] = A["set_names"].astype(str)
        A["meta"] = json.loads(str(A.pop("meta_json")))
        _DATA[key] = A
    return _DATA[key]


def a1_pair(A, p):
    iq, it = int(A["pair_q"][p]), int(A["pair_t"][p])
    return (np.asarray(A["q_pool"][A["q_off"][iq]:A["q_off"][iq + 1]]),
            np.asarray(A["c_pool"][A["c_off"][it]:A["c_off"][it + 1]]))


def a1_ref(A, p):
    return A["ref_flat"][A["ref_off"][p]:A["ref_off"][p + 1]]


def a1_stored(A, p):
    return A["stored_flat"][A["stored_off"][p]:A["stored_off"][p + 1]]


def load_q150(data):
    with open(os.path.join(data, "q150.json")) as f:
        return json.load(f)


def query_rows(spec, data):
    B = load_bench(data)
    nq = len(B["qlen"])
    if spec == "all":
        return np.arange(nq)
    if spec.startswith("q150"):
        rows = np.array(load_q150(data)["rows"], np.int64)
        if ":" in spec:
            rows = rows[spread(B["qlen"][rows], int(spec.split(":", 1)[1]))]
        return np.sort(rows)
    if spec.startswith("first:"):
        return np.arange(min(int(spec[6:]), nq))
    if spec.startswith("rows:"):
        return np.array(sorted({int(x) for x in spec[5:].split(",") if x}), np.int64)
    raise SystemExit(f"bad --queries {spec}")


def a1_selection(A, limit):
    P = len(A["pair_set"])
    if not limit:
        return np.arange(P)
    sets = np.unique(A["pair_set"])
    per = [limit // len(sets) + (1 if k < limit % len(sets) else 0) for k in range(len(sets))]
    sel = [np.flatnonzero(A["pair_set"] == s)[:c] for s, c in zip(sets, per)]
    return np.sort(np.concatenate(sel))


def ship_table(data):
    with open(os.path.join(data, "ship_final.json")) as f:
        S = json.load(f)
    t = S["tables"][SHIP_KEY]["all"]
    return dict(lnc=np.log(np.asarray(t["center_len"], np.float64)), lam=np.asarray(t["lam"], np.float64),
                lnK=np.asarray(t["lnK"], np.float64), N=float(S["meta"]["N_ann_fit_db"]), version=S["meta"]["version"])


def ln_evalue(tab, m, s_prime):
    lm = np.log(np.asarray(m, np.float64))
    lam, lnK = np.interp(lm, tab["lnc"], tab["lam"]), np.interp(lm, tab["lnc"], tab["lnK"])
    return lnK[..., None] + lm[..., None] + math.log(tab["N"]) - lam[..., None] * s_prime


# =====================================================================================================================
# blocks
# =====================================================================================================================
def blocks_prod(lengths, n, bf):
    """spike_lib.blocks / codepath._blocks: a candidate opens a new block when the current one is non-empty and would
    exceed bf similarity floats."""
    out, k0, acc = [], 0, 0
    for k, m in enumerate(np.asarray(lengths).tolist()):
        if k > k0 and acc + n * m > bf:
            out.append((k0, k))
            k0, acc = k, 0
        acc += n * m
    out.append((k0, len(lengths)))
    return out


def blocks_research(off, n, cells):
    """evalue/score/score.py chunks(): Lq x residues <= cells per GEMM (budget = cells // n residues)."""
    K = len(off) - 1
    budget = max(cells // n, 1)
    out, j = [], 0
    while j < K:
        k = int(np.searchsorted(off, off[j] + budget, side="right")) - 1
        k = min(max(k, j + 1), K)
        out.append((j, k))
        j = k
    return out


def make_blocks(cfg, off, n):
    if cfg["block_rule"] == "research":
        return blocks_research(off, n, cfg["block_floats"])
    return blocks_prod(np.diff(off), n, cfg["block_floats"])


# =====================================================================================================================
# sim: workers
# =====================================================================================================================
_G: dict = {}


def _setup(cfg):
    _G.clear()
    _G["cfg"] = cfg
    _G["SL"], _G["F"] = spike(cfg["lib"])
    _G["be"] = make_backend(cfg)
    F = _G["F"]
    _G["qprep"] = (lambda q16: F.query_prep(q16)) if cfg["qprep"] == "c" else qprep_numpy
    if cfg["work"] == "search":
        _G["B"] = load_bench(cfg["data"])
    elif cfg["work"] == "a1":
        _G["A1"] = load_a1(cfg["data"])


def _worker_init(cfg, counter):
    with counter.get_lock():
        wid = counter.value
        counter.value += 1
    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(THREADS)
    except Exception:
        pass
    _setup(dict(cfg, device_index=wid))


def _rep_stats():
    return dict(blocks=0, blocks_diff=0, cells=0, cells_diff=0, max_abs=0.0, max_ulp=0, cand=0, sf_diff=0,
                sr_diff=0, max_abs_sf=0.0, max_abs_sr=0.0, paths_diff=0)


def _repeat_check(rep, qh, codes, col, sim, sf, sr, path=None):
    F, be = _G["F"], _G["be"]
    for v in rep["variants"]:
        s2 = be.sim(qh, codes, v, col)
        st = rep["stats"].setdefault(v, _rep_stats())
        st["blocks"] += 1
        st["cells"] += int(sim.size)
        st["cand"] += len(col) - 1
        ne = ~bits_equal(sim, s2)
        nd = int(ne.sum())
        if not nd:
            continue
        st["blocks_diff"] += 1
        st["cells_diff"] += nd
        st["max_abs"] = max(st["max_abs"], float(np.abs(sim[ne].astype(np.float64) - s2[ne]).max()))
        st["max_ulp"] = max(st["max_ulp"], int(ulp_diff(sim[ne], s2[ne]).max()))
        a, b = F.score_fwd_rev(np.ascontiguousarray(s2), col)
        st["sf_diff"] += int((~bits_equal(a, sf)).sum())
        st["sr_diff"] += int((~bits_equal(b, sr)).sum())
        st["max_abs_sf"] = max(st["max_abs_sf"], float(np.abs(a.astype(np.float64) - sf).max()))
        st["max_abs_sr"] = max(st["max_abs_sr"], float(np.abs(b.astype(np.float64) - sr).max()))
        if path is not None:
            _, _, pf, po = F.align(np.ascontiguousarray(s2), col)
            st["paths_diff"] += int(not np.array_equal(pf[po[0]:po[1]], path))


def _score_set(qh, n, codes, off, sample=False, rep=None):
    """S_f, S_r of one query against every candidate of off (block by block)."""
    F, be, cfg = _G["F"], _G["be"], _G["cfg"]
    K = len(off) - 1
    sf, sr = np.empty(K, np.float32), np.empty(K, np.float32)
    blks = make_blocks(cfg, off, n)
    keep = {0, len(blks) // 2} if sample else set()
    kept, tsim, tsc, cells = [], 0.0, 0.0, 0
    for bi, (k0, k1) in enumerate(blks):
        j0, j1 = int(off[k0]), int(off[k1])
        col = (off[k0:k1 + 1] - j0).astype(np.int64)
        cb = codes[j0:j1]
        t = time.perf_counter()
        sim = be.sim(qh, cb, None, col)
        t2 = time.perf_counter()
        assert sim.shape == (n, j1 - j0) and sim.dtype == np.float32 and sim.flags.c_contiguous, (sim.shape, sim.dtype)
        assert np.isfinite(sim).all(), "non-finite similarity"
        a, b = F.score_fwd_rev(sim, col)
        tsim += t2 - t
        tsc += time.perf_counter() - t2
        cells += sim.size
        sf[k0:k1], sr[k0:k1] = a, b
        if bi in keep:
            kept.append((k0, k1, sim.copy()))
        if rep is not None:
            _repeat_check(rep, qh, cb, col, sim, a, b)
    return sf, sr, kept, dict(blocks=len(blks), cells=cells, t_sim=tsim, t_score=tsc)


def _search_query(row):
    cfg, B, be = _G["cfg"], _G["B"], _G["be"]
    part = os.path.join(cfg["parts"], f"q{row:05d}.npz")
    if os.path.exists(part):
        return row, None
    t0 = time.perf_counter()
    q16 = np.asarray(B["q"][B["qoff"][row]:B["qoff"][row + 1]])
    qp = _G["qprep"](q16)
    assert np.isfinite(qp).all(), f"non-finite query_prep (query row {row})"
    qh = be.prepare(qp)
    n = len(qp)
    tq = time.perf_counter() - t0
    rep = dict(variants=cfg["repeat"], stats={}) if (cfg["repeat"] and row in cfg["repeat_rows"]) else None
    sample = row in cfg["sample_rows"]
    sf, sr, kept, tt = _score_set(qh, n, B["codes"], B["toff"][:cfg["nt"] + 1], sample, rep)
    blob = dict(sf=sf, sr=sr)
    tp = None
    if cfg["panel"]:
        pf, pr, _, tp = _score_set(qh, n, B["pcodes"], B["poff"])
        blob.update(pf=pf, pr=pr)
    if kept:
        blob.update(blk_k0=np.array([k[0] for k in kept]), blk_k1=np.array([k[1] for k in kept]),
                    blk_w=np.array([k[2].shape[1] for k in kept]),
                    blk_sim=np.concatenate([k[2].ravel() for k in kept]), qp=qp)
    st = dict(row=int(row), n=n, t_qprep=tq, targets=tt, panel=tp, wall=time.perf_counter() - t0,
              repeat=rep["stats"] if rep else None, pid=os.getpid())
    blob["stats_json"] = np.array(json.dumps(st))
    tmp = part[:-4] + f".tmp{os.getpid()}.npz"
    np.savez(tmp, **blob)
    os.replace(tmp, part)
    return row, st


def _a1_chunk(ci):
    cfg, A, be, F = _G["cfg"], _G["A1"], _G["be"], _G["F"]
    part = os.path.join(cfg["parts"], f"a1_{ci:05d}.npz")
    if os.path.exists(part):
        return ci, None
    t0 = time.perf_counter()
    idx = np.asarray(cfg["a1_chunks"][ci], np.int64)
    K = len(idx)
    sf, sr, tb = np.empty(K, np.float32), np.empty(K, np.float32), np.empty(K, np.float32)
    ends = np.zeros((K, 2), np.int32)
    paths, rep = [], (dict(variants=cfg["repeat"], stats={}) if cfg["repeat"] else None)
    tsim = 0.0
    for k, p in enumerate(idx):
        q16, codes = a1_pair(A, p)
        qp = _G["qprep"](q16)
        assert np.isfinite(qp).all(), f"non-finite query_prep (a1 pair {p})"
        qh = be.prepare(qp)
        col = np.array([0, len(codes)], np.int64)
        t = time.perf_counter()
        sim = be.sim(qh, codes, None, col)
        tsim += time.perf_counter() - t
        assert sim.shape == (len(qp), len(codes)) and sim.flags.c_contiguous and np.isfinite(sim).all()
        a, b = F.score_fwd_rev(sim, col)
        sc, e, pf, po = F.align(sim, col)
        sf[k], sr[k], tb[k], ends[k] = a[0], b[0], sc[0], e[0]
        path = pf[po[0]:po[1]].copy()
        paths.append(path)
        if rep is not None:
            _repeat_check(rep, qh, codes, col, sim, a, b, path)
    off = np.zeros(K + 1, np.int64)
    off[1:] = np.cumsum([len(x) for x in paths])
    flat = np.concatenate(paths) if off[-1] else np.zeros((0, 2), np.int32)
    st = dict(chunk=int(ci), pairs=K, t_sim=tsim, wall=time.perf_counter() - t0, repeat=rep["stats"] if rep else None,
              pid=os.getpid())
    tmp = part[:-4] + f".tmp{os.getpid()}.npz"
    np.savez(tmp, idx=idx, sf=sf, sr=sr, tb=tb, ends=ends, path_flat=flat.astype(np.int32), path_off=off,
             stats_json=np.array(json.dumps(st)))
    os.replace(tmp, part)
    return ci, st


def _run_pool(cfg, func, items, nwork):
    """items in order (longest first); returns stats of the computed items (None for resumed ones)."""
    out, t, done = [], time.time(), 0
    if nwork <= 1:
        _setup(dict(cfg, device_index=0))
        it = map(func, items)
        pool = None
    else:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        counter = ctx.Value("i", 0)
        pool = ctx.Pool(nwork, initializer=_worker_init, initargs=(cfg, counter))
        it = pool.imap_unordered(func, items, chunksize=1)
    try:
        for key, st in it:
            done += 1
            if st is not None:
                out.append(st)
            if done % max(1, len(items) // 20) == 0 or done == len(items):
                log(f"  {done}/{len(items)} done, {time.time() - t:.0f}s")
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    return out


def _merge_repeat(stats_list):
    agg = {}
    for st in stats_list:
        for v, s in (st.get("repeat") or {}).items():
            a = agg.setdefault(v, _rep_stats())
            for k, x in s.items():
                a[k] = max(a[k], x) if k.startswith("max") else a[k] + x
    return agg


def cmd_sim(a):
    if a.backend not in BACKENDS:
        raise SystemExit(f"unknown backend {a.backend} (choose from {', '.join(BACKENDS)})")
    data = os.path.abspath(a.data)
    lib = os.path.abspath(a.lib or default_lib())
    cfg = dict(backend=a.backend, qprep=a.qprep, work=a.work, data=data, lib=lib, threads=THREADS,
               block_floats=int(a.block_floats), block_rule=a.block_rule, panel=not a.no_panel)
    # availability (fails clearly before any work)
    try:
        be = make_backend(dict(cfg, device_index=0))
        be_info = be.info()
        variants_ok = be.variants
        del be
    except Unavailable as e:
        print(f"BACKEND UNAVAILABLE: {a.backend}: {e}", file=sys.stderr)
        sys.exit(2)
    rep = [] if a.repeat in ("", "none") else (list(variants_ok) if a.repeat == "all" else a.repeat.split(","))
    bad = [v for v in rep if v not in variants_ok]
    if bad:
        raise SystemExit(f"--repeat {bad} not supported by {a.backend} (supported: {variants_ok})")
    cfg["repeat"] = rep
    host_tag = a.host_tag or f"{platform.system().lower()}-{platform.machine()}-{socket.gethostname().split('.')[0]}"
    lab = [a.backend, f"qp{a.qprep}", f"t{THREADS}"]
    if a.block_rule != "prod" or a.block_floats != BLOCK_FLOATS:
        lab.append(f"{a.block_rule}{a.block_floats}")
    if rep:
        lab.append("rep")
    if a.tag:
        lab.append(a.tag)
    label = a.label or "-".join(lab)
    if a.work == "search":
        rows = query_rows(a.queries, data)
        B = load_bench(data)
        nt = min(int(a.tmax or len(B["tlen"])), len(B["tlen"]))
        wdir = f"search-{a.queries.replace(':', '_').replace(',', '_')}-T{nt}" + ("" if cfg["panel"] else "-nopanel")
    else:
        A1 = load_a1(data)
        sel = a1_selection(A1, a.a1_limit)
        wdir = f"a1-{a.a1_limit or 'all'}"
    out = os.path.abspath(a.out or os.path.join(ROOT, "runs", host_tag, wdir, label))
    final = os.path.join(out, "search.npz" if a.work == "search" else "a1.npz")
    if os.path.exists(final) and not a.force:
        raise SystemExit(f"{final} exists (use --force to recompute)")
    if a.force and os.path.isdir(out):
        if not os.path.exists(os.path.join(out, "meta.json")):
            raise SystemExit(f"--force: {out} is not a blas_div run dir (no meta.json); not removing it")
        shutil.rmtree(out)
    parts = os.path.join(out, "parts")
    os.makedirs(parts, exist_ok=True)
    cfg["parts"] = parts
    meta = dict(what="blas_div sim run", created=time.strftime("%Y-%m-%d %H:%M:%S"), status="running",
                argv=sys.argv, cfg=dict(cfg), host_tag=host_tag, label=label, host=host_info(), env=env_info(),
                numpy=numpy_info(), backend_info=be_info, spike=spike_info(lib),
                data=dict(dir=data, manifest_md5=(md5(os.path.join(data, "MANIFEST.json"))
                                                  if os.path.exists(os.path.join(data, "MANIFEST.json")) else None)),
                setting_D=SETTING_D)
    if a.backend.startswith("torch_"):
        meta["torch"] = torch_info()
    t0 = time.time()
    if a.work == "search":
        lens = B["qlen"][rows]
        cfg["nt"] = nt
        cfg["sample_rows"] = set(rows[spread(lens, a.sample_queries)].tolist()) if a.sample_queries else set()
        cfg["repeat_rows"] = (set(rows.tolist()) if not a.repeat_queries
                              else set(rows[spread(lens, a.repeat_queries)].tolist()))
        meta["work"] = dict(kind="search", queries=a.queries, rows=rows.tolist(), n_queries=len(rows), nt=nt,
                            panel=cfg["panel"], rows_sha256=hashlib.sha256(np.asarray(rows, np.int64).tobytes()).hexdigest(),
                            q150_md5=md5(os.path.join(data, "q150.json")) if os.path.exists(
                                os.path.join(data, "q150.json")) else None,
                            sample_rows=sorted(cfg["sample_rows"]), repeat_rows=len(cfg["repeat_rows"]),
                            residues=int(lens.sum()))
        jdump(meta, os.path.join(out, "meta.json"))
        order = [int(r) for r in rows[np.argsort(-lens, kind="stable")]]
        log(f"sim {a.backend} qprep={a.qprep} search {len(rows)} queries x {nt} targets"
            f"{' + panel' if cfg['panel'] else ''}; {a.workers} workers x {THREADS} threads -> {out}")
        stats = _run_pool(cfg, _search_query, order, a.workers)
        _merge_search(out, parts, rows, B, cfg)
    else:
        chunks = [sel[i:i + a.chunk] for i in range(0, len(sel), a.chunk)]
        cfg["a1_chunks"] = [c.tolist() for c in chunks]
        meta["work"] = dict(kind="a1", limit=a.a1_limit, n_pairs=len(sel), n_chunks=len(chunks),
                            sel_sha256=hashlib.sha256(np.asarray(sel, np.int64).tobytes()).hexdigest())
        jdump(meta, os.path.join(out, "meta.json"))
        log(f"sim {a.backend} qprep={a.qprep} a1 {len(sel)} pairs in {len(chunks)} chunks; {a.workers} workers -> {out}")
        cost = [-sum(int(A1["q_off"][A1["pair_q"][p] + 1] - A1["q_off"][A1["pair_q"][p]]) *
                     int(A1["c_off"][A1["pair_t"][p] + 1] - A1["c_off"][A1["pair_t"][p]]) for p in c) for c in chunks]
        order = [int(i) for i in np.argsort(cost, kind="stable")]
        stats = _run_pool(cfg, _a1_chunk, order, a.workers)
        _merge_a1(out, parts, sel, len(chunks))
    # timing / repeat summaries from the part files (also covers resumed parts)
    allst = []
    for f in sorted(os.listdir(parts)):
        if f.endswith(".npz"):
            allst.append(json.loads(str(np.load(os.path.join(parts, f))["stats_json"])))
    meta = json.load(open(os.path.join(out, "meta.json")))
    meta["timing"] = dict(wall_s=time.time() - t0, computed_items=len(stats), items=len(allst),
                          worker_sum_s=float(sum(s["wall"] for s in allst)),
                          t_sim_s=float(sum((s.get("targets") or {}).get("t_sim", 0) + (s.get("panel") or {}).get(
                              "t_sim", 0) for s in allst)) if a.work == "search" else float(
                              sum(s["t_sim"] for s in allst)),
                          t_score_s=float(sum((s.get("targets") or {}).get("t_score", 0) + (s.get("panel") or {}).get(
                              "t_score", 0) for s in allst)) if a.work == "search" else None,
                          workers=a.workers)
    if rep:
        r = _merge_repeat(allst)
        jdump(dict(variants=rep, stats=r, rows_or_pairs=len(allst), backend=a.backend, qprep=a.qprep),
              os.path.join(out, "repeat.json"))
        meta["repeat"] = r
        log("repeat:", json.dumps({v: {k: r[v][k] for k in ("blocks", "blocks_diff", "cells_diff", "max_ulp",
                                                            "sf_diff", "sr_diff", "paths_diff")} for v in r}))
    meta["status"] = "done"
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    jdump(meta, os.path.join(out, "meta.json"))
    if not a.keep_parts:
        shutil.rmtree(parts)
    log(f"done: {out}")


def _merge_search(out, parts, rows, B, cfg):
    nq, nt = len(rows), cfg["nt"]
    Sf, Sr = np.empty((nq, nt), np.float32), np.empty((nq, nt), np.float32)
    Pf = np.empty((nq, 1000), np.float32) if cfg["panel"] else None
    Pr = np.empty((nq, 1000), np.float32) if cfg["panel"] else None
    blk = dict(row=[], k0=[], k1=[], n=[], w=[], sim=[], qp_rows=[], qp=[])
    for i, r in enumerate(rows):
        z = np.load(os.path.join(parts, f"q{int(r):05d}.npz"))
        Sf[i], Sr[i] = z["sf"], z["sr"]
        if cfg["panel"]:
            Pf[i], Pr[i] = z["pf"], z["pr"]
        if "blk_k0" in z.files:
            n = int(B["qlen"][r])
            for k0, k1, w in zip(z["blk_k0"], z["blk_k1"], z["blk_w"]):
                blk["row"].append(int(r)); blk["k0"].append(int(k0)); blk["k1"].append(int(k1))
                blk["n"].append(n); blk["w"].append(int(w))
            blk["sim"].append(z["blk_sim"])
            blk["qp_rows"].append(int(r)); blk["qp"].append(z["qp"])
    blob = dict(rows=np.asarray(rows, np.int64), q_ids=B["qids"][rows], q_len=B["qlen"][rows], nt=np.int64(nt),
                Sf=Sf, Sr=Sr)
    if cfg["panel"]:
        blob.update(Pf=Pf, Pr=Pr)
    np.savez(os.path.join(out, "search.npz.tmp.npz"), **blob)
    os.replace(os.path.join(out, "search.npz.tmp.npz"), os.path.join(out, "search.npz"))
    if blk["row"]:
        sizes = np.array(blk["n"]) * np.array(blk["w"])
        off = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        qoff = np.concatenate([[0], np.cumsum([len(q) for q in blk["qp"]])]).astype(np.int64)
        np.savez(os.path.join(out, "simblocks.npz"), row=np.array(blk["row"]), k0=np.array(blk["k0"]),
                 k1=np.array(blk["k1"]), n=np.array(blk["n"]), w=np.array(blk["w"]), off=off,
                 sim=np.concatenate(blk["sim"]).astype(np.float32), qp_rows=np.array(blk["qp_rows"]),
                 qp_off=qoff, qp=np.concatenate(blk["qp"]).astype(np.float32),
                 block_rule=np.array(cfg["block_rule"]), block_floats=np.int64(cfg["block_floats"]))


def _merge_a1(out, parts, sel, nchunks):
    P = len(sel)
    pos = {int(p): k for k, p in enumerate(sel)}
    sf, sr, tb = np.empty(P, np.float32), np.empty(P, np.float32), np.empty(P, np.float32)
    ends = np.zeros((P, 2), np.int32)
    paths = [None] * P
    for ci in range(nchunks):
        z = np.load(os.path.join(parts, f"a1_{ci:05d}.npz"))
        po, pf = z["path_off"], z["path_flat"]
        for k, p in enumerate(z["idx"]):
            j = pos[int(p)]
            sf[j], sr[j], tb[j], ends[j] = z["sf"][k], z["sr"][k], z["tb"][k], z["ends"][k]
            paths[j] = pf[po[k]:po[k + 1]]
    off = np.zeros(P + 1, np.int64)
    off[1:] = np.cumsum([len(x) for x in paths])
    flat = np.concatenate(paths).astype(np.int32) if off[-1] else np.zeros((0, 2), np.int32)
    np.savez(os.path.join(out, "a1.npz.tmp.npz"), pairs=np.asarray(sel, np.int64), Sf=sf, Sr=sr, tb=tb, ends=ends,
             path_flat=flat, path_off=off)
    os.replace(os.path.join(out, "a1.npz.tmp.npz"), os.path.join(out, "a1.npz"))


# =====================================================================================================================
# float64 oracle
# =====================================================================================================================
def oracle_sim(q16, codes):
    q = np.asarray(q16, np.float64)
    nrm = np.sqrt((q * q).sum(1))
    qp = np.where(nrm[:, None] < 1e-12, 0.0, q / (8.0 * np.where(nrm < 1e-12, 1.0, nrm))[:, None])
    return qp @ decode(codes).astype(np.float64).T


def oracle_z(s):
    """S7 rowcol z in float64 (population std, eps 1e-8, degenerate row / column (std < 1e-6) contributes 0)."""
    mu, sd = s.mean(1, keepdims=True), s.std(1, keepdims=True)
    nu, tau = s.mean(0, keepdims=True), s.std(0, keepdims=True)
    rz = np.where(sd < 1e-6, 0.0, (s - mu) / (sd + 1e-8))
    cz = np.where(tau < 1e-6, 0.0, (s - nu) / (tau + 1e-8))
    return 0.5 * (rz + cz)


def oracle_dp(cell, go=SETTING_D[1], ge=SETTING_D[2]):
    """Float64 3-state Gotoh local DP with the spike kernel's recurrences, tie rules and traceback (fmsw_spike.c
    ad_core / traceback): X = max(H_up - go, X_up - ge); V = max(X, H_diag + s, 0); Y = max(V_left - go, Y_left - ge);
    H = max(V, Y); best = first maximal H > 0 in row-major order.  Returns score, end (1-based), forward path [L, 2]
    (-1 = gap), best-cell margin (H_best - max H over every other cell) and traceback margin (min |a - b| over the
    binary decisions taken along the traceback: H<-Y vs V, start (v0 <= 0), X vs M, X extend vs open, Y extend vs open).
    """
    n, m = cell.shape
    NEG = -np.inf
    H = np.zeros((n + 1, m + 1))
    X = np.full((n + 1, m + 1), NEG)
    V = np.zeros((n + 1, m + 1))
    Y = np.full((n + 1, m + 1), NEG)
    MM = np.full((n + 1, m + 1), NEG)
    ar = np.arange(m, dtype=np.float64)
    for i in range(1, n + 1):
        x = np.maximum(H[i - 1, 1:] - go, X[i - 1, 1:] - ge)
        mm = H[i - 1, :-1] + cell[i - 1]
        v = np.maximum(np.maximum(x, mm), 0.0)
        t = np.concatenate([[0.0], v[:-1]]) + ar * ge          # V[i, k] + k ge, k = 0 .. m-1 (V[i, 0] = 0)
        y = np.maximum.accumulate(t) - go - ar * ge             # Y[i, j] = max_k<j V[i, k] - go - (j-1-k) ge
        H[i, 1:], X[i, 1:], V[i, 1:], Y[i, 1:], MM[i, 1:] = np.maximum(v, y), x, v, y, mm
    Hc = H[1:, 1:]
    idx = int(np.argmax(Hc))
    best = float(Hc.flat[idx])
    if best <= 0:
        return dict(score=0.0, end=(0, 0), path=np.zeros((0, 2), np.int32), margin_best=0.0, margin_tb=math.inf)
    bi, bj = divmod(idx, m)
    tmp = Hc.copy()
    tmp[bi, bj] = NEG
    margin_best = best - float(tmp.max()) if tmp.size > 1 else math.inf
    bi, bj = bi + 1, bj + 1
    i, j, state, path, mt = bi, bj, 0, [], math.inf

    def mg(a, b):
        return abs(a - b) if (math.isfinite(a) and math.isfinite(b)) else math.inf
    while True:
        bnd = i <= 0 or j <= 0
        if state == 0:
            if bnd:
                state = 1
                continue
            mt = min(mt, mg(Y[i, j], V[i, j]))
            state = 4 if Y[i, j] > V[i, j] else 1
        elif state == 1:
            if bnd:
                break
            v0 = max(X[i, j], MM[i, j])
            mt = min(mt, abs(v0))
            if v0 <= 0:
                break
            mt = min(mt, mg(X[i, j], MM[i, j]))
            state = 3 if X[i, j] > MM[i, j] else 2
        elif state == 2:
            path.append((i - 1, j - 1))
            i, j, state = i - 1, j - 1, 0
        elif state == 3:
            path.append((i - 1, -1))
            xo, xe = H[i - 1, j] - go, X[i - 1, j] - ge
            mt = min(mt, mg(xe, xo))
            state = 3 if xe > xo else 0
            i -= 1
        else:
            path.append((-1, j - 1))
            yo, ye = V[i, j - 1] - go, Y[i, j - 1] - ge
            mt = min(mt, mg(ye, yo))
            state = 4 if ye > yo else 1
            j -= 1
    return dict(score=best, end=(bi, bj), path=np.array(path[::-1], np.int32).reshape(-1, 2), margin_best=margin_best,
                margin_tb=mt)


def matched(path):
    path = np.asarray(path).reshape(-1, 2)
    return path[(path[:, 0] >= 0) & (path[:, 1] >= 0)]


def prf(pred, ref):
    """harness.prf: precision, recall, F1 of aligned residue pairs (exact match); two empty alignments -> 1."""
    n_p, n_r = len(pred), len(ref)
    if n_p + n_r == 0:
        return 1.0, 1.0, 1.0
    cp = pred[:, 0].astype(np.int64) * 1_000_000 + pred[:, 1]
    cr = ref[:, 0].astype(np.int64) * 1_000_000 + ref[:, 1]
    tp = np.intersect1d(cp, cr).size
    return (tp / n_p if n_p else 0.0), (tp / n_r if n_r else 0.0), 2.0 * tp / (n_p + n_r)


# =====================================================================================================================
# statistics helpers
# =====================================================================================================================
TH_ABS = (1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


class Hist:
    """Streaming summary of non-negative values: exact n / zeros / max / mean / exceedance counts, percentiles from
    a log histogram (50 bins per decade, 1e-15 .. 1e4; a percentile is the upper edge of its bin, <= 4.7% high)."""
    EDGES = 10.0 ** (np.arange(-15 * 50, 4 * 50 + 1) / 50.0)

    def __init__(self):
        self.n = self.zeros = 0
        self.max, self.sum = 0.0, 0.0
        self.counts = np.zeros(len(self.EDGES) + 1, np.int64)
        self.exceed = np.zeros(len(TH_ABS), np.int64)

    def add(self, x):
        x = np.abs(np.asarray(x, np.float64)).ravel()
        if not x.size:
            return
        self.n += x.size
        z = x == 0
        self.zeros += int(z.sum())
        self.max = max(self.max, float(x.max()))
        self.sum += float(x.sum())
        nz = x[~z]
        self.counts += np.bincount(np.searchsorted(self.EDGES, nz, side="right"), minlength=len(self.counts))
        self.exceed += np.array([(nz > t).sum() for t in TH_ABS], np.int64)

    def pct(self, q):
        if not self.n:
            return None
        k = q * self.n
        if k <= self.zeros:
            return 0.0
        c = self.zeros + np.cumsum(self.counts)
        b = int(np.searchsorted(c, k))
        return float(self.EDGES[min(b, len(self.EDGES) - 1)])

    def summary(self):
        if not self.n:
            return dict(n=0)
        return dict(n=self.n, frac_zero=self.zeros / self.n, max=self.max, mean=self.sum / self.n,
                    p50=self.pct(.5), p90=self.pct(.9), p95=self.pct(.95), p99=self.pct(.99), p999=self.pct(.999),
                    p9999=self.pct(.9999), frac_gt={f"{t:g}": float(c) / self.n for t, c in zip(TH_ABS, self.exceed)})


class UlpHist:
    def __init__(self):
        self.counts = np.zeros(40, np.int64)
        self.max = 0

    def add(self, u):
        u = np.asarray(u).ravel()
        if not u.size:
            return
        cat = np.zeros(u.size, np.int64)
        nz = u > 0
        cat[nz] = 1 + np.ceil(np.log2(u[nz].astype(np.float64))).astype(np.int64)
        self.counts += np.bincount(np.minimum(cat, 39), minlength=40)
        self.max = max(self.max, int(u.max()))

    def summary(self):
        n = int(self.counts.sum())
        if not n:
            return dict(n=0)
        c = np.cumsum(self.counts) / n
        return dict(n=n, max=self.max, frac_eq=float(c[0]), frac_le1=float(c[1]), frac_le2=float(c[2]),
                    frac_le4=float(c[3]), frac_le16=float(c[5]), frac_le256=float(c[9]), frac_le65536=float(c[17]))


def exact_stats(d):
    d = np.abs(np.asarray(d, np.float64)).ravel()
    if not d.size:
        return dict(n=0)
    return dict(n=int(d.size), frac_zero=float((d == 0).mean()), max=float(d.max()), mean=float(d.mean()),
                median=float(np.median(d)), p90=float(np.percentile(d, 90)), p95=float(np.percentile(d, 95)),
                p99=float(np.percentile(d, 99)))


class PairStats:
    """abs / floored-rel (|d| / max(|ref|, 1)) / pure-rel (|d| / |ref|, ref != 0) / ulp (float32 inputs)."""

    def __init__(self, ulp=False):
        self.abs, self.rel, self.relp = Hist(), Hist(), Hist()
        self.ulp = UlpHist() if ulp else None
        self.bit_eq = self.n = 0
        self.zero_ref_nonzero_diff = 0

    def add(self, ref, x):
        r64, x64 = np.asarray(ref, np.float64), np.asarray(x, np.float64)
        d = np.abs(x64 - r64)
        self.n += d.size
        if self.ulp is not None:
            self.bit_eq += int(bits_equal(ref, x).sum())
            self.ulp.add(ulp_diff(ref, x))
        else:
            self.bit_eq += int((d == 0).sum())
        self.abs.add(d)
        self.rel.add(d / np.maximum(np.abs(r64), 1.0))
        nz = r64 != 0
        self.relp.add(d[nz] / np.abs(r64[nz]))
        self.zero_ref_nonzero_diff += int(((~nz) & (d > 0)).sum())

    def summary(self):
        s = dict(n=self.n, frac_bit_equal=self.bit_eq / self.n if self.n else None, abs=self.abs.summary(),
                 rel_floor1=self.rel.summary(), rel=self.relp.summary(),
                 zero_ref_nonzero_diff=self.zero_ref_nonzero_diff)
        if self.ulp is not None:
            s["ulp"] = self.ulp.summary()
        return s


# =====================================================================================================================
# compare
# =====================================================================================================================
def load_run(d):
    d = os.path.abspath(d)
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    R = dict(dir=d, meta=meta)
    for nm in ("search", "a1", "simblocks"):
        p = os.path.join(d, f"{nm}.npz")
        if os.path.exists(p):
            R[nm] = p
    if os.path.exists(os.path.join(d, "repeat.json")):
        with open(os.path.join(d, "repeat.json")) as f:
            R["repeat"] = json.load(f)
    return R


def run_brief(R):
    m = R["meta"]
    return dict(dir=R["dir"], host_tag=m.get("host_tag"), label=m.get("label"), backend=m["cfg"]["backend"],
                qprep=m["cfg"]["qprep"], threads=m["cfg"]["threads"], block_rule=m["cfg"]["block_rule"],
                block_floats=m["cfg"]["block_floats"], host=m["host"].get("hostname"), cpu=m["host"].get("cpu"),
                blas=(m.get("numpy") or {}).get("blas"), backend_info=m.get("backend_info"),
                env_blas={k: v for k, v in m["env"]["blas_related"].items()
                          if k.startswith(("OPENBLAS_CORETYPE", "MKL_", "OMP_", "KMP_", "VECLIB", "NVIDIA_TF32"))})


def _rank_order(S_row):
    return np.lexsort((np.arange(len(S_row)), -S_row))


def compare_search(A, B, data, top=(10, 100)):
    za, zb = np.load(A["search"]), np.load(B["search"])
    ra, rb = za["rows"], zb["rows"]
    common = np.intersect1d(ra, rb)
    ia = np.searchsorted(ra, common)
    ib = np.searchsorted(rb, common)
    NT = int(min(int(za["nt"]), int(zb["nt"])))
    SfA, SrA = npz_member(A["search"], "Sf"), npz_member(A["search"], "Sr")
    SfB, SrB = npz_member(B["search"], "Sf"), npz_member(B["search"], "Sr")
    Bn = load_bench(data)
    qlen = Bn["qlen"][common]
    tab = ship_table(data)
    panel = "Pf" in za.files and "Pf" in zb.files
    out = dict(n_queries=len(common), n_targets=NT, pairs=len(common) * NT, panel=panel,
               E=dict(table=SHIP_KEY, version=tab["version"], N=tab["N"]))
    st = {k: PairStats(ulp=True) for k in ("S_f", "S_r")}
    st.update({k: PairStats() for k in ("S", "S_prime")})
    dl = Hist()
    dl_cond = {f"E_ref<={t:g}": Hist() for t in (10.0, 1.0, 1e-3)}
    flips = {f"{t:g}": dict(ref_hits=0, test_hits=0, flips=0, ref_only=0, test_only=0) for t in E_THRESH}
    labels = None
    lp = os.path.join(data, "bench_labels.npz")
    if os.path.exists(lp):
        L = np.load(lp)
        labels = dict(TP=np.unpackbits(L["TP"][common], axis=1)[:, :NT].astype(bool),
                      FPA=np.unpackbits(L["FPA"][common], axis=1)[:, :NT].astype(bool))
        flips_lab = {f"{t:g}": dict(TP=0, FPA=0, other=0) for t in E_THRESH}
    rank = {f"top{k}": dict(order_diff=0, set_diff=0, max_displacement=0) for k in top}
    hitlist = {f"{t:g}": dict(queries_with_hits=0, set_diff=0, order_diff=0) for t in (1e-3, 1.0)}
    perq = dict(max_abs_dS=np.zeros(len(common)), max_abs_dlnE_hits=np.zeros(len(common)))
    q90 = None
    if panel:
        PA = npz_member(A["search"], "Pf")[ia].astype(np.float64) - npz_member(A["search"], "Pr")[ia].astype(np.float64)
        PB = npz_member(B["search"], "Pf")[ib].astype(np.float64) - npz_member(B["search"], "Pr")[ib].astype(np.float64)
        q90 = dict(A=np.quantile(PA, 0.9, axis=1), B=np.quantile(PB, 0.9, axis=1))
        dq = np.abs(q90["B"] - q90["A"])
        out["Q90"] = dict(exact_stats(dq), frac_equal=float((dq == 0).mean()),
                          panel_S_f=None)
        pst = PairStats(ulp=True)
        pst.add(npz_member(A["search"], "Pf")[ia], npz_member(B["search"], "Pf")[ib])
        out["Q90"]["panel_S_f"] = pst.summary()
        pst = PairStats()
        pst.add(PA, PB)
        out["Q90"]["panel_S"] = pst.summary()
    CH = 16
    for c0 in range(0, len(common), CH):
        sl = slice(c0, min(c0 + CH, len(common)))
        sfa, sra = np.asarray(SfA[ia[sl], :NT]), np.asarray(SrA[ia[sl], :NT])
        sfb, srb = np.asarray(SfB[ib[sl], :NT]), np.asarray(SrB[ib[sl], :NT])
        st["S_f"].add(sfa, sfb)
        st["S_r"].add(sra, srb)
        SA = sfa.astype(np.float64) - sra.astype(np.float64)
        SB = sfb.astype(np.float64) - srb.astype(np.float64)
        st["S"].add(SA, SB)
        perq["max_abs_dS"][sl] = np.abs(SB - SA).max(1)
        if panel:
            SpA, SpB = SA - q90["A"][sl, None], SB - q90["B"][sl, None]
            st["S_prime"].add(SpA, SpB)
            lA, lB = ln_evalue(tab, qlen[sl], SpA), ln_evalue(tab, qlen[sl], SpB)
            d = np.abs(lB - lA)
            dl.add(d)
            for t in (10.0, 1.0, 1e-3):
                dl_cond[f"E_ref<={t:g}"].add(d[lA <= math.log(t)])
            hit1 = lA <= 0.0
            perq["max_abs_dlnE_hits"][sl] = np.where(hit1, d, 0).max(1)
            for t in E_THRESH:
                ha, hb = lA <= math.log(t), lB <= math.log(t)
                f = flips[f"{t:g}"]
                f["ref_hits"] += int(ha.sum())
                f["test_hits"] += int(hb.sum())
                x = ha != hb
                f["flips"] += int(x.sum())
                f["ref_only"] += int((ha & ~hb).sum())
                f["test_only"] += int((hb & ~ha).sum())
                if labels is not None:
                    fl = flips_lab[f"{t:g}"]
                    tp, fpa = labels["TP"][sl], labels["FPA"][sl]
                    fl["TP"] += int((x & tp).sum())
                    fl["FPA"] += int((x & fpa).sum())
                    fl["other"] += int((x & ~tp & ~fpa).sum())
        for r in range(SA.shape[0]):
            oa, ob = _rank_order(SA[r]), _rank_order(SB[r])
            for k in top:
                R = rank[f"top{k}"]
                if not np.array_equal(oa[:k], ob[:k]):
                    R["order_diff"] += 1
                    if set(oa[:k].tolist()) != set(ob[:k].tolist()):
                        R["set_diff"] += 1
                    posb = np.empty(NT, np.int64)
                    posb[ob] = np.arange(NT)
                    R["max_displacement"] = max(R["max_displacement"], int(np.abs(posb[oa[:k]] - np.arange(k)).max()))
            if panel:
                for t in (1e-3, 1.0):
                    H = hitlist[f"{t:g}"]
                    ha = oa[lA[r][oa] <= math.log(t)]
                    hb = ob[lB[r][ob] <= math.log(t)]
                    if len(ha) or len(hb):
                        H["queries_with_hits"] += 1
                        if set(ha.tolist()) != set(hb.tolist()):
                            H["set_diff"] += 1
                        if not np.array_equal(ha, hb):
                            H["order_diff"] += 1
    out.update({k: v.summary() for k, v in st.items()})
    if panel:
        out["lnE"] = dict(abs=dl.summary(), **{k: v.summary() for k, v in dl_cond.items()})
        out["E_threshold_flips"] = flips
        if labels is not None:
            out["E_threshold_flips_by_label"] = flips_lab
        out["hit_lists"] = hitlist
    out["ranking"] = rank
    return out, dict(rows=common, max_abs_dS=perq["max_abs_dS"], max_abs_dlnE_hits=perq["max_abs_dlnE_hits"],
                     q90_A=q90["A"] if q90 else np.zeros(0), q90_B=q90["B"] if q90 else np.zeros(0))


def _paths(z):
    off, flat = z["path_off"], z["path_flat"]
    return [flat[off[k]:off[k + 1]] for k in range(len(off) - 1)]


def compare_a1(A, B, data, max_oracle=3000, oracle_all=False):
    A1 = load_a1(data)
    za, zb = dict(np.load(A["a1"])), dict(np.load(B["a1"]))
    pa, pb = za["pairs"], zb["pairs"]
    common = np.intersect1d(pa, pb)
    ia, ib = np.searchsorted(pa, common), np.searchsorted(pb, common)
    pathA, pathB = _paths(za), _paths(zb)
    sets = A1["set_names"][A1["pair_set"][common]]
    out = dict(n_pairs=len(common))
    for nm, key in (("S_f", "Sf"), ("S_r", "Sr")):
        s = PairStats(ulp=True)
        s.add(za[key][ia], zb[key][ib])
        out[nm] = s.summary()
    s = PairStats()
    s.add(za["Sf"][ia].astype(np.float64) - za["Sr"][ia], zb["Sf"][ib].astype(np.float64) - zb["Sr"][ib])
    out["S"] = s.summary()
    out["tb_eq_score_only"] = dict(A=int(bits_equal(za["tb"], za["Sf"]).sum()), B=int(bits_equal(zb["tb"], zb["Sf"]).sum()),
                                   n_A=len(pa), n_B=len(pb))
    per_set = {}
    diffs, f1A_all, f1B_all = [], [], []
    for k, p in enumerate(common):
        PA, PB = pathA[ia[k]], pathB[ib[k]]
        st = per_set.setdefault(str(sets[k]), dict(n=0, path_identical=0, aligned_pairs_identical=0,
                                                    A_eq_stored=0, B_eq_stored=0, n_ref=0, f1A=0.0, f1B=0.0,
                                                    f1_stored=0.0, f1_diff_pairs=0, max_abs_dF1=0.0))
        st["n"] += 1
        same = np.array_equal(PA, PB)
        st["path_identical"] += int(same)
        mA, mB = matched(PA), matched(PB)
        st["aligned_pairs_identical"] += int(np.array_equal(mA, mB))
        stored = a1_stored(A1, p)
        st["A_eq_stored"] += int(np.array_equal(mA, stored))
        st["B_eq_stored"] += int(np.array_equal(mB, stored))
        f1a = f1b = None
        if A1["has_ref"][p]:
            ref = a1_ref(A1, p)
            f1a, f1b = prf(mA, ref)[2], prf(mB, ref)[2]
            st["n_ref"] += 1
            st["f1A"] += f1a
            st["f1B"] += f1b
            st["f1_stored"] += float(A1["stored_sw"][p, 4])
            if f1a != f1b:
                st["f1_diff_pairs"] += 1
                st["max_abs_dF1"] = max(st["max_abs_dF1"], abs(f1b - f1a))
            f1A_all.append(f1a)
            f1B_all.append(f1b)
        if not same or oracle_all:
            ca = {tuple(x) for x in PA.tolist()}
            cb = {tuple(x) for x in PB.tolist()}
            ma = {tuple(x) for x in mA.tolist()}
            mb = {tuple(x) for x in mB.tolist()}
            diffs.append(dict(pair=int(p), set=str(sets[k]), qid=str(A1["pair_qid"][p]), tid=str(A1["pair_tid"][p]),
                              n=int(A1["q_off"][A1["pair_q"][p] + 1] - A1["q_off"][A1["pair_q"][p]]),
                              m=int(A1["c_off"][A1["pair_t"][p] + 1] - A1["c_off"][A1["pair_t"][p]]),
                              same=bool(same), Sf_A=float(za["Sf"][ia[k]]), Sf_B=float(zb["Sf"][ib[k]]),
                              end_A=za["ends"][ia[k]].tolist(), end_B=zb["ends"][ib[k]].tolist(),
                              len_A=len(PA), len_B=len(PB), n_cols_diff=len(ca ^ cb), n_pairs_diff=len(ma ^ mb),
                              f1_A=f1a, f1_B=f1b))
    for nm, st in per_set.items():
        if st["n_ref"]:
            for k in ("f1A", "f1B", "f1_stored"):
                st[k] /= st["n_ref"]
            st["f1_research_rounded"] = RESEARCH_F1.get(nm)
        else:
            for k in ("f1A", "f1B", "f1_stored"):
                st[k] = None
    out["per_set"] = per_set
    tot = {k: sum(s[k] for s in per_set.values()) for k in ("n", "path_identical", "aligned_pairs_identical",
                                                            "A_eq_stored", "B_eq_stored", "f1_diff_pairs")}
    out["totals"] = dict(tot, frac_path_identical=tot["path_identical"] / max(tot["n"], 1),
                         mean_abs_dF1=float(np.mean(np.abs(np.array(f1B_all) - np.array(f1A_all)))) if f1A_all else None,
                         max_abs_dF1=max([s["max_abs_dF1"] for s in per_set.values()] or [0.0]))
    # float64 oracle on the differing pairs
    todo = [d for d in diffs if not d["same"]]
    if len(todo) > max_oracle:
        rng = np.random.default_rng(0)
        todo = [todo[i] for i in sorted(rng.choice(len(todo), max_oracle, replace=False))]
    for d in (diffs if oracle_all else todo):
        p = d["pair"]
        q16, codes = a1_pair(A1, p)
        o = oracle_dp(oracle_z(oracle_sim(q16, codes)) - SETTING_D[0])
        kA = int(np.searchsorted(pa, p))
        kB = int(np.searchsorted(pb, p))
        d.update(oracle_score=o["score"], oracle_end=list(o["end"]), margin_best=o["margin_best"],
                 margin_tb=o["margin_tb"], margin=min(o["margin_best"], o["margin_tb"]),
                 oracle_eq_A=bool(np.array_equal(o["path"], pathA[kA])),
                 oracle_eq_B=bool(np.array_equal(o["path"], pathB[kB])))
    nd = [d for d in diffs if not d["same"]]
    withm = [d for d in nd if "margin" in d]
    out["differing_paths"] = dict(
        n=len(nd), oracle_evaluated=len(withm),
        margin_lt_1e_4=sum(d["margin"] < 1e-4 for d in withm), margin_lt_1e_3=sum(d["margin"] < 1e-3 for d in withm),
        margin_ge_1e_4_unexplained=sum(d["margin"] >= 1e-4 for d in withm),
        oracle_eq_A=sum(d["oracle_eq_A"] for d in withm), oracle_eq_B=sum(d["oracle_eq_B"] for d in withm),
        n_cols_diff=exact_stats([d["n_cols_diff"] for d in nd]) if nd else None,
        n_pairs_diff=exact_stats([d["n_pairs_diff"] for d in nd]) if nd else None,
        margins=exact_stats([d["margin"] for d in withm]) if withm else None,
        pairs=nd[:500])
    if oracle_all:
        out["oracle_all"] = dict(n=len(diffs), margins=exact_stats([d["margin"] for d in diffs]),
                                 oracle_eq_A=sum(d["oracle_eq_A"] for d in diffs),
                                 oracle_eq_B=sum(d["oracle_eq_B"] for d in diffs))
    return out


def compare_sim(A, B, data, oracle=True):
    za, zb = dict(np.load(A["simblocks"])), dict(np.load(B["simblocks"]))
    keyA = {(int(r), int(a), int(b)): k for k, (r, a, b) in enumerate(zip(za["row"], za["k0"], za["k1"]))}
    keyB = {(int(r), int(a), int(b)): k for k, (r, a, b) in enumerate(zip(zb["row"], zb["k0"], zb["k1"]))}
    common = sorted(set(keyA) & set(keyB))
    out = dict(n_blocks=len(common))
    if not common:
        return out
    Bn = load_bench(data)
    ps, ps_big = PairStats(ulp=True), PairStats(ulp=True)
    errA, errB = Hist(), Hist()
    crA = crB = cells = 0
    closer = dict(A=0, B=0, tie=0)
    for key in common:
        ka, kb = keyA[key], keyB[key]
        sa = za["sim"][za["off"][ka]:za["off"][ka + 1]]
        sb = zb["sim"][zb["off"][kb]:zb["off"][kb + 1]]
        ps.add(sa, sb)
        big = np.abs(sa) >= 1e-3
        ps_big.add(sa[big], sb[big])
        if oracle:
            r, k0, k1 = key
            q16 = Bn["q"][Bn["qoff"][r]:Bn["qoff"][r + 1]]
            s64 = oracle_sim(q16, Bn["codes"][Bn["toff"][k0]:Bn["toff"][k1]]).ravel()
            ea, eb = np.abs(sa - s64), np.abs(sb - s64)
            errA.add(ea)
            errB.add(eb)
            cr = s64.astype(np.float32)
            crA += int(bits_equal(sa, cr).sum())
            crB += int(bits_equal(sb, cr).sum())
            cells += sa.size
            closer["A"] += int((ea < eb).sum())
            closer["B"] += int((eb < ea).sum())
            closer["tie"] += int((ea == eb).sum())
    out["cells"] = ps.summary()
    out["cells_abs_ge_1e_3"] = ps_big.summary()
    if oracle:
        out["oracle"] = dict(cells=cells, A_abs_err=errA.summary(), B_abs_err=errB.summary(),
                             A_frac_correctly_rounded=crA / cells, B_frac_correctly_rounded=crB / cells,
                             closer_to_float64=closer)
    return out


def cmd_compare(a):
    A, B = load_run(a.A), load_run(a.B)
    data = os.path.abspath(a.data)
    out = dict(what="blas_div compare: divergence of run B (test) against run A (reference)",
               created=time.strftime("%Y-%m-%d %H:%M:%S"), A=run_brief(A), B=run_brief(B), host=host_info(),
               definitions=dict(rel_floor1="|B - A| / max(|A|, 1)", rel="|B - A| / |A| (A != 0)",
                                ulp="float32 ordinal distance", S="f64 S_f - f64 S_r", S_prime="S - Q90",
                                lnE="lnK(m) + ln m + ln N - lambda(m) S' (ship_final 'all', N = 8,557,236)",
                                flips="pairs with (E_A <= t) != (E_B <= t)",
                                margin=("float64 oracle (sim64 -> S7 z -> spike-recurrence DP): min(best-cell margin, "
                                        "traceback decision margin)"),
                                ranking="per query: S descending, target index ascending; top-k lists"))
    extra = {}
    if "search" in A and "search" in B:
        log("compare search ...")
        out["search"], extra = compare_search(A, B, data)
    if "a1" in A and "a1" in B:
        log("compare a1 ...")
        out["a1"] = compare_a1(A, B, data, a.max_oracle, a.oracle_all)
    if "simblocks" in A and "simblocks" in B:
        log("compare sim blocks ...")
        out["sim"] = compare_sim(A, B, data, not a.no_oracle)
    for nm, R in (("A", A), ("B", B)):
        if "repeat" in R:
            out[f"repeat_{nm}"] = R["repeat"]
    out["headline"] = headline(out)
    wa, wb = (os.path.basename(os.path.dirname(R["dir"])) for R in (A, B))
    name = a.name or (f"{wa}__{A['meta'].get('host_tag')}__{A['meta'].get('label')}__VS__"
                      + (f"{wb}__" if wb != wa else "") + f"{B['meta'].get('host_tag')}__{B['meta'].get('label')}")
    path = a.out or os.path.join(ROOT, "compare", name + ".json")
    jdump(out, path)
    if extra:
        np.savez_compressed(path[:-5] + ".npz", **extra)
    print(json.dumps(out["headline"], indent=1, default=_jdefault))
    log(f"-> {path}")


def headline(o):
    h = {}
    s = o.get("search")
    if s:
        h["search_pairs"] = s["pairs"]
        for k in ("S_f", "S_r", "S"):
            h[f"{k}_frac_bit_equal"] = s[k]["frac_bit_equal"]
            h[f"{k}_max_abs"] = s[k]["abs"]["max"]
            h[f"{k}_max_rel_floor1"] = s[k]["rel_floor1"]["max"]
        if s.get("panel"):
            h["Q90_max_abs"] = s["Q90"]["max"]
            h["lnE_max_abs"] = s["lnE"]["abs"]["max"]
            h["lnE_hits_E<=1_median_p95_max"] = [s["lnE"]["E_ref<=1"].get(k) for k in ("p50", "p95", "max")]
            h["E_flips"] = {t: s["E_threshold_flips"][t]["flips"] for t in ("0.001", "0.01", "1")}
            h["hit_list_E<=1e-3_queries_differing"] = s["hit_lists"]["0.001"]["set_diff"]
        h["top10_order_diff_queries"] = s["ranking"]["top10"]["order_diff"]
    a1 = o.get("a1")
    if a1:
        t = a1["totals"]
        h["a1_pairs"] = t["n"]
        h["a1_paths_identical"] = t["path_identical"]
        h["a1_max_abs_dF1"] = t["max_abs_dF1"]
        dp = a1["differing_paths"]
        h["a1_differing_paths"] = dp["n"]
        h["a1_differing_margin_ge_1e-4"] = dp["margin_ge_1e_4_unexplained"]
        h["a1_S_f_frac_bit_equal"] = a1["S_f"]["frac_bit_equal"]
        h["a1_S_f_max_rel_floor1"] = a1["S_f"]["rel_floor1"]["max"]
    sm = o.get("sim")
    if sm and sm.get("n_blocks"):
        h["sim_cells_frac_bit_equal"] = sm["cells"]["frac_bit_equal"]
        h["sim_max_abs"] = sm["cells"]["abs"]["max"]
        if "oracle" in sm:
            h["sim_frac_correctly_rounded_A_B"] = [sm["oracle"]["A_frac_correctly_rounded"],
                                                   sm["oracle"]["B_frac_correctly_rounded"]]
    for nm in ("A", "B"):
        r = o.get(f"repeat_{nm}")
        if r:
            h[f"repeat_{nm}_blocks_diff"] = {v: x["blocks_diff"] for v, x in r["stats"].items()}
    return h


# =====================================================================================================================
# verify
# =====================================================================================================================
def _golden_items(data):
    z = np.load(os.path.join(data, "spike_inputs", "fixture_candidate.npz"))
    qi = {x: k for k, x in enumerate(z["q_ids"].astype(str))}
    ti = {x: k for k, x in enumerate(z["t_ids"].astype(str))}
    qo, to = z["q_offsets"], z["t_offsets"]
    for a_, b_ in z["pairs"].astype(str):
        ka, kb = qi[a_], ti[b_]
        yield "F1", z["q"][qo[ka]:qo[ka + 1]], z["codes"][to[kb]:to[kb + 1]]
    g = np.load(os.path.join(data, "spike_inputs", "golden_inputs.npz"))
    for p in range(len(g["q_off"]) - 1):
        yield str(g["kind"][p]), g["q"][g["q_off"][p]:g["q_off"][p + 1]], g["codes"][g["c_off"][p]:g["c_off"][p + 1]]


def _golden_run(cfg, data):
    """verify_spike golden definition, computed through this harness's backend + qprep + score/align calls."""
    _setup(dict(cfg, work="none", device_index=0))
    F, be = _G["F"], _G["be"]
    H = hashlib.sha256()
    recs = []
    for kind, q16, codes in _golden_items(data):
        qp = _G["qprep"](q16)
        col = np.array([0, len(codes)], np.int64)
        if not np.isfinite(qp).all():
            recs.append(dict(kind=kind, n=len(q16), m=len(codes), nonfinite_qp=True))
            continue
        sim = be.sim(be.prepare(qp), codes, None, col)
        sf, sr, ef, er = F.score_fwd_rev(np.ascontiguousarray(sim, np.float32), col, ends=True)
        stb, etb, pf, po = F.align(np.ascontiguousarray(sim, np.float32), col)
        path = pf[po[0]:po[1]]
        H.update(struct.pack("<ii", len(q16), len(codes)))
        H.update(np.float32(sf[0]).tobytes() + np.float32(sr[0]).tobytes())
        H.update(np.asarray(ef[0], "<i4").tobytes() + np.asarray(er[0], "<i4").tobytes())
        H.update(struct.pack("<i", len(path)))
        H.update(np.asarray(path, "<i4").tobytes())
        recs.append(dict(kind=kind, n=len(q16), m=len(codes), sf=sf[0], sr=sr[0], path=path.copy(),
                         score_eq_tb=bool(bits_equal(stb, sf)[0])))
    return H.hexdigest(), recs


def cmd_verify(a):
    data = os.path.abspath(a.data)
    if a.what == "golden":
        lib = os.path.abspath(a.lib or default_lib())
        base = dict(backend="fixed_c", qprep="c", lib=lib, data=data, threads=THREADS, block_floats=BLOCK_FLOATS,
                    block_rule="prod")
        sha_ref, ref = _golden_run(base, data)
        out = dict(what="golden", lib=spike_info(lib), host=host_info(), fixed_c_sha256=sha_ref,
                   expected=GOLDEN_SHA, fixed_c_equals_golden=sha_ref == GOLDEN_SHA, n_pairs=len(ref),
                   score_eq_tb=sum(r.get("score_eq_tb", False) for r in ref))
        if a.backend != "fixed_c" or a.qprep != "c":
            try:
                sha, recs = _golden_run(dict(base, backend=a.backend, qprep=a.qprep), data)
            except Unavailable as e:
                raise SystemExit(f"BACKEND UNAVAILABLE: {e}")
            per = {}
            for r0, r1 in zip(ref, recs):
                k = "F1" if r0["kind"] == "F1" else ("edge" if r0["kind"].startswith("edge") else r0["kind"])
                d = per.setdefault(k, dict(n=0, nonfinite=0, sf_bit_eq=0, sr_bit_eq=0, path_eq=0, max_rel_sf=0.0,
                                           max_rel_sr=0.0))
                d["n"] += 1
                if r1.get("nonfinite_qp"):
                    d["nonfinite"] += 1
                    continue
                d["sf_bit_eq"] += int(np.float32(r0["sf"]) == np.float32(r1["sf"]))
                d["sr_bit_eq"] += int(np.float32(r0["sr"]) == np.float32(r1["sr"]))
                d["path_eq"] += int(np.array_equal(r0["path"], r1["path"]))
                d["max_rel_sf"] = max(d["max_rel_sf"], abs(float(r1["sf"]) - float(r0["sf"])) / max(abs(float(r0["sf"])), 1))
                d["max_rel_sr"] = max(d["max_rel_sr"], abs(float(r1["sr"]) - float(r0["sr"])) / max(abs(float(r0["sr"])), 1))
            out["test"] = dict(backend=a.backend, qprep=a.qprep, sha256=sha, per_kind=per,
                               numpy=numpy_info() if a.backend.startswith("numpy") else None)
        path = a.out or os.path.join(ROOT, "verify", f"golden_{a.backend}_qp{a.qprep}_{socket.gethostname().split('.')[0]}.json")
        jdump(out, path)
        print(json.dumps({k: v for k, v in out.items() if k not in ("lib", "host")}, indent=1, default=_jdefault))
        if sha_ref != GOLDEN_SHA:
            raise SystemExit("FAIL: fixed_c golden sha256 differs from the spike golden hash")
        return
    # verify run
    R = load_run(a.run)
    cfg = R["meta"]["cfg"]
    out = dict(what="verify run", run=run_brief(R), created=time.strftime("%Y-%m-%d %H:%M:%S"))
    SL, F = spike(a.lib or cfg["lib"])
    qprep = (lambda q16: F.query_prep(q16)) if cfg["qprep"] == "c" else qprep_numpy
    if "search" in R:
        z = dict(np.load(R["search"]))
        rows, nt = z["rows"], int(z["nt"])
        Sf, Sr = z["Sf"], z["Sr"]
        Bn = load_bench(data)
        if cfg["backend"] == "fixed_c":
            sel = np.arange(len(rows)) if len(rows) * nt <= a.max_cells else spread(Bn["qlen"][rows], 3)
            res = dict(rows=len(sel), sf_eq=0, sr_eq=0, pf_eq=0, pr_eq=0, n=0, np=0)
            for k in sel:
                r = int(rows[k])
                qp = qprep(Bn["q"][Bn["qoff"][r]:Bn["qoff"][r + 1]])
                off = Bn["toff"][:nt + 1]
                a_, b_ = np.empty(nt, np.float32), np.empty(nt, np.float32)
                F.list_fwd_rev(qp, Bn["codes"], off, cfg["block_floats"], a_, b_)
                res["sf_eq"] += int(bits_equal(a_, Sf[k]).sum())
                res["sr_eq"] += int(bits_equal(b_, Sr[k]).sum())
                res["n"] += nt
                if "Pf" in z:
                    a_, b_ = np.empty(1000, np.float32), np.empty(1000, np.float32)
                    F.list_fwd_rev(qp, Bn["pcodes"], Bn["poff"], cfg["block_floats"], a_, b_)
                    res["pf_eq"] += int(bits_equal(a_, z["Pf"][k]).sum())
                    res["pr_eq"] += int(bits_equal(b_, z["Pr"][k]).sum())
                    res["np"] += 1000
            res["pass"] = res["sf_eq"] == res["n"] and res["sr_eq"] == res["n"] and res["pf_eq"] == res["np"] and \
                res["pr_eq"] == res["np"]
            out["fixed_c_vs_spike_list_driver"] = res
        rp = os.path.join(data, "research_SD_q150.npz")
        if os.path.exists(rp):
            rz = np.load(rp)
            rr = rz["rows"]
            both = np.intersect1d(rows, rr)
            if len(both):
                ia, ib = np.searchsorted(rows, both), np.searchsorted(rr, both)
                res = {}
                for nm, mine, theirs in (("S_f", Sf, rz["Sf"]), ("S_r", Sr, rz["Sr"])):
                    s = PairStats(ulp=True)
                    s.add(theirs[ib, :nt], mine[ia, :nt])
                    res[nm] = s.summary()
                if "Pf" in z:
                    q90r = np.load(os.path.join(data, "research_q90.npy"))[both]
                    q90 = np.quantile(z["Pf"][ia].astype(np.float64) - z["Pr"][ia], 0.9, axis=1)
                    res["Q90"] = exact_stats(q90 - q90r)
                res["rows"] = len(both)
                res["max_rel_floor1"] = max(res["S_f"]["rel_floor1"]["max"], res["S_r"]["rel_floor1"]["max"])
                res["pass_rel_1e-5"] = res["max_rel_floor1"] <= 1e-5
                res["research_made_by"] = rz["made_by"].item() if "made_by" in rz.files else None
                out["vs_research_stored_scores"] = res
    if "a1" in R:
        A1 = load_a1(data)
        z = dict(np.load(R["a1"]))
        paths = _paths(z)
        res = dict(n=len(z["pairs"]), eq_stored=0, n_ref=0, f1=0.0, f1_stored=0.0, max_abs_dF1_vs_stored=0.0)
        direct = dict(n=0, sf_eq=0, sr_eq=0, path_eq=0, tb_eq=0) if cfg["backend"] == "fixed_c" else None
        per = {}
        for k, p in enumerate(z["pairs"]):
            m = matched(paths[k])
            st = per.setdefault(str(A1["set_names"][A1["pair_set"][p]]), dict(n=0, eq_stored=0))
            st["n"] += 1
            e = int(np.array_equal(m, a1_stored(A1, p)))
            st["eq_stored"] += e
            res["eq_stored"] += e
            if A1["has_ref"][p]:
                f = prf(m, a1_ref(A1, p))[2]
                res["n_ref"] += 1
                res["f1"] += f
                res["f1_stored"] += float(A1["stored_sw"][p, 4])
                res["max_abs_dF1_vs_stored"] = max(res["max_abs_dF1_vs_stored"], abs(f - float(A1["stored_sw"][p, 4])))
            if direct is not None:
                q16, codes = a1_pair(A1, p)
                sim = F.sim_codes(qprep(q16), codes)
                col = np.array([0, len(codes)], np.int64)
                a_, b_ = F.score_fwd_rev(sim, col)
                sc, _, pf, po = F.align(sim, col)
                direct["n"] += 1
                direct["sf_eq"] += int(bits_equal(a_, z["Sf"][k:k + 1])[0])
                direct["sr_eq"] += int(bits_equal(b_, z["Sr"][k:k + 1])[0])
                direct["tb_eq"] += int(bits_equal(sc, z["tb"][k:k + 1])[0])
                direct["path_eq"] += int(np.array_equal(pf[po[0]:po[1]], paths[k]))
        if res["n_ref"]:
            res["f1"] /= res["n_ref"]
            res["f1_stored"] /= res["n_ref"]
        res["per_set"] = per
        out["a1_vs_stored_setting_D_alignments"] = res
        if direct is not None:
            direct["pass"] = all(direct[k] == direct["n"] for k in ("sf_eq", "sr_eq", "path_eq", "tb_eq"))
            out["fixed_c_vs_spike_direct"] = direct
    path = a.out or os.path.join(R["dir"], "verify.json")
    jdump(out, path)
    print(json.dumps(out, indent=1, default=_jdefault))


# =====================================================================================================================
# prep (NERSC)
# =====================================================================================================================
def make_q150(data):
    """25 queries per M3 length bin (rlib EDGES, left-closed), numpy default_rng(Q150_SEED).choice without replacement
    per bin, bins in order; written once (an existing q150.json must match)."""
    Q = np.load(os.path.join(data, "queries.L2m_B8_s0.npz"))
    qlen, ids = Q["lengths"], Q["ids"].astype(str)
    rng = np.random.default_rng(Q150_SEED)
    rows, bins = [], []
    for b in range(len(EDGES) - 1):
        cand = np.flatnonzero((qlen >= EDGES[b]) & (qlen < EDGES[b + 1]))
        pick = np.sort(rng.choice(cand, Q150_PER_BIN, replace=False))
        rows += pick.tolist()
        bins += [b] * len(pick)
    o = np.argsort(rows)
    rows = np.asarray(rows)[o]
    bins = np.asarray(bins)[o]
    q = dict(name="q150", rule=(f"{Q150_PER_BIN} queries per M3 length bin {EDGES[:-1]}+ (left-closed), numpy "
                                f"default_rng({Q150_SEED}).choice(rows of the bin, {Q150_PER_BIN}, replace=False), bins "
                                f"in order; rows = indices into queries.L2m_B8_s0.npz (= queries.fasta order)"),
             seed=Q150_SEED, per_bin=Q150_PER_BIN, edges=EDGES[:-1], queries_file="queries.L2m_B8_s0.npz",
             queries_md5=md5(os.path.join(data, "queries.L2m_B8_s0.npz")), rows=rows.tolist(),
             ids=ids[rows].tolist(), lengths=qlen[rows].tolist(), bins=bins.tolist(), residues=int(qlen[rows].sum()),
             rows_sha256=hashlib.sha256(np.asarray(rows, np.int64).tobytes()).hexdigest())
    p = os.path.join(data, "q150.json")
    if os.path.exists(p):
        old = load_q150(data)
        if old["rows"] != q["rows"]:
            raise SystemExit("existing q150.json differs from the deterministic rule: refusing to overwrite")
        log("q150.json exists and matches")
        return old
    jdump(q, p)
    log(f"q150.json written: {len(rows)} rows, {q['residues']} residues, sha256 {q['rows_sha256'][:16]}")
    return q


def build_a1(src, out):
    P = np.load(f"{src}/{READ_ONLY['pairs'][0]}")
    ENC = {s: np.load(f"{src}/{READ_ONLY['enc_' + s][0]}") for s in ("val", "test")}
    F1D = {s: np.load(f"{src}/{READ_ONLY['f1_' + s][0]}") for s in ("val", "test")}
    CH = np.load(f"{src}/{READ_ONLY['enc_chains'][0]}")
    with open(f"{src}/{READ_ONLY['chain_pairs'][0]}") as f:
        CP = json.load(f)
    CR = np.load(f"{src}/{READ_ONLY['f1_chains_D'][0]}")
    qpool, qoff, qkeys, cpool, coff, ckeys = [], [0], {}, [], [0], {}

    def add(pool, off, keys, key, rows):
        if key not in keys:
            keys[key] = len(keys)
            pool.append(np.asarray(rows))
            off.append(off[-1] + len(rows))
        return keys[key]
    R = dict(set=[], kind=[], qid=[], tid=[], pq=[], pt=[], has_ref=[], refs=[], stored=[], sw=[])
    for split, sets in (("val", ("related", "remote", "unrelated")), ("test", ("remote", "unrelated"))):
        E, F1 = ENC[split], F1D[split]
        Qr, Cr = E["q"], E["codes"]
        for s in sets:
            k = f"{split}/{s}"
            a_, b_, qa, tb = P[k + "/a"], P[k + "/b"], P[k + "/qa"], P[k + "/tb"]
            assert np.array_equal(F1[f"{s}/a"], a_) and np.array_equal(F1[f"{s}/b"], b_), k
            meta = json.loads(str(F1["meta_json"]))
            assert meta["tags"][3] == "D" and tuple(meta["sw_params"][3]) == SETTING_D
            sw = F1[f"{s}/sw"][:, 3, :]
            fl, fo = F1[f"{s}/aln_D_flat"], F1[f"{s}/aln_D_off"]
            rf, ro, hr = P[k + "/ref_flat"], P[k + "/ref_off"], P[k + "/has_ref"]
            assert np.array_equal(F1[f"{s}/len_a"], qa[:, 1] - qa[:, 0]) and np.array_equal(F1[f"{s}/len_b"],
                                                                                         tb[:, 1] - tb[:, 0])
            for p in range(len(a_)):
                R["set"].append(A1_SETS.index(k))
                R["kind"].append(s)
                R["qid"].append(str(a_[p]))
                R["tid"].append(str(b_[p]))
                R["pq"].append(add(qpool, qoff, qkeys, f"{split}:{a_[p]}", Qr[qa[p, 0]:qa[p, 1]]))
                R["pt"].append(add(cpool, coff, ckeys, f"{split}:{b_[p]}", Cr[tb[p, 0]:tb[p, 1]]))
                R["has_ref"].append(bool(hr[p]))
                R["refs"].append(np.asarray(rf[ro[p]:ro[p + 1]], np.int32).reshape(-1, 2))
                R["stored"].append(np.asarray(fl[fo[p]:fo[p + 1]], np.int32).reshape(-1, 2))
                R["sw"].append(sw[p])
    idx = {c: k for k, c in enumerate(CH["ids"].astype(str))}
    off = CH["offsets"]
    aoff, aflat = CR["aln_off"], CR["aln_flat"]
    cmeta = json.loads(str(CR["meta_json"]))
    assert tuple(cmeta["sw"]) == SETTING_D and len(CP) == len(aoff) - 1 and CR["evaluated"].all()
    for p, r in enumerate(CP):
        ka, kb = idx[r["a"]], idx[r["b"]]
        R["set"].append(A1_SETS.index("chains"))
        R["kind"].append(r["kind"])
        R["qid"].append(r["a"])
        R["tid"].append(r["b"])
        R["pq"].append(add(qpool, qoff, qkeys, f"chains:{r['a']}", CH["q"][off[ka]:off[ka + 1]]))
        R["pt"].append(add(cpool, coff, ckeys, f"chains:{r['b']}", CH["codes"][off[kb]:off[kb + 1]]))
        ref = r.get("ref_primary")
        has = ref is not None and len(ref) > 0
        R["has_ref"].append(bool(has))
        R["refs"].append(np.asarray(ref if has else np.zeros((0, 2)), np.int32).reshape(-1, 2))
        R["stored"].append(np.asarray(aflat[aoff[p]:aoff[p + 1]], np.int32).reshape(-1, 2))
        R["sw"].append([float(CR[f"f/{x}"][p]) for x in ("sw_score", "sw_n", "prec", "rec", "f1")])
        assert int(CR["f/len_a"][p]) == off[ka + 1] - off[ka] and int(CR["f/len_b"][p]) == off[kb + 1] - off[kb]
    n = len(R["set"])
    sw = np.asarray(R["sw"], np.float64)
    # stored F1 must be reproduced by prf of the stored alignment (both are harness.prf)
    chk, mism = 0, 0
    means = {}
    for p in range(n):
        if R["has_ref"][p]:
            f = prf(R["stored"][p], R["refs"][p])[2]
            chk += 1
            mism += int(f != sw[p, 4])
            means.setdefault(A1_SETS[R["set"][p]], []).append(f)
    assert mism == 0, f"{mism} stored F1 values not reproduced"
    gaps = sum(int((x < 0).any()) for x in R["stored"])
    rflat = np.concatenate(R["refs"]).astype(np.int32)
    sflat = np.concatenate(R["stored"]).astype(np.int32)
    roff = np.concatenate([[0], np.cumsum([len(x) for x in R["refs"]])]).astype(np.int64)
    soff = np.concatenate([[0], np.cumsum([len(x) for x in R["stored"]])]).astype(np.int64)
    meta = dict(what="A1 pair bundle: 12,228 CATH domain pairs + 732 chain pairs (query q fp16, target codes, US-align "
                     "refs, research stored setting-D alignments = float64 harness)",
                sources={k: dict(path=f"{src}/{v[0]}", md5=v[1]) for k, v in READ_ONLY.items()
                         if k in ("pairs", "enc_val", "enc_test", "enc_chains", "f1_val", "f1_test", "f1_chains_D",
                                  "chain_pairs")},
                set_counts={nm: int(np.sum(np.asarray(R["set"]) == k)) for k, nm in enumerate(A1_SETS)},
                stored_F1_mean={k: float(np.mean(v)) for k, v in means.items()}, stored_F1_checked=chk,
                stored_alignments_with_gap_entries=gaps,
                notes=("stored_flat = matched residue pairs only (harness pairs_array); domain refs from "
                       "pairs_L2m_B8_s0.npz (usalign_refs*.json via harness.load_pairs), chain refs = pairs.json "
                       "ref_primary; stored_sw = [score, n aligned, P, R, F1] at setting D"))
    np.savez(out, set_names=np.array(A1_SETS), pair_set=np.asarray(R["set"], np.int8),
             pair_kind=np.array(R["kind"]), pair_qid=np.array(R["qid"]), pair_tid=np.array(R["tid"]),
             pair_q=np.asarray(R["pq"], np.int32), pair_t=np.asarray(R["pt"], np.int32),
             q_pool=np.concatenate(qpool).astype(np.float16), q_off=np.asarray(qoff, np.int64),
             q_keys=np.array(list(qkeys)), c_pool=np.concatenate(cpool).astype(np.uint8),
             c_off=np.asarray(coff, np.int64), c_keys=np.array(list(ckeys)), has_ref=np.asarray(R["has_ref"]),
             ref_flat=rflat, ref_off=roff, stored_flat=sflat, stored_off=soff, stored_sw=sw,
             meta_json=np.array(json.dumps(meta)))
    log(f"a1.npz: {n} pairs, {len(qkeys)} query items ({qoff[-1]} rows), {len(ckeys)} target items ({coff[-1]} rows); "
        f"stored F1 means {meta['stored_F1_mean']}")
    return meta


def cmd_prep(a):
    src, data = a.src, os.path.abspath(a.data)
    os.makedirs(os.path.join(data, "spike_inputs"), exist_ok=True)
    man = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), host=host_info(), src=src, files={}, read_only_sources={})
    for name, (rel, want) in SOURCES.items():
        sp, dp = f"{src}/{rel}", os.path.join(data, name)
        got = md5(sp)
        if want and got != want:
            raise SystemExit(f"md5 mismatch {sp}: {got} != {want}")
        if not (os.path.exists(dp) and md5(dp) == got):
            shutil.copy2(sp, dp)
        man["read_only_sources"][name] = dict(path=sp, md5=got)
        log(f"copied {name} ({os.path.getsize(dp) / 1e6:.1f} MB) md5 {got}")
    for k, (rel, want) in READ_ONLY.items():
        got = md5(f"{src}/{rel}")
        if want and got != want:
            raise SystemExit(f"md5 mismatch {src}/{rel}: {got} != {want}")
        man["read_only_sources"][k] = dict(path=f"{src}/{rel}", md5=got)
    # benchmark N (ship.py N_ANN over all 50,000 columns) == ship_final meta
    L = np.load(os.path.join(data, "bench_labels.npz"))
    T = np.load(os.path.join(data, "targets.L2m_B8_s0.npz"))
    n_ann = float(T["lengths"][~L["rev"]].sum())
    assert n_ann == ship_table(data)["N"], (n_ann, ship_table(data)["N"])
    man["N_benchmark"] = n_ann
    q = make_q150(data)
    rows = np.asarray(q["rows"], np.int64)
    idx = np.load(f"{src}/{READ_ONLY['score_index'][0]}")
    Q = np.load(os.path.join(data, "queries.L2m_B8_s0.npz"))
    assert np.array_equal(idx["query_ids"].astype(str), Q["ids"].astype(str))
    assert np.array_equal(idx["target_ids"].astype(str), T["ids"].astype(str))
    with open(f"{src}/{READ_ONLY['score_manifest'][0]}") as f:
        sman = json.load(f)
    made_by = dict(job="58885229", host=sman["timing"]["score_timing.58885229.json"]["host"],
                   numpy_blas="foldmatch env numpy (OpenBLAS; research job) - see score_manifest",
                   gemm_chunk_floats=1 << 20, kernel="swmulti.c swz_ad_multi_f32 (swkern.c, gcc -O3 -march=znver3, "
                                                     "vector_lanes 8: 8-lane rowcol reductions)",
                   scoring=sman["scoring"])
    fw = np.load(f"{src}/{READ_ONLY['S_D_fwd'][0]}", mmap_mode="r")
    rv = np.load(f"{src}/{READ_ONLY['S_D_rev'][0]}", mmap_mode="r")
    np.savez(os.path.join(data, "research_SD_q150.npz"), rows=rows, Sf=np.asarray(fw[rows]), Sr=np.asarray(rv[rows]),
             made_by=np.array(json.dumps(made_by)))
    log("research_SD_q150.npz written")
    man["a1"] = build_a1(src, os.path.join(data, "a1.npz"))
    for root, _, files in os.walk(data):
        for f in sorted(files):
            p = os.path.join(root, f)
            if f == "MANIFEST.json":
                continue
            man["files"][os.path.relpath(p, data)] = dict(md5=md5(p), bytes=os.path.getsize(p))
    man["total_bytes"] = int(sum(v["bytes"] for v in man["files"].values()))
    jdump(man, os.path.join(data, "MANIFEST.json"))
    log(f"MANIFEST.json: {len(man['files'])} files, {man['total_bytes'] / 1e6:.1f} MB")


def cmd_info(a):
    d = dict(host=host_info(), env=env_info(), numpy=numpy_info(), root=ROOT, data=os.path.abspath(a.data))
    try:
        d["spike"] = spike_info(a.lib)
    except SystemExit as e:
        d["spike"] = f"unavailable: {e}"
    if a.torch:
        d["torch"] = torch_info()
    avail = {}
    for b in BACKENDS if a.torch else ("fixed_c", "numpy", "numpy_rowwise"):
        try:
            make_backend(dict(backend=b, lib=a.lib, device_index=0))
            avail[b] = True
        except (Unavailable, SystemExit) as e:
            avail[b] = f"no: {e}"
    d["backends"] = avail
    print(json.dumps(d, indent=1, default=_jdefault))


# =====================================================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("info")
    s.add_argument("--data", default=DATA_DEFAULT)
    s.add_argument("--lib")
    s.add_argument("--torch", action="store_true")
    s.add_argument("--threads", type=int, default=1)
    s = sub.add_parser("prep")
    s.add_argument("--src", default=EMB_NERSC)
    s.add_argument("--data", default=DATA_DEFAULT)
    s = sub.add_parser("sim")
    s.add_argument("--backend", required=True, help=" | ".join(BACKENDS))
    s.add_argument("--qprep", choices=("c", "numpy"), default="c")
    s.add_argument("--work", choices=("search", "a1"), default="search")
    s.add_argument("--queries", default="q150")
    s.add_argument("--tmax", type=int, default=0)
    s.add_argument("--no-panel", action="store_true")
    s.add_argument("--a1-limit", type=int, default=0)
    s.add_argument("--chunk", type=int, default=64, help="A1 pairs per task")
    s.add_argument("--block-floats", type=int, default=BLOCK_FLOATS)
    s.add_argument("--block-rule", choices=("prod", "research"), default="prod")
    s.add_argument("--workers", type=int, default=1)
    s.add_argument("--threads", type=int, default=1)
    s.add_argument("--repeat", default="none", help="none | all | comma list of " + ",".join(ALL_VARIANTS))
    s.add_argument("--repeat-queries", type=int, default=0, help="0 = every work query")
    s.add_argument("--sample-queries", type=int, default=16)
    s.add_argument("--host-tag")
    s.add_argument("--label")
    s.add_argument("--tag")
    s.add_argument("--out")
    s.add_argument("--data", default=DATA_DEFAULT)
    s.add_argument("--lib")
    s.add_argument("--force", action="store_true")
    s.add_argument("--keep-parts", action="store_true")
    s = sub.add_parser("compare")
    s.add_argument("A")
    s.add_argument("B")
    s.add_argument("--data", default=DATA_DEFAULT)
    s.add_argument("--out")
    s.add_argument("--name")
    s.add_argument("--max-oracle", type=int, default=3000)
    s.add_argument("--oracle-all", action="store_true", help="oracle margins for every A1 pair (slow)")
    s.add_argument("--no-oracle", action="store_true", help="skip the float64 sim oracle for sim blocks")
    s.add_argument("--threads", type=int, default=1)
    s = sub.add_parser("verify")
    s.add_argument("what", choices=("golden", "run"))
    s.add_argument("run", nargs="?")
    s.add_argument("--backend", default="fixed_c")
    s.add_argument("--qprep", choices=("c", "numpy"), default="c")
    s.add_argument("--data", default=DATA_DEFAULT)
    s.add_argument("--lib")
    s.add_argument("--out")
    s.add_argument("--max-cells", type=float, default=2e7)
    s.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    {"info": cmd_info, "prep": cmd_prep, "sim": cmd_sim, "compare": cmd_compare, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    main()
