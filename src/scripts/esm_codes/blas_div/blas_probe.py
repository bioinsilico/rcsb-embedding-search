#!/usr/bin/env python
"""blas_probe.py -- which BLAS does THIS python's numpy (and optionally torch) use, with which core type / threads?

Companion of blas_div.py (BLAS-divergence study).  Run it with the same python and the same environment
(OPENBLAS_CORETYPE, OPENBLAS_NUM_THREADS, MKL_CBWR, ...) as a blas_div.py run; it prints one JSON document:

  python / host, BLAS-related environment, numpy version + np.show_config BLAS entry, threadpoolctl (if importable),
  every BLAS-like shared object mapped into the process (/proc/self/maps, after a GEMM has run),
  OpenBLAS openblas_get_corename / _get_config / _get_num_threads / _get_parallel through ctypes (any of the symbol
  spellings '', 'scipy_' x '', '64_'), MKL mkl_get_version_string / mkl_get_max_threads when an MKL library is mapped,
  and a fingerprint: sha256 of the float32 result of a fixed [257, 64] x [64, 1031] GEMM (+-1 right operand, the
  shape class of the study) -- two builds with different fingerprints round differently.

  --torch   also torch version, torch.__config__.show() BLAS lines, threads and the same fingerprint through
            torch.matmul on the CPU.
  --threads T  sets every BLAS thread variable to T before numpy is imported (as blas_div.py does).
"""
import ctypes
import hashlib
import json
import os
import platform
import socket
import sys

THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
               "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
if "--threads" in sys.argv:
    _t = sys.argv[sys.argv.index("--threads") + 1]
    for _v in THREAD_VARS:
        os.environ[_v] = str(int(_t))

import numpy as np  # noqa: E402

KEYS = ("blas", "mkl", "lapack", "openblas", "libsci", "sci_", "flexiblas", "blis", "aocl", "gomp", "iomp", "omp")


def mapped_libs():
    out = []
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 6:
                    continue
                p = parts[-1]
                b = os.path.basename(p).lower()
                if (".so" in b) and any(k in b for k in KEYS) and p not in out:
                    out.append(p)
    except OSError:
        pass
    return out


def _sym(lib, names):
    for nm in names:
        for pre in ("", "scipy_"):
            for suf in ("", "64_"):
                s = f"{pre}{nm}{suf}"
                try:
                    return s, getattr(lib, s)
                except AttributeError:
                    continue
    return None, None


def openblas_ctypes(paths):
    res = []
    for p in paths:
        try:
            lib = ctypes.CDLL(p)
        except OSError:
            continue
        d = dict(path=p)
        nm, f = _sym(lib, ["openblas_get_corename"])
        if f is None:
            continue
        f.restype = ctypes.c_char_p
        d["corename"], d["symbol"] = f().decode(), nm
        nm, f = _sym(lib, ["openblas_get_config"])
        if f is not None:
            f.restype = ctypes.c_char_p
            d["config"] = f().decode()
        for key in ("openblas_get_num_threads", "openblas_get_parallel"):
            nm, f = _sym(lib, [key])
            if f is not None:
                f.restype = ctypes.c_int
                d[key.replace("openblas_get_", "")] = int(f())
        res.append(d)
    return res


def mkl_ctypes(paths):
    res = []
    for p in paths:
        if "mkl" not in os.path.basename(p).lower():
            continue
        try:
            lib = ctypes.CDLL(p)
        except OSError:
            continue
        d = dict(path=p)
        for nm in ("mkl_get_version_string", "MKL_Get_Version_String"):
            f = getattr(lib, nm, None)
            if f is not None:
                buf = ctypes.create_string_buffer(256)
                f(buf, 256)
                d["version"] = buf.value.decode(errors="replace")
                break
        f = getattr(lib, "mkl_get_max_threads", None) or getattr(lib, "MKL_Get_Max_Threads", None)
        if f is not None:
            f.restype = ctypes.c_int
            d["max_threads"] = int(f())
        if len(d) > 1:
            res.append(d)
    return res


def operands():
    rng = np.random.default_rng(20261012)
    a = rng.standard_normal((257, 64)).astype(np.float32)
    a /= 8.0 * np.sqrt((a.astype(np.float64) ** 2).sum(1, keepdims=True)).astype(np.float32)
    b = (rng.integers(0, 2, (1031, 64)) * 2 - 1).astype(np.float32)
    return a, b


def fingerprint(c):
    c = np.ascontiguousarray(c, np.float32)
    return hashlib.sha256(c.tobytes()).hexdigest()


def main():
    out = dict(host=socket.gethostname(), platform=platform.platform(), python=sys.version.split()[0],
               executable=sys.executable,
               env={k: v for k, v in sorted(os.environ.items())
                    if k.startswith(("OPENBLAS", "MKL_", "OMP_", "KMP_", "GOMP", "BLIS", "OPENMP", "LIBSCI",
                                     "CRAY_LIBSCI", "SLURM_JOB_ID"))})
    a, b = operands()
    c = a @ b.T
    out["numpy"] = dict(version=np.__version__, file=np.__file__, gemm_fingerprint=fingerprint(c))
    try:
        cfg = np.show_config(mode="dicts")
        bd = cfg.get("Build Dependencies", {})
        out["numpy"]["blas"] = bd.get("blas")
        out["numpy"]["simd"] = cfg.get("SIMD Extensions")
    except Exception as e:  # noqa: BLE001  (numpy < 1.25: text config only)
        out["numpy"]["show_config_error"] = str(e)
    try:
        import threadpoolctl
        out["threadpoolctl"] = threadpoolctl.threadpool_info()
    except Exception as e:  # noqa: BLE001
        out["threadpoolctl"] = f"unavailable: {e}"
    if "--torch" in sys.argv:
        try:
            import torch
            t = dict(version=torch.__version__, num_threads=torch.get_num_threads(),
                     mkl=bool(torch.backends.mkl.is_available()), mkldnn=bool(torch.backends.mkldnn.is_available()))
            cfg = torch.__config__.show()
            t["config_blas_lines"] = [ln.strip() for ln in cfg.splitlines()
                                      if any(k in ln for k in ("Math Kernel", "MKL-DNN", "OpenMP", "BLAS_INFO",
                                                               "CPU capability", "LAPACK"))][:12]
            try:
                t["mkldnn_matmul_fp32_precision"] = str(torch.backends.mkldnn.matmul.fp32_precision)
            except Exception:  # noqa: BLE001
                pass
            ct = torch.matmul(torch.from_numpy(a), torch.from_numpy(b).T).numpy()
            t["gemm_fingerprint"] = fingerprint(ct)
            t["gemm_bit_equal_numpy"] = bool((ct.view(np.int32) == c.view(np.int32)).all())
            out["torch"] = t
        except Exception as e:  # noqa: BLE001
            out["torch"] = f"unavailable: {e}"
    libs = mapped_libs()
    out["mapped_libs"] = libs
    out["openblas"] = openblas_ctypes(libs)
    out["mkl"] = mkl_ctypes(libs)
    js = json.dumps(out, indent=1, default=str)
    if "--out" in sys.argv:
        p = sys.argv[sys.argv.index("--out") + 1]
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
        with open(p, "w") as f:
            f.write(js)
    print(js)


if __name__ == "__main__":
    main()
