#!/usr/bin/env python
"""Shared module for the P8 offline acceptance (A1-A9) of FoldMatch `esm-codes-v1` (plan FINAL v1, section 8 P8).

Every acceptance script imports this module for
  * paths: the staged inputs (``inp("<logical name>")``, staged by ``stage_inputs.py``) and the result directories
    (``result_dir("A5")``) on the Mac and on NERSC;
  * the provenance record (``provenance()``): FoldMatch commit / file / tree digest / install check, kernel info
    (``swkern.load().info()``), host, devices, numpy / BLAS / threadpool and (optionally) torch details;
  * JSON writing (``write_json``: atomic, numpy-aware, non-finite floats written as strings) and md5 / sha256 helpers;
  * the benchmark labels (``Bench``: rlib.Bench reproduced; ``label_matrices``: the plan.py / recon.load rules applied
    to any pair of FASTA header lists);
  * the research FP-ratio / sensitivity / per-query statistics, reproduced from the research protocol files:
      bench_validate.py  (md5 60f0977e37a5bd5c2477a26039dcac77)  tabp, query_folds, clusters_from_folds, counts,
                                                                 boot_idx, ci, summarize, paired, the protocol loop
      rlib.py            (md5 6bf2586b154391a9cdb9e9c5b2a280bc)  Bench, row_params, evaluate, boot_ratio, EDGES / BL
      plan.py            (md5 fe3b4633439e3bb444e3ab90a475a84c)  the TP / FP / FPA label rules (= recon.load,
                         recon.py md5 c7410a44958d2d28626fa95d99adacf3, MMseqs2 EvaluateResults.cpp rules)
      auc_final.py       (md5 66053b23d5fa367b59f5ffea853ea6bc)  read_fasta, sccs_of, lvl (label parsing)
      fit_final2.py      (md5 1afbdaa0b84ecc63c66269d0a94378d4)  Q90 = np.quantile(float64 S(q, panel), 0.9) (linear)
    (archive /global/homes/j/jsegura/esm_codes_archive/research_files/...; staged as inputs/research_files/...).

Self-test (run it before any FoldMatch number is computed with this module; NERSC, it needs the stored research
scores):  python acc_common.py selftest [--out FILE]
  reproduces, from the research stored S_D matrices, labels and Q90, the research bench_validate.json 'own' protocol
  of the NEW (shipped) table three ways - this module, the research functions themselves (bench_validate.py /
  rlib.py imported from the staged archive copies) and the stored bench_validate.json - and checks the headline
  numbers: real-null FP counts 38 / 151 / 1331 at E <= 0.01 / 0.1 / 1 ('all_insample new'), FP ratio at E <= 1
  0.97 (held-out union; main 0.82, swap 1.13) and sensitivity at E <= 1e-3 0.152 / 0.196 (main / swap halves); it
  also re-derives labels.npz from the benchmark FASTA headers (plan.py rules) and bench_q90_shipped.npy from the
  stored panel scores (fit_final2.py), and checks the md5 of every research source file cited above.
Other commands:  provenance (print the record),  check-inputs [--md5] (staged inputs against inputs_manifest.json).

Definitions (benchmark, MMseqs2 rules): TP = the query and target share a SCOP family (annotated target); FP = a
reversed decoy target, or an annotated target sharing no SCOP fold with the query and not excepted (both in the
b.67-b.70 propellers, or either in class e); everything else is ignored. The real null (FPA) = FP & annotated.
E = exp(lnK(m) + ln m + ln N - lambda(m) (S - Q90_q)); FP ratio at x = (real-null FPs with E <= x) / (n_queries x),
N = annotated residues of the evaluated columns; sensitivity at 1e-3 = TPs with E <= 1e-3 / all TPs, with N = all
residues of the evaluated columns (rlib.evaluate).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

T0 = time.time()
HERE = os.path.dirname(os.path.abspath(__file__))

# =====================================================================================================================
# hosts and paths
# =====================================================================================================================
IS_NERSC = os.path.isdir("/pscratch/sd/j/jsegura")
HOST = "nersc" if IS_NERSC else "mac"
SANDBOX = "/Users/joan/devel/foldmatch-p2p3-sandbox"
P8_ROOT = os.environ.get("P8_ROOT") or ("/pscratch/sd/j/jsegura/embcomp/p8" if IS_NERSC else f"{SANDBOX}/p8")
INPUTS = os.environ.get("P8_INPUTS") or os.path.join(P8_ROOT, "inputs")
INPUTS_MANIFEST = os.path.join(INPUTS, "inputs_manifest.json")
# FoldMatch under test (plan P8): commit cc55c01 on js/dev-esm-code-alignments (P1-P7 committed and pushed).
EXPECTED_FM_COMMIT = "cc55c019ad85285119655c31f58a1345cbd5dc7d"
# Mac: the detached sandbox worktree (PYTHONPATH=<wt-P8>/src); NERSC: the 'foldmatch' conda env, installed
# (non-editable) from this checkout - the install is verified file by file against it (installed_matches_checkout).
FM_CHECKOUT = os.environ.get("FM_CHECKOUT") or ("/global/homes/j/jsegura/devel/rcsb-embedding-model" if IS_NERSC
                                                else f"{SANDBOX}/wt-P8")
ARCHIVE = "/global/homes/j/jsegura/esm_codes_archive"            # NERSC only (durable research archive)
NS = "/pscratch/sd/j/jsegura/embcomp"                             # NERSC research data (purgeable scratch)
MAC_ARTEFACTS = f"{SANDBOX}/artefacts"                            # Mac: md5-verified Appendix A subset
RS_ESM_CODES = os.path.dirname(HERE)                              # RS/src/scripts/esm_codes (this repo)
# D4 (decided 2026-10-12): A4 parity thresholds, committed in RS (a18f829) before A4 runs.
A4_THRESHOLDS_SHA256 = "7eabf36ee251033a9ea1cf0b478b9766023f213f53dfd9e2bbdd6f3bc7b456ba"

# research protocol sources (archive research_files/ relative path, md5) - cited by the functions below
RESEARCH_SOURCES = {
    "bench_validate.py": ("embcomp/panel/fit2/scripts/bench_validate.py", "60f0977e37a5bd5c2477a26039dcac77"),
    "bench_validate.json": ("embcomp/panel/fit2/results/bench_validate.json", "0a364f0944a5e4ffc7bdb4c999f261bc"),
    "bench_validate.txt": ("embcomp/panel/fit2/results/bench_validate.txt", "41a695fdeaea8a42600f2170ff2f15b9"),
    "rlib.py": ("embcomp/evalue/reconcile/rlib.py", "6bf2586b154391a9cdb9e9c5b2a280bc"),
    "plan.py": ("embcomp/evalue/score/plan.py", "fe3b4633439e3bb444e3ab90a475a84c"),
    "auc_final.py": ("calib3di/auc1/reconcile/auc_final.py", "66053b23d5fa367b59f5ffea853ea6bc"),
    "recon.py": ("calib3di/fix_eval/reconcile/recon.py", "c7410a44958d2d28626fa95d99adacf3"),
    "fit_final2.py": ("embcomp/panel/fit2/scripts/fit_final2.py", "1afbdaa0b84ecc63c66269d0a94378d4"),
}

SHIP_KEY = "D_fmr/pnl_uniref50/M3t999"            # the shipped table (= bench_validate KNEW)
# the benchmark inputs (staged names mirror the research tree: SP/embcomp/... and SP/calib3di/results/...)
BENCH_IN = {"index": "embcomp/evalue/score/out/index.npz", "labels": "embcomp/evalue/score/results/labels.npz",
            "qfasta": "calib3di/results/queries.fasta", "tfasta": "calib3di/results/targets.fasta",
            "q90": "embcomp/panel/fit2/results/bench_q90_shipped.npy"}


def log(*a):
    print(f"[{time.time() - T0:8.1f}s] " + " ".join(str(x) for x in a), flush=True)


def inp(name: str, *, must_exist: bool = True) -> str:
    """Path of a staged input, by its logical name (the path relative to INPUTS, see inputs_manifest.json)."""
    p = os.path.join(INPUTS, name)
    if must_exist and not os.path.exists(p):
        raise FileNotFoundError(f"staged input {name!r} not found at {p} (host {HOST}; run stage_inputs.py, or the "
                                f"input is not staged on this host)")
    return p


def research_file(key: str) -> str:
    """Staged copy of a research protocol source (RESEARCH_SOURCES key)."""
    return inp("research_files/" + RESEARCH_SOURCES[key][0])


def result_dir(aid: str, *sub: str) -> str:
    """Result directory of an acceptance check: <P8_ROOT>/<aid>[/sub...] (created)."""
    d = os.path.join(P8_ROOT, aid, *sub)
    os.makedirs(d, exist_ok=True)
    return d


# =====================================================================================================================
# hashing and JSON
# =====================================================================================================================
def _digest_file(path: str, algo: str, bs: int = 1 << 24) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def md5_file(path: str) -> str:
    return _digest_file(path, "md5")


def sha256_file(path: str) -> str:
    return _digest_file(path, "sha256")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def array_sha256(a) -> str:
    """sha256 of an array's dtype, shape and C-order bytes."""
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(f"{a.dtype.str}|{a.shape}|".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def jsonable(o):
    """numpy / set / tuple -> JSON types; non-finite floats -> 'nan' / 'inf' / '-inf' strings (strict JSON)."""
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (set, frozenset)):
        return [jsonable(v) for v in sorted(o)]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        if math.isfinite(f):
            return f
        return "nan" if f != f else ("inf" if f > 0 else "-inf")
    return o


def write_json(obj, path: str, indent: int = 1) -> str:
    """Atomic strict-JSON write (tmp + os.replace); returns the file's sha256."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(jsonable(obj), fh, indent=indent, allow_nan=False)
        fh.write("\n")
    os.replace(tmp, path)
    return sha256_file(path)


def read_json(path: str):
    with open(path) as fh:
        return json.load(fh)


# =====================================================================================================================
# provenance
# =====================================================================================================================
def _run(cmd: Sequence[str], cwd: Optional[str] = None, timeout: float = 60) -> Optional[str]:
    try:
        r = subprocess.run(list(cmd), cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def _git(repo: str, *args: str) -> Optional[str]:
    # --no-optional-locks: read-only (git status does not refresh / write the index)
    return _run(["git", "--no-optional-locks", "-C", repo, *args])


def _source_files(pkg_dir: str) -> Dict[str, str]:
    """relpath -> sha256 of a package tree's sources (no __pycache__, compiled objects or editor droppings)."""
    out = {}
    for root, dirs, files in os.walk(pkg_dir):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        for f in sorted(files):
            if f.endswith((".pyc", ".so", ".dylib", ".dll", ".tmp")) or f.startswith("."):
                continue
            p = os.path.join(root, f)
            out[os.path.relpath(p, pkg_dir)] = sha256_file(p)
    return out


def _tree_digest(files: Dict[str, str]) -> str:
    return sha256_bytes("".join(f"{k}\0{v}\n" for k, v in sorted(files.items())).encode())


def foldmatch_provenance(require_commit: Optional[str] = EXPECTED_FM_COMMIT) -> dict:
    """FoldMatch commit, file, version and install integrity. Raises RuntimeError if require_commit is given and the
    imported FoldMatch is not that commit (clean), or (site-packages install) its sources differ from the checkout."""
    import foldmatch
    pkg = os.path.dirname(os.path.realpath(foldmatch.__file__))
    rec = {"file": foldmatch.__file__, "package_dir": pkg, "checkout": FM_CHECKOUT}
    rec["version"] = getattr(foldmatch, "__version__", None)
    top = _git(pkg, "rev-parse", "--show-toplevel")
    if top:                                     # imported from a git checkout / worktree (Mac: wt-P8/src)
        repo, rec["mode"] = top, "checkout"
    else:                                       # installed into site-packages (NERSC conda env)
        repo, rec["mode"] = FM_CHECKOUT, "installed"
    rec["repo"] = repo
    try:
        with open(os.path.join(repo, "pyproject.toml")) as fh:
            m = re.search(r'^version\s*=\s*"([^"]+)"', fh.read(), re.M)
        rec["pyproject_version"] = m.group(1) if m else None
    except OSError:
        rec["pyproject_version"] = None
    try:
        import importlib.metadata as md
        d = md.distribution("foldmatch")
        dist = {"version": d.version, "direct_url": json.loads(d.read_text("direct_url.json") or "null"),
                "location": str(d.locate_file(""))}
        # in checkout mode the env may hold an unrelated installed foldmatch: record it, but it is not under test
        dist["is_imported"] = os.path.realpath(os.path.join(dist["location"], "foldmatch")) == pkg
        rec["installed_distribution"] = dist
    except Exception as e:  # noqa: BLE001
        rec["installed_distribution"] = {"error": repr(e)}
    rec["commit"] = _git(repo, "rev-parse", "HEAD")
    rec["branch"] = _git(repo, "rev-parse", "--abbrev-ref", "HEAD")
    rec["tree"] = _git(repo, "rev-parse", "HEAD^{tree}")
    st = _git(repo, "status", "--porcelain")
    rec["dirty_entries"] = None if st is None else [x for x in st.splitlines() if x]
    files = _source_files(pkg)
    rec["source_tree_sha256"] = _tree_digest(files)
    rec["source_files"] = len(files)
    if rec["mode"] == "installed":
        ref = _source_files(os.path.join(repo, "src", "foldmatch"))
        diff = sorted(k for k in set(files) | set(ref) if files.get(k) != ref.get(k))
        rec["installed_matches_checkout"] = not diff
        rec["installed_vs_checkout_differences"] = diff[:50]
    else:
        rec["installed_matches_checkout"] = None
    if require_commit is not None:
        bad = []
        if rec["commit"] != require_commit:
            bad.append(f"commit {rec['commit']} != {require_commit}")
        if rec["dirty_entries"]:
            bad.append(f"dirty checkout: {rec['dirty_entries'][:5]}")
        if rec["mode"] == "installed" and not rec["installed_matches_checkout"]:
            bad.append(f"installed foldmatch differs from {repo}: {rec['installed_vs_checkout_differences'][:5]}")
        if bad:
            raise RuntimeError("FoldMatch under test is not the P8 commit: " + "; ".join(bad))
    return rec


def kernel_provenance(selftest: bool = False) -> dict:
    from foldmatch.search import swkern
    out = {"info": swkern.load().info()}
    if selftest:
        t = time.time()
        out["selftest"] = swkern.selftest()          # raises KernelError on failure
        out["selftest_ok"] = True
        out["selftest_wall_s"] = round(time.time() - t, 3)
    return out


def system_provenance(name: str = "esm-codes-v1") -> dict:
    """The FoldMatch registry entry under test: name, digests, calibration version and the pinned bundle."""
    from foldmatch.search import scoring_system as SSY
    s = SSY.get(name)
    return {"name": s.name, "digest": s.digest(), "encoding_digest": s.encoding_digest(),
            "calibration_version": s.calibration_version, "bundle_repo": s.bundle_repo,
            "bundle_revision": s.bundle_revision, "bundle_files": [list(x) for x in s.bundle_files],
            "default": SSY.DEFAULT_ESM_CODES}


def _cpu_model() -> Optional[str]:
    if sys.platform == "darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _mem_total_bytes() -> Optional[int]:
    if sys.platform == "darwin":
        v = _run(["sysctl", "-n", "hw.memsize"])
        return int(v) if v else None
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def host_provenance() -> dict:
    rec = {"hostname": socket.gethostname(), "host_kind": HOST, "platform": platform.platform(),
           "machine": platform.machine(), "python": sys.version.split()[0], "executable": sys.executable,
           "cpu_model": _cpu_model(), "logical_cpus": os.cpu_count(), "mem_total_bytes": _mem_total_bytes()}
    if hasattr(os, "sched_getaffinity"):
        rec["affinity_cpus"] = len(os.sched_getaffinity(0))
    rec["slurm"] = {k: v for k, v in sorted(os.environ.items()) if k in (
        "SLURM_JOB_ID", "SLURM_JOB_NAME", "SLURM_JOB_NODELIST", "SLURM_NNODES", "SLURM_NTASKS", "SLURM_CPUS_ON_NODE",
        "SLURM_JOB_PARTITION", "SLURM_JOB_QOS", "SLURM_PROCID", "SLURM_LOCALID", "SLURM_GPUS_ON_NODE",
        "SLURM_CLUSTER_NAME")}
    rec["env"] = {k: os.environ.get(k) for k in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "PYTHONPATH",
        "CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "HF_HUB_CACHE", "HF_HOME", "PYTORCH_ENABLE_MPS_FALLBACK",
        "CONDA_DEFAULT_ENV", "CONDA_PREFIX", "TMPDIR", "FOLDMATCH_STAGE2_PROFILE")}
    return rec


def devices_provenance(torch_info: bool) -> dict:
    rec = {}
    smi = _run(["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version,uuid", "--format=csv,noheader"],
               timeout=30)
    rec["nvidia_smi"] = smi.splitlines() if smi else []
    if torch_info or "torch" in sys.modules:
        try:
            import torch
            t = {"version": torch.__version__, "cuda": torch.version.cuda,
                 "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
                 "cuda_available": torch.cuda.is_available(),
                 "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                 if torch.cuda.is_available() else [],
                 "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
                 "num_threads": torch.get_num_threads(),
                 "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
                 "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
                 "float32_matmul_precision": torch.get_float32_matmul_precision()}
            rec["torch"] = t
        except Exception as e:  # noqa: BLE001
            rec["torch_error"] = repr(e)
    return rec


def blas_provenance() -> dict:
    rec = {"numpy": np.__version__}
    try:
        cfg = np.show_config(mode="dicts")
        deps = cfg.get("Build Dependencies", {})
        rec["numpy_build"] = {k: deps.get(k) for k in ("blas", "lapack")}
        rec["simd"] = cfg.get("SIMD Extensions")
    except Exception as e:  # noqa: BLE001
        rec["numpy_config_error"] = repr(e)
    try:
        import threadpoolctl
        rec["threadpoolctl"] = threadpoolctl.__version__
        rec["threadpools"] = threadpoolctl.threadpool_info()
    except Exception as e:  # noqa: BLE001
        rec["threadpoolctl_error"] = repr(e)
    try:
        import scipy
        rec["scipy"] = scipy.__version__
    except ImportError:
        rec["scipy"] = None
    return rec


def inputs_provenance() -> dict:
    if not os.path.exists(INPUTS_MANIFEST):
        return {"manifest": None}
    return {"manifest": INPUTS_MANIFEST, "manifest_sha256": sha256_file(INPUTS_MANIFEST)}


def provenance(*, fm: bool = True, kernel: bool = True, kernel_selftest: bool = False, torch_info: bool = False,
               require_commit: Optional[str] = EXPECTED_FM_COMMIT, extra: Optional[dict] = None) -> dict:
    """The provenance record every P8 output JSON carries (key 'provenance')."""
    script = os.path.abspath(sys.argv[0]) if sys.argv and sys.argv[0] else None
    rec = {"created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
           "argv": list(sys.argv), "cwd": os.getcwd(),
           "script": script, "script_md5": md5_file(script) if script and os.path.isfile(script) else None,
           "acc_common": os.path.abspath(__file__), "acc_common_md5": md5_file(os.path.abspath(__file__)),
           "host": host_provenance(), "devices": devices_provenance(torch_info), "blas": blas_provenance(),
           "inputs": inputs_provenance(), "p8_root": P8_ROOT}
    if fm:
        rec["foldmatch"] = foldmatch_provenance(require_commit)
        rec["system"] = system_provenance()
    if kernel:
        rec["kernel"] = kernel_provenance(kernel_selftest)
    if extra:
        rec.update(extra)
    return rec


def a4_thresholds() -> dict:
    """The pre-registered A4 thresholds (D4), sha256-checked against the adopted file."""
    p = inp("rs/A4_THRESHOLDS.json")
    h = sha256_file(p)
    if h != A4_THRESHOLDS_SHA256:
        raise RuntimeError(f"A4_THRESHOLDS.json sha256 {h} != adopted {A4_THRESHOLDS_SHA256}")
    return {"path": p, "sha256": h, "thresholds": read_json(p)}


# =====================================================================================================================
# FASTA, label parsing (auc_final.py md5 66053b23...) and the benchmark labels (plan.py md5 fe3b4633... rules)
# =====================================================================================================================
EDGES = [0, 40, 60, 100, 200, 400, 10 ** 9]                        # rlib.EDGES (M3 query-length bins)
BL = ["0-40", "40-60", "60-100", "100-200", "200-400", "400+"]     # rlib.BL
XS = (0.01, 0.1, 1.0)                                              # bench_validate.XS
NB = 2000                                                          # bench_validate.NB


def read_fasta(path: str) -> List[tuple]:
    """auc_final.read_fasta: [(header without '>', sequence)], blank lines skipped, sequence lines stripped."""
    recs, h, seq = [], None, []
    for line in open(path):
        line = line.rstrip("\n\r")
        if not line:
            continue
        if line.startswith(">"):
            if h is not None:
                recs.append((h, "".join(seq)))
            h, seq = line[1:], []
        else:
            seq.append(line.strip())
    recs.append((h, "".join(seq)))
    return recs


def read_headers(path: str) -> List[str]:
    """rlib.read_headers."""
    return [line[1:].rstrip("\n") for line in open(path) if line.startswith(">")]


def sccs_of(h: str) -> List[str]:
    """auc_final.sccs_of: SCOP sccs of a header ('rev_' decoys: none; MMseqs2-benchmark '>id s1 s2 |...': the tokens
    before the first '|'; otherwise the second token)."""
    toks = h.split()
    if toks[0].startswith("rev_"):
        return []
    if " |" in h:
        out = []
        for t in toks[1:]:
            if t.startswith("|"):
                break
            out.append(t)
        return sorted(set(out))
    return [toks[1]]


def lvl(s: str, n: int) -> str:
    """auc_final.lvl: the first n levels of an sccs (2 = fold, 3 = superfamily, 4 = family)."""
    return ".".join(s.split(".")[:n])


def classes_of(h: str) -> set:
    """rlib.classes_of."""
    toks = h.split()
    if toks[0].startswith("rev_"):
        return set()
    out = set()
    for t in toks[1:]:
        if t.startswith("|"):
            break
        out.add(t.split(".")[0])
    return out


_PROP = re.compile(r"^b\.(67|68|69|70)\.")
_CLE = re.compile(r"^e\.")


def label_matrices(q_headers: Sequence[str], t_headers: Sequence[str]) -> dict:
    """plan.py / recon.load labels for any query x target header lists (dense bool [nq, nt]):
    TP = shared SCOP family & annotated target; FP = reversed target, or annotated with no shared fold and not excepted
    (both b.67-70 propellers / either class e); FPA = FP & annotated (the real null); the rest is ignored."""
    q_sccs = [sccs_of(h) for h in q_headers]
    t_sccs = [sccs_of(h) for h in t_headers]
    t_ids = [h.split()[0] for h in t_headers]
    rev = np.array([t.startswith("rev_") for t in t_ids])
    nq, nt = len(q_sccs), len(t_sccs)

    def label_sets(lists, n):
        return [set(lvl(s, n) for s in l_) for l_ in lists]
    qf, tf = label_sets(q_sccs, 4), label_sets(t_sccs, 4)
    qd, td = label_sets(q_sccs, 2), label_sets(t_sccs, 2)

    def inv(tsets):
        d = {}
        for j, s in enumerate(tsets):
            for x in s:
                d.setdefault(x, []).append(j)
        return {k: np.array(v) for k, v in d.items()}
    ifam, ifold = inv(tf), inv(td)
    FAM = np.zeros((nq, nt), bool)
    FOLD = np.zeros((nq, nt), bool)
    for i in range(nq):
        for x in qf[i]:
            if x in ifam:
                FAM[i, ifam[x]] = True
        for x in qd[i]:
            if x in ifold:
                FOLD[i, ifold[x]] = True
    qp = np.array([any(_PROP.match(s) for s in l_) for l_ in q_sccs], bool)
    tp_ = np.array([any(_PROP.match(s) for s in l_) for l_ in t_sccs], bool)
    qe = np.array([any(_CLE.match(s) for s in l_) for l_ in q_sccs], bool)
    te = np.array([any(_CLE.match(s) for s in l_) for l_ in t_sccs], bool)
    EXC = (qp[:, None] & tp_[None, :]) | qe[:, None] | te[None, :]
    ann = ~rev
    TP = FAM & ann[None, :]
    FP = rev[None, :] | (ann[None, :] & ~FOLD & ~EXC)
    assert not (TP & FP).any()
    FPA = FP & ann[None, :]
    counts = dict(tp=int(TP.sum()), fp_ann=int(FPA.sum()), ignored=int((~TP & ~FP).sum()),
                  q_with_tp=int(TP.any(1).sum()))
    return dict(TP=TP, FP=FP, FPA=FPA, rev=rev, counts=counts, q_sccs=q_sccs, t_sccs=t_sccs)


# the research benchmark label counts (plan.py / rlib.Bench / recon.load asserts)
BENCH_COUNTS = dict(tp=31099, fp_ann=34948695, ignored=1645206, q_with_tp=959)


class Bench:
    """rlib.Bench (md5 6bf2586b...), loaded from the staged inputs: the 1,465 x 50,000 benchmark (calib3di queries /
    targets incl. reversed decoys), labels.npz (plan.py), index.npz (evalue/score). Same fields and asserts."""

    def __init__(self, index: Optional[str] = None, labels: Optional[str] = None, qfasta: Optional[str] = None,
                 tfasta: Optional[str] = None):
        self.paths = dict(index=index or inp(BENCH_IN["index"]), labels=labels or inp(BENCH_IN["labels"]),
                          qfasta=qfasta or inp(BENCH_IN["qfasta"]), tfasta=tfasta or inp(BENCH_IN["tfasta"]))
        ix = np.load(self.paths["index"])
        lab = np.load(self.paths["labels"])
        self.qids = ix["query_ids"].astype(str)
        self.tids = ix["target_ids"].astype(str)
        assert np.array_equal(self.qids, lab["query_ids"].astype(str))
        assert np.array_equal(self.tids, lab["target_ids"].astype(str))
        self.nq, self.nt = (int(x) for x in lab["shape"])
        self.qlen = ix["query_len"].astype(np.float64)
        self.tlen = ix["target_len"].astype(np.float64)
        self.rev = lab["rev"].astype(bool)
        un = lambda k: np.unpackbits(lab[k], axis=1, count=self.nt).astype(bool)  # noqa: E731
        self.TP, self.REAL = un("TP"), un("FPA")
        FP = un("FP")
        self.FP = FP
        self.REVM = FP & self.rev[None, :]
        assert int(self.TP.sum()) == BENCH_COUNTS["tp"] and int(self.REAL.sum()) == BENCH_COUNTS["fp_ann"]
        assert int(self.REVM.sum()) == self.nq * int(self.rev.sum())
        assert int(self.TP.any(1).sum()) == BENCH_COUNTS["q_with_tp"]
        self.qbin = np.digitize(self.qlen, EDGES[1:-1], right=False)   # 0..5 with [lo, hi)
        for k in range(6):
            m = self.qbin == k
            assert ((self.qlen[m] >= EDGES[k]) & (self.qlen[m] < EDGES[k + 1])).all()
        idx = np.arange(self.nq)
        self.splits = {"main": (idx[idx % 2 == 0], idx[idx % 2 == 1]), "swap": (idx[idx % 2 == 1], idx[idx % 2 == 0])}
        qh = read_headers(self.paths["qfasta"])
        th = read_headers(self.paths["tfasta"])
        assert [h.split()[0] for h in qh] == list(self.qids) and [h.split()[0] for h in th] == list(self.tids)
        self.q_headers, self.t_headers = qh, th
        self.qcls = [classes_of(h) for h in qh]
        self.tcls = [classes_of(h) for h in th]
        self.qclass = np.array([sorted(c)[0] if len(c) == 1 else ("multi" if c else "none") for c in self.qcls])
        self.N_ann = float(self.tlen[~self.rev].sum())
        self.N_all = float(self.tlen.sum())

    def research_scores(self, setting: str = "D", form: str = "fmr", rows=None) -> np.ndarray:
        """rlib.Bench.scores: float64(S_<setting>_fwd) [- float64(S_<setting>_rev)] of the stored research matrices."""
        f = np.load(inp(f"embcomp/evalue/score/out/S_{setting}_fwd.npy"), mmap_mode="r")
        f = np.asarray(f if rows is None else f[rows]).astype(np.float64)
        if form == "fwd":
            return f
        assert form == "fmr"
        r = np.load(inp(f"embcomp/evalue/score/out/S_{setting}_rev.npy"), mmap_mode="r")
        f -= np.asarray(r if rows is None else r[rows]).astype(np.float64)
        return f


# =====================================================================================================================
# E-values, Q90 (fit_final2.py md5 1afbdaa0...; bench_validate.py md5 60f0977e...; rlib.py md5 6bf2586b...)
# =====================================================================================================================
def ship_tables(path: Optional[str] = None, key: str = SHIP_KEY) -> dict:
    """ship_final.json tables[key]: sub-tables 'all' (shipped), 'main' (fitted on even rows), 'swap' (odd rows), each
    with center_len / lam / lnK; plus q90_summary."""
    return read_json(path or inp("embcomp/panel/fit2/results/ship_final.json"))["tables"][key]


def q90_of(S_panel) -> np.ndarray:
    """fit_final2.py: Q90_q = np.quantile(float64 S(q, panel members), 0.9, axis=1) (linear interpolation)."""
    return np.quantile(np.asarray(S_panel, np.float64), 0.9, axis=1)


def tabp(tab, m):
    """bench_validate.tabp: lambda(m), lnK(m) linear in ln m between the centres (np.interp clamps outside)."""
    lm = np.log(np.asarray(m, np.float64))
    lnc = np.log(np.asarray(tab["center_len"], np.float64))
    return np.interp(lm, lnc, tab["lam"]), np.interp(lm, lnc, tab["lnK"])


def tab_lnc(tab) -> dict:
    """bench_validate t2(): the rlib table form (lnc = [math.log(c)]) used for rlib.evaluate."""
    return dict(lnc=[math.log(c) for c in tab["center_len"]], lam=tab["lam"], lnK=tab["lnK"])


def row_params(tab, m):
    """rlib.row_params (tab with 'lnc')."""
    lm = np.log(np.asarray(m, np.float64))
    return np.interp(lm, tab["lnc"], tab["lam"]), np.interp(lm, tab["lnc"], tab["lnK"])


def ln_evalue(tab, m, S, q90, N: float) -> np.ndarray:
    """bench_validate.counts' L: lnK(m) + ln m + ln N - lambda(m) (S - Q90), rows = queries (float64)."""
    lam, lnK = tabp(tab, m)
    m = np.asarray(m, np.float64)
    return (lnK + np.log(m) + math.log(N))[:, None] - lam[:, None] * (np.asarray(S) - np.asarray(q90)[:, None])


def fp_counts(B: Bench, S, tab, rows, cols, q90, xs: Iterable[float] = XS) -> Dict[float, np.ndarray]:
    """bench_validate.counts: per-query real-null FP counts at each x (E with N = annotated residues of cols).
    S is the full [nq, nt] matrix (rows / cols index it) and q90 the full [nq] vector, as in the research."""
    rows, cols = np.asarray(rows), np.asarray(cols)
    lam, lnK = tabp(tab, B.qlen[rows])
    N = float(B.tlen[cols][~B.rev[cols]].sum())
    L = (lnK + np.log(B.qlen[rows]) + math.log(N))[:, None] - lam[:, None] * (S[np.ix_(rows, cols)]
                                                                            - q90[rows][:, None])
    REAL = B.REAL[np.ix_(rows, cols)]
    return {x: ((L <= math.log(x)) & REAL).sum(1) for x in xs}


def evaluate(B: Bench, S, tab, rows, cols=None, row_shift=None, perq: bool = False, lnk_off=None, lam_rows=None):
    """rlib.evaluate (md5 6bf2586b...), verbatim arithmetic: real-null and reversed-target FP counts / ratios at XS
    (N = annotated / reversed residues of cols), per-bin counts, the top-1 query and the top-5% share, sensitivity at
    E <= 1e-3 (N = all residues of cols), per-bin sensitivity, and the TP budget before the F-th real-null FP
    (F = 1, 10, 100; ties count half). tab is in the rlib form (lnc / lam / lnK; see tab_lnc)."""
    rows = np.asarray(rows)
    cols = np.arange(B.nt) if cols is None else np.asarray(cols)
    lam, lnK = row_params(tab, B.qlen[rows])
    if lam_rows is not None:
        lam = lam_rows
    if lnk_off is not None:
        lnK = lnK + lnk_off
    Sv = S[np.ix_(rows, cols)]
    if row_shift is not None:
        Sv = Sv - row_shift[rows][:, None]
    base = (lnK + np.log(B.qlen[rows]))[:, None] - lam[:, None] * Sv          # ln E minus ln N
    tl = B.tlen[cols]
    rv = B.rev[cols]
    N_ann, N_rev, N_all = float(tl[~rv].sum()), float(tl[rv].sum()), float(tl.sum())
    REAL, REVM, TP = B.REAL[np.ix_(rows, cols)], B.REVM[np.ix_(rows, cols)], B.TP[np.ix_(rows, cols)]
    b = B.qbin[rows]
    out = {"nq": int(len(rows)), "real": {}, "rev": {}, "bins": {}}
    pq = {}
    for x in [0.01, 0.1, 1.0]:                                                  # rlib.XS
        cr = ((base + math.log(N_ann) <= math.log(x)) & REAL).sum(1)
        cv = ((base + math.log(N_rev) <= math.log(x)) & REVM).sum(1)
        pq[f"real_{x}"], pq[f"rev_{x}"] = cr, cv
        srt = np.sort(cr)[::-1]
        k5 = max(1, int(round(0.05 * len(cr))))
        out["real"][str(x)] = dict(fp=int(cr.sum()), ratio=float(cr.sum() / (len(rows) * x)),
                                   fq=float((cr > 0).mean()), top1=int(srt[0]),
                                   top5pct_share=(float(srt[:k5].sum() / cr.sum()) if cr.sum() else None),
                                   ratio_wo_top1=float((cr.sum() - srt[0]) / ((len(rows) - 1) * x)))
        out["rev"][str(x)] = dict(fp=int(cv.sum()), ratio=float(cv.sum() / (len(rows) * x)))
        for k, g in enumerate(BL):
            m = b == k
            c = cr[m]
            s2 = np.sort(c)[::-1]
            out["bins"].setdefault(g, {"nq": int(m.sum())})[str(x)] = dict(
                fp=int(c.sum()), ratio=float(c.sum() / (m.sum() * x)), top1=int(s2[0]) if c.size else 0,
                fq=float((c > 0).mean()))
    L = base + math.log(N_all)
    found = ((L <= math.log(1e-3)) & TP).sum(1)
    out["sens"] = float(found.sum() / TP.sum())
    out["sens_bins"] = {g: float(found[b == k].sum() / max(TP[b == k].sum(), 1)) for k, g in enumerate(BL)}
    fpv = np.sort(L[REAL])
    tpv = L[TP]
    out["budget"] = {}
    for F in (1, 10, 100):
        z = fpv[F - 1]
        out["budget"][str(F)] = float((tpv < z).sum() + 0.5 * (tpv == z).sum())
    if perq:
        pq["found"] = found
        pq["ntp"] = TP.sum(1)
        return out, pq
    return out, None


def boot_ratio(c, x, nb=2000, seed=0):
    """rlib.boot_ratio: plain query-bootstrap 95% CI of an FP ratio."""
    n = len(c)
    ii = np.random.default_rng(seed).integers(0, n, size=(nb, n))
    r = c[ii].sum(1) / (n * x)
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


# ------------------------------------------------------------------------------------- bootstrap (bench_validate.py)
def query_folds(headers_or_path) -> List[set]:
    """bench_validate.query_folds: the SCOP folds (a.b) of each query header ('>id sccs1 sccs2 |...' or
    '>d1gdta1 a.4.1.2 ...')."""
    lines = (open(headers_or_path) if isinstance(headers_or_path, str)
             else (">" + h for h in headers_or_path))
    out = []
    for line in lines:
        if not line.startswith(">"):
            continue
        toks = line[1:].split()
        f = set()
        for t in (toks[1:] if " |" in line else toks[1:2]):
            if t.startswith("|"):
                break
            p = t.split(".")
            assert len(p) == 4, t
            f.add(".".join(p[:2]))
        assert f, line
        out.append(f)
    return out


def clusters_from_folds(folds: Sequence[set]) -> np.ndarray:
    """bench_validate.clusters_from_folds: connected components of queries sharing any SCOP fold (union-find)."""
    par = {}

    def find(a):
        par.setdefault(a, a)
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a
    for fs in folds:
        fs = sorted(fs)
        for f in fs[1:]:
            par[find(f)] = find(fs[0])
    roots = [find(sorted(fs)[0]) for fs in folds]
    u = {r: k for k, r in enumerate(sorted(set(roots)))}
    return np.array([u[r] for r in roots])


def boot_idx(groups, nb: int, seed: int) -> List[np.ndarray]:
    """bench_validate.boot_idx: index arrays (into rows) of a cluster bootstrap over the distinct groups."""
    ug = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in ug]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(nb):
        pick = rng.integers(0, len(ug), len(ug))
        out.append(np.concatenate([members[k] for k in pick]))
    return out


def query_boot_idx(n: int, nb: int = NB, seed: int = 2) -> List[np.ndarray]:
    """bench_validate: the plain query bootstrap (rngq = default_rng(2); one integers(0, n, n) per resample)."""
    rngq = np.random.default_rng(seed)
    return [rngq.integers(0, n, n) for _ in range(nb)]


def ci(v) -> List[float]:
    """bench_validate.ci: percentile 2.5 / 97.5."""
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def summarize(c, x, bidx_c, bidx_q, qbin) -> dict:
    """bench_validate.summarize: per-query FP counts c at threshold x -> nq, fp, ratio, fq (fraction of queries with
    >= 1 FP), fold-cluster and query bootstrap CIs, fq CI, and per query-length bin (nq, fp, ratio, fq, fold CI)."""
    n = len(c)
    o = dict(nq=int(n), fp=int(c.sum()), ratio=float(c.sum() / (n * x)), fq=float((c > 0).mean()),
             ci_fold=ci([c[i].sum() / (len(i) * x) for i in bidx_c]),
             ci_query=ci([c[i].sum() / (len(i) * x) for i in bidx_q]),
             fq_ci_fold=ci([(c[i] > 0).mean() for i in bidx_c]), bins={})
    for k, g in enumerate(BL):
        m = qbin == k
        if not m.any():
            continue
        vals = []
        for i in bidx_c:
            ii = i[m[i]]
            vals.append(c[ii].sum() / (max(len(ii), 1) * x))
        o["bins"][g] = dict(nq=int(m.sum()), fp=int(c[m].sum()), ratio=float(c[m].sum() / (m.sum() * x)),
                            fq=float((c[m] > 0).mean()), ci_fold=ci(vals))
    return o


def paired(cn, co, x, bidx) -> dict:
    """bench_validate.paired: the same cluster resample for two systems (here 'new' = cn, 'old' = co) -> the ratio
    difference and its CI, cn/co and its CI, P(cn < co)."""
    dn = np.array([cn[i].sum() / (len(i) * x) for i in bidx])
    do = np.array([co[i].sum() / (len(i) * x) for i in bidx])
    with np.errstate(divide="ignore", invalid="ignore"):
        rr = dn / do
    return dict(diff=float(cn.sum() / (len(cn) * x) - co.sum() / (len(co) * x)), diff_ci_fold=ci(dn - do),
                new_over_old=float(cn.sum() / max(co.sum(), 1)), new_over_old_ci_fold=ci(rr[np.isfinite(rr)]),
                p_new_lt_old=float((dn < do).mean()))


# ------------------------------------------------------------------------------------- the bench_validate protocol
def bench_protocol(B: Bench, S, tabs: dict, q90, cols=None, *, groups=None, nboot: int = NB,
                   with_sensitivity: bool = True, xs: Sequence[float] = XS) -> dict:
    """bench_validate.main, 'own' protocol, one system (the research 'new' table): for each evaluation
    main (table 'main' on odd rows), swap ('swap' on even rows), all_insample ('all' on all rows) and heldout_union
    (main counts on odd rows + swap counts on even rows, in query order) -> summarize() at each x, with the
    fold-cluster bootstrap (boot_idx(groups[rows], nboot, 1)) and the query bootstrap (default_rng(2)); plus the
    held-out sensitivity at E <= 1e-3 and the TP budgets (rlib.evaluate on the validation rows of each split).
    S: [nq, nt] float64 S = S_f - S_r; tabs: ship_tables() (sub-tables main / swap / all); q90: [nq];
    cols: evaluated target columns (default all 50,000 = the research 'own' NEW protocol);
    groups: fold-cluster id per query (default clusters_from_folds(query_folds(B.q_headers))).
    Returns {'per_query': {eval: {x: counts}}, eval: {x: summary}, 'sensitivity': {split: {...}}, 'meta': {...}}."""
    cols = np.arange(B.nt) if cols is None else np.asarray(cols)
    if groups is None:
        groups = clusters_from_folds(query_folds(B.q_headers))
    idx = np.arange(B.nq)
    ev, od = idx[idx % 2 == 0], idx[idx % 2 == 1]
    evals = {"main": ("main", od), "swap": ("swap", ev), "all_insample": ("all", idx)}
    per = {}
    for ename, (sp, rows) in evals.items():
        per[ename] = fp_counts(B, S, tabs[sp], rows, cols, q90, xs)
    u = {}
    for x in xs:
        c = np.zeros(B.nq, np.int64)
        c[od] = per["main"][x]
        c[ev] = per["swap"][x]
        u[x] = c
    per["heldout_union"] = u
    rowsets = {"main": od, "swap": ev, "all_insample": idx, "heldout_union": idx}
    out = {"meta": dict(n_fold_clusters=int(groups.max() + 1), largest_cluster=int(np.bincount(groups).max()),
                        nboot=nboot, cols=int(len(cols)), N_ann=float(B.tlen[cols][~B.rev[cols]].sum()),
                        N_all=float(B.tlen[cols].sum())),
           "per_query": {e: {str(x): per[e][x] for x in xs} for e in per}}
    for ename, rows in rowsets.items():
        bc = boot_idx(groups[rows], nboot, 1)
        bq = query_boot_idx(len(rows), nboot, 2)
        qb = B.qbin[rows]
        out[ename] = {str(x): summarize(per[ename][x], x, bc, bq, qb) for x in xs}
    if with_sensitivity:
        sens = {}
        for sp, (fr, vr) in B.splits.items():
            o, _ = evaluate(B, S, tab_lnc(tabs[sp]), vr, cols=cols, row_shift=q90)
            sens[sp] = dict(sens_1e3=o["sens"], tps_before_F_fp=o["budget"])
        out["sensitivity"] = sens
    return out


def protocol_line(name: str, e: dict, xs: Sequence[float] = XS) -> str:
    """One bench_validate.txt-style line for a summarize() dict set (ratios with fold CIs, FPs, fq, nq)."""
    return (f"{name:13s}: FP/q/x " + "/".join(str(x) for x in xs) + " " + " / ".join(
        f"{e[str(x)]['ratio']:.2f} [{e[str(x)]['ci_fold'][0]:.2f},{e[str(x)]['ci_fold'][1]:.2f}]" for x in xs)
        + "  FPs " + "/".join(str(e[str(x)]["fp"]) for x in xs) + "  fq "
        + "/".join(f"{e[str(x)]['fq']:.3f}" for x in xs) + f"  nq {e[str(xs[-1])]['nq']}")


# ------------------------------------------------------------------------------------- per-query / threshold helpers
def threshold_counts(L, mask, xs: Sequence[float]) -> Dict[str, int]:
    """Number of pairs with ln E <= ln x inside a boolean mask (e.g. TP / REAL / REVM), per x."""
    return {str(x): int(((L <= math.log(x)) & mask).sum()) for x in xs}


def boundary_pairs(L_ref, xs: Sequence[float], tol: float = 1e-4) -> Dict[str, np.ndarray]:
    """S16 threshold-boundary pairs: |ln E_ref - ln t| < tol (their E-value decision may legitimately differ)."""
    return {str(x): np.abs(np.asarray(L_ref) - math.log(x)) < tol for x in xs}


# =====================================================================================================================
# A5 criteria (plan section 8 P8, A5 row; pre-registered - not to be changed)
# =====================================================================================================================
A5_CRITERIA = {
    "text": "FP ratio at E <= 1 in [0.7, 1.4] (main and swap); at E <= 0.01, <= 4.0; sensitivity at 1e-3 within 0.02 "
            "of 0.152 / 0.196",
    "fp_ratio_e1_range": (0.7, 1.4), "fp_ratio_e001_max": 4.0,
    "sens_1e3_ref": {"main": 0.152, "swap": 0.196}, "sens_1e3_tol": 0.02,
}


def a5_verdict(proto: dict) -> dict:
    """Apply the A5 criteria to a bench_protocol() result (main and swap halves)."""
    c = A5_CRITERIA
    v = {}
    for sp in ("main", "swap"):
        r1 = proto[sp]["1.0"]["ratio"]
        r001 = proto[sp]["0.01"]["ratio"]
        s = proto["sensitivity"][sp]["sens_1e3"]
        lo, hi = c["fp_ratio_e1_range"]
        v[sp] = dict(fp_ratio_e1=r1, fp_ratio_e1_pass=bool(lo <= r1 <= hi),
                     fp_ratio_e001=r001, fp_ratio_e001_pass=bool(r001 <= c["fp_ratio_e001_max"]),
                     sens_1e3=s, sens_1e3_ref=c["sens_1e3_ref"][sp],
                     sens_1e3_pass=bool(abs(s - c["sens_1e3_ref"][sp]) <= c["sens_1e3_tol"]))
    v["pass"] = all(v[sp][k] for sp in ("main", "swap") for k in ("fp_ratio_e1_pass", "fp_ratio_e001_pass",
                                                                   "sens_1e3_pass"))
    v["criteria"] = c["text"]
    return v


# =====================================================================================================================
# inputs manifest checks
# =====================================================================================================================
def load_inputs_manifest() -> dict:
    return read_json(INPUTS_MANIFEST)


def check_inputs(names: Optional[Iterable[str]] = None, full_md5: bool = False) -> dict:
    """Staged inputs against inputs_manifest.json: existence, size and (full_md5) md5. Raises on a mismatch."""
    man = load_inputs_manifest()
    ent = {e["name"]: e for e in man["entries"] if e.get("kind", "file") == "file"}
    names = list(ent) if names is None else list(names)
    bad, ok = [], 0
    for n in names:
        e = ent.get(n)
        if e is None:
            bad.append(f"{n}: not in the manifest")
            continue
        p = inp(n, must_exist=False)
        if not os.path.exists(p):
            bad.append(f"{n}: missing")
            continue
        if os.path.getsize(p) != e["bytes"]:
            bad.append(f"{n}: size {os.path.getsize(p)} != {e['bytes']}")
            continue
        if full_md5 and md5_file(p) != e["md5"]:
            bad.append(f"{n}: md5 differs")
            continue
        ok += 1
    if bad:
        raise RuntimeError("staged inputs do not match inputs_manifest.json: " + "; ".join(bad[:20]))
    return {"checked": ok, "full_md5": full_md5, "manifest_sha256": sha256_file(INPUTS_MANIFEST)}


# =====================================================================================================================
# self-test: the research numbers from the research stored scores
# =====================================================================================================================
SELFTEST_EXPECT = {      # bench_validate.txt (md5 41a695fd...), [own] NEW rows and the sensitivity line
    "all_insample_fp": {"0.01": 38, "0.1": 151, "1.0": 1331},
    "heldout_union_ratio_e1": 0.97, "main_ratio_e1": 0.82, "swap_ratio_e1": 1.13,
    "sens_1e3": {"main": 0.1521, "swap": 0.1964},
    "budget": {"main": {"1": 2079.0, "10": 3230.0, "100": 5953.0}, "swap": {"1": 1685.0, "10": 3891.0, "100": 7228.0}},
}


def _cmp_tree(a, b, path="", out=None):
    """Recursive comparison of two JSON-like trees: counts of compared numbers, max abs / rel float difference,
    list of structural or integer mismatches."""
    if out is None:
        out = {"numbers": 0, "max_abs": 0.0, "max_rel": 0.0, "mismatches": []}
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            out["mismatches"].append(f"{path}: keys {sorted(set(a) ^ set(b))[:6]}")
        for k in set(a) & set(b):
            _cmp_tree(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out["mismatches"].append(f"{path}: len {len(a)} != {len(b)}")
        for k, (x, y) in enumerate(zip(a, b)):
            _cmp_tree(x, y, f"{path}[{k}]", out)
    elif isinstance(a, (int, float, np.integer, np.floating)) and isinstance(b, (int, float, np.integer, np.floating)):
        out["numbers"] += 1
        if isinstance(a, (int, np.integer)) and isinstance(b, (int, np.integer)):
            if int(a) != int(b):
                out["mismatches"].append(f"{path}: {a} != {b}")
        else:
            d = abs(float(a) - float(b))
            out["max_abs"] = max(out["max_abs"], d)
            out["max_rel"] = max(out["max_rel"], d / max(abs(float(b)), 1e-300) if d else 0.0)
    elif a != b:
        out["mismatches"].append(f"{path}: {a!r} != {b!r}")
    return out


def _import_research():
    """Import the research bench_validate.py / rlib.py from the staged archive copies (md5-checked first)."""
    for k in ("bench_validate.py", "rlib.py"):
        got = md5_file(research_file(k))
        assert got == RESEARCH_SOURCES[k][1], f"{k}: md5 {got} != {RESEARCH_SOURCES[k][1]}"
    rdir = os.path.dirname(research_file("rlib.py"))
    bdir = os.path.dirname(research_file("bench_validate.py"))
    for d in (rdir, bdir):           # appended (not prepended): the research dirs must not shadow any module
        if d not in sys.path:
            sys.path.append(d)
    import rlib as R  # noqa: E402
    import bench_validate as BV  # noqa: E402
    assert os.path.realpath(R.__file__) == os.path.realpath(research_file("rlib.py"))
    assert os.path.realpath(BV.__file__) == os.path.realpath(research_file("bench_validate.py"))
    assert BV.R is R
    return R, BV


def selftest(out_path: Optional[str] = None) -> dict:
    res = {"what": "acc_common self-test: research FP ratio / sensitivity / per-query statistics reproduced from the "
                   "research stored scores (before any FoldMatch number)",
           "provenance": provenance(), "checks": {}}
    ok = True

    def chk(name, cond, **info):
        nonlocal ok
        res["checks"][name] = dict(ok=bool(cond), **info)
        ok &= bool(cond)
        log(f"{'PASS' if cond else 'FAIL'} {name}", json.dumps(jsonable(info))[:400])

    # 0. research source files (cited above) and the data inputs
    src = {k: dict(path=research_file(k), md5=md5_file(research_file(k)), expected=v[1])
           for k, v in RESEARCH_SOURCES.items()}
    chk("research_sources_md5", all(s["md5"] == s["expected"] for s in src.values()), files=src)
    exp = {"embcomp/evalue/score/out/index.npz": "ff0f64fc242625b6c7dfce5082a57abb",
           "calib3di/results/queries.fasta": "cf6bb37859a48052875b4a2912080b1b",
           "calib3di/results/targets.fasta": "a34ca17a20cfba6994a33012e064f4a5",
           "embcomp/evalue/score/out/S_D_fwd.npy": "1d015c63cb870524664442bd3784384e",
           "embcomp/evalue/score/out/S_D_rev.npy": "70f3a0f75bf58243a9697da73d449cbd",
           "embcomp/panel/fit2/results/ship_final.json": "c51735f962c04bd8c441b9625c4cf30a",
           "embcomp/panel/score2/out/S_D_fwd.npy": "361fbdf30e81613b902eb3f094e0aa00",
           "embcomp/panel/score2/out/S_D_rev.npy": "a07c4f1b38bc113fa36860207c0351ab",
           "embcomp/panel/score2/out/index.npz": "4f4ac7887a8f3c7233bd493c10646d6c",
           "embcomp/panel/fit2/results/shipped_panel.ids": "68463819ee52018dccb6b4fab3d5467a"}
    data = {n: md5_file(inp(n)) for n in list(exp) + [BENCH_IN["labels"], BENCH_IN["q90"]]}
    chk("data_md5", all(data[k] == v for k, v in exp.items()), md5=data,
        reference="evalue/score/out/manifest.json (index, S_D_*, FASTAs), fit_final2.py MD5 (panel scores), "
                  "plan Appendix A (ship_final.json, shipped_panel.ids); labels.npz / bench_q90_shipped.npy are "
                  "checked by content below")

    # 1. labels: Bench (rlib asserts) and the plan.py rules re-derived from the FASTA headers
    t = time.time()
    B = Bench()
    lm = label_matrices(B.q_headers, B.t_headers)
    same = (np.array_equal(lm["TP"], B.TP) and np.array_equal(lm["FPA"], B.REAL) and np.array_equal(lm["FP"], B.FP)
            and np.array_equal(lm["rev"], B.rev))
    chk("labels_from_fasta_equal_labels_npz", same and lm["counts"] == BENCH_COUNTS, counts=lm["counts"],
        expected=BENCH_COUNTS, wall_s=round(time.time() - t, 1))
    del lm

    # 2. Q90 (fit_final2.py) from the stored panel scores == bench_q90_shipped.npy
    q90 = np.load(inp(BENCH_IN["q90"]))
    ids = [x for x in open(inp("embcomp/panel/fit2/results/shipped_panel.ids")).read().split("\n") if x]
    pix = np.load(inp("embcomp/panel/score2/out/index.npz"))
    assert np.array_equal(pix["query_ids"].astype(str), B.qids)
    pos = {x: k for k, x in enumerate(pix["member_ids"].astype(str))}
    pc = np.array([pos[x] for x in ids])
    P = np.load(inp("embcomp/panel/score2/out/S_D_fwd.npy"), mmap_mode="r")[:, pc].astype(np.float64)
    P -= np.load(inp("embcomp/panel/score2/out/S_D_rev.npy"), mmap_mode="r")[:, pc].astype(np.float64)
    q90_re = q90_of(P)
    del P
    tabs = ship_tables()
    chk("q90_from_panel_scores", np.array_equal(q90_re, q90) and len(ids) == 1000,
        max_abs=float(np.abs(q90_re - q90).max()), n_members=len(ids),
        median=float(np.median(q90)), ship_q90_median=float(tabs["q90_summary"]["pct"][2]))
    chk("q90_median_vs_ship_final", abs(float(np.median(q90)) - float(tabs["q90_summary"]["pct"][2])) < 1e-9)

    # 3. the bench_validate 'own' protocol, NEW table: this module
    t = time.time()
    S = B.research_scores("D", "fmr")
    mine = bench_protocol(B, S, tabs, q90)
    res["wall_protocol_s"] = round(time.time() - t, 1)
    stored = read_json(research_file("bench_validate.json"))
    st_own = stored["protocols"]["own"]

    # 3a. vs the stored bench_validate.json
    cmp_stored = {}
    for e in ("main", "swap", "all_insample", "heldout_union"):
        cmp_stored[e] = _cmp_tree(jsonable(mine[e]), st_own[e]["new"])
    for sp in ("main", "swap"):
        cmp_stored[f"sens/{sp}"] = _cmp_tree(jsonable(mine["sensitivity"][sp]),
                                             stored["sensitivity_heldout_own"][f"new/{sp}"])
    cmp_stored["meta"] = _cmp_tree({"n_fold_clusters": mine["meta"]["n_fold_clusters"],
                                    "largest_cluster": mine["meta"]["largest_cluster"]},
                                   {k: stored["meta"][k] for k in ("n_fold_clusters", "largest_cluster")})
    worst = max(v["max_rel"] for v in cmp_stored.values())
    nmis = sum(len(v["mismatches"]) for v in cmp_stored.values())
    chk("protocol_vs_stored_bench_validate_json", nmis == 0 and worst <= 1e-12,
        numbers=sum(v["numbers"] for v in cmp_stored.values()), max_rel=worst,
        mismatches=[m for v in cmp_stored.values() for m in v["mismatches"]][:20])

    # 3b. vs the research functions themselves, re-run now on the same data (bench_validate.py + rlib.py)
    R, BV = _import_research()
    t = time.time()
    grp = BV.clusters_from_folds(BV.query_folds(inp(BENCH_IN["qfasta"])))
    idx = np.arange(B.nq)
    ev, od = idx[idx % 2 == 0], idx[idx % 2 == 1]
    cols = np.arange(B.nt)
    rper = {e: BV.counts(B, S, tabs[sp], rows, cols, q90)
            for e, (sp, rows) in {"main": ("main", od), "swap": ("swap", ev), "all_insample": ("all", idx)}.items()}
    rper["heldout_union"] = {}
    for x in BV.XS:
        c = np.zeros(B.nq, np.int64)
        c[od] = rper["main"][x]
        c[ev] = rper["swap"][x]
        rper["heldout_union"][x] = c
    rres = {}
    for e, rows in {"main": od, "swap": ev, "all_insample": idx, "heldout_union": idx}.items():
        bc = BV.boot_idx(grp[rows], BV.NB, 1)
        rngq = np.random.default_rng(2)
        bq = [rngq.integers(0, len(rows), len(rows)) for _ in range(BV.NB)]
        rres[e] = {str(x): BV.summarize(rper[e][x], x, bc, bq, B.qbin[rows]) for x in BV.XS}
    t2 = lambda tb: dict(lnc=[math.log(c) for c in tb["center_len"]], lam=tb["lam"], lnK=tb["lnK"])  # noqa: E731
    rsens = {}
    for sp, (fr, vr) in B.splits.items():
        o, _ = R.evaluate(B, S, t2(tabs[sp]), vr, cols=cols, row_shift=q90)
        rsens[sp] = dict(sens_1e3=o["sens"], tps_before_F_fp=o["budget"])
    res["wall_research_rerun_s"] = round(time.time() - t, 1)
    cmp_rerun = {e: _cmp_tree(jsonable(mine[e]), jsonable(rres[e])) for e in rres}
    for sp in rsens:
        cmp_rerun[f"sens/{sp}"] = _cmp_tree(jsonable(mine["sensitivity"][sp]), jsonable(rsens[sp]))
    perq_equal = all(np.array_equal(mine["per_query"][e][str(x)], rper[e][x]) for e in rper for x in BV.XS)
    chk("protocol_vs_research_functions_rerun", perq_equal and np.array_equal(grp, clusters_from_folds(
        query_folds(B.q_headers))) and all(v["max_abs"] == 0.0 and not v["mismatches"] for v in cmp_rerun.values()),
        numbers=sum(v["numbers"] for v in cmp_rerun.values()), max_abs=max(v["max_abs"] for v in cmp_rerun.values()),
        per_query_counts_equal=perq_equal)
    # the full rlib.evaluate output (all fields, per-query arrays) on each held-out half and in-sample
    ev_cmp = {}
    for name, tb, rows in (("main", tabs["main"], od), ("swap", tabs["swap"], ev), ("all", tabs["all"], idx)):
        a, pa = evaluate(B, S, tab_lnc(tb), rows, row_shift=q90, perq=True)
        b, pb = R.evaluate(B, S, t2(tb), rows, row_shift=q90, perq=True)
        c = _cmp_tree(jsonable(a), jsonable(b))
        c["perq_equal"] = all(np.array_equal(pa[k], pb[k]) for k in pb)
        ev_cmp[name] = c
        res.setdefault("evaluate", {})[name] = a
    chk("evaluate_vs_rlib_evaluate", all(v["max_abs"] == 0.0 and not v["mismatches"] and v["perq_equal"]
                                         for v in ev_cmp.values()), cmp=ev_cmp)

    # 4. the headline numbers
    E = SELFTEST_EXPECT
    fp_all = {x: mine["all_insample"][x]["fp"] for x in ("0.01", "0.1", "1.0")}
    chk("headline_all_insample_FP_38_151_1331", fp_all == E["all_insample_fp"], got=fp_all)
    hu = mine["heldout_union"]["1.0"]["ratio"]
    chk("headline_heldout_union_ratio_e1_0.97", round(hu, 2) == E["heldout_union_ratio_e1"], got=hu,
        ci_fold=mine["heldout_union"]["1.0"]["ci_fold"])
    rm, rs = mine["main"]["1.0"]["ratio"], mine["swap"]["1.0"]["ratio"]
    chk("headline_main_swap_ratio_e1_0.82_1.13",
        round(rm, 2) == E["main_ratio_e1"] and round(rs, 2) == E["swap_ratio_e1"],
        main=rm, swap=rs)
    sm, ss = mine["sensitivity"]["main"]["sens_1e3"], mine["sensitivity"]["swap"]["sens_1e3"]
    chk("headline_sens_1e3_0.152_0.196", round(sm, 4) == E["sens_1e3"]["main"] and round(ss, 4) == E["sens_1e3"]["swap"]
        and {sp: mine["sensitivity"][sp]["tps_before_F_fp"] for sp in ("main", "swap")} == E["budget"],
        main=sm, swap=ss, budgets={sp: mine["sensitivity"][sp]["tps_before_F_fp"] for sp in ("main", "swap")})
    # the A5 criteria applied to the research encodings (sanity: the research path passes its own criteria)
    res["a5_verdict_research"] = a5_verdict(mine)
    chk("a5_criteria_pass_on_research_scores", res["a5_verdict_research"]["pass"], verdict=res["a5_verdict_research"])

    res["summary"] = {e: {x: {k: mine[e][x][k] for k in ("fp", "ratio", "ci_fold", "fq")} for x in mine[e]}
                      for e in ("main", "swap", "all_insample", "heldout_union")}
    res["sensitivity"] = mine["sensitivity"]
    res["meta"] = mine["meta"]
    res["lines"] = [protocol_line(e, mine[e]) for e in ("main", "swap", "all_insample", "heldout_union")]
    res["ok"] = bool(ok)
    for line in res["lines"]:
        log(line)
    log(f"sensitivity @1e-3 main {sm:.4f} swap {ss:.4f}")
    log("SELFTEST", "PASS" if ok else "FAIL")
    if out_path:
        h = write_json(res, out_path)
        log(f"wrote {out_path} sha256 {h}")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("selftest")
    s.add_argument("--out", default=None, help="JSON output (default <P8_ROOT>/prep/acc_common_selftest.json)")
    p = sub.add_parser("provenance")
    p.add_argument("--kernel-selftest", action="store_true")
    p.add_argument("--torch", action="store_true")
    p.add_argument("--no-require", action="store_true", help="do not require the P8 FoldMatch commit")
    p.add_argument("--out", default=None)
    c = sub.add_parser("check-inputs")
    c.add_argument("--md5", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        r = selftest(a.out or os.path.join(result_dir("prep"), "acc_common_selftest.json"))
        return 0 if r["ok"] else 1
    if a.cmd == "provenance":
        r = provenance(kernel_selftest=a.kernel_selftest, torch_info=a.torch,
                       require_commit=None if a.no_require else EXPECTED_FM_COMMIT)
        if a.out:
            print(a.out, write_json(r, a.out))
        else:
            print(json.dumps(jsonable(r), indent=1))
        return 0
    if a.cmd == "check-inputs":
        print(json.dumps(check_inputs(full_md5=a.md5), indent=1))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
