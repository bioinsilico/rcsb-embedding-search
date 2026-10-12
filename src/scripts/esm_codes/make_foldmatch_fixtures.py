#!/usr/bin/env python
"""P0.8 (FoldMatch esm-codes plan): freeze the FoldMatch test fixtures in FM/tests/resources/esm_codes/.

Expected values come ONLY from the research reference code:
  * numpy harness (embcomp/harness/harness.py; normalize_scores + smith_waterman copied verbatim from
    src/scripts/esm3_needleman_wunsch.py md5 7ec19b7e...) = float64 oracle; the S5 zero-norm rule and the S7
    degenerate-row/column rule are thin wrappers around it (ref_qprep64 / ref_rowcol_s7 below) and are proven to be
    no-ops on F1;
  * research C kernel (throughput/build swkern.c md5 438d5a98... via swfast.py md5 80a81355...) for S_f, S_r and the
    forward traceback, exactly as inventory_probe/probe.py;
  * the panel driver that produced the shipped Q90s (panel/fit/scripts/cath_score.py query_scores: swmulti.c
    swz_ad_multi_f32, R = 2 runs) + np.quantile(S_f - S_r, 0.9), as panel/fit2/scripts/panel_local.py;
  * the E-value table ship_final.json through panel/fit2/scripts/bench_validate.tabp (np.interp in ln m, clamped);
  * the research encoders sw_mlp/sw_common.py (Whitener + Model, L2m_B8_s0.pt) on CPU fp32, each item alone.
FoldMatch code is used only where the fixture IS FoldMatch's output: its existing sequence-only ESM3 path on CPU fp32
(utils.model.get_residue_model, modules.esm_module.EsmModule.predict_step, dataset.EsmProtFromFasta) for F2c and the
real-ESM3 F-deg rows, and the ESM3 tokenizer it ships with (F-tok). FoldMatch is imported read-only from FM/src; no new
FoldMatch code exists or is used.

Stages (run in order; each writes WORK/stage_<name>.* and prints a JSON summary):
  esm3    foldmatch env:  PYTHONPATH=<FM>/src HF_HUB_OFFLINE=1 .../envs/foldmatch/bin/python THIS esm3
  encode  research env:   .../envs/rcsb-embedding-search/bin/python THIS encode --tag ref
          (optional cross-torch check in the foldmatch env: ... THIS encode --tag fm)
  build   research env:   .../envs/rcsb-embedding-search/bin/python THIS build
Outputs: FM/tests/resources/esm_codes/{pairs_v1.npz, encoder_rows_v1.npz, ood_rows_codes_v1.npz,
fasta_rows_codes_v1.npz, q90_queries_v1.npz, degenerate_v1.npz, evalue_grid_v1.json, tokenizer_cases_v1.json,
FIXTURES.json}; <track>/report.json (all verification numbers). npz files are written deterministically (fixed zip
timestamps, deflate level 9) and load with np.load(..., allow_pickle=False).
Local processes: at most 4 (build: 2 spawn workers for the float64 harness Q90 + the parent + the idle multiprocessing
resource tracker).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import platform
import subprocess
import sys
import time
import zipfile

_STAGE = sys.argv[1] if len(sys.argv) > 1 else ""
if _STAGE != "esm3":  # research convention: single-threaded BLAS; esm3 keeps FoldMatch's own thread setting
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_v] = "1"

import numpy as np  # noqa: E402

# =============================================================================
# Locations and constants
# =============================================================================
SP = os.environ.get("FMFIX_SP", "/private/tmp/claude-501/-Users-joan-devel-rcsb-embedding-search/"
                                "d9266e69-77ea-4121-a428-2e6f3bcc4ca2/scratchpad")
EMB = f"{SP}/embcomp"
FM = os.environ.get("FMFIX_FM", "/Users/joan/devel/rcsb-embedding-model")
RS = os.environ.get("FMFIX_RS", "/Users/joan/devel/rcsb-embedding-search")
OUT = os.environ.get("FMFIX_OUT", f"{FM}/tests/resources/esm_codes")
TRACK = os.environ.get("FMFIX_TRACK", f"{SP}/p0/fixtures")
WORK = f"{TRACK}/work"
FASTA = f"{FM}/tests/resources/fasta/test_sequences.fasta"

SYSTEM = "esm-codes-v1"
CAL_VERSION = "foldmatch-evalue-M8_s0-D_fmr-pnlU50s1000-M3t999-v1"
TABLE_KEY = "D_fmr/pnl_uniref50/M3t999"
OFF, GO, GE = 1.0, 5.0, 0.5            # setting D: cell = z - 1.0, gap of length L costs 5.0 + (L - 1) 0.5
N_FIX = 1.0e9                          # F1 lnE database size
S7_EPS = 1e-6                          # S7: row / column with sigma < 1e-6 contributes exactly 0
QPREP_EPS = 1e-12                      # S5: |q_i| < 1e-12 -> qp_i = 0
ESM_REVISION = "636acb420daf674df37ab0220f56061a5de2f169"
ESM_WEIGHTS_SHA256 = "5ead5a135c658068db6a4f1b933e72d6110992c4668822e1c0e2dcc53e38acd9"
PANEL_CODES_BYTES_MD5 = "885dfc88590e0b270addcd4b61c3a7a7"
STD20 = frozenset("ACDEFGHIKLMNPQRSTVWY")
DIMS_DROPPED = (1361, 1489)
NWORK = 2                              # spawn workers; + parent + multiprocessing resource tracker = 4 processes

# F-deg real ESM3 inputs (plan P0.8: all-X x30, HHHHHH, poly-A x50, M, GS, a 10-aa peptide = angiotensin I)
DEG_SEQS = [("allX30", "X" * 30), ("His6", "HHHHHH"), ("polyA50", "A" * 50), ("Met1", "M"), ("GS", "GS"),
            ("angiotensinI", "DRVYIHPFHL")]

# F-tok cases (plan P0.8 / S15: AC<mask>DE, _, *, J, digits, space, -.|, lowercase; plus edge cases)
FTOK_CASES = [
    ("mask_literal", "AC<mask>DE"), ("mask_literal_lowercase", "ac<mask>de"), ("mask_short", "_"),
    ("mask_short_inside", "AC_DE"), ("stop", "*"), ("trailing_stop", "MKT*"), ("J", "J"), ("digits", "0123456789"),
    ("space", "AC DE"), ("whitespace_mixed", " A\tC\nD\rE\x0bF\x0c "), ("gap_dot_bar", "-.|"), ("lowercase", "acde"),
    ("mixed_case", "AcDe"), ("nonstandard_real_tokens", "XBZUO"), ("non_ascii", "ACDÉ"),
    ("lt_only", "A<C"), ("gt_only", "A>C"), ("special_cls", "<cls>A"), ("special_eos", "A<eos>"),
    ("special_pad", "A<pad>C"), ("special_unk", "A<unk>C"), ("standard20", "ACDEFGHIKLMNPQRSTVWY"),
]

INPUTS = {  # name: (path, expected md5 from PLAN Appendix A / ARCHIVE or None = recorded only)
    "fixture_candidate": (f"{SP}/inventory_probe/fixture_candidate.npz", "d724b769bcf26178cc20a3bf157a64bf"),
    "probe_report": (f"{SP}/inventory_probe/probe_report.json", "c35ed1589908808e31afe3dfd562f776"),
    "probe_py": (f"{SP}/inventory_probe/probe.py", None),
    "enc_val": (f"{EMB}/throughput/build/data/L2m_B8_s0.val.npz", "5aebc14942665748313f22ca737ca26d"),
    "enc_test": (f"{EMB}/throughput/build/data/L2m_B8_s0.test.npz", "e570bc26e1a142183aaf85e9a7a57df7"),
    "f1_settings_val": (f"{EMB}/evalue/f1_settings/results/dom/M8_s0.val.npz", None),
    "f1_settings_test": (f"{EMB}/evalue/f1_settings/results/dom/M8_s0.test.npz", None),
    "cath_set": (f"{EMB}/panel/fit/data/cath_set.npz", "78b6c3fcf64d1ffcc9aa98d81c04c81c"),
    "cath_q90_local": (f"{EMB}/panel/fit2/results/cath_q90_shipped_local.npy", None),
    "val_emb": (f"{EMB}/val.npz", None),
    "enc_check": (f"{EMB}/sw_mlp/enc_check/L2m_B8_s0.npz", None),
    "enc_check_items": (f"{EMB}/sw_mlp/enc_check/check_items.json", None),
    "encoders": (f"{EMB}/sw_mlp/models/L2m_B8_s0.pt", "85fdd985087d6813e9994274d4fca6ab"),
    "sw_common": (f"{EMB}/sw_mlp/sw_common.py", "b8dab0a03907c9b7ed67663d96c55561"),
    "zcw50_clean": (f"{EMB}/clean_pairs/zcw50_clean.npz", "4c7692ed66e599aca906b41ec7c13f04"),
    "ship_final": (f"{EMB}/panel/fit2/results/ship_final.json", "c51735f962c04bd8c441b9625c4cf30a"),
    "panel_codes": (f"{EMB}/panel/fit2/results/shipped_panel_codes.npz", "15cc1c7b2d1dd488a5f9b742365ccb13"),
    "harness": (f"{EMB}/harness/harness.py", "05f387cb400cc85dbacaef771a275eeb"),
    "oracle_script": (f"{RS}/src/scripts/esm3_needleman_wunsch.py", "7ec19b7e5ff89dc7b44d2512aed3997e"),
    "swkern_c": (f"{EMB}/throughput/build/swkern.c", "438d5a9811153f61e66cee845f1ed037"),
    "swfast_py": (f"{EMB}/throughput/build/swfast.py", "80a813553d42801fce6b77462c5e1d0a"),
    "libswkern": (f"{EMB}/throughput/build/libswkern.dylib", None),
    "kdir_swkern_c": (f"{EMB}/panel/fit/kernel/swkern.c", "438d5a9811153f61e66cee845f1ed037"),
    "kdir_swfast_py": (f"{EMB}/panel/fit/kernel/swfast.py", "80a813553d42801fce6b77462c5e1d0a"),
    "kdir_swmulti_c": (f"{EMB}/panel/fit/kernel/swmulti.c", "b12fbc2c03638659440b12282a4d8670"),
    "kdir_libswkern": (f"{EMB}/panel/fit/kernel/libswkern.dylib", None),
    "kdir_libswmulti": (f"{EMB}/panel/fit/kernel/libswmulti.so", None),
    "cath_score_py": (f"{EMB}/panel/fit/scripts/cath_score.py", None),
    "panel_local_py": (f"{EMB}/panel/fit2/scripts/panel_local.py", None),
    "bench_validate_py": (f"{EMB}/panel/fit2/scripts/bench_validate.py", None),
    "rlib_py": (f"{EMB}/evalue/reconcile/rlib.py", None),
    "bench_queries": (f"{EMB}/evalue/encode/out/queries.L2m_B8_s0.npz", "e821497f59dda06a3899c18dce37286c"),
    "bench_q90_local": (f"{EMB}/panel/fit2/results/bench_q90_shipped_local.npy", None),
    "bench_q90_nersc": (f"{EMB}/panel/fit2/results/bench_q90_shipped.npy", None),
    "test_fasta": (FASTA, None),
    "pt_1acb_A": (f"{FM}/tests/resources/embeddings/1acb.A.pt", None),
    "pt_2uzi_A": (f"{FM}/tests/resources/embeddings/2uzi.A.pt", None),
}
STAGE_INPUTS = {
    "esm3": ["test_fasta"],
    "encode": ["encoders", "sw_common", "zcw50_clean", "val_emb", "enc_check", "enc_check_items", "pt_1acb_A",
               "pt_2uzi_A"],
    "build": [k for k in INPUTS if k not in ("val_emb",)],
}


# =============================================================================
# Small helpers
# =============================================================================
def md5_file(p, bs=1 << 24):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def sha256_file(p, bs=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def verify_inputs(names):
    out = {}
    for nm in names:
        p, exp = INPUTS[nm]
        h = md5_file(p)
        if exp is not None and h != exp:
            raise SystemExit(f"input md5 mismatch for {nm} ({p}): {h} != expected {exp}")
        out[nm] = dict(path=p, md5=h, expected_md5=exp, bytes=os.path.getsize(p))
    return out


def jdefault(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _clean(o):
    """strict JSON: numpy -> python, non-finite floats -> None."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.floating, float)):
        o = float(o)
        return o if math.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def jdump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_clean(obj), f, indent=1, default=jdefault, ensure_ascii=True, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)


def meta_arr(meta):
    """0-d unicode array holding strict JSON (json.loads(str(z['meta_json'])))."""
    return np.array(json.dumps(_clean(meta), allow_nan=False, ensure_ascii=True))


def generator_md5():
    return md5_file(os.path.abspath(__file__))


def q_diff_stats(a, b):
    """fp16 q differences: per-element ulp count (misleading near 0) and |dq| in ulps of the ROW's max |q|."""
    a16, b16 = np.asarray(a, np.float16), np.asarray(b, np.float16)
    d = np.abs(a16.astype(np.float64) - b16.astype(np.float64))
    ul = fp16_ulp_dist(a16, b16)
    big = np.maximum(np.abs(a16.astype(np.float32)), np.abs(b16.astype(np.float32)))
    rowmax = big.max(axis=1, keepdims=True)

    def ulp_at(x):
        return np.float32(2.0) ** (np.floor(np.log2(np.maximum(x, 2.0 ** -14))) - 10)

    return dict(n=int(a16.size), bit_equal=int((a16.view(np.uint16) == b16.view(np.uint16)).sum()),
                max_elem_ulp=int(ul.max()), n_elem_ulp_gt1=int((ul > 1).sum()), max_abs=float(d.max()),
                max_abs_over_row_ulp=float((d / ulp_at(rowmax)).max()),
                n_abs_gt_elem_ulp=int((d > ulp_at(big)).sum()))


def write_npz(path, arrays: dict):
    """Deterministic npz: fixed zip timestamps, deflate 9, npy format 1.0, no pickles."""
    tmp = path + ".tmp"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for k, v in arrays.items():
            v = np.asarray(v)
            if not v.flags.c_contiguous:          # (np.ascontiguousarray would turn 0-d arrays into shape (1,))
                v = v.copy(order="C")
            assert v.dtype != object, k
            buf = io.BytesIO()
            np.lib.format.write_array(buf, v, version=(1, 0), allow_pickle=False)
            zi = zipfile.ZipInfo(k + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            zf.writestr(zi, buf.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    os.replace(tmp, path)


def npz_manifest(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: [list(z[k].shape), str(z[k].dtype)] for k in z.files}


def spec_normalize(raw: str):
    """S15 normalize_sequence, transcribed from the spec: refuse non-ASCII; drop ASCII whitespace; ASCII upper."""
    if any(ord(c) > 127 for c in raw):
        return None
    return "".join(c for c in raw if c not in " \t\n\r\x0b\x0c").upper()


def fp16_ordered(a):
    u = np.asarray(a, np.float16).view(np.uint16).astype(np.int32)
    mag = u & 0x7FFF
    return np.where(u & 0x8000, -mag, mag)


def fp16_ulp_dist(a, b):
    return np.abs(fp16_ordered(a) - fp16_ordered(b))


def offsets_of(lengths):
    return np.concatenate([[0], np.cumsum(np.asarray(lengths, np.int64))]).astype(np.int64)


def git_info(repo):
    def g(*a):
        return subprocess.run(["git", "--no-optional-locks", "-C", repo, *a], capture_output=True, text=True).stdout
    return dict(head=g("rev-parse", "HEAD").strip(), branch=g("rev-parse", "--abbrev-ref", "HEAD").strip(),
                dirty_files=[x for x in g("status", "--porcelain").splitlines() if x.strip()])


def base_env():
    return dict(python=sys.version.split()[0], executable=sys.executable, numpy=np.__version__,
                platform=platform.platform(), machine=platform.machine(),
                threads_env={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")})


# =============================================================================
# Reference arithmetic (float64 oracle = harness + S5 zero rule + S7 degenerate rule)
# =============================================================================
def ref_qprep64(q16):
    """harness: Q = fp64(q); Q /= |Q| (rows).  S5 adds: |q_i| < 1e-12 -> 0 row; NaN / Inf is an error."""
    Q = np.asarray(q16, dtype=np.float64)
    if not np.isfinite(Q).all():
        raise FloatingPointError("non-finite query encoding (S5: error)")
    nrm = np.linalg.norm(Q, axis=1, keepdims=True)
    zero = nrm < QPREP_EPS
    return np.where(zero, 0.0, Q / np.where(zero, 1.0, nrm))


def decode64(codes):
    """harness / probe: t = (unpackbits(codes)[:, :64] * 2 - 1) / 8 (unit rows)."""
    return (np.unpackbits(np.asarray(codes, np.uint8), axis=1)[:, :64].astype(np.float64) * 2 - 1) / 8.0


def ref_rowcol_s7(sim):
    """harness.normalize_scores(sim, 'rowcol') with S7: a row (column) with population std < 1e-6 contributes exactly 0.
    Identical expression order to the verbatim harness, so it is bit-identical to it when nothing is degenerate."""
    sim = np.asarray(sim, dtype=np.float64)
    eps = 1e-8
    mu, sd = sim.mean(axis=1, keepdims=True), sim.std(axis=1, keepdims=True)
    nu, tau = sim.mean(axis=0, keepdims=True), sim.std(axis=0, keepdims=True)
    z_row = (sim - mu) / (sd + eps)
    z_col = (sim - nu) / (tau + eps)
    dr, dc = sd < S7_EPS, tau < S7_EPS
    if dr.any():
        z_row = np.where(dr, 0.0, z_row)
    if dc.any():
        z_col = np.where(dc, 0.0, z_col)
    return 0.5 * (z_row + z_col), int(dr.sum()), int(dc.sum())


def aln_path(aln):
    return np.array([(-1 if i is None else i, -1 if j is None else j) for i, j in aln.pairs],
                    np.int32).reshape(-1, 2)


def aligned_pairs(path):
    p = np.asarray(path).reshape(-1, 2)
    return p[(p[:, 0] >= 0) & (p[:, 1] >= 0)]


def ref_pair64(H, q16, codes, s7=True, path=True):
    """Float64 oracle for one pair: S_f (+ forward path), S_r = same on the row-reversed query."""
    Q = ref_qprep64(q16)
    T = decode64(codes)
    res = {}
    for d, Qd in (("f", Q), ("r", Q[::-1])):
        sim = Qd @ T.T
        if s7:
            z, nr, nc = ref_rowcol_s7(sim)
        else:
            z, nr, nc = H.normalize_scores(sim, "rowcol"), -1, -1
        a = H.smith_waterman(z - OFF, GO, GE)
        res[f"S_{d}"] = float(a.score)
        res[f"ndeg_rows_{d}"], res[f"ndeg_cols_{d}"] = nr, nc
        if d == "f" and path:
            res["path"] = aln_path(a)
    return res


# ---- spawn workers: float64 harness Q90 over the shipped panel ------------------------------------------------------
_W: dict = {}


def _w_init(emb):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[v] = "1"
    os.environ["EMBCOMP_DATA"] = emb
    sys.path.insert(0, f"{emb}/harness")
    import harness as H
    z = np.load(f"{emb}/panel/fit2/results/shipped_panel_codes.npz")
    _W.update(H=H, codes=z["codes"], off=z["offsets"].astype(np.int64))


def _w_q90(task):
    key, q16, s7 = task
    H, codes, off = _W["H"], _W["codes"], _W["off"]
    NT = len(off) - 1
    f, r = np.empty(NT), np.empty(NT)
    ndr = ndc = 0
    for j in range(NT):
        res = ref_pair64(H, q16, codes[off[j]:off[j + 1]], s7=s7, path=False)
        f[j], r[j] = res["S_f"], res["S_r"]
        ndr += max(res["ndeg_rows_f"], 0) + max(res["ndeg_rows_r"], 0)
        ndc += max(res["ndeg_cols_f"], 0) + max(res["ndeg_cols_r"], 0)
    return key, float(np.quantile(f - r, 0.9)), f, r, ndr, ndc


def harness_q90_all(tasks):
    """tasks: [(key, q16, s7)] -> {key: dict(q90, S_f, S_r, ndeg_rows, ndeg_cols)}; LPT order, NWORK spawn workers."""
    from multiprocessing import get_context
    tasks = sorted(tasks, key=lambda t: -len(t[1]))
    out = {}
    t0 = time.time()
    with get_context("spawn").Pool(NWORK, initializer=_w_init, initargs=(EMB,)) as pool:
        for k, (key, q90, f, r, ndr, ndc) in enumerate(pool.imap_unordered(_w_q90, tasks, chunksize=1)):
            out[key] = dict(q90=q90, S_f=f, S_r=r, ndeg_rows=ndr, ndeg_cols=ndc)
            if (k + 1) % 10 == 0 or k + 1 == len(tasks):
                log(f"  harness float64 Q90: {k + 1}/{len(tasks)} queries, {time.time() - t0:.0f}s")
    return out


T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s] " + " ".join(str(x) for x in a), flush=True)


# =============================================================================
# Stage esm3 (foldmatch env): FoldMatch's own sequence-only ESM3 on CPU fp32; ESM3 tokenizer ids
# =============================================================================
def stage_esm3(args):
    os.environ["HF_HUB_OFFLINE"] = "1"          # no network: the pinned snapshot must already be cached
    if f"{FM}/src" not in sys.path:
        sys.path.insert(0, f"{FM}/src")
    ins = verify_inputs(STAGE_INPUTS["esm3"])
    import foldmatch  # noqa: F401  (FoldMatch sets OMP_NUM_THREADS=1 on macOS at import: its real CPU setting)
    assert os.path.realpath(foldmatch.__file__).startswith(os.path.realpath(f"{FM}/src")), foldmatch.__file__
    import torch
    import importlib.metadata as md
    from foldmatch.utils.model import get_residue_model
    from foldmatch.utils.esm import loaders as FL
    from foldmatch.modules.esm_module import EsmModule
    from foldmatch.dataset.esm_prot_from_fasta import EsmProtFromFasta
    from esm.sdk.api import ESMProtein
    from esm.tokenization import EsmSequenceTokenizer
    from esm.utils import encoding as ENC

    root = FL.data_root()
    wfile = os.path.join(str(root), "data", "weights", "esm3_sm_open_v1.pth")
    blob = os.path.basename(os.path.realpath(wfile))
    snap = os.path.basename(str(root).rstrip("/"))
    if blob != ESM_WEIGHTS_SHA256 or snap != ESM_REVISION:
        raise SystemExit(f"ESM3 snapshot {snap} / weights blob {blob} differ from the pinned ones")
    default_threads = torch.get_num_threads()
    t = time.time()
    model = get_residue_model(torch.device("cpu"))
    load_s = time.time() - t
    pdt = sorted({str(p.dtype) for p in model.parameters()})
    module = EsmModule(model)
    module.eval()
    ds = EsmProtFromFasta(FASTA)
    fasta_items = [ds[i] for i in range(len(ds))]
    deg_items = [(ESMProtein(sequence=s), name) for name, s in DEG_SEQS]
    items = fasta_items + deg_items

    def run():
        t0 = time.time()
        with torch.inference_mode():
            embs, names = module.predict_step(items, 0)
        return [e.detach().to("cpu").numpy().copy() for e in embs], list(names), time.time() - t0

    embs_a, names, sec_a = run()                       # FoldMatch default CPU threading
    torch.set_num_threads(args.threads)
    embs_b, _, sec_b = run()                           # explicit multi-thread repeat (determinism check)
    torch.set_num_threads(default_threads)
    seqs = [p.sequence for p, _ in items]
    tok = []
    for (p, _), e in zip(items, embs_a):
        ids = model.encode(p).sequence.tolist()
        tok.append(ids)
        assert e.shape == (len(p.sequence) + 2, 1536) and e.dtype == np.float32, (e.shape, e.dtype)
        assert len(ids) == len(p.sequence) + 2
    rows16 = [e.astype(np.float16) for e in embs_a]
    th_equal = [bool(torch.equal(torch.from_numpy(e).half(), torch.from_numpy(r))) for e, r in zip(embs_a, rows16)]
    item_stats = []
    for k, (nm, s, e, e2) in enumerate(zip(names, seqs, embs_a, embs_b)):
        d = np.abs(e.astype(np.float64) - e2.astype(np.float64))
        item_stats.append(dict(
            name=nm, length=len(s), rows=int(e.shape[0]), max_abs=float(np.abs(e).max()),
            fp16_finite=bool(np.isfinite(rows16[k]).all()), fp16_cast_numpy_eq_torch=th_equal[k],
            threads_repeat_max_abs_diff=float(d.max()), threads_repeat_bit_identical=bool(np.array_equal(e, e2)),
            threads_repeat_fp16_identical=bool(np.array_equal(rows16[k], e2.astype(np.float16)))))

    # ---- F-tok: tokenizer cases (weights-free path esm.utils.encoding.tokenize_sequence == model.encode) ----
    st = EsmSequenceTokenizer()
    vocab = {int(i): t for t, i in st.get_vocab().items()}

    def toks(s):
        ids = ENC.tokenize_sequence(s, st, add_special_tokens=True).tolist()
        ids_m = model.encode(ESMProtein(sequence=s)).sequence.tolist()
        assert ids == ids_m, (s, ids, ids_m)
        return ids, st.convert_ids_to_tokens(ids)

    cases = []
    for cid, raw in FTOK_CASES:
        norm = spec_normalize(raw)
        ids_raw, str_raw = toks(raw)
        c = dict(id=cid, raw=raw, raw_is_ascii=all(ord(ch) < 128 for ch in raw), normalized=norm,
                 normalize_refuses=norm is None,
                 strict_refusal=("non_ascii" if norm is None else
                                 ("angle_bracket" if ("<" in norm or ">" in norm) else None)),
                 tokens_raw=ids_raw, tokens_raw_str=str_raw, n_residue_tokens_raw=len(ids_raw) - 2)
        if norm is not None:
            ids_n, str_n = toks(norm)
            c.update(tokens_normalized=ids_n, tokens_normalized_str=str_n, len_normalized=len(norm),
                     n_residue_tokens_normalized=len(ids_n) - 2, length_preserved=(len(ids_n) - 2 == len(norm)),
                     n_nonstandard=sum(ch not in STD20 for ch in norm),
                     n_unk=sum(t == "<unk>" for t in str_n[1:-1]), n_mask=sum(t == "<mask>" for t in str_n[1:-1]))
        cases.append(c)

    nf = len(fasta_items)
    arrays = dict(
        names=np.array(names), sequences=np.array(seqs), kind=np.array(["fasta"] * nf + ["deg"] * len(deg_items)),
        rows_offsets=offsets_of([len(r) for r in rows16]), rows=np.concatenate(rows16),
        tokens_offsets=offsets_of([len(x) for x in tok]), tokens=np.concatenate([np.array(x, np.int64) for x in tok]))
    os.makedirs(WORK, exist_ok=True)
    write_npz(f"{WORK}/stage_esm3.npz", arrays)
    info = dict(
        stage="esm3", generator_md5=generator_md5(), created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        inputs=ins,
        env=dict(base_env(), torch=torch.__version__, esm=md.version("esm"),
                 foldmatch_dist_version=md.version("foldmatch"),
                 foldmatch_imported_from=os.path.dirname(os.path.realpath(foldmatch.__file__)),
                 foldmatch_git=git_info(FM), hf_hub_offline=os.environ.get("HF_HUB_OFFLINE")),
        esm3=dict(snapshot_path=str(root), snapshot_revision=snap, weights_file=wfile, weights_blob_sha256=blob,
                  param_dtypes=pdt, device="cpu", autocast=False, precision="fp32 weights, fp32 compute (CPU)",
                  default_torch_threads=default_threads, repeat_torch_threads=args.threads,
                  load_s=round(load_s, 1), pass_s_default=round(sec_a, 1), pass_s_repeat=round(sec_b, 1),
                  call=("foldmatch.modules.esm_module.EsmModule(get_residue_model(cpu)).predict_step(items, 0) "
                        "under torch.inference_mode(); per_residue_embedding [L+2, 1536] fp32 -> stored fp16"),
                  fasta_reader="foldmatch.dataset.esm_prot_from_fasta.EsmProtFromFasta(test_sequences.fasta)"),
        items=item_stats, tokenizer=dict(vocab=vocab, cases=cases, esm_mask_str_short="_"),
        stage_npz_md5=md5_file(f"{WORK}/stage_esm3.npz"))
    jdump(info, f"{WORK}/stage_esm3.json")
    log(json.dumps(dict(items=item_stats, default_threads=default_threads), default=jdefault))


# =============================================================================
# Stage encode (research env): research reference encoders on CPU fp32, each item alone
# =============================================================================
def f2_domains():
    it = json.load(open(INPUTS["enc_check_items"][0]))["val"]
    return it


def stage_encode(args):
    import torch
    import copy
    ins = verify_inputs(STAGE_INPUTS["encode"])
    s3 = f"{WORK}/stage_esm3.npz"
    ins["stage_esm3"] = dict(path=s3, md5=md5_file(s3), expected_md5=None, bytes=os.path.getsize(s3))
    os.environ["EMBCOMP_DATA"] = EMB
    sys.path.insert(0, f"{EMB}/sw_mlp")
    import sw_common as C
    assert os.path.realpath(C.__file__) == os.path.realpath(INPUTS["sw_common"][0])
    torch.set_num_threads(args.threads)
    st = torch.load(INPUTS["encoders"][0], map_location="cpu", weights_only=False)  # research checkpoint (md5-checked)
    model = C.build_model(st["cfg"])
    model.load_state_dict(st["state"])
    model.eval()
    m64 = copy.deepcopy(model).double().eval()
    Wt = C.Whitener(torch.device("cpu"))
    dims, off64, M64 = C.zcw50_clean_map()
    assert len(dims) == 1534 and not set(DIMS_DROPPED) & set(dims.tolist())

    # ---- rows per set -------------------------------------------------------------------------------------------
    sets = {}
    z = np.load(INPUTS["val_emb"][0])
    ids, voff = z["domain_ids"].astype(str), z["offsets"]
    pos = {d: k for k, d in enumerate(ids.tolist())}
    cand = f2_domains()
    lens = {d: int(voff[pos[d] + 1] - voff[pos[d]]) for d in cand}
    pick = sorted(cand, key=lambda d: (lens[d], d))[:5]           # the 5 shortest enc_check val domains
    emb = z["emb"]
    assert emb.dtype == np.float16
    sets["F2"] = [(d, np.ascontiguousarray(emb[voff[pos[d]]:voff[pos[d] + 1]]), str(z["sequences"][pos[d]]))
                  for d in pick]
    del emb, z
    import torch as _t
    f2b, f2b_inner = [], []
    for nm in ("1acb.A", "2uzi.A"):
        x = _t.load(INPUTS[f"pt_{nm.replace('.', '_')}"][0], map_location="cpu", weights_only=True)
        x16 = x.to(_t.float16).numpy()
        assert np.array_equal(x16, x.numpy().astype(np.float16))
        f2b.append((nm, x16, ""))
        f2b_inner.append((nm, np.ascontiguousarray(x16[1:-1]), ""))
    sets["F2b"], sets["F2b_inner"] = f2b, f2b_inner
    e3 = np.load(s3)
    names, seqs, kind = e3["names"].astype(str), e3["sequences"].astype(str), e3["kind"].astype(str)
    ro, rows = e3["rows_offsets"], e3["rows"]
    for tag, kd in (("F2c", "fasta"), ("Fdeg", "deg")):
        sets[tag] = [(names[k], np.ascontiguousarray(rows[ro[k] + 1:ro[k + 1] - 1]), seqs[k])
                     for k in range(len(names)) if kind[k] == kd]
        for nm, r, s in sets[tag]:
            assert len(r) == len(s), (nm, len(r), len(s))

    # ---- encode --------------------------------------------------------------------------------------------------
    def enc_item(r16):
        with torch.no_grad():
            xw = Wt(r16)                                    # S2 (research Whitener: fp32 math, fp16 out)
            hq = model.f_q(xw.float())                      # model.query(xw)
            ht = model.f_t(xw.float())
            q = hq.half().numpy()
            code = model.quant.encode(ht)                   # np.packbits((h >= 0), axis=1), MSB first
            x64 = xw.double()
            hq64, ht64 = m64.f_q(x64).numpy(), m64.f_t(x64).numpy()
        # float64 whitening path (for fp16-boundary sensitivity of xw only; not a spec path)
        xw64 = ((np.asarray(r16, np.float64)[:, dims] - off64) @ M64).astype(np.float16)
        return dict(xw=xw.numpy(), hq=hq.numpy(), ht=ht.numpy(), q=q, code=code, hq64=hq64, ht64=ht64,
                    xw_ulp_vs_f64path=fp16_ulp_dist(xw.numpy(), xw64))

    def encode_all():
        res = {}
        for sname, items in sets.items():
            res[sname] = [enc_item(r) for _, r, _ in items]
        return res

    t = time.time()
    R = encode_all()
    enc_s = time.time() - t
    torch.set_num_threads(1 if args.threads != 1 else 4)
    R2 = encode_all()
    torch.set_num_threads(args.threads)

    arrays, stats = {}, {}
    for sname, items in sets.items():
        res = R[sname]
        codes = np.concatenate([x["code"] for x in res])
        q = np.concatenate([x["q"] for x in res])
        ht64 = np.concatenate([x["ht64"] for x in res])
        ht = np.concatenate([x["ht"] for x in res])
        hq64 = np.concatenate([x["hq64"] for x in res])
        margin = np.abs(ht64).min(axis=1).astype(np.float32)
        sign_mis = int(((ht >= 0) != (ht64 >= 0)).sum())
        q_ulp_f64 = fp16_ulp_dist(q, hq64.astype(np.float16))
        xw_ulp = np.concatenate([x["xw_ulp_vs_f64path"].ravel() for x in res])
        codes2 = np.concatenate([x["code"] for x in R2[sname]])
        q2 = np.concatenate([x["q"] for x in R2[sname]])
        arrays[f"{sname}/ids"] = np.array([i for i, _, _ in items])
        arrays[f"{sname}/sequences"] = np.array([s for _, _, s in items])
        arrays[f"{sname}/offsets"] = offsets_of([len(r) for _, r, _ in items])
        arrays[f"{sname}/codes"] = codes
        arrays[f"{sname}/q"] = q
        arrays[f"{sname}/margin_t"] = margin
        stats[sname] = dict(
            n_items=len(items), n_rows=int(len(codes)), n_bits=int(codes.size * 8),
            code_bits_vs_f64_sign_mismatch=sign_mis, min_abs_h_t=float(np.abs(ht64).min()),
            n_abs_h_t_lt_1e_4=int((np.abs(ht64) < 1e-4).sum()), n_abs_h_t_lt_1e_3=int((np.abs(ht64) < 1e-3).sum()),
            q_max_ulp_vs_fp16_of_f64=int(q_ulp_f64.max()), q_frac_ulp0_vs_f64=float((q_ulp_f64 == 0).mean()),
            xw_elems=int(xw_ulp.size), xw_ulp_gt0_vs_f64_whitening=int((xw_ulp > 0).sum()),
            xw_max_ulp_vs_f64_whitening=int(xw_ulp.max()),
            thread_repeat_codes_identical=bool(np.array_equal(codes, codes2)),
            thread_repeat_q_identical=bool(np.array_equal(q, q2)))
    # F2b_inner vs F2b[1:-1] (row independence under a different GEMM shape)
    o, oi = arrays["F2b/offsets"], arrays["F2b_inner/offsets"]
    inner_eq = []
    for k in range(len(o) - 1):
        a = arrays["F2b/codes"][o[k] + 1:o[k + 1] - 1]
        b = arrays["F2b_inner/codes"][oi[k]:oi[k + 1]]
        qa = arrays["F2b/q"][o[k] + 1:o[k + 1] - 1]
        qb = arrays["F2b_inner/q"][oi[k]:oi[k + 1]]
        inner_eq.append(dict(codes_identical=bool(np.array_equal(a, b)),
                             q_max_ulp=int(fp16_ulp_dist(qa, qb).max())))
    stats["F2b_inner_vs_full_slice"] = inner_eq
    # debugging heads (S2/S3 intermediates) for the first F2 domain, first 8 rows
    h = R["F2"][0]
    arrays["F2/head_xw"] = h["xw"][:8]
    arrays["F2/head_h_q"] = h["hq"][:8].astype(np.float32)
    arrays["F2/head_h_t"] = h["ht"][:8].astype(np.float32)
    tag = args.tag
    write_npz(f"{WORK}/stage_encode_{tag}.npz", arrays)
    info = dict(stage="encode", tag=tag, generator_md5=generator_md5(),
                created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                inputs=ins, env=dict(base_env(), torch=torch.__version__, torch_threads=args.threads,
                                     repeat_threads=(1 if args.threads != 1 else 4)),
                encoder=dict(checkpoint_epoch=int(st["epoch"]), cfg=st["cfg"],
                             code=("Whitener(x16) -> f_t(xw.float()) -> np.packbits(h >= 0, axis=1) (MSB first), "
                                   "q = f_q(xw.float()).half(); each item encoded alone; torch CPU fp32, no autocast"),
                             f2_domains=pick, f2_domain_lengths=[lens[d] for d in pick], encode_s=round(enc_s, 1)),
                stats=stats, stage_npz_md5=md5_file(f"{WORK}/stage_encode_{tag}.npz"))
    jdump(info, f"{WORK}/stage_encode_{tag}.json")
    log(json.dumps(stats, default=jdefault))


# =============================================================================
# Stage build (research env): expected values + fixture files
# =============================================================================
class Ctx:
    pass


def research_modules():
    c = Ctx()
    os.environ["EMBCOMP_DATA"] = EMB
    sys.path.insert(0, f"{EMB}/harness")
    import harness as H
    sys.path.insert(0, f"{EMB}/panel/fit/scripts")
    import cath_score as CS                      # loads panel/fit/kernel swfast.py + libswkern.dylib + libswmulti.so
    sys.path.insert(0, f"{EMB}/panel/fit2/scripts")
    import bench_validate as BV
    c.H, c.CS, c.F, c.BV = H, CS, CS.F, BV
    c.verbatim = bool(H.check_verbatim())
    c.paths = dict(harness=H.__file__, cath_score=CS.__file__, swfast=CS.F.__file__, bench_validate=BV.__file__,
                   libswkern=os.path.join(os.path.dirname(CS.F.__file__), "libswkern.dylib"),
                   libswmulti=os.path.join(CS.KDIR, "libswmulti.so"))
    c.vector_lanes = int(CS.F.VECTOR_LANES)
    return c


def kernel_pair_diag(F, q16, codes):
    """research kernel without S7 (diagnostic only); None when the research query_prep is non-finite (zero-norm row
    divides by 0, NaN / Inf input), which the research kernel was never meant to receive."""
    with np.errstate(all="ignore"):
        qp = F.query_prep(np.asarray(q16).astype(np.float32))
        if not np.isfinite(qp).all():
            return None, None
        sf, _, sr, _ = kernel_pair(F, q16, codes)
    return sf, sr


def kernel_pair(F, q16, codes):
    """inventory_probe/probe.py kernel(): research C kernel, fp32; S_r from the row-reversed similarity copy."""
    qp = F.query_prep(np.asarray(q16).astype(np.float32))
    sim = np.ascontiguousarray(qp @ F.decode_unpackbits(codes).T)
    col = np.array([0, len(codes)], np.int64)
    sf, ef, pflat, poff = F.swz_ad_batch(sim, col, OFF, GO, GE, tb=True)
    simr = np.ascontiguousarray(sim[::-1])
    sr, _ = F.swz_ad_batch(simr, col, OFF, GO, GE, tb=False)
    return float(sf[0]), pflat[poff[0]:poff[1]].copy(), float(sr[0]), ef[0].astype(np.int32)


def q90_research(c, P, q_list):
    """panel/fit2/scripts/panel_local.py arithmetic: cath_score.query_scores (swmulti swz_ad_multi_f32, R = 2) per
    query vs the shipped panel, Q90 = np.quantile(f64(S_f) - f64(S_r), 0.9). Non-finite scores -> NaN (diagnostic)."""
    CS, F = c.CS, c.F
    q16 = np.concatenate([np.asarray(q, np.float16) for q in q_list])
    qoff = offsets_of([len(q) for q in q_list])
    with np.errstate(all="ignore"):
        CS.G["QP"] = F.query_prep(q16)
    CS.G["qoff"], CS.G["codes"], CS.G["toff"], CS.G["NT"] = qoff, np.ascontiguousarray(P["codes"]), P["off"], P["NT"]
    out = []
    for i in range(len(q_list)):
        if not np.isfinite(CS.G["QP"][qoff[i]:qoff[i + 1]]).all():
            out.append(float("nan"))
            continue
        try:
            with np.errstate(all="ignore"):
                _, o = CS.query_scores(i)
            out.append(float(np.quantile(o[0].astype(np.float64) - o[1].astype(np.float64), 0.9)))
        except AssertionError:
            out.append(float("nan"))
    return np.array(out, np.float64)


def lnE_research(BV, tab, S_prime, m, N):
    """bench_validate.counts(): L = (lnK + ln m + ln N) - lam * S'  with lam, lnK = tabp(tab, m) (np.interp, clamped)."""
    lam, lnk = BV.tabp(tab, np.asarray(m, np.float64))
    return (lnk + np.log(np.asarray(m, np.float64)) + math.log(N)) - lam * np.asarray(S_prime, np.float64)


def rel(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float((np.abs(a - b) / np.maximum(np.abs(b), 1.0)).max()) if a.size else 0.0


def build_F1(c, P, tab, R):
    z = np.load(INPUTS["fixture_candidate"][0])
    rep = json.load(open(INPUTS["probe_report"][0]))["rows"]
    pairs = z["pairs"].astype(str)
    n = len(pairs)
    q_ids, t_ids = z["q_ids"].astype(str), z["t_ids"].astype(str)
    assert np.array_equal(q_ids, pairs[:, 0]) and np.array_equal(t_ids, pairs[:, 1]) and n == 50
    assert all(r["a"] == a and r["b"] == b for r, (a, b) in zip(rep, pairs))
    split = np.array([r["split"] for r in rep])
    stratum = np.array([r["stratum"] for r in rep])
    qo, to = z["q_offsets"].astype(np.int64), z["t_offsets"].astype(np.int64)
    enc = {s: np.load(INPUTS[f"enc_{s}"][0]) for s in ("val", "test")}
    eidx = {s: {x: k for k, x in enumerate(enc[s]["ids"].astype(str))} for s in enc}
    f1s = {s: np.load(INPUTS[f"f1_settings_{s}"][0]) for s in ("val", "test")}
    cs = np.load(INPUTS["cath_set"][0])
    cpos = {x: k for k, x in enumerate(cs["ids"].astype(str))}
    cath_q90 = np.load(INPUTS["cath_q90_local"][0])
    qs, cds, qseq, tseq = [], [], [], []
    S_f, S_r, S_f64, S_r64, ends, paths = [], [], [], [], [], []
    chk = dict(src_q_equal=0, src_codes_equal=0, cand_Sf_bit_equal=0, cand_Sr_bit_equal=0, cand_path_equal=0,
               path_eq_f64=0, path_forward=0, path_eq_stored_nersc=0, s7_noop_scores=0, s7_noop_paths=0,
               seq_len_ok=0)
    n_al, n_cols, n_id = [], [], []
    for k in range(n):
        a, b = pairs[k]
        s = split[k]
        qa = z["q"][qo[k]:qo[k + 1]]
        cb = z["codes"][to[k]:to[k + 1]]
        o = enc[s]["offsets"]
        ia, ib = eidx[s][a], eidx[s][b]
        chk["src_q_equal"] += int(np.array_equal(qa, enc[s]["q"][o[ia]:o[ia + 1]]))
        chk["src_codes_equal"] += int(np.array_equal(cb, enc[s]["codes"][o[ib]:o[ib + 1]]))
        sf, path, sr, end = kernel_pair(c.F, qa, cb)
        cp = z["path_flat"][z["path_off"][k]:z["path_off"][k + 1]]
        chk["cand_Sf_bit_equal"] += int(sf == z["S_f"][k])
        chk["cand_Sr_bit_equal"] += int(sr == z["S_r"][k])
        chk["cand_path_equal"] += int(np.array_equal(path, cp))
        ref = ref_pair64(c.H, qa, cb, s7=True)
        plain = ref_pair64(c.H, qa, cb, s7=False)
        chk["s7_noop_scores"] += int(ref["S_f"] == plain["S_f"] and ref["S_r"] == plain["S_r"])
        chk["s7_noop_paths"] += int(np.array_equal(ref["path"], plain["path"]))
        chk["path_eq_f64"] += int(np.array_equal(path, ref["path"]))
        kp = aligned_pairs(path)
        fwd = len(path) > 0 and path[0, 0] <= path[-1, 0] and path[0, 1] <= path[-1, 1] and \
            bool(np.all(np.diff(kp[:, 0]) > 0)) and bool(np.all(np.diff(kp[:, 1]) > 0))
        chk["path_forward"] += int(fwd)
        g = f1s[s]
        ga, gb = g[f"{stratum[k]}/a"].astype(str), g[f"{stratum[k]}/b"].astype(str)
        gi = int(np.flatnonzero((ga == a) & (gb == b))[0])
        go = g[f"{stratum[k]}/aln_D_off"]
        chk["path_eq_stored_nersc"] += int(np.array_equal(kp, g[f"{stratum[k]}/aln_D_flat"][go[gi]:go[gi + 1]]))
        sa, sb = str(cs["sequences"][cpos[a]]), str(cs["sequences"][cpos[b]])
        chk["seq_len_ok"] += int(len(sa) == len(qa) and len(sb) == len(cb))
        n_al.append(len(kp))
        n_cols.append(len(path))
        n_id.append(int(sum(sa[i] == sb[j] for i, j in kp)))
        qs.append(qa)
        cds.append(cb)
        qseq.append(sa)
        tseq.append(sb)
        S_f.append(sf)
        S_r.append(sr)
        S_f64.append(ref["S_f"])
        S_r64.append(ref["S_r"])
        ends.append(end)
        paths.append(path)
    S_f, S_r, S_f64, S_r64 = map(lambda v: np.array(v, np.float64), (S_f, S_r, S_f64, S_r64))
    La = np.array([len(q) for q in qs], np.int64)
    Lb = np.array([len(x) for x in cds], np.int64)
    log("F1: kernel / harness pairs done; research-driver Q90 ...")
    q90 = q90_research(c, P, qs)
    q90_f64 = np.array([R[("F1", k)]["q90"] for k in range(n)])
    S = S_f - S_r
    Sp = S - q90
    lnE = lnE_research(c.BV, tab, Sp, La, N_FIX)
    arrays = dict(
        pair_ids=pairs, pair_split=split, pair_stratum=stratum, pair_q=np.arange(n, dtype=np.int32),
        pair_t=np.arange(n, dtype=np.int32),
        q_ids=q_ids, q_offsets=offsets_of(La), q=np.concatenate(qs).astype(np.float16), q_sequences=np.array(qseq),
        t_ids=t_ids, t_offsets=offsets_of(Lb), codes=np.concatenate(cds).astype(np.uint8), t_sequences=np.array(tseq),
        S_f=S_f, S_r=S_r, S=S, S_f_f64=S_f64, S_r_f64=S_r64,
        end_1based=np.array(ends, np.int32), path_flat=np.concatenate(paths).astype(np.int32),
        path_off=offsets_of([len(p) for p in paths]),
        n_aligned=np.array(n_al, np.int32), aln_cols=np.array(n_cols, np.int32), n_ident=np.array(n_id, np.int32),
        q90=q90, q90_f64=q90_f64, S_prime=Sp, lnE_N1e9=lnE)
    ev_le = {f"{s}/{t}": int(((split == s) & (stratum == t) & (lnE <= math.log(1e-3))).sum())
             for s, t in (("val", "related"), ("val", "remote"), ("test", "remote"), ("val", "unrelated"),
                          ("test", "unrelated"))}
    cand_lnE = np.log(z["E_N1e9"])
    chk.update(
        n=n, max_rel_Sf_kernel_vs_f64=rel(S_f, S_f64), max_rel_Sr_kernel_vs_f64=rel(S_r, S_r64),
        max_abs_S_kernel_vs_f64=float(np.abs(S - (S_f64 - S_r64)).max()),
        q90_bit_equal_candidate=int((q90 == z["q90"]).sum()), max_abs_q90_vs_candidate=float(np.abs(q90 - z["q90"]).max()),
        max_abs_q90_research_vs_f64=float(np.abs(q90 - q90_f64).max()),
        max_abs_q90_vs_cath_q90_local=float(np.abs(q90 - np.array([cath_q90[cpos[a]] for a in q_ids])).max()),
        max_abs_lnE_vs_candidate=float(np.abs(lnE - cand_lnE).max()),
        harness_q90_degenerate_rows_cols=[int(sum(R[("F1", k)]["ndeg_rows"] for k in range(n))),
                                          int(sum(R[("F1", k)]["ndeg_cols"] for k in range(n)))],
        n_lnE_le_ln_1e_3=int((lnE <= math.log(1e-3)).sum()), lnE_le_1e_3_by_stratum=ev_le,
        lnE_range=[float(lnE.min()), float(lnE.max())], La_range=[int(La.min()), int(La.max())],
        Lb_range=[int(Lb.min()), int(Lb.max())], n_La_below_first_centre=int((La < tab["center_len"][0]).sum()),
        n_La_above_last_centre=int((La > tab["center_len"][-1]).sum()),
        empty_alignments=int(sum(x == 0 for x in n_al)))
    meta = dict(
        fixture="F1", system=SYSTEM, calibration_version=CAL_VERSION, table=TABLE_KEY, N=N_FIX,
        setting=dict(offset=OFF, gap_open=GO, gap_extend=GE),
        source="inventory_probe/fixture_candidate.npz (50 CATH pairs: 10 each val/related, val/remote, test/remote, "
               "val/unrelated, test/unrelated; q = sw_mlp enc fp16, codes = sw_mlp enc)",
        arrays={
            "pair_q/pair_t": "pair k uses query pair_q[k] and target pair_t[k] (here k, k)",
            "q/q_offsets": "query encodings fp16 [sum La, 64]", "codes/t_offsets": "target codes uint8 [sum Lb, 8]",
            "S_f/S_r": "research C kernel (fp32 result as float64): swz_ad_batch on qp @ decode(codes).T; S_r on the "
                       "row-reversed similarity (= S9 reversed view)",
            "S": "fp64(S_f) - fp64(S_r)", "S_f_f64/S_r_f64": "float64 harness oracle (+S5/S7 wrappers, no-ops here)",
            "end_1based": "kernel best cell (i, j), 1-based",
            "path_flat/path_off": "forward traceback of S_f exactly as the research kernel writes it: [k, 2] "
                                  "(query i, target j), 0-based, -1 = gap, FORWARD order (first column first)",
            "n_aligned/aln_cols/n_ident": "aligned pairs, path columns, exact-letter identities along the path "
                                          "(q_sequences / t_sequences from cath_set.npz)",
            "q90": "research panel driver (cath_score.query_scores, shipped panel) np.quantile(f64 S, 0.9)",
            "q90_f64": "float64 harness Q90 over the same panel (diagnostic)",
            "S_prime/lnE_N1e9": "S - q90; lnE = (lnK(m) + ln m + ln N) - lam(m) S' at N = 1e9, m = La, "
                                "bench_validate.tabp (np.interp in ln m, clamped)"},
        checks=chk)
    arrays["meta_json"] = meta_arr(meta)
    return arrays, chk


def load_encode(tag):
    p = f"{WORK}/stage_encode_{tag}.npz"
    if not os.path.exists(p):
        return None, None
    return dict(np.load(p)), json.load(open(f"{WORK}/stage_encode_{tag}.json"))


def cross_env(E, Efm, sname):
    """reference encoder under the research env torch vs the foldmatch env torch (same sw_common code, same Mac)."""
    if Efm is None:
        return None
    a, b = E[f"{sname}/codes"], Efm[f"{sname}/codes"]
    ba, bb = np.unpackbits(a, axis=1), np.unpackbits(b, axis=1)
    return dict(bits=int(ba.size), bits_equal=int((ba == bb).sum()), q=q_diff_stats(E[f"{sname}/q"], Efm[f"{sname}/q"]))


def build_F2(E, Efm, einfo):
    ids = E["F2/ids"].astype(str)
    off = E["F2/offsets"]
    ec = np.load(INPUTS["enc_check"][0])
    z = np.load(INPUTS["val_emb"][0])
    pos = {d: k for k, d in enumerate(z["domain_ids"].astype(str).tolist())}
    rows = np.concatenate([z["emb"][z["offsets"][pos[d]]:z["offsets"][pos[d] + 1]] for d in ids])
    del z
    cn = np.concatenate([ec[f"val/code/{d}"] for d in ids])
    qn = np.concatenate([ec[f"val/q/{d}"] for d in ids])
    codes, q = E["F2/codes"], E["F2/q"]
    assert len(rows) == off[-1] == len(cn)
    b1, b2 = np.unpackbits(codes, axis=1), np.unpackbits(cn, axis=1)
    qs = q_diff_stats(q, qn)
    chk = dict(domains=ids.tolist(), lengths=np.diff(off).tolist(), rows=int(off[-1]), bits=int(b1.size),
               bits_equal_ref_vs_nersc=int((b1 == b2).sum()), q_max_ulp_ref_vs_nersc=qs["max_elem_ulp"],
               q_ref_vs_nersc=qs, ref_stats=einfo["stats"]["F2"], cross_torch_env=cross_env(E, Efm, "F2"))
    meta = dict(fixture="F2", source="embcomp/val.npz emb rows (fp16, sequence-only ESM3 store, BOS/EOS removed) of the "
                                     "5 shortest sw_mlp/enc_check val domains",
                arrays={"rows/offsets": "raw fp16 [sum L, 1536] (S1 rows 1..L)",
                        "codes/q": "research reference encoder (sw_common, CPU fp32 torch, each domain alone): "
                                   "codes uint8 [L, 8] = packbits(f_t(xw) >= 0) MSB-first, q = fp16(f_q(xw))",
                        "codes_nersc/q_nersc": "NERSC stored encodings (sw_mlp/enc_check/L2m_B8_s0.npz val/*)",
                        "margin_t": "per row min_k |h_t,k| of a float64 re-evaluation from the same xw (fragility)",
                        "head_*": "first domain, first 8 rows: xw fp16 [8, 1534] (S2 output), f_q / f_t outputs fp32"},
                checks=chk)
    arrays = dict(ids=ids, sequences=E["F2/sequences"], offsets=off, rows=rows.astype(np.float16), codes=codes, q=q,
                  codes_nersc=cn, q_nersc=qn, margin_t=E["F2/margin_t"], head_xw=E["F2/head_xw"],
                  head_h_q=E["F2/head_h_q"], head_h_t=E["F2/head_h_t"],
                  meta_json=meta_arr(meta))
    return arrays, chk


def build_F2b(E, Efm, einfo, ins):
    ids = E["F2b/ids"].astype(str)
    chk = dict(rows=np.diff(E["F2b/offsets"]).tolist(), ref_stats=einfo["stats"]["F2b"],
               inner_slice=einfo["stats"]["F2b_inner_vs_full_slice"], cross_torch_env=cross_env(E, Efm, "F2b"))
    meta = dict(fixture="F2b", note="ARITHMETIC ONLY, out of distribution: structure-derived FoldMatch ESM3 rows of other "
                                    "chains (not the FASTA sequences); rows are NOT stored here, load them with "
                                    "torch.load(path, map_location='cpu', weights_only=True) (fp32 [R, 1536]); every "
                                    "row is encoded (no BOS/EOS stripping); S2 starts with fp16(x)",
                source_files=[f"tests/resources/embeddings/{i}.pt" for i in ids],
                source_md5=[ins[f"pt_{i.replace('.', '_')}"]["md5"] for i in ids],
                arrays={"codes/q/offsets": "research reference encoder on all rows of each file, each file alone",
                        "margin_t": "per row min_k |h_t,k| (float64 re-evaluation)"}, checks=chk)
    arrays = dict(ids=ids, source_files=np.array(meta["source_files"]), source_md5=np.array(meta["source_md5"]),
                  offsets=E["F2b/offsets"], codes=E["F2b/codes"], q=E["F2b/q"], margin_t=E["F2b/margin_t"],
                  meta_json=meta_arr(meta))
    return arrays, chk


def build_F2c(E, Efm, einfo, e3, e3info):
    k = [i for i, x in enumerate(e3["kind"].astype(str)) if x == "fasta"]
    ro = e3["rows_offsets"]
    rows = np.concatenate([e3["rows"][ro[i]:ro[i + 1]] for i in k])
    ids = e3["names"].astype(str)[k]
    seqs = e3["sequences"].astype(str)[k]
    assert np.array_equal(ids, E["F2c/ids"].astype(str))
    items = [x for x in e3info["items"] if x["name"] in set(ids.tolist())]
    chk = dict(ids=ids.tolist(), lengths=[len(s) for s in seqs], total_residues=int(sum(len(s) for s in seqs)),
               esm3_items=items, ref_stats=einfo["stats"]["F2c"], cross_torch_env=cross_env(E, Efm, "F2c"))
    meta = dict(fixture="F2c", source="FoldMatch sequence-only ESM3 on CPU fp32 (existing FoldMatch code, read-only) for "
                                      "tests/resources/fasta/test_sequences.fasta",
                esm3={k: v for k, v in e3info["esm3"].items()
                      if not (k.endswith("_s") or k.startswith("pass_s"))},   # no timings: reproducible bytes
                foldmatch=dict(git=e3info["env"]["foldmatch_git"],
                                                    dist_version=e3info["env"]["foldmatch_dist_version"]),
                env=dict(torch=e3info["env"]["torch"], esm=e3info["env"]["esm"], platform=e3info["env"]["platform"]),
                arrays={"rows/rows_offsets": "per_residue_embedding [L+2, 1536] (BOS, residues, EOS) cast to fp16",
                        "codes/q/offsets": "research reference encoder on rows[1:-1] of each entry, each entry alone",
                        "margin_t": "per row min_k |h_t,k| (float64 re-evaluation)"}, checks=chk)
    arrays = dict(ids=ids, sequences=seqs, lengths=np.array([len(s) for s in seqs], np.int64),
                  rows_offsets=offsets_of([ro[i + 1] - ro[i] for i in k]), rows=rows, offsets=E["F2c/offsets"],
                  codes=E["F2c/codes"], q=E["F2c/q"], margin_t=E["F2c/margin_t"],
                  meta_json=meta_arr(meta))
    return arrays, chk


def f3_select(tab):
    zq = np.load(INPUTS["bench_queries"][0])
    L, off, ids = zq["lengths"].astype(np.int64), zq["offsets"].astype(np.int64), zq["ids"].astype(str)
    sc = zq["seq_concat"].tobytes().decode("ascii")
    seqs = [sc[off[i]:off[i + 1]] for i in range(len(ids))]
    seen, elig = set(), []
    for i, s in enumerate(seqs):
        if set(s) <= STD20 and s not in seen:
            elig.append(i)
        seen.add(s)
    elig = np.array(elig)
    cen = list(tab["center_len"])
    edges = [21.0] + cen + [967.0]
    counts = [3, 3, 3, 3, 3, 3, 2]
    rng = np.random.default_rng(20260927)
    pick = []
    for b in range(7):
        cand = elig[(L[elig] >= edges[b]) & (L[elig] < edges[b + 1])]
        forced = []
        if b == 0:
            forced = [int(cand[np.argmin(L[cand])])]
        if b == 6:
            forced = [int(cand[np.argmax(L[cand])])]
        rest = np.setdiff1d(cand, forced)
        pick += forced + sorted(rng.choice(rest, counts[b] - len(forced), replace=False).tolist())
    pick = sorted(pick, key=lambda i: (L[i], ids[i]))
    return zq, pick, seqs


def build_F3(c, P, tab, R):
    zq, pick, seqs = f3_select(tab)
    off = zq["offsets"]
    qs = [zq["q"][off[i]:off[i + 1]] for i in pick]
    q90 = q90_research(c, P, qs)
    q90_local = np.load(INPUTS["bench_q90_local"][0])[pick]
    q90_nersc = np.load(INPUTS["bench_q90_nersc"][0])[pick]
    q90_f64 = np.array([R[("F3", k)]["q90"] for k in range(len(pick))])
    L = np.array([len(q) for q in qs], np.int64)
    chk = dict(n=len(pick), bench_index=pick, lengths=L.tolist(), total_residues=int(L.sum()),
               q90_bit_equal_stored_local=int((q90 == q90_local).sum()),
               max_abs_q90_vs_stored_local=float(np.abs(q90 - q90_local).max()),
               max_abs_q90_vs_nersc_score2=float(np.abs(q90 - q90_nersc).max()),
               max_abs_q90_research_vs_f64=float(np.abs(q90 - q90_f64).max()),
               harness_q90_degenerate_rows_cols=[int(sum(R[("F3", k)]["ndeg_rows"] for k in range(len(pick)))),
                                                 int(sum(R[("F3", k)]["ndeg_cols"] for k in range(len(pick))))],
               q90_range=[float(q90.min()), float(q90.max())])
    meta = dict(fixture="F3", source="evalue/encode/out/queries.L2m_B8_s0.npz (1,465 benchmark queries); 20 chosen: "
                                     "3 per table length bin [21, c1), [c1, c2) ... [c5, c6) and 2 in [c6, 966] "
                                     "(c = ship_final centres), standard-20 letters, first copy of each sequence, "
                                     "shortest (21) and longest (966) forced, rest default_rng(20260927)",
                arrays={"q/offsets": "query encodings fp16", "q90": "research panel driver (cath_score.query_scores) "
                        "vs shipped_panel_codes.npz, np.quantile(f64 S_f - f64 S_r, 0.9); = panel_local.py",
                        "q90_stored_local": "panel/fit2/results/bench_q90_shipped_local.npy (Mac rescoring)",
                        "q90_nersc": "panel/fit2/results/bench_q90_shipped.npy (NERSC score2, used by fit_final2.py)",
                        "q90_f64": "float64 harness Q90 (diagnostic)"}, checks=chk)
    arrays = dict(ids=zq["ids"].astype(str)[pick], bench_index=np.array(pick, np.int32), lengths=L,
                  sequences=np.array([seqs[i] for i in pick]), offsets=offsets_of(L),
                  q=np.concatenate(qs).astype(np.float16), q90=q90, q90_stored_local=q90_local, q90_nersc=q90_nersc,
                  q90_f64=q90_f64, meta_json=meta_arr(meta))
    return arrays, chk


def build_F4(c, tab, ins):
    cen = np.array(tab["center_len"], np.float64)
    lam, lnk = np.array(tab["lam"], np.float64), np.array(tab["lnK"], np.float64)
    lnc = np.log(cen)
    s_lam = (lam[5] - lam[4]) / (lnc[5] - lnc[4])
    s_lnk = (lnk[5] - lnk[4]) / (lnc[5] - lnc[4])
    ms = [10.0, 32.58, 50.0, 100.0, 594.92, 1000.0, 5000.0] + [float(x) for x in cen]
    Ns = [1.0, 3.6e5, 1e9, 1e11]
    Sps = [-5.0, 0.0, 12.3, 80.0]
    rows = []
    ln2 = math.log(2.0)

    def vals(la, lk, m, N, Sp):
        lnE = (lk + math.log(m) + math.log(N)) - la * Sp
        bits = (la * Sp - lk) / ln2
        try:
            E = math.exp(lnE)
            E = E if 0.0 < E < float("inf") else None
        except OverflowError:
            E = None
        return dict(lnE=lnE, E=E, bits=bits, render_bits=math.floor(bits + 0.5))

    for m in ms:
        la_a, lk_a = c.BV.tabp(tab, m)
        la, lk = float(la_a), float(lk_a)
        region = "below_first_centre" if m < cen[0] else ("above_last_centre" if m > cen[-1] else "inside")
        at_centre = [i for i in range(6) if m == cen[i]]
        for N in Ns:
            for Sp in Sps:
                r = dict(m=m, N=N, S_prime=Sp, region=region, at_centre=(at_centre[0] if at_centre else None),
                         lam=la, lnK=lk, **vals(la, lk, m, N, Sp))
                if region == "above_last_centre":
                    lx = float(lam[5] + s_lam * (math.log(m) - lnc[5]))
                    kx = float(lnk[5] + s_lnk * (math.log(m) - lnc[5]))
                    r["extrapolated"] = dict(lam=lx, lnK=kx, **vals(lx, kx, m, N, Sp))
                rows.append(r)
    doc = dict(
        fixture="F4", system=SYSTEM, calibration_version=CAL_VERSION, table_key=TABLE_KEY,
        source=dict(path="panel/fit2/results/ship_final.json", md5=ins["ship_final"]["md5"],
                    function="panel/fit2/scripts/bench_validate.py tabp (np.interp(ln m, ln centre, table))"),
        table=dict(center_len=tab["center_len"], lam=tab["lam"], lnK=tab["lnK"]),
        formula=dict(lnE="(lnK(m) + ln m + ln N) - lam(m) * S'   (float64, left to right)",
                     bits="(lam(m) * S' - lnK(m)) / ln 2", render_bits="floor(bits + 0.5)",
                     E="exp(lnE) when finite and > 0, else null (render E from lnE)"),
        rules=dict(
            default_columns="lam/lnK/lnE/bits/render_bits: research rule = np.interp in ln m, CLAMPED at both ends "
                            "(S11 low clamp is normative; the high-m clamp is only valid if D15 keeps it)",
            extrapolated="rows with m > last centre also carry 'extrapolated': lam and lnK continued linearly in ln m "
                         "with the last segment's slope (centres 5 -> 6), UNCAPPED; D15 (P0.7) decides between the "
                         "clamp and a capped extrapolation (cap = validated length, replace m by min(m, cap) in the "
                         "extrapolation only; ln m and ln N still use the true m)",
            note="m = 32.58 < centre 1 (32.5844) and m = 594.92 > centre 6 (594.9176): both are just outside the table"),
        last_segment_slopes=dict(lam=float(s_lam), lnK=float(s_lnk)),
        tolerance="plan: E arithmetic rel <= 1e-12 vs ship_final.json", rows=rows)
    return doc, dict(n_rows=len(rows), n_above=sum(r["region"] == "above_last_centre" for r in rows),
                     n_below=sum(r["region"] == "below_first_centre" for r in rows))


def build_Fdeg(c, P, tab, R, E, e3, einfo, F1a):
    H = c.H
    # base normal pair = F1 pair 0 (real CATH domains)
    bq = F1a["q"][F1a["q_offsets"][0]:F1a["q_offsets"][1]].copy()
    bt = F1a["codes"][F1a["t_offsets"][0]:F1a["t_offsets"][1]].copy()
    bq_id, bt_id = str(F1a["q_ids"][0]), str(F1a["t_ids"][0])
    zero_row = bq.copy()
    zero_row[5] = 0
    sub = bq.copy()
    sub[7] = 0
    sub[7, 3] = np.float16(2.0 ** -24)
    nanq = bq.copy()
    nanq[3, 7] = np.float16(np.nan)
    infq = bq.copy()
    infq[3, 7] = np.float16(np.inf)
    alt = np.stack([bt[0], bt[1]] * 20)
    rep_q, rep_t = np.repeat(bq[:1], 40, 0), np.repeat(bt[:1], 40, 0)
    syn = [
        ("n1", bq[:1], bt, False, "query n = 1 (every column std = 0)"),
        ("m1", bq, bt[:1], False, "target m = 1 (every row std = 0)"),
        ("n1_m1", bq[:1], bt[:1], False, "1 x 1"),
        ("identical_codes", bq, rep_t, False, "target of 40 identical codes (every row std = 0); S_f = S_r expected"),
        ("two_distinct_codes", bq, alt, False, "target alternating 2 codes (D14 boundary: 2 distinct codes)"),
        ("identical_q_rows", rep_q, bt, False, "query of 40 identical rows (every column std = 0); S_f = S_r"),
        ("identical_both", rep_q, rep_t, False, "both constant: z = 0 everywhere -> S = 0, empty"),
        ("zero_row_q", zero_row, bt, False, "query row 5 all zero (S5: qp row = 0; that row std = 0)"),
        ("all_zero_q", np.zeros((30, 64), np.float16), bt, False, "all-zero query: qp = 0, sim = 0, z = 0, S = 0"),
        ("subnormal_row_q", sub, bt, False, "row 7 = single fp16 subnormal 2^-24 (|q| = 6e-8 > 1e-12: normal prep)"),
        ("nan_q", nanq, bt, True, "NaN in q -> error (S5; kernel status 2)"),
        ("inf_q", infq, bt, True, "Inf in q -> error (S5; kernel status 2)"),
    ]
    sq, sc, s_res, diag = [], [], [], {}
    for name, q, t, err, _ in syn:
        sq.append(np.asarray(q, np.float16))
        sc.append(np.asarray(t, np.uint8))
        if err:
            try:
                ref_pair64(H, q, t)
                raise AssertionError("expected an error")
            except FloatingPointError:
                pass
            s_res.append(dict(S_f=np.nan, S_r=np.nan, path=np.zeros((0, 2), np.int32), ndeg_rows_f=-1, ndeg_cols_f=-1,
                              ndeg_rows_r=-1, ndeg_cols_r=-1))
        else:
            s_res.append(ref_pair64(H, q, t, s7=True))
        kf, kr = kernel_pair_diag(c.F, q, t)
        with np.errstate(all="ignore"):
            try:
                pl = ref_pair64(H, q, t, s7=False, path=False)
                pl = (pl["S_f"], pl["S_r"])
            except FloatingPointError:
                pl = (None, None)
        diag[name] = dict(research_kernel_noS7=[kf, kr], harness_f64_noS7=list(pl))
    # real ESM3 items
    ids = E["Fdeg/ids"].astype(str)
    seqs = E["Fdeg/sequences"].astype(str)
    off = E["Fdeg/offsets"]
    codes, qq = E["Fdeg/codes"], E["Fdeg/q"]
    kk = [i for i, x in enumerate(e3["kind"].astype(str)) if x == "deg"]
    ro = e3["rows_offsets"]
    rows = np.concatenate([e3["rows"][ro[i]:ro[i + 1]] for i in kk])
    assert np.array_equal(e3["names"].astype(str)[kk], ids)
    n_items = len(ids)
    rq = [qq[off[i]:off[i + 1]] for i in range(n_items)]
    rc = [codes[off[i]:off[i + 1]] for i in range(n_items)]
    n_dc = np.array([len(np.unique(x, axis=0)) for x in rc], np.int32)
    n_dq = np.array([len(np.unique(x, axis=0)) for x in rq], np.int32)
    nonstd = np.array([sum(ch not in STD20 for ch in s) / len(s) for s in seqs], np.float64)
    q90_f64 = np.array([R[("Fdeg", i)]["q90"] for i in range(n_items)])
    q90_kernel = q90_research(c, P, rq)
    base_q90 = float(F1a["q90"][0])
    # pairs: all item x item, item x base target, base query x item
    Q = rq + [bq]
    T = rc + [bt]
    qlab = list(ids) + [f"base:{bq_id}"]
    tlab = list(ids) + [f"base:{bt_id}"]
    q90s = list(q90_f64) + [base_q90]
    plist = [(i, j) for i in range(n_items) for j in range(n_items)] + [(i, n_items) for i in range(n_items)] + \
            [(n_items, j) for j in range(n_items)]
    pres, pdiag = [], []
    for i, j in plist:
        r = ref_pair64(H, Q[i], T[j], s7=True)
        pres.append(r)
        pdiag.append(list(kernel_pair_diag(c.F, Q[i], T[j])))
    pS_f = np.array([r["S_f"] for r in pres])
    pS_r = np.array([r["S_r"] for r in pres])
    pq90 = np.array([q90s[i] for i, _ in plist])
    m_len = np.array([len(Q[i]) for i, _ in plist], np.float64)
    lnE = lnE_research(c.BV, tab, (pS_f - pS_r) - pq90, m_len, N_FIX)

    def ecal(i, j):     # provisional: S13 + D14 recommended default
        ql, tl = len(Q[i]), len(T[j])
        qns = nonstd[i] if i < n_items else 0.0
        tns = nonstd[j] if j < n_items else 0.0
        tdc = n_dc[j] if j < n_items else len(np.unique(bt, axis=0))
        return int(21 <= ql <= 966 and tl <= 1000 and qns <= 0.01 and tns <= 0.01 and tdc >= 2)

    items_chk = [dict(id=ids[i], sequence=seqs[i], length=len(seqs[i]), n_distinct_codes=int(n_dc[i]),
                      n_distinct_q_rows=int(n_dq[i]), nonstandard_frac=float(nonstd[i]),
                      min_margin_t=float(E["Fdeg/margin_t"][off[i]:off[i + 1]].min()),
                      q90_f64_s7=float(q90_f64[i]), q90_research_kernel_noS7=float(q90_kernel[i]),
                      harness_q90_degenerate_rows_cols=[R[("Fdeg", i)]["ndeg_rows"], R[("Fdeg", i)]["ndeg_cols"]])
                 for i in range(n_items)]
    chk = dict(items=items_chk, synthetic={nm: dict(S_f=s["S_f"], S_r=s["S_r"], ndeg=[s["ndeg_rows_f"], s["ndeg_cols_f"],
                                                                                     s["ndeg_rows_r"], s["ndeg_cols_r"]],
                                                    **diag[nm]) for (nm, *_), s in zip(syn, s_res)},
               pairs_all_finite=bool(np.isfinite(pS_f).all() and np.isfinite(pS_r).all() and np.isfinite(lnE).all()),
               pairs_max_abs_S_f_minus_S_r_when_target_codes_single=float(max(
                   [abs(pres[k]["S_f"] - pres[k]["S_r"]) for k, (i, j) in enumerate(plist)
                    if j < n_items and n_dc[j] < 2] or [0.0])),
               pairs_research_kernel_noS7=[[qlab[i], tlab[j], pdiag[k]] for k, (i, j) in enumerate(plist)],
               esm3_items=[x for x in json.load(open(f"{WORK}/stage_esm3.json"))["items"] if x["name"] in set(ids)])
    meta = dict(
        fixture="F-deg", setting=dict(offset=OFF, gap_open=GO, gap_extend=GE), s7_eps=S7_EPS, qprep_eps=QPREP_EPS,
        base_pair=dict(query=bq_id, target=bt_id, note="F1 pair 0 (real CATH domains); base_q90 = F1 q90[0]"),
        synthetic_cases=[dict(id=nm, expect_error=err, description=d) for nm, _, _, err, d in syn],
        real_items="FoldMatch sequence-only ESM3 CPU fp32 rows (as F2c) + research reference encoder on rows[1:-1]",
        arrays={
            "syn_*": "synthetic cases: q fp16 / codes by syn_q_offsets / syn_t_offsets; expected S_f, S_r, forward path "
                     "from the float64 oracle WITH S5 and S7; syn_expect_error = NaN/Inf q must raise; ndeg = number "
                     "of rows / columns with std < 1e-6 in the forward (f) and reversed (r) z",
            "real_*": "real ESM3 items: rows fp16 [L+2, 1536], codes, q, margin_t, n_distinct_codes, q90 (float64 "
                      "oracle + S7 vs the shipped panel), q90_kernel_noS7 (research kernel without S7; diagnostic)",
            "pair_*": "pairs (query index, target index) over real items + base (index n_items = base query / base "
                      "target): expected S_f, S_r, path (float64 + S7), q90 of the query, lnE at N = 1e9, "
                      "ecal_provisional (S13 + D14 recommended default; recompute if D14 changes)"},
        checks=dict(pairs_all_finite=chk["pairs_all_finite"]))
    arrays = dict(
        syn_ids=np.array([x[0] for x in syn]), syn_expect_error=np.array([x[3] for x in syn]),
        syn_q_offsets=offsets_of([len(x) for x in sq]), syn_q=np.concatenate(sq),
        syn_t_offsets=offsets_of([len(x) for x in sc]), syn_codes=np.concatenate(sc),
        syn_S_f=np.array([s["S_f"] for s in s_res]), syn_S_r=np.array([s["S_r"] for s in s_res]),
        syn_ndeg=np.array([[s["ndeg_rows_f"], s["ndeg_cols_f"], s["ndeg_rows_r"], s["ndeg_cols_r"]] for s in s_res],
                          np.int32),
        syn_path_flat=np.concatenate([s["path"] for s in s_res]).astype(np.int32).reshape(-1, 2),
        syn_path_off=offsets_of([len(s["path"]) for s in s_res]),
        real_ids=ids, real_sequences=seqs, real_rows_offsets=offsets_of([ro[i + 1] - ro[i] for i in kk]),
        real_rows=rows, real_offsets=off, real_codes=codes, real_q=qq, real_margin_t=E["Fdeg/margin_t"],
        real_n_distinct_codes=n_dc, real_n_distinct_q_rows=n_dq, real_nonstandard_frac=nonstd, real_q90=q90_f64,
        real_q90_kernel_noS7=q90_kernel,
        base_q=bq, base_codes=bt, base_ids=np.array([bq_id, bt_id]), base_q90=np.float64(base_q90),
        pair_q=np.array([i for i, _ in plist], np.int32), pair_t=np.array([j for _, j in plist], np.int32),
        pair_S_f=pS_f, pair_S_r=pS_r, pair_q90=pq90, pair_lnE_N1e9=lnE,
        pair_ndeg=np.array([[r["ndeg_rows_f"], r["ndeg_cols_f"], r["ndeg_rows_r"], r["ndeg_cols_r"]] for r in pres],
                           np.int32),
        pair_path_flat=np.concatenate([r["path"] for r in pres]).astype(np.int32).reshape(-1, 2),
        pair_path_off=offsets_of([len(r["path"]) for r in pres]),
        pair_ecal_provisional=np.array([ecal(i, j) for i, j in plist], np.int8),
        meta_json=meta_arr(meta))
    return arrays, chk


def stage_build(args):
    ins = verify_inputs(STAGE_INPUTS["build"])
    ins["val_emb"] = dict(path=INPUTS["val_emb"][0], md5=None, note="md5 recorded by the encode stage")
    for nm in ("stage_esm3.npz", "stage_esm3.json", "stage_encode_ref.npz", "stage_encode_ref.json",
               "stage_encode_fm.npz", "stage_encode_fm.json"):
        p = f"{WORK}/{nm}"
        if os.path.exists(p):
            ins[f"work/{nm}"] = dict(path=p, md5=md5_file(p), bytes=os.path.getsize(p))
    e3info = json.load(open(f"{WORK}/stage_esm3.json"))
    E, einfo = load_encode("ref")
    Efm, efminfo = load_encode("fm")
    gm = generator_md5()
    stage_md5 = dict(esm3=e3info.get("generator_md5"), encode_ref=einfo.get("generator_md5"),
                     encode_fm=(efminfo or {}).get("generator_md5", gm))
    if any(v != gm for v in stage_md5.values()):
        raise SystemExit(f"stale stage outputs (generator md5 {gm}): {stage_md5}; rerun esm3 and encode")
    ins["val_emb"]["md5"] = einfo["inputs"]["val_emb"]["md5"]
    e3 = dict(np.load(f"{WORK}/stage_esm3.npz"))
    c = research_modules()
    ship = json.load(open(INPUTS["ship_final"][0]))
    assert ship["meta"]["version"] == CAL_VERSION
    tab = ship["tables"][TABLE_KEY]["all"]
    zp = np.load(INPUTS["panel_codes"][0])
    P = dict(codes=zp["codes"], off=zp["offsets"].astype(np.int64), NT=len(zp["offsets"]) - 1)
    assert hashlib.md5(zp["codes"].tobytes()).hexdigest() == PANEL_CODES_BYTES_MD5 and P["NT"] == 1000

    # ---- float64 harness Q90 for F1, F3, F-deg real queries (spawn pool) -----------------------------------------
    zc = np.load(INPUTS["fixture_candidate"][0])
    tasks = [(("F1", k), zc["q"][zc["q_offsets"][k]:zc["q_offsets"][k + 1]], True) for k in range(len(zc["q_ids"]))]
    zq, pick, _ = f3_select(tab)
    tasks += [(("F3", k), zq["q"][zq["offsets"][i]:zq["offsets"][i + 1]], True) for k, i in enumerate(pick)]
    do = E["Fdeg/offsets"]
    tasks += [(("Fdeg", i), E["Fdeg/q"][do[i]:do[i + 1]], True) for i in range(len(do) - 1)]
    log(f"harness float64 Q90: {len(tasks)} queries, {sum(len(t[1]) for t in tasks)} rows x 1,000 panel members x 2")
    t = time.time()
    R = harness_q90_all(tasks)
    q90_wall = time.time() - t

    os.makedirs(OUT, exist_ok=True)
    report = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), harness_q90_wall_s=round(q90_wall, 1),
                  research_modules=c.paths, harness_check_verbatim=c.verbatim, vector_lanes=c.vector_lanes)
    files = {}
    log("F1 ...")
    F1a, report["F1"] = build_F1(c, P, tab, R)
    write_npz(f"{OUT}/pairs_v1.npz", F1a)
    files["pairs_v1.npz"] = ("F1", "50 CATH pairs: q fp16, codes, sequences, S_f, S_r, forward paths, Q90, lnE at N=1e9")
    log("F2 / F2b / F2c ...")
    a, report["F2"] = build_F2(E, Efm, einfo)
    write_npz(f"{OUT}/encoder_rows_v1.npz", a)
    files["encoder_rows_v1.npz"] = ("F2", "raw fp16 rows of 5 val domains + reference and NERSC codes / q")
    a, report["F2b"] = build_F2b(E, Efm, einfo, ins)
    write_npz(f"{OUT}/ood_rows_codes_v1.npz", a)
    files["ood_rows_codes_v1.npz"] = ("F2b", "arithmetic-only OOD: reference codes / q of tests/resources/embeddings/"
                                             "{1acb.A,2uzi.A}.pt rows (rows not stored)")
    a, report["F2c"] = build_F2c(E, Efm, einfo, e3, e3info)
    write_npz(f"{OUT}/fasta_rows_codes_v1.npz", a)
    files["fasta_rows_codes_v1.npz"] = ("F2c", "FoldMatch CPU fp32 ESM3 rows [L+2,1536] fp16 of test_sequences.fasta + "
                                               "reference codes / q")
    log("F3 ...")
    a, report["F3"] = build_F3(c, P, tab, R)
    write_npz(f"{OUT}/q90_queries_v1.npz", a)
    files["q90_queries_v1.npz"] = ("F3", "20 benchmark queries q fp16 + expected Q90 vs the shipped panel")
    log("F4 ...")
    doc, report["F4"] = build_F4(c, tab, ins)
    jdump(doc, f"{OUT}/evalue_grid_v1.json")
    files["evalue_grid_v1.json"] = ("F4", "E arithmetic grid (lnE, bits) from ship_final.json")
    log("F-deg ...")
    a, report["Fdeg"] = build_Fdeg(c, P, tab, R, E, e3, einfo, F1a)
    write_npz(f"{OUT}/degenerate_v1.npz", a)
    files["degenerate_v1.npz"] = ("F-deg", "synthetic degenerate cases + real ESM3 degenerate / tiny inputs")
    tokdoc = dict(
        fixture="F-tok",
        spec="S15 normalize_sequence + G9 strict refusals; ids from the ESM3 tokenizer FoldMatch ships "
             "(esm.utils.encoding.tokenize_sequence == ESM3.encode(ESMProtein(sequence=s)).sequence)",
        esm=e3info["env"]["esm"],
        # sorted by token id: get_vocab() insertion order is not stable, which changed the file bytes run to run
        tokenizer_vocab={str(i): t for i, t in sorted((int(k), v) for k, v in e3info["tokenizer"]["vocab"].items())},
        fields={"normalized": "S15 transcription: null = refused (non-ASCII); ASCII whitespace removed; ASCII upper",
                "strict_refusal": "non_ascii | angle_bracket ('<' or '>' after normalisation) | null",
                "tokens_raw / tokens_normalized": "ids incl. BOS (0) and EOS (2); *_str = token strings",
                "length_preserved": "len(tokens_normalized) - 2 == len(normalized)",
                "n_nonstandard": "characters of normalized outside the 20 standard amino acids"},
        cases=e3info["tokenizer"]["cases"])
    jdump(tokdoc, f"{OUT}/tokenizer_cases_v1.json")
    files["tokenizer_cases_v1.json"] = ("F-tok", "tokenizer / normalisation cases")
    report["Ftok"] = dict(n_cases=len(tokdoc["cases"]),
                          not_length_preserving=[x["id"] for x in tokdoc["cases"]
                                                 if x.get("length_preserved") is False],
                          strict_refused=[x["id"] for x in tokdoc["cases"] if x["strict_refusal"]])

    # ---- manifest ----------------------------------------------------------------------------------------------------
    me = os.path.abspath(__file__)
    outs = {}
    total = 0
    for fn, (fid, desc) in files.items():
        p = f"{OUT}/{fn}"
        sz = os.path.getsize(p)
        total += sz
        o = dict(fixture=fid, content=desc, bytes=sz, sha256=sha256_file(p))
        if fn.endswith(".npz"):
            o["arrays"] = npz_manifest(p)
        outs[fn] = o
    manifest = dict(
        fixture_set=f"FoldMatch {SYSTEM} test fixtures v1 (PLAN P0.8)",
        created_utc=report["created_utc"], system=SYSTEM, calibration_version=CAL_VERSION, table_key=TABLE_KEY,
        setting=dict(offset=OFF, gap_open=GO, gap_extend=GE, s7_eps=S7_EPS, qprep_eps=QPREP_EPS),
        generator=dict(path=f"{RS}/src/scripts/esm_codes/make_foldmatch_fixtures.py", md5=md5_file(me),
                       sha256=sha256_file(me), stages=["esm3 (foldmatch env)", "encode --tag ref (research env)",
                                                       "encode --tag fm (foldmatch env, cross-check)",
                                                       "build (research env)"]),
        environments=dict(esm3=e3info["env"], encode_ref=einfo["env"], encode_fm=(efminfo or {}).get("env"),
                          build=dict(base_env(), research_modules=c.paths, vector_lanes=c.vector_lanes)),
        esm3=e3info["esm3"], reference_code=dict(
            oracle="harness.py normalize_scores + smith_waterman (verbatim copy of esm3_needleman_wunsch.py, "
                   f"check_verbatim={c.verbatim}) + S5 zero-row and S7 degenerate wrappers in this script",
            kernel="research swkern.c via swfast.swz_ad_batch (probe.py kernel)",
            q90="cath_score.query_scores (swmulti.c) + np.quantile(f64 S, 0.9) (panel_local.py)",
            evalue="bench_validate.tabp + lnE = (lnK + ln m + ln N) - lam S'",
            encoders="sw_common.Whitener + Model (L2m_B8_s0.pt), CPU fp32 torch, each item alone"),
        inputs=ins, outputs=outs, total_bytes=total,
        notes=[
            "Every npz carries meta_json (0-d str): json.loads(str(z['meta_json'])) documents each array and the checks.",
            "Paths (F1 path_flat, F-deg *_path_flat) are [k, 2] (query i, target j), 0-based, -1 = gap, FORWARD order.",
            "F2/F2b/F2c q: per-element fp16 ulp counts are misleading near 0 (F2 reference vs NERSC: max "
            f"{report['F2']['q_ref_vs_nersc']['max_elem_ulp']} element ulps, "
            f"{report['F2']['q_ref_vs_nersc']['n_elem_ulp_gt1']} elements > 1 ulp) while every difference is <= "
            f"{report['F2']['q_ref_vs_nersc']['max_abs_over_row_ulp']} ulp of the ROW's max |q| (q_diff_stats).",
            "margin_t = per residue min_k |h_t,k| (float64 re-evaluation). Bits with |h_t| < 1e-4: "
            + ", ".join(f"{s} {einfo['stats'][s]['n_abs_h_t_lt_1e_4']}" for s in ("F2", "F2b", "F2c", "Fdeg"))
            + "; an encoder with a different fp32 summation order may flip exactly those.",
            "F-deg expected values come from the float64 oracle WITH S5/S7. The research fp32 kernel (no S7) is wrong "
            "on degenerate rows/columns (affine-coefficient cancellation): e.g. case n1 kernel "
            f"{report['Fdeg']['synthetic']['n1']['research_kernel_noS7'][0]} vs oracle "
            f"{report['Fdeg']['synthetic']['n1']['S_f']:.6f}; kept only as a diagnostic (real_q90_kernel_noS7).",
            "Real ESM3 distinct codes per item (length): " + ", ".join(
                f"{x['id']} {x['n_distinct_codes']}/{x['length']}" for x in report["Fdeg"]["items"]) + ".",
            "F-tok: S15 upper-cases '<mask>' to '<MASK>', which ESM3 does not treat as special, so every normalised "
            "case is length-preserving; the multi-character collapse only happens on raw (un-normalised) input.",
            "F4: m = 32.58 and 594.92 lie just outside the table (centres 32.5844 / 594.9176). High-m rows carry both "
            "the research clamp and an UNCAPPED ln-m extrapolation; D15 decides.",
            "F2b rows are not stored: load tests/resources/embeddings/{1acb.A,2uzi.A}.pt with map_location='cpu'."],
        plan_tolerances={
            "F1": "paths 50/50 identical; S_f/S_r rel <= 1e-5; Q90 abs <= 1e-3; lnE abs <= 1e-3 at N=1e9; 3 pairs "
                  "survive E <= 1e-3",
            "F2": "code bits >= 0.99999; q within 1 fp16 ulp", "F2b/F2c": "codes identical to the reference encoder",
            "F3": "Q90 abs <= 1e-3", "F4": "rel <= 1e-12", "F-deg": "synthetic: abs <= 1e-5 vs reference with S7; no "
                                                                   "NaN; NaN q raises"},
        summary=dict(
            F1={k: report["F1"][k] for k in ("path_eq_f64", "cand_path_equal", "max_rel_Sf_kernel_vs_f64",
                                              "max_rel_Sr_kernel_vs_f64", "n_lnE_le_ln_1e_3",
                                              "max_abs_q90_research_vs_f64")},
            F2={k: report["F2"][k] for k in ("bits", "bits_equal_ref_vs_nersc", "q_ref_vs_nersc")},
            F3={k: report["F3"][k] for k in ("q90_bit_equal_stored_local", "max_abs_q90_vs_nersc_score2",
                                              "max_abs_q90_research_vs_f64")}))
    jdump(manifest, f"{OUT}/FIXTURES.json")
    report["outputs"] = outs
    report["total_bytes"] = total
    report["encode_stage"] = dict(ref=einfo["stats"], fm=(efminfo or {}).get("stats"))
    report["esm3_stage"] = dict(items=e3info["items"], esm3=e3info["esm3"])
    jdump(report, f"{TRACK}/report.json")
    # round-trip every npz with allow_pickle=False
    for fn in files:
        if fn.endswith(".npz"):
            with np.load(f"{OUT}/{fn}", allow_pickle=False) as zz:
                for k in zz.files:
                    _ = zz[k]
    log(json.dumps(dict(total_bytes=total, files={k: v["bytes"] for k, v in outs.items()}), default=jdefault))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["esm3", "encode", "build"])
    ap.add_argument("--threads", type=int, default=4, help="torch threads (esm3: repeat pass; encode: main pass)")
    ap.add_argument("--tag", default="ref", help="encode: output tag (ref = research env; fm = foldmatch env check)")
    args = ap.parse_args()
    os.makedirs(WORK, exist_ok=True)
    {"esm3": stage_esm3, "encode": stage_encode, "build": stage_build}[args.stage](args)
    log("done")


if __name__ == "__main__":
    main()
