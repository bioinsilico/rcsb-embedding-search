#!/usr/bin/env python
"""Export the FoldMatch ESM3 residue-code bundle ``esm-codes-v1`` (plan P2.1, Appendix C) for the Hugging Face model
repo ``rcsb/foldmatch-esm-codes``.

Outputs (all in --out; every byte is deterministic, so a rerun on the same machine and torch gives the same sha256s):

  esm_codes_v1_encoders.pt   torch.load(..., weights_only=True). A dict of primitives and torch tensors:
                             format tag, ``architecture``, the zcw50_clean ``transform`` (dims int64, offset fp32,
                             M fp32 cast once from float64; NOT folded into the first layer), the ``f_q`` and ``f_t``
                             state_dicts, the code and query ``conventions`` and ``provenance``.
  esm_codes_v1_scoring.npz   numpy.load(..., allow_pickle=False), so Stage-2 workers never import torch: panel codes,
                             offsets, lengths and UniRef50 ids, the E-value table (bin_centres, lam, lnK as f64[6]) and
                             ``meta_json`` (a 0-d string with every scoring and statistics constant).
  bundle.json                system name, calibration version, file sha256s, content digests, ESM3 identity, DP id,
                             licence (MIT) and attribution (UniRef50 CC BY 4.0, ESM3 upstream).
  README.md, LICENSE, SHA256SUMS, UPLOAD.md
                             model card, MIT text (as FoldMatch's LICENSE.md), checksums of the files to upload, and the
                             upload steps for the user (UPLOAD.md itself is not uploaded).

Inputs are the research artefacts (md5-checked against PLAN Appendix A) and the frozen FoldMatch fixtures (sha256-checked
against FIXTURES.json, P0.8). Nothing is downloaded and nothing is uploaded.

Self-checks, all before any sha256 is printed (the run stops on the first failure and leaves no bundle behind):
  * the .pt reloads with weights_only=True and rebuilds with strict=True; every tensor equals the research checkpoint
    (L2m_B8_s0.pt) and the fp32 cast of zcw50_clean.npz; serialising twice gives identical bytes;
  * F2 (encoder_rows_v1.npz): re-encoding with the bundle reproduces the reference codes and q exactly; the research
    sw_common encoder on the original checkpoint agrees bit for bit; F2c (fasta_rows_codes_v1.npz) and F2b
    (ood_rows_codes_v1.npz) likewise;
  * F1 (pairs_v1.npz): S_f, S_r and paths with the research C kernel (swkern.c via swfast.py, compiled into
    --kernel-dir) and Q90 over the bundle panel, ln E from the bundle table, all within the S16 tolerance contract;
    F3 Q90 and the F4 E-value grid as well;
  * the scoring npz loads with allow_pickle=False in a process where torch cannot be imported.

Run (research env, torch 2.9.0: the env that produced the fixtures' reference encodings):
  /Users/joan/app/miniconda3/envs/rcsb-embedding-search/bin/python src/scripts/save_esm_codes_for_production.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import zipfile

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"  # research convention: single-threaded BLAS (the F1 / F3 numpy GEMMs); torch threads set below

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# =====================================================================================================================
# Locations (overridable on the command line)
# =====================================================================================================================
SANDBOX = "/Users/joan/devel/foldmatch-p2p3-sandbox"
DEFAULT_ARTEFACTS = f"{SANDBOX}/artefacts"          # research artefacts: embcomp/... and inventory_probe/...
DEFAULT_FM = f"{SANDBOX}/fm"                        # a FoldMatch checkout holding tests/resources (read only)
DEFAULT_KERNEL_DIR = f"{SANDBOX}/build"              # research kernel build (swkern.c + swfast.py + libswkern)
DEFAULT_OUT = "/Users/joan/devel/foldmatch-esm-codes-hf"
DEFAULT_REPORT = f"{SANDBOX}/p2/export_report.json"

# =====================================================================================================================
# System constants (plan §2, S1-S16)
# =====================================================================================================================
SYSTEM = "esm-codes-v1"
CAL_VERSION = "foldmatch-evalue-M8_s0-D_fmr-pnlU50s1000-M3t999-v1"
TABLE_KEY = "D_fmr/pnl_uniref50/M3t999"
HF_REPO = "rcsb/foldmatch-esm-codes"
ENCODERS_FILE = "esm_codes_v1_encoders.pt"
SCORING_FILE = "esm_codes_v1_scoring.npz"
BUNDLE_FILE = "bundle.json"
ENCODERS_FORMAT = "foldmatch-esm-codes-encoders/1"
SCORING_FORMAT = "foldmatch-esm-codes-scoring/1"
BUNDLE_FORMAT = "foldmatch-esm-codes-bundle/1"
DIGEST_METHOD = "foldmatch-tensor-digest/1"
DP_ID = "fmsw/1"
OFF, GO, GE = 1.0, 5.0, 0.5             # setting D: cell = z - 1.0; a gap of length L costs 5.0 + (L - 1) 0.5
ROWCOL_EPS, S7_EPS, QPREP_EPS = 1e-8, 1e-6, 1e-12
N_FIX = 1.0e9                           # F1 lnE database size
DIMS_DROPPED = (1361, 1489)
PANEL_CODES_BYTES_MD5 = "885dfc88590e0b270addcd4b61c3a7a7"
COPYRIGHT = "Copyright (c) 2026 - now, RCSB PDB and contributors"

ESM3 = {
    "model": "esm3_sm_open_v1",
    "repo": "rcsb/rcsb-esm",
    "revision": "636acb420daf674df37ab0220f56061a5de2f169",
    "weights_file": "data/weights/esm3_sm_open_v1.pth",
    "weights_sha256": "5ead5a135c658068db6a4f1b933e72d6110992c4668822e1c0e2dcc53e38acd9",
    "upstream_repo": "EvolutionaryScale/esm3-sm-open-v1",
    "upstream_revision": "47f0545b2b6daf26a93439a3cd610f4f7f3d5478",
    "embedding_tap": "per_residue_embedding",
    "embedding_tap_detail": ("ESM3.forward_and_sample(..., SamplingConfig(return_per_residue_embeddings=True))"
                             ".per_residue_embedding: the last transformer block's output before the final "
                             "LayerNorm; rows 1..L (BOS and EOS removed), [L, 1536]; sequence-only input"),
    "input_normalisation": "ascii_upper_nows_v1",
    "input_normalisation_detail": "ASCII whitespace removed, ASCII letters upper-cased, non-ASCII refused (S15)",
}

#: The only encoder architecture of this system. FoldMatch's loader compares a bundle's ``architecture`` to its own
#: copy of this dict exactly (lists, not tuples, because that is what survives the round trip).
ARCHITECTURE = {
    "name": "esm-codes-mlp-v1",
    "d_embed": 1536,
    "dims_dropped": list(DIMS_DROPPED),
    "d_in": 1534,
    "hidden": 1024,
    "d_out": 64,
    "activation": "gelu_exact",
    "skip": "linear",
    "bias": True,
    "dropout": 0.0,
    "encoders": ["f_q", "f_t"],
}
STATE_KEYS = ["lin.weight", "lin.bias", "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias"]

CONVENTIONS = {
    "input": {"esm3_model": ESM3["model"], "esm3_repo": ESM3["repo"], "esm3_revision": ESM3["revision"],
              "esm3_weights_sha256": ESM3["weights_sha256"], "embedding_tap": ESM3["embedding_tap"],
              "embedding_tap_detail": ESM3["embedding_tap_detail"],
              "input_normalisation": ESM3["input_normalisation"]},
    "transform": {"name": "zcw50_clean",
                  "formula": "x16 = fp16(x); xs = fp32(x16[:, dims]) - offset; xw = fp32(fp16(xs @ M))",
                  "input_cast": "float16", "compute_dtype": "float32", "output_cast": "float16",
                  "folded_into_encoders": False},
    "encoder": {"formula": "f(x) = lin(x) + fc2(GELU_exact(fc1(x)))", "dtype": "float32", "tf32": False,
                "autocast": False, "dropout": 0.0},
    "code": {"encoder": "f_t", "bits": 64, "bytes": 8, "sign_rule": "bit_k = (h_k >= 0)",
             "packing": "numpy.packbits(bits, axis=1)", "bitorder": "big", "dtype": "uint8", "shape": "[L, 8]",
             "decode": "d = 2 * bit - 1"},
    "query": {"encoder": "f_q", "formula": "q = fp16(f_q(xw))", "dtype": "float16", "shape": "[L, 64]"},
}

#: Every number that shapes Stage-2 output, shared by bundle.json and the scoring npz meta_json (the P4 registry entry
#: must equal it). Table values and panel numbers are filled in from ship_final.json / the panel at run time.
SCORING = {
    "dp": DP_ID,
    "offset": OFF,
    "gap_open": GO,
    "gap_extend": GE,
    "gap_cost": "a gap of length L costs gap_open + (L - 1) * gap_extend",
    "cell_score": "z_ij - offset",
    "score_form": "fwd_minus_rev_view",
    "score_form_detail": "S = fp64(S_f) - fp64(S_r); S_r = Smith-Waterman on the row-reversed view of the same "
                         "similarity, z recomputed in its own row order, score only",
    "score_diff": "float64",
    "normalisation": {"name": "rowcol_z", "std": "population", "eps": ROWCOL_EPS, "degenerate_std_lt": S7_EPS,
                      "degenerate_rule": "a row (column) with std < 1e-6 contributes exactly 0",
                      "formula": "z_ij = 0.5 ((s_ij - mu_i) / (sigma_i + 1e-8) + (s_ij - nu_j) / (tau_j + 1e-8))"},
    "query_prep": "fp64_seq_k_asc",
    "query_prep_detail": "qp_i = fp32(fp64(q_i) / (8 sqrt(sum_k fp64(q_ik)^2))), k ascending; |q_i| < 1e-12 -> 0; "
                         "NaN / Inf is an error",
    "query_prep_eps": QPREP_EPS,
    "similarity": "pm1_fp32_blas_gemm_per_pair",
    "determinism": "tolerance_s16_v1",
}


# =====================================================================================================================
# Helpers
# =====================================================================================================================
def md5_file(p, bs=1 << 24):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p, bs=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(bs), b""):
            h.update(b)
    return h.hexdigest()


def tensor_digest(named) -> str:
    """Content digest of a set of named arrays (``foldmatch-tensor-digest/1``): sha256 over the method tag and newline,
    then, for every name in sorted order, an 8-byte little-endian length + the compact sorted-key JSON header
    {"dtype": <numpy dtype str, little-endian>, "name": ..., "shape": [...]} and an 8-byte length + the raw C-order
    little-endian bytes. Independent of the container (.pt / .npz) and of torch vs numpy."""
    h = hashlib.sha256(DIGEST_METHOD.encode() + b"\n")
    for name in sorted(named):
        v = named[name]
        a = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
        a = np.ascontiguousarray(a)
        a = a.astype(a.dtype.newbyteorder("<"), copy=False)
        head = json.dumps({"dtype": a.dtype.str, "name": name, "shape": list(a.shape)}, sort_keys=True,
                          separators=(",", ":")).encode()
        raw = a.tobytes(order="C")
        h.update(len(head).to_bytes(8, "little"))
        h.update(head)
        h.update(len(raw).to_bytes(8, "little"))
        h.update(raw)
    return h.hexdigest()


def write_npz_bytes(arrays: dict) -> bytes:
    """Deterministic npz: fixed zip timestamps, stored (the codes are high-entropy), npy format 1.0, no pickles."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_STORED) as zf:
        for k, v in arrays.items():
            v = np.asarray(v)
            if not v.flags.c_contiguous:
                v = v.copy(order="C")
            assert v.dtype != object, k
            buf = io.BytesIO()
            np.lib.format.write_array(buf, v, version=(1, 0), allow_pickle=False)
            zi = zipfile.ZipInfo(k + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_STORED
            zi.external_attr = 0o644 << 16
            zf.writestr(zi, buf.getvalue())
    return out.getvalue()


def torch_save_bytes(obj) -> bytes:
    """torch.save into memory: the zip record prefix is then 'archive/' whatever the final file name is."""
    buf = io.BytesIO()
    torch.save(obj, buf)
    return buf.getvalue()


def jdump_str(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=True, allow_nan=False) + "\n"


def write_atomic(path, data: bytes):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def fp16_ordered(a):
    u = np.asarray(a, np.float16).view(np.uint16).astype(np.int32)
    mag = u & 0x7FFF
    return np.where(u & 0x8000, -mag, mag)


def q_diff(a, b):
    """fp16 q differences: bit-equal count, max per-element ulp distance, max |dq| in ulps of the row's max |q|."""
    a16, b16 = np.asarray(a, np.float16), np.asarray(b, np.float16)
    ul = np.abs(fp16_ordered(a16) - fp16_ordered(b16))
    d = np.abs(a16.astype(np.float64) - b16.astype(np.float64))
    rowmax = np.maximum(np.abs(a16.astype(np.float32)), np.abs(b16.astype(np.float32))).max(axis=1, keepdims=True)
    ulp_row = np.float64(2.0) ** (np.floor(np.log2(np.maximum(rowmax, 2.0 ** -14))) - 10)
    return dict(n=int(a16.size), bit_equal=int((a16.view(np.uint16) == b16.view(np.uint16)).sum()),
                max_elem_ulp=int(ul.max()) if ul.size else 0,
                max_abs_over_row_ulp=float((d / ulp_row).max()) if d.size else 0.0)


def bits_equal(a, b):
    ba, bb = np.unpackbits(np.asarray(a, np.uint8), axis=1), np.unpackbits(np.asarray(b, np.uint8), axis=1)
    return int((ba == bb).sum()), int(ba.size)


def offsets_of(lengths):
    return np.concatenate([[0], np.cumsum(np.asarray(lengths, np.int64))]).astype(np.int64)


class Checks:
    """Collects every self-check number; ``require`` stops the run on a failed condition."""

    def __init__(self):
        self.out = {}

    def put(self, section, **kv):
        self.out.setdefault(section, {}).update(kv)

    def require(self, cond, msg):
        if not cond:
            raise SystemExit(f"SELF-CHECK FAILED: {msg}")


# =====================================================================================================================
# Inputs
# =====================================================================================================================
def input_table(artefacts, fm):
    emb = f"{artefacts}/embcomp"
    res = f"{fm}/tests/resources"
    md5 = {  # name: (path, md5); PLAN Appendix A
        "encoders": (f"{emb}/sw_mlp/models/L2m_B8_s0.pt", "85fdd985087d6813e9994274d4fca6ab"),
        "encoders_history": (f"{emb}/sw_mlp/models/L2m_B8_s0.json", "19e58f8956cf24404def09aac56e00b5"),
        "sw_common": (f"{emb}/sw_mlp/sw_common.py", "b8dab0a03907c9b7ed67663d96c55561"),
        "zcw50_clean": (f"{emb}/clean_pairs/zcw50_clean.npz", "4c7692ed66e599aca906b41ec7c13f04"),
        "ship_final": (f"{emb}/panel/fit2/results/ship_final.json", "c51735f962c04bd8c441b9625c4cf30a"),
        "panel_codes": (f"{emb}/panel/fit2/results/shipped_panel_codes.npz", "15cc1c7b2d1dd488a5f9b742365ccb13"),
        "panel_ids": (f"{emb}/panel/fit2/results/shipped_panel.ids", "68463819ee52018dccb6b4fab3d5467a"),
        "swkern_c": (f"{emb}/throughput/build/swkern.c", "438d5a9811153f61e66cee845f1ed037"),
        "swfast_py": (f"{emb}/throughput/build/swfast.py", "80a813553d42801fce6b77462c5e1d0a"),
        "emb_1acb_A": (f"{res}/embeddings/1acb.A.pt", "d3c93088f2048a2e83869133d6a0238e"),
        "emb_2uzi_A": (f"{res}/embeddings/2uzi.A.pt", "fe5006f58f4f61331d9b3a481aed76d9"),
    }
    sha = {  # name: (path, sha256); FoldMatch tests/resources/esm_codes/FIXTURES.json (P0.8)
        "F1": (f"{res}/esm_codes/pairs_v1.npz", "9ccf7f840e11ff612359896a4e3b73d43f0192578dfbbdea0f4d54fd9042c48e"),
        "F2": (f"{res}/esm_codes/encoder_rows_v1.npz",
               "3197bbbbadc12c033d8aa54f7f13efcc72390a6bf4eedf45ccd83c25d69df657"),
        "F2b": (f"{res}/esm_codes/ood_rows_codes_v1.npz",
                "36f092dbb906f7cf679079b93173b5ab8216e7c90b7552f43aec2a779fab748e"),
        "F2c": (f"{res}/esm_codes/fasta_rows_codes_v1.npz",
                "f40f7a22b8e8736b4be89d2392f26da18aec65da7783afe02a755ee87f6fc497"),
        "F3": (f"{res}/esm_codes/q90_queries_v1.npz",
               "3ab44ad70d930b664899ad3294180f1af5af354670ebb30fddb0ee95bc1d96f5"),
        "F4": (f"{res}/esm_codes/evalue_grid_v1.json",
               "4dffd04c17c6a7a0ccf8bfbda915d055ad1c6febe3092c8db6c02a46e02ef563"),
    }
    return md5, sha


def verify_inputs(md5, sha):
    rec = {}
    for nm, (p, exp) in md5.items():
        h = md5_file(p)
        if h != exp:
            raise SystemExit(f"input md5 mismatch for {nm} ({p}): {h} != {exp}")
        rec[nm] = dict(file=os.path.basename(p), md5=h, bytes=os.path.getsize(p))
    for nm, (p, exp) in sha.items():
        h = sha256_file(p)
        if h != exp:
            raise SystemExit(f"fixture sha256 mismatch for {nm} ({p}): {h} != {exp}")
        rec[nm] = dict(file=os.path.basename(p), sha256=h, bytes=os.path.getsize(p))
    return rec


def load_research_checkpoint(path):
    """L2m_B8_s0.pt (md5-checked): {'state': f_q.* / f_t.* tensors, 'cfg', 'args', 'epoch'}."""
    try:
        st = torch.load(path, map_location="cpu", weights_only=True)
        how = "weights_only=True"
    except Exception:  # noqa: BLE001  (research checkpoint; md5 verified above)
        st = torch.load(path, map_location="cpu", weights_only=False)
        how = "weights_only=False (md5-verified research checkpoint)"
    return st, how


# =====================================================================================================================
# Bundle construction
# =====================================================================================================================
def primitives(o):
    """Deep copy keeping only what weights_only=True accepts besides tensors: dict / list / str / int / float / bool /
    None. Tuples become lists; anything else is refused."""
    if isinstance(o, dict):
        return {str(k): primitives(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [primitives(v) for v in o]
    if o is None or isinstance(o, (bool, int, float, str)):
        return o
    raise TypeError(f"not a primitive: {type(o)}")


def build_encoder_bundle(st, zc, inputs_rec):
    dims = np.asarray(zc["dims"])
    keep = np.setdiff1d(np.arange(ARCHITECTURE["d_embed"]), np.array(DIMS_DROPPED))
    if not np.array_equal(dims, keep) or dims.dtype != np.int64:
        raise SystemExit("zcw50_clean dims are not 0..1535 without 1361 and 1489 (int64)")
    if zc["offset"].dtype != np.float64 or zc["M"].dtype != np.float64:
        raise SystemExit("zcw50_clean offset / M are expected as float64 (S2 casts them once to fp32)")
    transform = {
        "dims": torch.from_numpy(dims.astype(np.int64)).clone(),
        "offset": torch.from_numpy(zc["offset"].astype(np.float32)).clone(),      # S2: fp32, cast once from f64
        "M": torch.from_numpy(np.ascontiguousarray(zc["M"].astype(np.float32))).clone(),
    }
    cfg = st["cfg"]
    want_cfg = {"arch": "mlp", "code": "sign", "B": 8, "ste": "tanh", "hidden": 1024, "init": "pca", "skip": True,
                "d": 64}
    if cfg != want_cfg:
        raise SystemExit(f"unexpected research cfg {cfg}")
    encs = {}
    for e in ("f_q", "f_t"):
        sd = {k: st["state"][f"{e}.{k}"].detach().to("cpu").clone().contiguous() for k in STATE_KEYS}
        extra = sorted(k for k in st["state"] if k.startswith(f"{e}.") and k[len(e) + 1:] not in STATE_KEYS)
        if extra:
            raise SystemExit(f"unexpected research parameters {extra}")
        for k, v in sd.items():
            if v.dtype != torch.float32 or not torch.isfinite(v).all():
                raise SystemExit(f"{e}.{k}: expected finite float32")
        encs[e] = sd
    rest = sorted(k for k in st["state"] if not k.startswith(("f_q.", "f_t.")))
    if rest:
        raise SystemExit(f"unexpected research state entries {rest}")
    provenance = {
        "system": SYSTEM,
        "research_model": "L2m_B8_s0",
        "scoring_unit": "M8_s0 (L2m_B8_s0), setting D",
        "checkpoint_epoch": int(st["epoch"]),
        "checkpoint_cfg": primitives(cfg),
        # every training argument except "out" (a cluster path that says nothing about the model)
        "training_args": primitives({k: v for k, v in st["args"].items() if k != "out"}),
        "sources": {k: inputs_rec[k] for k in ("encoders", "encoders_history", "sw_common", "zcw50_clean")},
        "zcw50_clean": {"n_res": int(zc["n_res"]), "n_dom": int(zc["n_dom"]), "note": str(zc["note"])},
        "exported_by": "rcsb-embedding-search src/scripts/save_esm_codes_for_production.py (plan P2.1)",
        "torch": str(torch.__version__),
        "n_parameters": int(sum(v.numel() for e in encs.values() for v in e.values())),
    }
    return {
        "format": ENCODERS_FORMAT,
        "system": SYSTEM,
        "architecture": ARCHITECTURE,
        "transform": transform,
        "f_q": encs["f_q"],
        "f_t": encs["f_t"],
        "conventions": CONVENTIONS,
        "provenance": provenance,
    }


def build_scoring(zp, ids_file_lines, ship, digests_panel_table):
    tab = ship["tables"][TABLE_KEY]["all"]
    codes = np.ascontiguousarray(zp["codes"], dtype=np.uint8)
    off = zp["offsets"].astype(np.int64)
    lens = zp["lengths"].astype(np.int64)
    ids = zp["ids"].astype(str)
    if hashlib.md5(codes.tobytes()).hexdigest() != PANEL_CODES_BYTES_MD5:
        raise SystemExit("panel codes bytes md5 differs from 885dfc88...")
    if not (np.array_equal(np.diff(off), lens) and off[0] == 0 and off[-1] == len(codes) and codes.shape[1] == 8):
        raise SystemExit("panel offsets / lengths / codes inconsistent")
    if ids.tolist() != ids_file_lines or ids.tolist() != ship["meta"]["panel_ids"]:
        raise SystemExit("panel ids differ between shipped_panel_codes.npz, shipped_panel.ids and ship_final.json")
    if int(off[-1]) != int(ship["meta"]["panel_residues"]) or len(ids) != int(ship["meta"]["S_star"]):
        raise SystemExit("panel size / residues differ from ship_final.json")
    cen = np.array(tab["center_len"], np.float64)
    lam = np.array(tab["lam"], np.float64)
    lnk = np.array(tab["lnK"], np.float64)
    if not (len(cen) == 6 and np.all(np.diff(cen) > 0) and np.all(cen > 0) and np.all(lam > 0)):
        raise SystemExit("E-value table is not 6 increasing positive centres with lam > 0")
    width = max(len(s) for s in ids)
    arrays = {
        "panel_codes": codes,
        "panel_offsets": off,
        "panel_lengths": lens,
        "panel_ids": np.array(ids, dtype=f"<U{width}"),
        "bin_centres": cen,
        "lam": lam,
        "lnK": lnk,
    }
    panel_sha = tensor_digest({"codes": codes, "offsets": off})
    table_sha = tensor_digest({"bin_centres": cen, "lam": lam, "lnK": lnk})
    digests_panel_table.update(panel_sha256=panel_sha, table_sha256=table_sha)
    return arrays, statistics_block(ship, cen, lam, lnk), panel_block(ship, off, panel_sha), table_sha


def statistics_block(ship, cen, lam, lnk):
    return {
        "table_key": TABLE_KEY,
        "bin_centres": cen.tolist(),
        "lam": lam.tolist(),
        "lnK": lnk.tolist(),
        "interpolation": "lam(m), lnK(m) linear in ln m between the six bin centres (numpy.interp(ln m, ln centres, "
                         "table))",
        "low_m_rule": "clamp",
        "high_m_rule": "clamp",
        "m": "query length (residues)",
        "N": "database residues (manifest total_residues)",
        "S_prime": "S - Q90(query)",
        "lnE": "lnK(m) + ln m + ln N - lam(m) * S'  (float64)",
        "bits": "(lam(m) * S' - lnK(m)) / ln 2, rendered floor(bits + 0.5)",
        "fit": ship["meta"]["fit"],
        "fit_db_residues": float(ship["meta"]["N_ann_fit_db"]),
    }


def panel_block(ship, off, panel_sha):
    return {
        "size": int(len(off) - 1),
        "residues": int(off[-1]),
        "quantile": 0.9,
        "quantile_method": "linear",
        "q90": "numpy.quantile(float64 S(query, members), 0.9), linear interpolation",
        "source": ship["meta"]["panel_source"],
        "codes_bytes_md5": PANEL_CODES_BYTES_MD5,
        "idlist_md5": ship["meta"]["panel_idlist_md5"],
        "panel_sha256": panel_sha,
    }


CALIBRATED_DOMAIN = {
    "query_min_length": 21,
    "query_max_length": 966,
    "target_max_length": 3000,
    "max_nonstandard_fraction": 0.01,
    "target_min_distinct_codes": 2,
    "nonstandard": "any character outside the 20 standard amino acids ACDEFGHIKLMNPQRSTVWY",
    "rule": "ecal = 1 only when every condition holds; otherwise ecal = 0, the alignment is still reported and a "
            "once-per-run warning gives the counts",
}
DEGENERATE_QUERY_RULE = "no_hits"
DEGENERATE_QUERY_DETAIL = ("a query with n = 1 or whose q fp16 rows are all bitwise identical reports no hits "
                           "(S13b), whatever max_evalue")


def licence_block():
    """P0.3: the bundle (encoders, transform, statistics, panel codes) is MIT; the panel ids are UniProt data and keep
    their CC BY 4.0 licence and attribution (they are not relicensed)."""
    return {
        "spdx": "MIT",
        "file": "LICENSE",
        "copyright": COPYRIGHT,
        "scope": "MIT: the encoders, the zcw50_clean transform, the statistics table, the panel codes and the "
                 "documentation. Not covered by the MIT licence: the panel member identifiers (panel_ids), which are "
                 "UniProt data under CC BY 4.0 (see attribution.uniref50).",
        "exceptions": {
            "panel_ids": {"spdx": "CC-BY-4.0", "source": "UniProt Consortium, UniRef50 release 2026_03",
                          "attribution": "attribution.uniref50"},
        },
        "attribution": {
            "uniref50": ("Panel member identifiers are UniRef50 cluster identifiers (UniProt release 2026_03) from the "
                         "UniProt Consortium, https://www.uniprot.org, distributed under the Creative Commons "
                         "Attribution 4.0 International licence (CC BY 4.0, "
                         "https://creativecommons.org/licenses/by/4.0/). The panel codes were computed from ESM3 "
                         "embeddings of those representative sequences; the sequences themselves are not included."),
            "esm3": (f"The encoders take per-residue embeddings of ESM3 {ESM3['model']} (EvolutionaryScale) as input; "
                     "no ESM3 weights are included. FoldMatch loads the weights from "
                     f"{ESM3['repo']}@{ESM3['revision']} (MIT; weights sha256 {ESM3['weights_sha256']}), the same "
                     f"blob as {ESM3['upstream_repo']}@{ESM3['upstream_revision']}, whose model card points to the "
                     "MIT licence of the evolutionaryscale/esm repository. Older releases of the same weights (e.g. "
                     f"the gated {ESM3['upstream_repo']} card at 2feda652 and the LICENSE.md shipped in the esm 3.2.3 "
                     "wheel) carry the EvolutionaryScale Cambrian Non-Commercial License; each copy of the weights is "
                     "governed by the licence it was obtained under."),
        },
    }


# =====================================================================================================================
# Reference encoder rebuilt from the bundle (spec S2-S5, the exact research op sequence of sw_common.py)
# =====================================================================================================================
class _Mlp(nn.Module):
    def __init__(self, d_in, hidden, d_out):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, d_out)

    def forward(self, x):
        y = self.lin(x)
        return y + self.fc2(F.gelu(self.fc1(x)))


class BundleEncoder:
    """Rebuilt from a weights_only-loaded bundle dict with strict=True; CPU fp32."""

    def __init__(self, b):
        a = b["architecture"]
        self.dims = b["transform"]["dims"]
        self.offset = b["transform"]["offset"]
        self.M = b["transform"]["M"]
        self.f_q = _Mlp(a["d_in"], a["hidden"], a["d_out"])
        self.f_t = _Mlp(a["d_in"], a["hidden"], a["d_out"])
        self.f_q.load_state_dict(b["f_q"], strict=True)
        self.f_t.load_state_dict(b["f_t"], strict=True)
        self.f_q.eval()
        self.f_t.eval()

    @torch.no_grad()
    def transform(self, x):
        x16 = torch.as_tensor(x).to(torch.float16)                               # S2: x16 = fp16(x)
        xs = x16.index_select(1, self.dims).to(torch.float32) - self.offset    # fp32(x16[:, dims]) - offset
        return (xs @ self.M).to(torch.float16)                                  # fp16(xs @ M)

    @torch.no_grad()
    def encode(self, x):
        xw = self.transform(x).to(torch.float32)                                # xw = fp32(fp16(...))
        ht = self.f_t(xw)
        hq = self.f_q(xw)
        code = np.packbits((ht >= 0).numpy().astype(np.uint8), axis=1)          # S4, MSB first
        return code, hq.to(torch.float16).numpy(), xw, hq, ht                   # S5: q = fp16(f_q(xw))


def research_encoder(artefacts, st):
    """sw_common.Whitener + Model on the ORIGINAL research checkpoint (cross-check of the bundle)."""
    emb = f"{artefacts}/embcomp"
    os.environ["EMBCOMP_DATA"] = emb
    spec = importlib.util.spec_from_file_location("sw_common_md5_checked", f"{emb}/sw_mlp/sw_common.py")
    C = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(C)
    model = C.build_model(st["cfg"])
    model.load_state_dict(st["state"])
    model.eval()
    wt = C.Whitener(torch.device("cpu"))

    @torch.no_grad()
    def enc(x16):
        xw = wt(np.asarray(x16, np.float16))
        q = model.f_q(xw.float()).half().numpy()
        code = model.quant.encode(model.f_t(xw.float()))
        return code, q
    return enc


def encode_items(enc, items):
    codes, qs = [], []
    for x in items:
        r = enc(x)
        codes.append(r[0])
        qs.append(r[1])
    return np.concatenate(codes), np.concatenate(qs)


# =====================================================================================================================
# Research kernel (swkern.c via swfast.py) for F1 / F3
# =====================================================================================================================
def kernel_module(kernel_dir, md5tab, checks):
    kernel_dir = os.path.abspath(kernel_dir)          # the compiler runs with cwd=kernel_dir
    os.makedirs(kernel_dir, exist_ok=True)
    for nm, fn in (("swkern_c", "swkern.c"), ("swfast_py", "swfast.py")):
        src, exp = md5tab[nm]
        dst = os.path.join(kernel_dir, fn)
        if not os.path.exists(dst) or md5_file(dst) != exp:
            shutil.copyfile(src, dst)
        if md5_file(dst) != exp:
            raise SystemExit(f"{dst}: md5 mismatch")
    lib = os.path.join(kernel_dir, "libswkern.dylib" if sys.platform == "darwin" else "libswkern.so")
    if sys.platform == "darwin" and platform.machine() == "arm64":
        cmd = ["cc", "-O3", "-mcpu=apple-m1", "-ffp-contract=off", "-shared", "-fPIC", "-o", lib, "swkern.c"]
    else:
        cmd = ["gcc", "-O3", "-ffp-contract=off", "-shared", "-fPIC", "-o", lib, "swkern.c", "-lm"]
    if not os.path.exists(lib) or os.path.getmtime(lib) < os.path.getmtime(os.path.join(kernel_dir, "swkern.c")):
        subprocess.run(cmd, cwd=kernel_dir, check=True)
    spec = importlib.util.spec_from_file_location("swfast_md5_checked", os.path.join(kernel_dir, "swfast.py"))
    Fk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(Fk)
    cc = subprocess.run(["cc", "--version"], capture_output=True, text=True).stdout.splitlines()[:1]
    checks.put("kernel", dir=kernel_dir, build_command=" ".join(cmd), compiler=cc, vector_lanes=int(Fk.VECTOR_LANES),
               lib_sha256=sha256_file(lib))
    return Fk


def kernel_pair(Fk, q16, codes):
    """Exactly the fixture generator's kernel_pair (inventory_probe/probe.py kernel)."""
    qp = Fk.query_prep(np.asarray(q16).astype(np.float32))
    sim = np.ascontiguousarray(qp @ Fk.decode_unpackbits(codes).T)
    col = np.array([0, len(codes)], np.int64)
    sf, ef, pflat, poff = Fk.swz_ad_batch(sim, col, OFF, GO, GE, tb=True)
    simr = np.ascontiguousarray(sim[::-1])
    sr, _ = Fk.swz_ad_batch(simr, col, OFF, GO, GE, tb=False)
    return float(sf[0]), pflat[poff[0]:poff[1]].copy(), float(sr[0])


def panel_q90(Fk, q16, codes, off, chunk=50):
    """Q90 of S = fp64(S_f) - fp64(S_r) over the bundle panel; research kernel (swz_ad_batch score only, R = forward
    and row-reversed similarity), one GEMM per chunk of members."""
    qp = Fk.query_prep(np.asarray(q16).astype(np.float32))
    NT = len(off) - 1
    sf = np.empty(NT, np.float32)
    sr = np.empty(NT, np.float32)
    for c0 in range(0, NT, chunk):
        c1 = min(NT, c0 + chunk)
        a, b = int(off[c0]), int(off[c1])
        sim = np.ascontiguousarray(qp @ Fk.decode_unpackbits(codes[a:b]).T)
        col = (off[c0:c1 + 1] - a).astype(np.int64)
        s, _ = Fk.swz_ad_batch(sim, col, OFF, GO, GE, tb=False)
        r, _ = Fk.swz_ad_batch(np.ascontiguousarray(sim[::-1]), col, OFF, GO, GE, tb=False)
        sf[c0:c1], sr[c0:c1] = s, r
    return float(np.quantile(sf.astype(np.float64) - sr.astype(np.float64), 0.9))


def tabp(cen, tab, m):
    return np.interp(np.log(np.asarray(m, np.float64)), np.log(cen), tab)


def lnE_of(cen, lam_t, lnk_t, Sp, m, N):
    lam, lnk = tabp(cen, lam_t, m), tabp(cen, lnk_t, m)
    return (lnk + np.log(np.asarray(m, np.float64)) + math.log(N)) - lam * np.asarray(Sp, np.float64)


# =====================================================================================================================
# Self-checks
# =====================================================================================================================
def check_encoders(pt_bytes, pt_bytes_again, st, zc, checks):
    checks.require(pt_bytes == pt_bytes_again, "serialising the encoder bundle twice gave different bytes")
    b = torch.load(io.BytesIO(pt_bytes), map_location="cpu", weights_only=True)
    checks.require(b["format"] == ENCODERS_FORMAT and b["system"] == SYSTEM, "format / system tag lost")
    checks.require(b["architecture"] == ARCHITECTURE, "architecture changed in the round trip")
    checks.require(set(b) == {"format", "system", "architecture", "transform", "f_q", "f_t", "conventions",
                              "provenance"}, f"unexpected top-level keys {sorted(b)}")
    enc = BundleEncoder(b)  # strict=True rebuild
    n_eq = 0
    for e in ("f_q", "f_t"):
        for k in STATE_KEYS:
            n_eq += int(torch.equal(b[e][k], st["state"][f"{e}.{k}"]))
    checks.require(n_eq == 12, "bundle tensors differ from the research checkpoint")
    t_eq = (torch.equal(b["transform"]["dims"], torch.from_numpy(zc["dims"]))
            and np.array_equal(b["transform"]["offset"].numpy(), zc["offset"].astype(np.float32))
            and np.array_equal(b["transform"]["M"].numpy(), zc["M"].astype(np.float32)))
    checks.require(t_eq, "bundle transform differs from the fp32 cast of zcw50_clean.npz")
    dtypes = {f"transform.{k}": str(v.dtype) for k, v in b["transform"].items()}
    checks.put("encoders_file", weights_only_reload=True, strict_rebuild=True, tensors_equal_research=n_eq,
               transform_equal_fp32_cast=bool(t_eq), serialisation_deterministic=True, transform_dtypes=dtypes,
               n_parameters=b["provenance"]["n_parameters"], bytes=len(pt_bytes))
    return b, enc


def check_parity(enc, artefacts, st, fixtures, checks, strict_q):
    ref = research_encoder(artefacts, st)
    out = {}

    def compare(tag, items, codes_ref, q_ref):
        c, q = encode_items(lambda x: enc.encode(x)[:2], items)
        c2, q2 = encode_items(ref, items)
        be, nb = bits_equal(c, codes_ref)
        qd = q_diff(q, q_ref)
        rb, _ = bits_equal(c, c2)
        rq = q_diff(q, q2)
        out[tag] = dict(items=len(items), rows=int(len(c)), bits=nb, bits_equal_fixture=be,
                        codes_identical_fixture=bool(np.array_equal(c, codes_ref)), q_vs_fixture=qd,
                        bits_equal_research_encoder=rb, q_bit_equal_research_encoder=rq["bit_equal"])
        checks.require(np.array_equal(c, codes_ref), f"{tag}: codes differ from the fixture ({be}/{nb} bits)")
        checks.require(rb == nb and rq["bit_equal"] == rq["n"], f"{tag}: bundle and research encoder disagree")
        if strict_q:
            checks.require(qd["bit_equal"] == qd["n"], f"{tag}: q differs from the fixture: {qd}")
        else:
            checks.require(qd["max_abs_over_row_ulp"] <= 1.0, f"{tag}: q differs by more than 1 ulp: {qd}")
        return c, q

    z = np.load(fixtures["F2"][0], allow_pickle=False)
    o = z["offsets"]
    items = [z["rows"][o[k]:o[k + 1]] for k in range(len(o) - 1)]
    compare("F2", items, z["codes"], z["q"])
    bn, nb = bits_equal(encode_items(lambda x: enc.encode(x)[:2], items)[0], z["codes_nersc"])
    out["F2"]["bits_equal_nersc_enc_check"] = bn
    # S2 / S3 intermediates of the first F2 domain (first 8 rows)
    _, _, xw, hq, ht = enc.encode(items[0])
    out["F2"]["head_xw_identical"] = bool(np.array_equal(xw[:8].to(torch.float16).numpy(), z["head_xw"]))
    out["F2"]["head_h_q_identical"] = bool(np.array_equal(hq[:8].numpy(), z["head_h_q"]))
    out["F2"]["head_h_t_identical"] = bool(np.array_equal(ht[:8].numpy(), z["head_h_t"]))
    if strict_q:
        checks.require(out["F2"]["head_xw_identical"] and out["F2"]["head_h_q_identical"]
                       and out["F2"]["head_h_t_identical"], "F2 S2/S3 intermediates differ")

    z = np.load(fixtures["F2c"][0], allow_pickle=False)
    ro = z["rows_offsets"]
    items = [z["rows"][ro[k] + 1:ro[k + 1] - 1] for k in range(len(ro) - 1)]   # S1: BOS and EOS removed
    compare("F2c", items, z["codes"], z["q"])

    z = np.load(fixtures["F2b"][0], allow_pickle=False)
    items = []
    for nm in z["ids"].astype(str):
        x = torch.load(fixtures[f"emb_{nm.replace('.', '_')}"][0], map_location="cpu", weights_only=True)
        items.append(x.numpy())                                                 # fp32 rows; S2 casts to fp16
    compare("F2b", items, z["codes"], z["q"])
    checks.put("encoder_parity", **out)


def check_scoring_file(npz_bytes, out_path_tmp, ship, ids_lines, digests, checks):
    z = np.load(io.BytesIO(npz_bytes), allow_pickle=False)
    names = sorted(z.files)
    want = sorted(["panel_codes", "panel_offsets", "panel_lengths", "panel_ids", "bin_centres", "lam", "lnK",
                   "meta_json"])
    checks.require(names == want, f"scoring npz arrays {names}")
    codes, off = z["panel_codes"], z["panel_offsets"]
    meta = json.loads(str(z["meta_json"]))
    tab = ship["tables"][TABLE_KEY]["all"]
    ok = dict(
        codes_bytes_md5=hashlib.md5(codes.tobytes()).hexdigest(),
        dtypes={k: str(z[k].dtype) for k in names},
        shapes={k: list(z[k].shape) for k in names},
        members=int(len(off) - 1), residues=int(off[-1]),
        ids_equal_ids_file=z["panel_ids"].tolist() == ids_lines,
        table_equal_ship_final=(z["bin_centres"].tolist() == tab["center_len"] and z["lam"].tolist() == tab["lam"]
                                and z["lnK"].tolist() == tab["lnK"]),
        panel_sha256_recomputed=tensor_digest({"codes": codes, "offsets": off}) == digests["panel_sha256"]
        == meta["panel"]["panel_sha256"],
        meta_table_equal_arrays=(meta["statistics"]["bin_centres"] == z["bin_centres"].tolist()
                                 and meta["statistics"]["lam"] == z["lam"].tolist()
                                 and meta["statistics"]["lnK"] == z["lnK"].tolist()),
    )
    checks.require(ok["codes_bytes_md5"] == PANEL_CODES_BYTES_MD5 and ok["members"] == 1000
                   and ok["residues"] == 355626 and ok["ids_equal_ids_file"] and ok["table_equal_ship_final"]
                   and ok["panel_sha256_recomputed"] and ok["meta_table_equal_arrays"], f"scoring npz: {ok}")
    # Stage-2 workers never import torch: load the written file in a process where torch cannot be imported.
    code = ("import sys, json; sys.modules['torch'] = None; import numpy as np; "
            f"z = np.load({out_path_tmp!r}, allow_pickle=False); a = {{k: z[k] for k in z.files}}; "
            "json.loads(str(a['meta_json'])); print(len(a), int(a['panel_offsets'][-1]))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    ok["loads_without_torch"] = r.returncode == 0 and r.stdout.split() == ["8", "355626"]
    checks.require(ok["loads_without_torch"], f"scoring npz without torch: {r.stderr[-500:]}")
    checks.put("scoring_file", **ok, bytes=len(npz_bytes))
    return z


def check_scores(Fk, zs, fixtures, checks):
    codes, off = zs["panel_codes"], zs["panel_offsets"].astype(np.int64)
    cen, lam_t, lnk_t = zs["bin_centres"], zs["lam"], zs["lnK"]
    # ---- F1 ----------------------------------------------------------------------------------------------------
    z = np.load(fixtures["F1"][0], allow_pickle=False)
    qo, to, po = z["q_offsets"], z["t_offsets"], z["path_off"]
    n = len(qo) - 1
    sf, sr, paths_eq, q90 = np.empty(n), np.empty(n), 0, np.empty(n)
    for k in range(n):
        q16 = z["q"][qo[k]:qo[k + 1]]
        a, path, b = kernel_pair(Fk, q16, z["codes"][to[k]:to[k + 1]])
        sf[k], sr[k] = a, b
        paths_eq += int(np.array_equal(path, z["path_flat"][po[k]:po[k + 1]]))
        q90[k] = panel_q90(Fk, q16, codes, off)
    La = np.diff(qo).astype(np.float64)
    lnE = lnE_of(cen, lam_t, lnk_t, (sf - sr) - q90, La, N_FIX)

    def relmax(x, ref):
        return float((np.abs(x - ref) / np.maximum(np.abs(ref), 1.0)).max())
    surv = np.flatnonzero(lnE <= math.log(1e-3))
    surv_ref = np.flatnonzero(z["lnE_N1e9"] <= math.log(1e-3))
    f1 = dict(pairs=n, paths_identical=paths_eq,
              S_f_bit_equal=int((sf == z["S_f"]).sum()), S_r_bit_equal=int((sr == z["S_r"]).sum()),
              S_f_max_rel=relmax(sf, z["S_f"]), S_r_max_rel=relmax(sr, z["S_r"]),
              q90_bit_equal=int((q90 == z["q90"]).sum()), q90_max_abs=float(np.abs(q90 - z["q90"]).max()),
              lnE_max_abs=float(np.abs(lnE - z["lnE_N1e9"]).max()),
              survivors_E_le_1e_3=surv.tolist(), survivors_fixture=surv_ref.tolist(),
              S16_bounds=dict(S_rel=5e-5, q90_abs=1e-4, lnE_abs=1e-4))
    checks.require(paths_eq == n, f"F1 paths {paths_eq}/{n}")
    checks.require(f1["S_f_max_rel"] <= 5e-5 and f1["S_r_max_rel"] <= 5e-5, f"F1 scores outside S16: {f1}")
    checks.require(f1["q90_max_abs"] <= 1e-4 and f1["lnE_max_abs"] <= 1e-4, f"F1 Q90 / lnE outside S16: {f1}")
    checks.require(surv.tolist() == surv_ref.tolist() and len(surv) == 3, f"F1 survivors at E <= 1e-3: {f1}")
    # ---- F3 ----------------------------------------------------------------------------------------------------
    z = np.load(fixtures["F3"][0], allow_pickle=False)
    o = z["offsets"]
    q3 = np.array([panel_q90(Fk, z["q"][o[k]:o[k + 1]], codes, off) for k in range(len(o) - 1)])
    f3 = dict(queries=int(len(o) - 1), q90_bit_equal=int((q3 == z["q90"]).sum()),
              q90_max_abs=float(np.abs(q3 - z["q90"]).max()), bound_S16=1e-4, bound_plan_F3=1e-3)
    checks.require(f3["q90_max_abs"] <= 1e-4, f"F3 Q90 outside S16: {f3}")
    # ---- F4 ----------------------------------------------------------------------------------------------------
    g = json.load(open(fixtures["F4"][0]))
    worst = dict(lam=0.0, lnK=0.0, lnE=0.0, bits=0.0)
    bits_render_eq = 0
    for r in g["rows"]:
        la, lk = float(tabp(cen, lam_t, r["m"])), float(tabp(cen, lnk_t, r["m"]))
        lnE_ = (lk + math.log(r["m"]) + math.log(r["N"])) - la * r["S_prime"]
        bits = (la * r["S_prime"] - lk) / math.log(2.0)
        for key, v in (("lam", la), ("lnK", lk), ("lnE", lnE_), ("bits", bits)):
            worst[key] = max(worst[key], abs(v - r[key]) / max(abs(r[key]), 1e-300))
        bits_render_eq += int(math.floor(bits + 0.5) == r["render_bits"])
    f4 = dict(rows=len(g["rows"]), max_rel=worst, render_bits_equal=bits_render_eq, bound=1e-12)
    checks.require(max(worst.values()) <= 1e-12 and bits_render_eq == len(g["rows"]), f"F4: {f4}")
    checks.put("scores", F1=f1, F3=f3, F4=f4)


# =====================================================================================================================
# Documents
# =====================================================================================================================
MIT_TEXT = """The MIT License

    {copyright}

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""


def readme_text(files, digests, checks, lic):
    c = checks.out
    p = c["encoder_parity"]
    s = c["scores"]
    rows = "\n".join(f"| `{fn}` | {v['bytes']:,} | `{v['sha256']}` |" for fn, v in files.items())
    return f"""---
license: mit
tags:
- biology
- protein
- esm3
- sequence-alignment
- foldmatch
---

# FoldMatch ESM3 residue codes (`esm-codes-v1`)

Encoders, statistics table and calibration panel for FoldMatch's ESM3 residue-code Stage-2 scoring. Every residue of a
database sequence is stored as an 8-byte binary code computed from its ESM3 embedding; a query keeps a 64-dimensional
fp16 encoding. Stage 2 aligns query and target with Smith-Waterman on the row/column-normalised code similarity and
reports calibrated E-values. It is an alternative to BLOSUM62, trained so that its alignments reproduce structural
(US-align) residue correspondences.

## Intended use

These files are made for the ESM3 residue-code scoring that is being added to
[FoldMatch](https://github.com/rcsb/foldmatch); FoldMatch's release notes will say from which version on it is
available. That FoldMatch downloads the files from this repository at a pinned revision and checks their sha256s
before use. It adds two options:

- `fm-search build sequences --residue-codes` writes the per-residue codes next to the sequence database (8 bytes per
  residue, from the same ESM3 pass that computes the chain embeddings);
- `fm-search query sequences --scoring esm-codes` scores the Stage-1 candidates with the codes (one scoring per query;
  BLOSUM62 scoring, `--scoring blosum62`, is unchanged).

The files are not meant to be used outside FoldMatch: the exact precision chain below is part of the method.

## Inputs

- ESM3 `{ESM3['model']}`, sequence-only, through `{ESM3['repo']}@{ESM3['revision']}` (weights file
  `{ESM3['weights_file']}`, sha256 `{ESM3['weights_sha256']}`, the same blob as
  `{ESM3['upstream_repo']}@{ESM3['upstream_revision']}`).
- Tap: `per_residue_embedding` from `forward_and_sample(..., SamplingConfig(return_per_residue_embeddings=True))`, i.e.
  the last block's output before the final LayerNorm; rows 1..L (BOS and EOS removed), `[L, 1536]`.
- Sequences are normalised first: ASCII whitespace removed, ASCII letters upper-cased, non-ASCII refused.

## Method (system `esm-codes-v1`)

1. Transform (zcw50_clean, not folded): `x16 = fp16(x)`; `xs = fp32(x16[:, dims]) - offset` (dims = 0..1535 without
   1361 and 1489); `xw = fp32(fp16(xs @ M))`; `offset` fp32[1534], `M` fp32[1534, 1534].
2. Encoders `f_q`, `f_t`: `f(x) = lin(x) + fc2(GELU_exact(fc1(x)))`, 1534 -> 1024 -> 64 with a linear skip, fp32 with
   no TF32, no autocast and no dropout ({c['encoders_file']['n_parameters']:,} parameters in total).
3. Target code: `bit_k = (f_t(xw)_k >= 0)`, `numpy.packbits(bits, axis=1)` MSB-first -> `uint8[L, 8]`; decoded as
   `2 bit - 1`.
4. Query encoding: `q = fp16(f_q(xw))`, `[L, 64]`.
5. Scoring (setting D): cosine similarity of `q` and the decoded codes (fp32 GEMM per pair), row/column z-scores
   (population std; a row or column with std < 1e-6 contributes 0), Smith-Waterman on `z - 1.0` with gap open 5.0 and
   extend 0.5 (DP id `fmsw/1`). Score `S = S_f - S_r`, where `S_r` aligns the row-reversed similarity.
6. Statistics: `S' = S - Q90(query)`, with Q90 the 0.9 quantile of S over a fixed panel of 1,000 UniRef50 sequences;
   `ln E = lnK(m) + ln m + ln N - lambda(m) S'` with lambda and lnK linear in ln m between six length bins and clamped
   outside them (m = query length, N = database residues).

Calibration version: `{CAL_VERSION}`.

## Files

| File | Bytes | sha256 |
|---|---:|---|
{rows}

- `{ENCODERS_FILE}`: `torch.load(path, weights_only=True)`. Format `{ENCODERS_FORMAT}`; the transform, both
  state_dicts, the architecture, conventions and provenance.
- `{SCORING_FILE}`: `numpy.load(path, allow_pickle=False)`. Format `{SCORING_FORMAT}`; `panel_codes` uint8
  [355626, 8], `panel_offsets`, `panel_lengths`, `panel_ids` (UniRef50), `bin_centres`, `lam`, `lnK` (float64[6]) and
  `meta_json` with every scoring constant.
- `{BUNDLE_FILE}`: system name, calibration version, the sha256s above, content digests (transform
  `{digests['transform_sha256'][:16]}...`, query encoder `{digests['query_encoder_sha256'][:16]}...`, target encoder
  `{digests['target_encoder_sha256'][:16]}...`, panel `{digests['panel_sha256'][:16]}...`), the ESM3 identity, the DP
  id, licence and attribution.
- `SHA256SUMS`: `shasum -a 256 -c SHA256SUMS` checks every file.

## Training and calibration

The encoders (research model L2m_B8_s0, epoch 7) were trained on sequence-only ESM3 embeddings of CATH domains to
reproduce US-align residue correspondences (a cross-entropy on distance-filtered US-align targets, distillation from the
full-precision whitened ESM3 similarity and a Smith-Waterman margin term), on a training pool that excludes every
evaluation domain and its superfamily; the checkpoint and the Smith-Waterman setting were chosen on development pairs
only. Smith-Waterman F1 against US-align at the shipped setting: 0.640 on validation related CATH pairs, 0.266 on
validation remote pairs, 0.262 on test remote pairs and 0.613 on 582 similar multi-domain chain pairs (predicted 3Di at
its own tuned setting: 0.604, 0.212, 0.203 and 0.536). E-values were fitted (censored Gumbel ML at q = 0.999 per
query-length bin) on 1,465 SCOP-annotated benchmark queries against 50,000 targets (8.6 million residues), with the
real null = targets that share no SCOP fold with the query; the per-query location Q90 comes from 1,000 UniRef50
release 2026_03 cluster representatives (amended stratified draw, 355,626 residues). On held-out query halves the
false-positive ratio (observed / expected) was 0.82 and 1.13 at E <= 1, 1.26 and 0.93 at E <= 0.1, and 2.7 and 2.2
at E <= 0.01; sensitivity at E <= 1e-3 was 0.15 and 0.20. FoldMatch's own ESM3 path reproduces the research codes on
99.5-99.6% of bits on GPU, CPU and Apple MPS (median |delta ln E| about 0.05, 0.049-0.051), and Mac and Linux CPUs
agree on 99.997% of bits.

## Limitations

- Sequence (FASTA) inputs only. The encoders were trained on sequence-only ESM3 embeddings; structure-conditioned
  embeddings are out of distribution, and FoldMatch's residue-code options are for sequence inputs only.
- Calibrated domain: queries of 21-966 residues, targets up to 3,000 residues, at most 1% non-standard residues on
  either side and targets with at least 2 distinct codes. Outside it FoldMatch still reports the alignment but sets
  `ecal = 0` and warns.
- Long queries: for queries of 1,000-3,000 residues the E-values are optimistic (about 2x at E <= 1, 38x at E <= 0.01
  and 180x at E <= 1e-3 on an 8.6-million-residue database); such hits need E far below the threshold or independent
  confirmation. Long targets (1,000-3,000 residues) are calibrated (false-positive ratio 1.00 at E <= 1).
- In general E <= 0.01 is about 2.5x optimistic while E <= 1 is about right; the panel and table are fixed, so any
  change is a new system name.
- A query of one residue, or whose encoding rows are all identical, carries no information and reports no hits.
- Results are bit-identical on one machine and BLAS build; across machines they agree within FoldMatch's tolerance
  contract (scores within 5e-5 relative, Q90 and ln E within 1e-4).
- Only the ESM3 weights with sha256 `{ESM3['weights_sha256'][:16]}...` are supported.
- The codes are trained for alignment quality, not for homology detection; candidates come from FoldMatch's Stage-1
  chain-embedding search, which bounds recall.

## Licence and attribution

{lic['copyright']}. The encoders, the transform, the statistics table, the panel codes and this documentation are
released under the MIT licence (see `LICENSE`), like FoldMatch. The panel member identifiers (`panel_ids`) are UniProt
data and are not covered by the MIT licence: they stay under CC BY 4.0, with the attribution below.

- UniRef50: {lic['attribution']['uniref50']}
- ESM3: {lic['attribution']['esm3']}

## Citation

- FoldMatch: Segura, J., et al. (2026). Multi-scale structural similarity embedding search across entire proteomes.
  https://doi.org/10.1093/bioinformatics/btag058
- ESM3: Hayes, T., et al. (2025). Simulating 500 million years of evolution with a language model. Science 387,
  850-858. https://doi.org/10.1126/science.ads0018
- UniProt: The UniProt Consortium (2025). UniProt: the Universal Protein Knowledgebase in 2025. Nucleic Acids Research
  53(D1), D609-D617. https://doi.org/10.1093/nar/gkae1010

## Verification

Before writing these files the export script (`rcsb-embedding-search src/scripts/save_esm_codes_for_production.py`)
checked that the encoders reload with `weights_only=True`, rebuild with `strict=True` and equal the research
checkpoint, and that re-encoding FoldMatch's test fixtures gives the reference codes (F2
{p['F2']['bits_equal_fixture']:,}/{p['F2']['bits']:,} bits, F2c {p['F2c']['bits_equal_fixture']:,}/{p['F2c']['bits']:,},
F2b {p['F2b']['bits_equal_fixture']:,}/{p['F2b']['bits']:,}) and query encodings ({p['F2']['q_vs_fixture']['bit_equal']:,} of
{p['F2']['q_vs_fixture']['n']:,} F2 values bit-equal). With this panel and table the research kernel reproduces the 50
fixture pairs ({s['F1']['paths_identical']}/50 paths; max |delta Q90| {s['F1']['q90_max_abs']:.2g}, max |delta ln E|
{s['F1']['lnE_max_abs']:.2g}) and 20 benchmark Q90s (max |delta| {s['F3']['q90_max_abs']:.2g}), and the scoring file
loads with `allow_pickle=False` in a process without torch. These numbers come from the export machine (Apple silicon
CPU). On other CPUs (checked on Linux x86-64) the codes are identical and the query encodings agree within one fp16 ulp
of each row's largest value.
"""


def upload_text(files_to_upload, sums_sha):
    lst = "\n".join(f"- `{f}`" for f in files_to_upload)
    return f"""# Uploading the esm-codes-v1 bundle

This folder is ready to upload as the **public** Hugging Face model repo `{HF_REPO}`. Upload these
{len(files_to_upload)} files, i.e. everything here except this `UPLOAD.md`:

{lst}

Nothing has been uploaded yet. The commands use the `hf` CLI and Python of the `foldmatch` conda env
(huggingface-hub 0.36.2, the version FoldMatch pins):

```bash
conda activate foldmatch
unset HF_HUB_OFFLINE
```

## 1. Check the files

```bash
cd /Users/joan/devel/foldmatch-esm-codes-hf
shasum -a 256 -c SHA256SUMS        # every line must say OK
shasum -a 256 SHA256SUMS            # must print {sums_sha}
```

## 2. Log in with a token that can write to the `rcsb` organisation

```bash
hf auth whoami      # if not logged in: hf auth login
```

## 3. Create the public model repo

```bash
hf repo create {HF_REPO}
python -c "from huggingface_hub import HfApi; print('private =', HfApi().model_info('{HF_REPO}').private)"
```

It must print `private = False`: FoldMatch downloads the bundle without a token. If it prints `private = True`
(an organisation default), make it public in the repo's Settings or with:

```bash
python -c "from huggingface_hub import HfApi; HfApi().update_repo_settings('{HF_REPO}', private=False)"
```

## 4. Upload in one commit

```bash
hf upload {HF_REPO} /Users/joan/devel/foldmatch-esm-codes-hf . --exclude UPLOAD.md ".*" --commit-message "esm-codes-v1 bundle"
```

`--exclude` keeps this file and hidden files such as `.DS_Store` out of the repo. `hf upload` prints the URL of the
uploaded folder, not the commit hash; step 5 reads the hash.

## 5. Read back the 40-hex commit hash

```bash
python -c "from huggingface_hub import HfApi; c = HfApi().list_repo_commits('{HF_REPO}')[0]; print(c.commit_id, c.title)"
# or, without Python:
git ls-remote https://huggingface.co/{HF_REPO} refs/heads/main
```

The 40-hex hash of the commit titled "esm-codes-v1 bundle" is the revision to send back for the P4 registry entry.

## 6. Check the anonymous download at that revision

```bash
REV=<the 40-hex commit hash>
rm -rf /tmp/esm-codes-check && mkdir /tmp/esm-codes-check
HF_HUB_DISABLE_IMPLICIT_TOKEN=1 hf download {HF_REPO} --revision "$REV" --local-dir /tmp/esm-codes-check
cd /tmp/esm-codes-check && shasum -a 256 -c SHA256SUMS
```

`HF_HUB_DISABLE_IMPLICIT_TOKEN=1` makes the download anonymous, so it also proves that the repo is public.

After the upload, leave `{ENCODERS_FILE}`, `{SCORING_FILE}` and `{BUNDLE_FILE}` unchanged: FoldMatch pins the revision
and their sha256s. A later README-only edit is harmless but creates another commit; send the hash from step 5.
"""


# =====================================================================================================================
# Main
# =====================================================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--artefacts", default=os.environ.get("ESMCODES_ARTEFACTS", DEFAULT_ARTEFACTS))
    ap.add_argument("--fm", default=os.environ.get("ESMCODES_FM", DEFAULT_FM),
                    help="FoldMatch checkout with tests/resources/esm_codes (read only)")
    ap.add_argument("--kernel-dir", default=DEFAULT_KERNEL_DIR,
                    help="where swkern.c / swfast.py are copied and libswkern is compiled")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--report", default=DEFAULT_REPORT, help="JSON with every self-check number")
    ap.add_argument("--threads", type=int, default=4, help="torch threads for the encoder checks (fixtures: 4)")
    ap.add_argument("--q-tolerance", choices=["exact", "1ulp"], default="exact",
                    help="exact: q must equal the fixtures bit for bit (Apple silicon CPU with torch 2.9.0, the "
                         "fixtures' reference env); 1ulp: within 1 fp16 ulp of the row maximum (any other CPU or "
                         "torch build: on Linux x86-64 even torch 2.9.0 differs in the last q bits; the codes must "
                         "still be identical)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    checks = Checks()
    md5tab, shatab = input_table(args.artefacts, args.fm)
    inputs_rec = verify_inputs(md5tab, shatab)
    fixtures = {**shatab, **{k: v for k, v in md5tab.items() if k.startswith("emb_")}}

    st, how = load_research_checkpoint(md5tab["encoders"][0])
    zc = dict(np.load(md5tab["zcw50_clean"][0], allow_pickle=False))
    ship = json.load(open(md5tab["ship_final"][0]))
    if ship["meta"]["version"] != CAL_VERSION:
        raise SystemExit(f"ship_final.json version {ship['meta']['version']} != {CAL_VERSION}")
    zp = np.load(md5tab["panel_codes"][0], allow_pickle=False)
    ids_lines = [x for x in open(md5tab["panel_ids"][0]).read().split("\n") if x]

    # ---- build -------------------------------------------------------------------------------------------------
    bundle = build_encoder_bundle(st, zc, inputs_rec)
    pt_bytes = torch_save_bytes(bundle)
    pt_bytes_again = torch_save_bytes(build_encoder_bundle(st, zc, inputs_rec))
    digests = {
        "method": DIGEST_METHOD,
        "method_detail": ("sha256 over the ASCII tag 'foldmatch-tensor-digest/1' and a newline, then for each array "
                          "name in sorted order: an 8-byte little-endian length and the compact sorted-key JSON "
                          "{dtype (numpy dtype str, little-endian), name, shape}, an 8-byte little-endian length and "
                          "the raw C-order little-endian bytes"),
        "transform_sha256": tensor_digest(bundle["transform"]),
        "query_encoder_sha256": tensor_digest(bundle["f_q"]),
        "target_encoder_sha256": tensor_digest(bundle["f_t"]),
        "transform_arrays": ["M", "dims", "offset"],
        "encoder_arrays": sorted(STATE_KEYS),
        "panel_arrays": ["codes (panel_codes)", "offsets (panel_offsets)"],
        "table_arrays": ["bin_centres", "lam", "lnK"],
    }
    arrays, stats, panel, _ = build_scoring(zp, ids_lines, ship, digests)
    lic = licence_block()
    meta = {
        "format": SCORING_FORMAT,
        "system": SYSTEM,
        "calibration_version": CAL_VERSION,
        "arrays": {
            "panel_codes": "uint8 [N_res, 8]: packed target codes of the 1,000 panel members (S4), concatenated",
            "panel_offsets": "int64 [1001]: member k = panel_codes[panel_offsets[k]:panel_offsets[k + 1]]",
            "panel_lengths": "int64 [1000]",
            "panel_ids": "UniRef50 cluster ids, in panel order",
            "bin_centres": "float64 [6]: E-value table bin centres (query length, geometric mean per bin)",
            "lam": "float64 [6]: lambda at the bin centres",
            "lnK": "float64 [6]: ln K at the bin centres",
        },
        "code": {k: CONVENTIONS["code"][k] for k in ("bits", "bytes", "bitorder", "sign_rule", "decode")},
        "scoring": SCORING,
        "statistics": stats,
        "panel": panel,
        "calibrated_domain": CALIBRATED_DOMAIN,
        "degenerate_query_rule": DEGENERATE_QUERY_RULE,
        "degenerate_query_detail": DEGENERATE_QUERY_DETAIL,
        "esm3": ESM3,
        "licence": lic,
        "provenance": {"ship_final.json": inputs_rec["ship_final"], "shipped_panel_codes.npz": inputs_rec["panel_codes"],
                       "shipped_panel.ids": inputs_rec["panel_ids"], "fit_inputs": ship["meta"]["inputs"]},
    }
    arrays["meta_json"] = np.array(json.dumps(meta, ensure_ascii=True, allow_nan=False, sort_keys=False))
    npz_bytes = write_npz_bytes(arrays)
    checks.require(npz_bytes == write_npz_bytes(arrays), "npz serialisation is not deterministic")

    # ---- self-checks (nothing final is written before they pass) ------------------------------------------------
    os.makedirs(args.out, exist_ok=True)
    tmp_pt = os.path.join(args.out, ENCODERS_FILE + ".tmp")
    tmp_npz = os.path.join(args.out, SCORING_FILE + ".tmp")
    try:
        write_atomic(tmp_pt, pt_bytes)
        write_atomic(tmp_npz, npz_bytes)
        with open(tmp_pt, "rb") as f:
            checks.require(f.read() == pt_bytes, "written .pt differs")
        _, enc = check_encoders(pt_bytes, pt_bytes_again, st, zc, checks)
        check_parity(enc, args.artefacts, st, fixtures, checks, strict_q=(args.q_tolerance == "exact"))
        zs = check_scoring_file(npz_bytes, tmp_npz, ship, ids_lines, digests, checks)
        Fk = kernel_module(args.kernel_dir, md5tab, checks)
        check_scores(Fk, zs, fixtures, checks)
    except BaseException:
        for p in (tmp_pt, tmp_npz):
            if os.path.exists(p):
                os.remove(p)
        raise
    os.replace(tmp_pt, os.path.join(args.out, ENCODERS_FILE))
    os.replace(tmp_npz, os.path.join(args.out, SCORING_FILE))

    # ---- bundle.json, documents -----------------------------------------------------------------------------------
    files = {ENCODERS_FILE: dict(sha256=sha256_bytes(pt_bytes), bytes=len(pt_bytes)),
             SCORING_FILE: dict(sha256=sha256_bytes(npz_bytes), bytes=len(npz_bytes))}
    me = os.path.abspath(__file__)
    bundle_json = {
        "bundle_format": BUNDLE_FORMAT,
        "system": SYSTEM,
        "calibration_version": CAL_VERSION,
        "hf_repo": HF_REPO,
        "files": {
            ENCODERS_FILE: {**files[ENCODERS_FILE], "format": ENCODERS_FORMAT,
                            "load": "torch.load(path, map_location='cpu', weights_only=True)"},
            SCORING_FILE: {**files[SCORING_FILE], "format": SCORING_FORMAT,
                           "load": "numpy.load(path, allow_pickle=False)"},
        },
        "content_digests": digests,
        "esm3": ESM3,
        "encoding": {
            "architecture": ARCHITECTURE,
            "transform": CONVENTIONS["transform"]["formula"],
            "input_cast": "float16",
            "transform_output_cast": "float16",
            "encoder": CONVENTIONS["encoder"],
            "code_bits": 64,
            "code_bytes": 8,
            "bitorder": CONVENTIONS["code"]["bitorder"],
            "sign_rule": CONVENTIONS["code"]["sign_rule"],     # one canonical string in .pt, npz and bundle.json
            "query_dtype": "float16",
        },
        "scoring": SCORING,
        "statistics": stats,
        "panel": panel,
        "calibrated_domain": CALIBRATED_DOMAIN,
        "degenerate_query_rule": DEGENERATE_QUERY_RULE,
        "degenerate_query_detail": DEGENERATE_QUERY_DETAIL,
        "licence": lic,
        "provenance": {
            "research_model": "L2m_B8_s0 (sw_mlp), checkpoint epoch %d" % int(st["epoch"]),
            "inputs": {k: inputs_rec[k] for k in ("encoders", "encoders_history", "sw_common", "zcw50_clean",
                                                  "ship_final", "panel_codes", "panel_ids")},
            "generator": {"script": "rcsb-embedding-search src/scripts/save_esm_codes_for_production.py",
                          "sha256": sha256_file(me)},
            "environment": {"python": sys.version.split()[0], "torch": str(torch.__version__),
                            "numpy": np.__version__, "platform": f"{platform.system()}-{platform.machine()}"},
        },
    }
    # one value per field across the three files (P4 compares bundle.json and meta_json with the registry entry)
    for k in ("system", "calibration_version", "scoring", "statistics", "panel", "calibrated_domain",
              "degenerate_query_rule", "degenerate_query_detail", "esm3", "licence"):
        checks.require(bundle_json[k] == meta[k], f"bundle.json and meta_json disagree on {k}")
    enc_, code_ = bundle_json["encoding"], meta["code"]
    checks.require((enc_["code_bits"], enc_["code_bytes"], enc_["bitorder"], enc_["sign_rule"])
                   == (code_["bits"], code_["bytes"], code_["bitorder"], code_["sign_rule"])
                   == tuple(bundle["conventions"]["code"][k] for k in ("bits", "bytes", "bitorder", "sign_rule")),
                   "code conventions differ between bundle.json, meta_json and the .pt")
    checks.require(enc_["architecture"] == bundle["architecture"]
                   and enc_["encoder"] == bundle["conventions"]["encoder"]
                   and enc_["transform"] == bundle["conventions"]["transform"]["formula"],
                   "encoding differs between bundle.json and the .pt")
    checks.put("consistency", bundle_json_meta_json_pt="identical shared fields")
    bj = jdump_str(bundle_json).encode()
    write_atomic(os.path.join(args.out, BUNDLE_FILE), bj)
    files[BUNDLE_FILE] = dict(sha256=sha256_bytes(bj), bytes=len(bj))
    lic_text = (MIT_TEXT.format(copyright=COPYRIGHT) + "\n").encode()
    write_atomic(os.path.join(args.out, "LICENSE"), lic_text)
    readme = readme_text(files, digests, checks, lic).encode()
    write_atomic(os.path.join(args.out, "README.md"), readme)
    upload = [ENCODERS_FILE, SCORING_FILE, BUNDLE_FILE, "README.md", "LICENSE", "SHA256SUMS"]
    sums = "".join(f"{sha256_file(os.path.join(args.out, f))}  {f}\n" for f in upload[:-1]).encode()
    write_atomic(os.path.join(args.out, "SHA256SUMS"), sums)
    write_atomic(os.path.join(args.out, "UPLOAD.md"), upload_text(upload, sha256_bytes(sums)).encode())
    extra = sorted(set(os.listdir(args.out)) - set(upload) - {"UPLOAD.md"})
    checks.require(not extra, f"unexpected files in {args.out}: {extra}")

    # ---- report and sha256s ----------------------------------------------------------------------------------------
    sha_all = {f: sha256_file(os.path.join(args.out, f)) for f in upload + ["UPLOAD.md"]}
    report = dict(system=SYSTEM, out=args.out, research_checkpoint_load=how, inputs=inputs_rec,
                  checks=checks.out, content_digests=digests, files={f: dict(sha256=sha_all[f], bytes=os.path.getsize(
                      os.path.join(args.out, f))) for f in sha_all},
                  environment=bundle_json["provenance"]["environment"], torch_threads=args.threads,
                  q_tolerance=args.q_tolerance)
    if args.report:
        os.makedirs(os.path.dirname(args.report), exist_ok=True)
        write_atomic(args.report, jdump_str(report).encode())
    print(jdump_str(dict(checks=checks.out, content_digests={k: v for k, v in digests.items()
                                                             if k.endswith("sha256")})))
    print("All self-checks passed.")
    for f in upload + ["UPLOAD.md"]:
        print(f"{sha_all[f]}  {f}")


if __name__ == "__main__":
    main()
