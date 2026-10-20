#!/usr/bin/env python
"""P8 A2: encoder parity (plan FINAL v1, section 8 P8, A2 row; method S2-S5; plan section 6 P2 'F2' measure).

The production encoder module (foldmatch.model.residue_code_encoder, loaded through the registry bundle:
scoring_system.prepare(esm-codes-v1).load_head(), which checks the bundle sha256s and content digests) against the
research encodings of the 50 enc_check items (sw_mlp/enc_check/L2m_B8_s0.npz, written by sw_extract.py from the stored
research encodings sw_mlp/enc/L2m_B8_s0.{val,test,chains}.npz): 15 val + 15 test CATH domains and 20 chains, 9,072
residues = 580,608 code bits. Input rows = the rows the research encoder read (sw_train.py encode: val.npz / test.npz
'emb' for the domains, chains/data/chain_emb.npz 'emb' for the chains; fp16 [L, 1536], BOS / EOS already removed).
Each item is encoded alone (ResidueCodeHead.encode_target / encode_query, one protein at a time, as production does).

Pass (plan A2): code bits >= 0.99999 identical (pooled over the 580,608 bits) and q within 1 fp16 ulp, measured as in
P2 (FoldMatch tests/test_esm_codes_encoder.py _row_ulps): max |q - q_ref| / ulp(row max), ulp(row max) =
2^(floor(log2(max(|q|, |q_ref|) over the row)) - 10) (per-element ulps are misleading near 0).

Usage:
  python a2_encoder_parity.py extract     NERSC: the 50 items' input rows -> <P8_ROOT>/A2/reference_extract/ (the Mac
                                          has neither val.npz / test.npz nor chain_emb.npz; it uses a copy, sha256
                                          checked)
  python a2_encoder_parity.py run [--rows source|extract] [--device cpu|mps|cuda] [--tag TAG]
                                          -> <P8_ROOT>/A2/<TAG>/a2_result.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import acc_common as C  # noqa: E402
import numpy as np  # noqa: E402

BITS_MIN = 0.99999
ULP_MAX = 1.0
N_BITS = 580_608
INP = {
    "enc_check": "embcomp/sw_mlp/enc_check/L2m_B8_s0.npz",
    "items": "embcomp/sw_mlp/enc_check/check_items.json",
    "val": "embcomp/val.npz",
    "test": "embcomp/test.npz",
    "chains": "embcomp/chains/data/chain_emb.npz",
}
SETS = ("val", "test", "chains")
EXTRACT_DIR = os.path.join(C.P8_ROOT, "A2", "reference_extract")
EXTRACT = os.path.join(EXTRACT_DIR, "a2_rows_extract.npz")
EXTRACT_RECORD = os.path.join(EXTRACT_DIR, "a2_rows_extract.json")
log = C.log


def _npz_memmap(path, name):
    """A STORED npz member as a read-only memmap (chain_eval._npz_memmap); None when it is compressed."""
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(name + ".npy")
        if info.compress_type != zipfile.ZIP_STORED:
            return None
    with open(path, "rb") as fh:
        fh.seek(info.header_offset)
        loc = fh.read(30)
        fh.seek(info.header_offset + 30 + int.from_bytes(loc[26:28], "little") + int.from_bytes(loc[28:30], "little"))
        ver = np.lib.format.read_magic(fh)
        shape, fortran, dtype = (np.lib.format.read_array_header_1_0 if ver == (1, 0)
                                 else np.lib.format.read_array_header_2_0)(fh)
        off = fh.tell()
    return np.memmap(path, dtype=dtype, mode="r", shape=shape, offset=off, order="F" if fortran else "C")


def source_rows(items: dict) -> dict:
    """{set: {id: fp16 [L, 1536]}} from the research row files (val.npz / test.npz 'domain_ids', chain_emb.npz
    'chain_ids'; 'offsets'; 'emb'), exactly the rows sw_train.py encoded."""
    out = {}
    for s in SETS:
        p = C.inp(INP[s])
        z = np.load(p)
        ids = (z["chain_ids"] if s == "chains" else z["domain_ids"]).astype(str).tolist()
        off = z["offsets"]
        emb = _npz_memmap(p, "emb")
        if emb is None:
            emb = z["emb"]
        assert emb.dtype == np.float16 and emb.shape[1] == 1536 and off[-1] == len(emb)
        idx = {d: k for k, d in enumerate(ids)}
        out[s] = {d: np.array(emb[off[idx[d]]:off[idx[d] + 1]]) for d in items[s]}
    return out


def extract(args) -> int:
    items = C.read_json(C.inp(INP["items"]))
    t0 = time.time()
    rows = source_rows(items)
    blob, sha = {}, {}
    for s in SETS:
        for d, x in rows[s].items():
            blob[f"{s}/rows/{d}"] = x
            sha[f"{s}/{d}"] = C.array_sha256(x)
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    tmp = EXTRACT + f".tmp{os.getpid()}.npz"
    np.savez(tmp, **blob)
    os.replace(tmp, EXTRACT)
    rec = dict(what="A2 input rows: the 50 enc_check items' research ESM3 rows (fp16 [L, 1536]) copied from the "
                    "research row files", items=items, item_rows_sha256=sha,
               source={s: dict(path=C.inp(INP[s]), realpath=os.path.realpath(C.inp(INP[s])),
                               md5=C.md5_file(C.inp(INP[s]))) for s in SETS},
               extract=EXTRACT, sha256=C.sha256_file(EXTRACT), md5=C.md5_file(EXTRACT),
               wall_s=round(time.time() - t0, 1), provenance=C.provenance(kernel=False))
    h = C.write_json(rec, EXTRACT_RECORD)
    log(f"wrote {EXTRACT} sha256 {rec['sha256']}; record {EXTRACT_RECORD} sha256 {h}")
    return 0


def extract_rows() -> tuple:
    rec = C.read_json(EXTRACT_RECORD)
    got = C.sha256_file(EXTRACT)
    if got != rec["sha256"]:
        raise RuntimeError(f"{EXTRACT}: sha256 {got} != the record's {rec['sha256']}")
    z = np.load(EXTRACT)
    items = rec["items"]
    rows = {s: {d: z[f"{s}/rows/{d}"] for d in items[s]} for s in SETS}
    for s in SETS:
        for d, x in rows[s].items():
            assert C.array_sha256(x) == rec["item_rows_sha256"][f"{s}/{d}"], (s, d)
    return rows, rec


def row_ulps(a, b) -> float:
    """FoldMatch tests/test_esm_codes_encoder.py _row_ulps (the P2 'within 1 fp16 ulp' measure)."""
    a32, b32 = np.asarray(a, np.float32), np.asarray(b, np.float32)
    rowmax = np.maximum(np.abs(a32), np.abs(b32)).max(axis=1, keepdims=True)
    ulp = 2.0 ** (np.floor(np.log2(np.maximum(rowmax, 2.0 ** -14))) - 10)
    return float((np.abs(a32 - b32) / ulp).max())


def row_ulps_ref(q, ref) -> float:
    """tests/test_esm_codes_build_channel.py _row_ulps (ulp of the reference row's largest |value|; reported)."""
    q32, r32 = q.astype(np.float32), ref.astype(np.float32)
    scale = np.spacing(np.abs(ref).max(axis=1).astype(np.float16)).astype(np.float32)[:, None]
    return float((np.abs(q32 - r32) / scale).max())


def run(args) -> int:
    t_start = time.time()
    tag = args.tag or (C.HOST if args.device == "cpu" else f"{C.HOST}_{args.device}")
    rdir = C.result_dir("A2", tag)
    import torch
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    from foldmatch.search import scoring_system as ss
    res = {"check": "A2", "what": __doc__.split("\n\n")[0], "tag": tag, "device": args.device,
           "rows_source": args.rows, "torch_threads_requested": args.torch_threads,
           "criteria": {"code_bits_min": BITS_MIN, "q_row_max_ulps_max": ULP_MAX, "bits_total": N_BITS}}
    res["provenance"] = C.provenance(kernel=False, torch_info=True)
    items = C.read_json(C.inp(INP["items"]))
    ref = np.load(C.inp(INP["enc_check"]))
    res["inputs"] = {k: dict(name=v, md5=C.md5_file(C.inp(v))) for k, v in INP.items()
                     if k in ("enc_check", "items")}
    if args.rows == "source":
        rows = source_rows(items)
        for s in SETS:
            res["inputs"][s] = dict(name=INP[s], realpath=os.path.realpath(C.inp(INP[s])))
        if os.path.exists(EXTRACT_RECORD):        # the Mac's copy is checked against the source rows here
            ex, rec = extract_rows()
            res["extract_rows_equal_source"] = all(np.array_equal(ex[s][d].view(np.uint16), rows[s][d].view(np.uint16))
                                                   for s in SETS for d in items[s])
            res["inputs"]["reference_extract"] = dict(path=EXTRACT, sha256=rec["sha256"])
    else:
        rows, rec = extract_rows()
        res["inputs"]["reference_extract"] = dict(path=EXTRACT, sha256=rec["sha256"], record=EXTRACT_RECORD,
                                                  made_on=rec["provenance"]["host"]["hostname"],
                                                  source_md5={s: v["md5"] for s, v in rec["source"].items()})
    system = ss.get(ss.DEFAULT_ESM_CODES)
    prepared = ss.prepare(system, local_files_only=True)
    t = time.perf_counter()
    head = prepared.load_head(device=args.device)
    res["head"] = dict(system=system.name, encoders=str(prepared.encoders_path),
                       encoders_sha256=system.bundle_sha256(prepared.encoders_file),
                       content_digests=head.content_digests(), load_s=round(time.perf_counter() - t, 3),
                       torch_threads=torch.get_num_threads())
    per, eq_bits, n_bits, q_eq, q_n = {}, 0, 0, 0, 0
    worst_ulps = worst_ulps_ref = 0.0
    t = time.perf_counter()
    for s in SETS:
        for d in items[s]:
            x = torch.from_numpy(np.ascontiguousarray(rows[s][d]))
            codes = head.encode_target(x).cpu().numpy()
            q = head.encode_query(x).cpu().numpy()
            c_ref, q_ref = ref[f"{s}/code/{d}"], ref[f"{s}/q/{d}"]
            assert codes.shape == c_ref.shape and q.shape == q_ref.shape and q.dtype == np.float16, (s, d)
            be = int((np.unpackbits(codes, axis=1) == np.unpackbits(c_ref, axis=1)).sum())
            nb = int(codes.size * 8)
            qe = int((q.view(np.uint16) == q_ref.view(np.uint16)).sum())
            u, ur = row_ulps(q, q_ref), row_ulps_ref(q, q_ref)
            per[f"{s}/{d}"] = dict(L=int(len(q)), bits_equal=be, bits=nb, q_bit_equal=qe, q_values=int(q.size),
                                   q_row_max_ulps=u, q_ref_row_max_ulps=ur,
                                   q_max_abs=float(np.abs(q.astype(np.float32) - q_ref.astype(np.float32)).max()))
            eq_bits += be
            n_bits += nb
            q_eq += qe
            q_n += int(q.size)
            worst_ulps, worst_ulps_ref = max(worst_ulps, u), max(worst_ulps_ref, ur)
    res["encode_s"] = round(time.perf_counter() - t, 3)
    frac = eq_bits / n_bits
    res["totals"] = dict(items=len(per), residues=int(sum(v["L"] for v in per.values())), bits=n_bits,
                         bits_equal=eq_bits, bits_fraction=frac, q_values=q_n, q_bit_equal=q_eq,
                         q_row_max_ulps_max=worst_ulps, q_ref_row_max_ulps_max=worst_ulps_ref,
                         items_codes_identical=sum(v["bits_equal"] == v["bits"] for v in per.values()))
    res["per_item"] = per
    verdict = dict(bits_total_ok=bool(n_bits == N_BITS), bits_ok=bool(frac >= BITS_MIN),
                   q_ok=bool(worst_ulps <= ULP_MAX))
    if "extract_rows_equal_source" in res:
        verdict["extract_rows_equal_source"] = bool(res["extract_rows_equal_source"])
    verdict["pass"] = bool(verdict["bits_total_ok"] and verdict["bits_ok"] and verdict["q_ok"])
    res["verdict"] = verdict
    res["wall_s"] = round(time.time() - t_start, 1)
    out = os.path.join(rdir, "a2_result.json")
    h = C.write_json(res, out)
    T = res["totals"]
    log(f"A2 {tag}: code bits {T['bits_equal']}/{T['bits']} ({T['bits_fraction']:.7f}); q bit-equal "
        f"{T['q_bit_equal']}/{T['q_values']}; max {T['q_row_max_ulps_max']} fp16 ulp of the row max "
        f"(ref-row convention {T['q_ref_row_max_ulps_max']}); items with identical codes {T['items_codes_identical']}/50")
    log(f"VERDICT A2 {tag}: {'PASS' if verdict['pass'] else 'FAIL'} {json.dumps(verdict)}")
    log(f"wrote {out} sha256 {h}")
    return 0 if verdict["pass"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("extract")
    r = sub.add_parser("run")
    r.add_argument("--rows", choices=("source", "extract"), default="source")
    r.add_argument("--device", default="cpu")
    r.add_argument("--tag", default=None)
    r.add_argument("--torch-threads", type=int, default=None,
                   help="torch.set_num_threads (default: torch's own, 1 under acc_common's OMP_NUM_THREADS=1)")
    a = ap.parse_args(argv)
    return extract(a) if a.cmd == "extract" else run(a)


if __name__ == "__main__":
    sys.exit(main())
