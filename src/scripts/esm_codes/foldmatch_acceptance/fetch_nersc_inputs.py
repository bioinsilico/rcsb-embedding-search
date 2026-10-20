#!/usr/bin/env python
"""Mac side of P8: copy the staged NERSC inputs the Mac runs need but does not have, and the reference extracts.

stage_inputs.py on the Mac links only the local artefacts; A1 and A2 also need inputs that exist only on NERSC (and in
the archive). This script copies them (scp, read-only on NERSC) into <P8_ROOT>/nersc_copies/<logical name>, checks
each copy's md5 against the NERSC inputs_manifest.json entry, and links <P8_ROOT>/inputs/<logical name> to it (the
same logical names, so acc_common.inp() works unchanged). The reference extracts made on NERSC
(A1/reference_extract, A2/reference_extract) are copied to the same paths under the local P8_ROOT and checked against
their records' sha256. Record: <P8_ROOT>/prep/mac_nersc_copies.json.

Usage (Mac): python fetch_nersc_inputs.py [--names NAME ...] [--extracts A1 A2]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import acc_common as C  # noqa: E402

SSH = ["ssh", "-i", os.path.expanduser("~/.ssh/nersc"), "-o", "BatchMode=yes", "-o", "LogLevel=ERROR"]
SCP = ["scp", "-i", os.path.expanduser("~/.ssh/nersc"), "-o", "BatchMode=yes", "-o", "LogLevel=ERROR"]
HOST = "jsegura@perlmutter.nersc.gov"
NERSC_P8 = "/pscratch/sd/j/jsegura/embcomp/p8"
DEFAULT_NAMES = [
    # A1
    "embcomp/throughput/build/data/L2m_B8_s0.val.npz", "embcomp/throughput/build/data/L2m_B8_s0.test.npz",
    "embcomp/evalue/f1_settings/results/dom/M8_s0.val.npz", "embcomp/evalue/f1_settings/results/dom/M8_s0.test.npz",
    "embcomp/evalue/f1_settings/results/check.json",
    "embcomp/chains/data/pairs.json", "embcomp/chains/data/chains_meta.json",
    # A2
    "embcomp/sw_mlp/enc_check/L2m_B8_s0.npz", "embcomp/sw_mlp/enc_check/check_items.json",
]
EXTRACTS = {"A1": ("A1/reference_extract", "a1_reference_extract.npz", "a1_reference_extract.json"),
            "A2": ("A2/reference_extract", "a2_rows_extract.npz", "a2_rows_extract.json")}


def main(argv=None) -> int:
    if C.IS_NERSC:
        raise SystemExit("run this on the Mac")
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--names", nargs="*", default=DEFAULT_NAMES)
    ap.add_argument("--extracts", nargs="*", default=list(EXTRACTS))
    a = ap.parse_args(argv)
    with tempfile.TemporaryDirectory(dir=os.path.join(C.P8_ROOT, "prep")) as td:
        man_local = os.path.join(td, "nersc_inputs_manifest.json")
        subprocess.run(SCP + [f"{HOST}:{NERSC_P8}/inputs/inputs_manifest.json", man_local], check=True)
        man = C.read_json(man_local)
        man_sha = C.sha256_file(man_local)
    ent = {e["name"]: e for e in man["entries"] if e.get("kind", "file") == "file"}
    rec = dict(what=__doc__.split("\n\n")[0], nersc_manifest_sha256=man_sha, copies=[], extracts=[])
    copies = os.path.join(C.P8_ROOT, "nersc_copies")
    for name in a.names:
        e = ent[name]
        dst = os.path.join(copies, name)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not (os.path.exists(dst) and os.path.getsize(dst) == e["bytes"] and C.md5_file(dst) == e["md5"]):
            subprocess.run(SCP + [f"{HOST}:{NERSC_P8}/inputs/{name}", dst + ".part"], check=True)
            os.replace(dst + ".part", dst)
        got = C.md5_file(dst)
        link = os.path.join(C.INPUTS, name)
        os.makedirs(os.path.dirname(link), exist_ok=True)
        if os.path.islink(link) or os.path.exists(link):
            if os.path.realpath(link) != os.path.realpath(dst):
                raise SystemExit(f"{link} exists and is not the copy; not touching it")
        else:
            os.symlink(dst, link)
        rec["copies"].append(dict(name=name, nersc_source=e["source"], nersc_realpath=e.get("realpath"),
                                  bytes=e["bytes"], nersc_md5=e["md5"], md5=got, match=got == e["md5"], local=dst,
                                  link=link))
        C.log(f"{name}: {e['bytes']:,} B md5 {got} {'OK' if got == e['md5'] else 'MISMATCH'}")
    for k in a.extracts:
        sub, npz, js = EXTRACTS[k]
        d = os.path.join(C.P8_ROOT, sub)
        os.makedirs(d, exist_ok=True)
        for f in (js, npz):
            subprocess.run(SCP + [f"{HOST}:{NERSC_P8}/{sub}/{f}", os.path.join(d, f)], check=True)
        r = C.read_json(os.path.join(d, js))
        got = C.sha256_file(os.path.join(d, npz))
        rec["extracts"].append(dict(check=k, path=os.path.join(d, npz), sha256=got, record_sha256=r["sha256"],
                                    match=got == r["sha256"], record=os.path.join(d, js),
                                    record_file_sha256=C.sha256_file(os.path.join(d, js))))
        C.log(f"extract {k}: sha256 {got} {'OK' if got == r['sha256'] else 'MISMATCH'}")
    ok = all(c["match"] for c in rec["copies"]) and all(x["match"] for x in rec["extracts"])
    rec["ok"] = ok
    out = os.path.join(C.P8_ROOT, "prep", "mac_nersc_copies.json")
    h = C.write_json(rec, out)
    C.log(f"{'OK' if ok else 'PROBLEMS'}; wrote {out} sha256 {h}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
