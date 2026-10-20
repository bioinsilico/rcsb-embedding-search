#!/usr/bin/env python
"""Stage the P8 acceptance inputs (A1-A9) under <P8_ROOT>/inputs/ as symlinks and write inputs/inputs_manifest.json.

Every input gets a logical name = its path under inputs/, mirroring the research tree (embcomp/..., calib3di/results/,
p0/parity/..., p0/long/..., research_files/...), so a script asks for acc_common.inp("embcomp/.../ship_final.json")
on the Mac and on NERSC alike. For each name the catalogue lists candidate sources (NERSC research scratch first, then
the durable archive); the first that exists - and, when the plan's Appendix A gives an md5, matches it - is linked.

The manifest records per input: the source, its realpath, whether it is durable (/global/homes, /global/cfs) or in
purgeable scratch, size, mtime, md5 (sha256 for files <= 64 MiB), and the comparisons
  * appendix_a  - the md5 in plan Appendix A (PLAN.md, 'Research artefacts (SP-relative; md5 verified 2026-09-27)');
  * archive     - the md5 listed for the corresponding archive file in the archive's MD5SUMS files
                  (/global/homes/j/jsegura/esm_codes_archive/MD5SUMS, p0/parity/MD5SUMS, nersc_only/..., ...);
                  where the archive copy is a compacted version (an npz without the large arrays), the arrays the two
                  have in common are compared instead (archive.common_arrays_equal);
  * refs        - other recorded md5s / sha256s (evalue/score/out/manifest.json, f1_settings nersc_md5.txt, the
                  fit_final2.py MD5 table, the encode README, bundle_v1/SHA256SUMS, the FoldMatch registry,
                  the adopted A4_THRESHOLDS.json sha256).
Usage:  python stage_inputs.py [--dry-run] [--only PREFIX]     (re-running replaces the links and the manifest)
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acc_common as C  # noqa: E402  (first: it sets the BLAS thread variables before numpy loads)
import numpy as np  # noqa: E402

NS, ARCH, MAC = C.NS, C.ARCHIVE, C.MAC_ARTEFACTS
FMDB = "/pscratch/sd/j/jsegura/fm-db/calib3di/data"
RS_NERSC = "/global/homes/j/jsegura/devel/rcsb-embedding-search/src/scripts/esm_codes"
SHA_LIMIT = 64 << 20
ARCHIVE_MD5_MIRROR = os.path.join(C.P8_ROOT, "prep", "archive_md5sums")   # Mac only (copied from the archive)

# references recorded by the research (md5)
SCORE_MANIFEST = {     # evalue/score/out/manifest.json ('files', query/target FASTA md5)
    "S_D_fwd.npy": "1d015c63cb870524664442bd3784384e", "S_D_rev.npy": "70f3a0f75bf58243a9697da73d449cbd",
    "index.npz": "ff0f64fc242625b6c7dfce5082a57abb", "queries.fasta": "cf6bb37859a48052875b4a2912080b1b",
    "targets.fasta": "a34ca17a20cfba6994a33012e064f4a5"}
F1_NERSC_MD5 = {       # artefacts/embcomp/evalue_f1_settings_results/nersc_md5.txt
    "dom/M8_s0.val.npz": "67a6ee1d9d9825978c8cd7a25869c8a9", "dom/M8_s0.test.npz": "f4225dd075ef82288748749dcbb016ef",
    "chains/M8_s0_D.npz": "9931e3c82bcde55dc0bde12fd75fcad5", "check.json": "18b1667a158de37de18c2b41f79a0825"}
FIT_FINAL2_MD5 = {     # panel/fit2/scripts/fit_final2.py MD5 (inputs of the shipped-panel Q90 / table fit)
    "S_D_fwd.npy": "361fbdf30e81613b902eb3f094e0aa00", "S_D_rev.npy": "a07c4f1b38bc113fa36860207c0351ab",
    "index.npz": "4f4ac7887a8f3c7233bd493c10646d6c", "rlib.py": "6bf2586b154391a9cdb9e9c5b2a280bc"}
ENCODE_README = {"targets.L2m_B8_s0.npz": "cfee06cd"}   # evalue/score/README.txt: 'targets md5 cfee06cd...' (prefix)


def E(name, nersc=(), mac=(), appendix_a=None, archive=None, refs=None, checks=(), note=""):
    return dict(name=name, nersc=list(nersc), mac=list(mac), appendix_a=appendix_a, archive=archive,
                refs=dict(refs or {}), checks=list(checks), note=note)


def art(rel, md5, checks, ap_rel=None, arch_rel=None, mac_rel=None, nersc_extra=(), note=""):
    """Appendix A artefact (SP/embcomp-relative)."""
    arch_rel = arch_rel or f"artefacts/embcomp/{rel}"
    return E(f"embcomp/{rel}", nersc=[f"{NS}/{rel}", *nersc_extra, f"{ARCH}/{arch_rel}"],
             mac=[f"{MAC}/{mac_rel or ('embcomp/' + rel)}"], appendix_a=md5, archive=arch_rel, checks=checks,
             note=note or f"plan Appendix A: {ap_rel or rel}")


def ns(rel, checks, refs=None, archive="auto", note="", mac=()):
    """A research file in the NERSC embcomp tree."""
    return E(f"embcomp/{rel}", nersc=[f"{NS}/{rel}"], mac=list(mac), archive=archive, refs=refs, checks=checks,
             note=note)


def rf(rel, checks, note=""):
    """A research protocol / report file from the archive research_files/."""
    return E(f"research_files/{rel}", nersc=[f"{ARCH}/research_files/{rel}"], archive=f"research_files/{rel}",
             checks=checks, note=note)


def catalogue():
    A = []
    # ------------------------------------------------------------------ Appendix A artefacts (shared by A1-A9)
    A += [
        art("sw_mlp/models/L2m_B8_s0.pt", "85fdd985087d6813e9994274d4fca6ab", ["A2", "A4"], note="encoders M8_s0"),
        art("sw_mlp/models/L2m_B8_s0.json", "19e58f8956cf24404def09aac56e00b5", ["A2", "A4"]),
        art("sw_mlp/sw_common.py", "b8dab0a03907c9b7ed67663d96c55561", ["A2", "A4"]),
        art("clean_pairs/zcw50_clean.npz", "4c7692ed66e599aca906b41ec7c13f04", ["A2", "A4"]),
        art("panel/fit2/results/ship_final.json", "c51735f962c04bd8c441b9625c4cf30a", ["A3", "A4", "A5", "A8", "A9"]),
        art("panel/fit2/results/shipped_panel_codes.npz", "15cc1c7b2d1dd488a5f9b742365ccb13", ["A3", "A4", "A5"],
            note="plan Appendix A (codes-bytes md5 885dfc88590e0b270addcd4b61c3a7a7)"),
        art("panel/fit2/results/shipped_panel.ids", "68463819ee52018dccb6b4fab3d5467a", ["A3", "A4", "A5"]),
        art("panel/fit2/results/shipped_panel.fasta", "b7e41050760592b65d4ce026d7174590", ["A4"]),
        art("panel/fit2/results/shipped_panel.tsv", "b79790e58644d3e78b1fa094d0d13dd0", ["A4"]),
        art("throughput/build/swkern.c", "438d5a9811153f61e66cee845f1ed037", ["A1", "A3"]),
        art("throughput/build/swfast.py", "80a813553d42801fce6b77462c5e1d0a", ["A1", "A3"]),
        art("evalue/score/swmulti.c", "b12fbc2c03638659440b12282a4d8670", ["A3"]),
        art("harness/harness.py", "05f387cb400cc85dbacaef771a275eeb", ["A1"], note="plan Appendix A: oracle copy"),
        art("evalue/encode/out/queries.L2m_B8_s0.npz", "e821497f59dda06a3899c18dce37286c", ["A3", "A4", "A7"]),
        art("throughput/build/data/L2m_B8_s0.val.npz", "5aebc14942665748313f22ca737ca26d", ["A1"]),
        art("throughput/build/data/L2m_B8_s0.test.npz", "e570bc26e1a142183aaf85e9a7a57df7", ["A1"]),
        art("throughput/bench/data/L2m_B8_s0.chains.npz", "5519866044ab0f348c74b9319125acd2", ["A1", "A4", "A7"],
            nersc_extra=[f"{NS}/sw_mlp/enc/L2m_B8_s0.chains.npz"]),
        art("panel/fit/data/cath_set.npz", "78b6c3fcf64d1ffcc9aa98d81c04c81c", ["A6"]),
        art("panel/amend/AMENDMENT.txt", "589708cc06a710d8007c7161190e175c", ["A5"]),
        art("panel/pool/PREREGISTRATION.txt", "90e19e019ec6fd688d21bc58fcd5ae4c", ["A5"]),
        art("panel/score2/enc/results/verify_panel.58937318.json", "02b918bdc8635ec2fb9c0e262a2c4e79", ["A4"]),
        art("throughput/bench/results/summary.txt", "b8f85dde9201d13d59c2461dbe4a231a", ["A7"]),
    ]
    for sub in ("dom/M8_s0.val.npz", "dom/M8_s0.test.npz", "chains/M8_s0_D.npz", "check.json"):
        A.append(E(f"embcomp/evalue/f1_settings/results/{sub}",
                   nersc=[f"{NS}/evalue/f1_settings/results/{sub}",
                          f"{ARCH}/artefacts/embcomp/evalue_f1_settings_results/{sub}"],
                   mac=[f"{MAC}/embcomp/evalue_f1_settings_results/{sub}"],
                   archive=f"artefacts/embcomp/evalue_f1_settings_results/{sub}",
                   refs={"f1_settings nersc_md5.txt": F1_NERSC_MD5[sub]}, checks=["A1", "A4"],
                   note="plan Appendix A: stored setting-D alignments (md5 in nersc_md5.txt)"))
    for f, md5 in (("esm3_needleman_wunsch.py", "7ec19b7e5ff89dc7b44d2512aed3997e"),
                   ("esm3_sequence_embeddings.py", "7e51a13b9c11fddc532b1ff072a9446c")):
        A.append(E(f"oracle/{f}", nersc=[f"{ARCH}/artefacts/{f}"], mac=[f"{MAC}/{f}"], appendix_a=md5,
                   archive=f"artefacts/{f}", checks=["A1"],
                   note="plan Appendix A: oracle source / research extraction"))
    for f, md5 in (("fixture_candidate.npz", "d724b769bcf26178cc20a3bf157a64bf"),
                   ("probe_report.json", "c35ed1589908808e31afe3dfd562f776")):
        A.append(E(f"inventory_probe/{f}", nersc=[f"{ARCH}/artefacts/inventory_probe/{f}"],
                   mac=[f"{MAC}/inventory_probe/{f}"], appendix_a=md5, archive=f"artefacts/inventory_probe/{f}",
                   checks=["A1"], note="plan Appendix A: F1 candidate / cross-ISA probe"))

    # ------------------------------------------------------------------ benchmark (A3, A5, A8; labels; self-test)
    for f in ("queries.fasta", "targets.fasta"):
        A.append(E(f"calib3di/results/{f}", nersc=[f"{FMDB}/{f}"], archive=None,
                   refs={"evalue/score/out/manifest.json": SCORE_MANIFEST[f]}, checks=["A3", "A5", "A8", "A9", "self"],
                   note="the benchmark FASTAs (SP/calib3di/results/ in the research; NERSC fm-db/calib3di/data/)"))
    A += [
        ns("evalue/score/out/S_D_fwd.npy", ["A3", "A5", "self"], {"evalue/score/out/manifest.json": SCORE_MANIFEST[
            "S_D_fwd.npy"]}, note="research S_f, setting D [1465, 50000] float32"),
        ns("evalue/score/out/S_D_rev.npy", ["A3", "A5", "self"], {"evalue/score/out/manifest.json": SCORE_MANIFEST[
            "S_D_rev.npy"]}, note="research S_r, setting D"),
        ns("evalue/score/out/index.npz", ["A3", "A5", "A8", "self"],
           {"evalue/score/out/manifest.json": SCORE_MANIFEST["index.npz"]}),
        ns("evalue/score/out/manifest.json", ["A3", "self"]),
        ns("evalue/score/results/labels.npz", ["A3", "A5", "A8", "self"],
           note="TP / FP / FPA packbits (plan.py); content checked by acc_common selftest"),
        ns("evalue/score/results/plan_info.json", ["self"]),
        ns("evalue/score/results/check.58885229.json", ["A3"]),
        ns("evalue/encode/out/targets.L2m_B8_s0.npz", ["A3", "A5"],
           {"evalue/score/README.txt (md5 prefix)": ENCODE_README["targets.L2m_B8_s0.npz"]},
           note="research target encodings, 50,000 targets incl. reversed decoys"),
        ns("panel/fit2/results/bench_q90_shipped.npy", ["A3", "A5", "self"],
           note="research Q90 of the 1,465 queries (fit_final2.py); content checked by acc_common selftest"),
        ns("panel/score2/out/S_D_fwd.npy", ["A3", "self"], {"fit_final2.py MD5": FIT_FINAL2_MD5["S_D_fwd.npy"]},
           note="research benchmark-query x panel-candidate S_f (for Q90)"),
        ns("panel/score2/out/S_D_rev.npy", ["A3", "self"], {"fit_final2.py MD5": FIT_FINAL2_MD5["S_D_rev.npy"]}),
        ns("panel/score2/out/index.npz", ["A3", "self"], {"fit_final2.py MD5": FIT_FINAL2_MD5["index.npz"]}),
    ]
    # ------------------------------------------------------------------ A1 harness data (pairs, US-align references)
    for f in ("cath_val_pairs.csv", "cath_val_remote50.csv", "cath_val_unrelated.csv", "cath_test_remote50.csv",
              "cath_test_unrelated.csv", "usalign_refs.json", "usalign_refs_test.json"):
        A.append(ns(f, ["A1"], note="harness pair lists / US-align reference alignments (harness.SETS)"))
    A += [
        ns("throughput/build/data/pairs_L2m_B8_s0.npz", ["A1"]),
        ns("chains/data/pairs.json", ["A1", "A4"], note="chain_eval ChainSet (732 chain pairs)"),
        ns("chains/data/chains_meta.json", ["A1", "A4"]),
        ns("chains/data/chain_emb.npz", ["A1", "A2", "A4"], note="chain-set ESM3 rows (enc_check chain items)"),
        ns("chains/chain_eval.py", ["A1", "A4"]),
        ns("evalue/f1_settings/f1_eval.py", ["A1"]),
        ns("evalue/f1_settings/README.txt", ["A1"]),
        ns("sw_eval/nersc/eval_all.py", ["A1"]),
        ns("reference/xharness.py", ["A1"], note="sw_eval harness driver (imported by eval_all.py / f1_eval.py)"),
        ns("clean_pairs/clean_io.py", ["A1"]),
    ]
    # ------------------------------------------------------------------ A2 encoder parity (enc_check)
    A += [
        ns("sw_mlp/enc_check/L2m_B8_s0.npz", ["A2"], note="enc_check codes and q of the 50 items (580,608 bits)"),
        ns("sw_mlp/enc_check/check_items.json", ["A2"]),
        ns("val.npz", ["A2"], note="research ESM3 rows, CATH val domains (enc_check val items)"),
        ns("test.npz", ["A2"], note="research ESM3 rows, CATH test domains (enc_check test items)"),
    ]
    # ------------------------------------------------------------------ A4 / A5: P0.5 parity work
    par = f"{NS}/p0/parity"
    for sub in ("inputs", "enc", "scores", "chainf1"):
        for p in sorted(glob.glob(f"{par}/{sub}/*")):
            rel = os.path.relpath(p, par)
            A.append(E(f"p0/parity/{rel}", nersc=[p], archive=f"p0/parity/{rel}",
                       checks=["A4", "A5", "A8"] if sub == "inputs" else ["A4", "A5"],
                       note="P0.5 parity work (" + {"inputs": "input FASTAs / sets.json", "enc": "P0.5 encodings; "
                            "gpu.a5q / gpu.a5t = the P0.5(d) A5 re-embedding",
                            "scores": "P0.5 scores", "chainf1": "P0.5 chain F1"}[sub] + ")"))
    for f in ("esm3_parity.json", "esm3_parity.py", "A4_THRESHOLDS.json", "FM_COMMIT"):
        A.append(E(f"p0/parity/{f}", nersc=[f"{par}/{f}", f"{ARCH}/p0/parity/{f}"], archive=f"p0/parity/{f}",
                   checks=["A4"], note="P0.5 outputs"))
    # ------------------------------------------------------------------ A9: P0.7 long slice
    lon = f"{NS}/p0/long"
    pats = ["*.fasta", "*.tsv", "*.json", "*.txt", "*.py", "out/*", "enc/out/*", "results/*"]
    for pat in pats:
        for p in sorted(glob.glob(f"{lon}/{pat}")):
            if os.path.isdir(p):
                continue
            rel = os.path.relpath(p, lon)
            A.append(E(f"p0/long/{rel}", nersc=[p], archive=f"p0/nersc_long/{rel}", checks=["A9"],
                       note="P0.7 long-sequence slice (FASTAs, research encodings and scores, results, scripts)"))
    # ------------------------------------------------------------------ A6 (optional): CATH null
    A += [
        ns("panel/fit2/results/cath_validate.json", ["A6"]),
        ns("panel/fit2/results/cath_validate.txt", ["A6"]),
        ns("panel/fit2/results/cath_q90_shipped_local.npy", ["A6"]),
        ns("panel/fit2/results/bench_q90_shipped_local.npy", ["A6"]),
        ns("panel/fit2/results/cath_scop_map.json", ["A6"]),
        ns("panel/fit2/results/panel_local_bench.json", ["A6"]),
        ns("panel/fit2/results/panel_local_cath.json", ["A6"]),
        ns("panel/score2/out/cath_strat_index.npz", ["A6"]),
        ns("panel/score2/out/cath_strat_manifest.json", ["A6"]),
        ns("panel/score2/out/cath_strat_S_D_fwd.npy", ["A6"]),
        ns("panel/score2/out/cath_strat_S_D_rev.npy", ["A6"]),
        ns("panel/score2/out/cath_cath_S_D_fwd.npy", ["A6"]),
        ns("panel/score2/out/cath_cath_S_D_rev.npy", ["A6"]),
    ]
    # ------------------------------------------------------------------ research protocol files (cited)
    for k, (rel, _md5) in C.RESEARCH_SOURCES.items():
        A.append(rf(rel, ["self", "A3", "A5", "A6", "A8", "A9"], note=f"research protocol source ({k})"))
    for rel in ("embcomp/evalue/reconcile/ship.py", "embcomp/evalue/score/README.txt", "embcomp/evalue/score/score.py",
                "embcomp/evalue/score/summarize.py", "embcomp/evalue/score/results/summary.txt",
                "embcomp/evalue/encode/README.txt", "embcomp/evalue/encode/encode.py",
                "embcomp/panel/fit2/scripts/cath_validate.py", "embcomp/panel/fit2/scripts/panel_local.py",
                "embcomp/panel/fit2/results/ship_final.txt", "embcomp/panel/fit2/results/panel_files.json",
                "embcomp/evalue/f1_settings/analyze_f1.py"):
        if os.path.exists(f"{ARCH}/research_files/{rel}") or not C.IS_NERSC:
            A.append(rf(rel, ["A1", "A3", "A5", "A6"], note="research protocol / report"))
    # ------------------------------------------------------------------ thresholds, bundle, phase reports
    A.append(E("rs/A4_THRESHOLDS.json", nersc=[f"{RS_NERSC}/A4_THRESHOLDS.json"],
               mac=[f"{C.RS_ESM_CODES}/A4_THRESHOLDS.json"], archive=None,
               refs={"plan D4 / P0.5 adopted sha256": "sha256:" + C.A4_THRESHOLDS_SHA256}, checks=["A4"],
               note="RS commit a18f829 (committed before A4 runs)"))
    for f in ("bundle.json", "esm_codes_v1_encoders.pt", "esm_codes_v1_scoring.npz", "README.md", "LICENSE"):
        A.append(E(f"bundle_v1/{f}", nersc=[f"{ARCH}/bundle_v1/{f}"], archive=f"bundle_v1/{f}", checks=["A2", "A4"],
                   note="rcsb/foldmatch-esm-codes@3a14f46b72f5 (archive copy; FoldMatch loads its own sha256-pinned "
                        "HF copy)"))
    for ph in ("p6/P6_REPORT.md", "p7/P7_REPORT.md", "p7/PLAN_after_P7.md"):
        A.append(E(f"reports/{os.path.basename(ph)}", nersc=[f"{ARCH}/{ph}"],
                   mac=[f"{C.SANDBOX}/{os.path.basename(ph)}"] if "REPORT" in ph else [], archive=ph,
                   checks=["A7", "A8"], note="phase report (P7 table for A7; notes for P8)"))
    return A


# =====================================================================================================================
def archive_md5_map():
    """archive-relative path -> md5, from every MD5SUMS* file in the archive (each relative to its own directory)."""
    out = {}
    base_dir = ARCH if os.path.isdir(ARCH) else ARCHIVE_MD5_MIRROR     # Mac: a mirror of the MD5SUMS text files
    if not os.path.isdir(base_dir):
        return out
    for root, _dirs, files in os.walk(base_dir):
        for f in files:
            if not f.startswith("MD5SUMS"):
                continue
            base = os.path.relpath(root, base_dir)
            for line in open(os.path.join(root, f), errors="replace"):
                parts = line.rstrip("\n").split(None, 1)
                if len(parts) != 2 or len(parts[0]) != 32:
                    continue
                p = parts[1].lstrip("*")
                p = p[2:] if p.startswith("./") else p
                rel = os.path.normpath(os.path.join(base, p)) if base != "." else os.path.normpath(p)
                out.setdefault(rel, parts[0])
    return out


def bundle_refs():
    """bundle_v1/SHA256SUMS and the FoldMatch registry's bundle_files (sha256)."""
    refs = {}
    p = f"{ARCH}/bundle_v1/SHA256SUMS"
    if os.path.exists(p):
        for line in open(p):
            parts = line.split()
            if len(parts) == 2:
                refs.setdefault(parts[1].lstrip("*").lstrip("./"), {})["bundle_v1/SHA256SUMS"] = "sha256:" + parts[0]
    try:
        from foldmatch.search import scoring_system as SSY
        s = SSY.get("esm-codes-v1")
        for name, sha in s.bundle_files:
            refs.setdefault(name, {})[f"FoldMatch registry esm-codes-v1@{s.bundle_revision[:12]}"] = "sha256:" + sha
    except Exception as e:  # noqa: BLE001
        refs["_error"] = repr(e)
    return refs


def npz_common_equal(a, b):
    """For two npz files: the arrays they share, compared for exact equality (keys only in one listed)."""
    try:
        za, zb = np.load(a, mmap_mode="r"), np.load(b, mmap_mode="r")
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}
    common = sorted(set(za.files) & set(zb.files))
    eq = {k: bool(np.array_equal(za[k], zb[k])) for k in common}
    return {"common_arrays_equal": all(eq.values()), "compared": eq,
            "only_source": sorted(set(za.files) - set(zb.files)),
            "only_archive": sorted(set(zb.files) - set(za.files))}


def digest(p, cache):
    rp = os.path.realpath(p)
    if rp not in cache:
        st = os.stat(rp)
        cache[rp] = dict(bytes=st.st_size, mtime=time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(st.st_mtime)),
                         md5=C.md5_file(rp), sha256=C.sha256_file(rp) if st.st_size <= SHA_LIMIT else None)
    return cache[rp]


def ref_match(expected, d):
    if expected.startswith("sha256:"):
        return None if d["sha256"] is None else d["sha256"] == expected[7:]
    return d["md5"].startswith(expected) if len(expected) < 32 else d["md5"] == expected


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", default=None, help="stage only names with this prefix (the manifest is still complete "
                                                 "only when run without --only)")
    a = ap.parse_args(argv)
    host_key = "nersc" if C.IS_NERSC else "mac"
    os.makedirs(C.INPUTS, exist_ok=True)
    amap = archive_md5_map()
    brefs = bundle_refs()
    cache, entries, problems = {}, [], []
    t0 = time.time()
    cat = catalogue()
    names = [e["name"] for e in cat]
    dup = sorted({n for n in names if names.count(n) > 1})
    assert not dup, f"duplicate names: {dup}"
    for e in cat:
        if a.only and not e["name"].startswith(a.only):
            continue
        cands = e[host_key]
        if not cands:
            continue
        tried, chosen = [], None
        for c in cands:
            if not os.path.isfile(c):
                tried.append(dict(path=c, exists=False))
                continue
            d = digest(c, cache)
            ok = e["appendix_a"] is None or d["md5"] == e["appendix_a"]
            tried.append(dict(path=c, exists=True, md5=d["md5"], appendix_a_match=ok if e["appendix_a"] else None))
            if ok:
                chosen = c
                break
        if chosen is None:
            existing = [t["path"] for t in tried if t["exists"]]
            if not existing:
                if C.IS_NERSC:
                    problems.append(f"{e['name']}: no source exists ({cands})")
                entries.append(dict(name=e["name"], kind="missing", candidates=tried, checks=e["checks"],
                                    note=e["note"]))
                continue
            chosen = existing[0]
            problems.append(f"{e['name']}: no source matches Appendix A md5 {e['appendix_a']}; staged {chosen}")
        d = digest(chosen, cache)
        rp = os.path.realpath(chosen)
        rec = dict(name=e["name"], kind="file", source=chosen, realpath=rp,
                   durable=not rp.startswith(("/pscratch/", "/private/tmp/")), **d,
                   checks=e["checks"], note=e["note"])
        if len(tried) > 1:
            rec["candidates"] = tried
        if e["appendix_a"]:
            rec["appendix_a"] = dict(expected=e["appendix_a"], match=d["md5"] == e["appendix_a"])
        # archive comparison
        arel = e["archive"]
        if arel == "auto":
            arel = next((r for r in (f"research_files/embcomp/{e['name'][8:]}", f"artefacts/{e['name']}",
                                     f"nersc_only/{e['name'][8:]}") if r in amap), None)
        if arel:
            am = amap.get(arel)
            ar = dict(relpath=arel, md5=am, exists=os.path.exists(f"{ARCH}/{arel}") if C.IS_NERSC else None)
            ar["match"] = None if am is None else (am == d["md5"])
            if ar["match"] is False and arel.endswith(".npz") and C.IS_NERSC and os.path.exists(f"{ARCH}/{arel}"):
                ar.update(npz_common_equal(rp, f"{ARCH}/{arel}"))
                ar["note"] = "archive copy differs (compacted npz): arrays in common compared"
            elif am is None and C.IS_NERSC and os.path.isfile(f"{ARCH}/{arel}"):
                ad = digest(f"{ARCH}/{arel}", cache)
                ar.update(md5=ad["md5"], match=ad["md5"] == d["md5"], md5_source="computed (archive file not listed "
                                                                                  "in any MD5SUMS)")
            elif am is None:
                ar["note"] = "not listed in any archive MD5SUMS"
            rec["archive"] = ar
        else:
            rec["archive"] = dict(relpath=None, match=None, note="no archive copy")
        refs = dict(e["refs"])
        base = os.path.basename(e["name"])
        if e["name"].startswith("bundle_v1/") and base in brefs:
            refs.update(brefs[base])
        if refs:
            rec["refs"] = [dict(label=k, expected=v, match=ref_match(v, d)) for k, v in refs.items()]
        for k in ("appendix_a",):
            if k in rec and rec[k]["match"] is False:
                problems.append(f"{e['name']}: Appendix A md5 mismatch")
        if rec["archive"].get("match") is False and not rec["archive"].get("common_arrays_equal"):
            problems.append(f"{e['name']}: archive md5 {rec['archive']['md5']} != {d['md5']}"
                            + ("" if "common_arrays_equal" not in rec["archive"] else " and common arrays differ"))
        for r in rec.get("refs", []):
            if r["match"] is False:
                problems.append(f"{e['name']}: {r['label']} {r['expected']} != staged")
        link = os.path.join(C.INPUTS, e["name"])
        if not a.dry_run:
            os.makedirs(os.path.dirname(link), exist_ok=True)
            if os.path.islink(link) or os.path.exists(link):
                os.remove(link)
            os.symlink(chosen, link)
        entries.append(rec)
        common = rec["archive"].get("common_arrays_equal")
        rmatch = ",".join(str(r["match"]) for r in rec.get("refs", []))
        C.log(f"{e['name']}  {d['bytes']:>12,d}  {d['md5']}  "
              f"A:{rec.get('appendix_a', {}).get('match', '-')} arch:{rec['archive'].get('match')}"
              + (f" common:{common}" if common is not None else "") + (f" refs:{rmatch}" if rmatch else ""))
    by_check = {}
    for r in entries:
        for c in r.get("checks", []):
            by_check.setdefault(c, 0)
            by_check[c] += 1
    summary = dict(entries=len(entries), files=sum(r["kind"] == "file" for r in entries),
                   missing=sum(r["kind"] == "missing" for r in entries),
                   bytes=sum(r.get("bytes", 0) for r in entries if r["kind"] == "file"),
                   appendix_a_checked=sum("appendix_a" in r for r in entries),
                   appendix_a_match=sum(r.get("appendix_a", {}).get("match") is True for r in entries),
                   archive_listed=sum(r.get("archive", {}).get("md5") is not None for r in entries),
                   archive_match=sum(r.get("archive", {}).get("match") is True for r in entries),
                   archive_compacted_common_equal=sum(r.get("archive", {}).get("common_arrays_equal") is True
                                                      for r in entries),
                   refs_checked=sum(len(r.get("refs", [])) for r in entries),
                   refs_match=sum(x["match"] is True for r in entries for x in r.get("refs", [])),
                   non_durable=sum(r["kind"] == "file" and not r["durable"] for r in entries),
                   by_check=dict(sorted(by_check.items())), problems=problems, wall_s=round(time.time() - t0, 1))
    man = dict(what="P8 acceptance inputs: symlinks under this directory; md5s compared with plan Appendix A, the "
                    "archive MD5SUMS and other recorded references", inputs_dir=C.INPUTS, host=host_key,
               stage_script=os.path.abspath(__file__), stage_script_md5=C.md5_file(os.path.abspath(__file__)),
               only=a.only, summary=summary, entries=entries,
               provenance=C.provenance(kernel=True, require_commit=C.EXPECTED_FM_COMMIT))
    out = C.INPUTS_MANIFEST if not a.dry_run else os.path.join(C.INPUTS, "inputs_manifest.dryrun.json")
    h = C.write_json(man, out)
    lines = [f"{'name':78s} {'bytes':>14s} {'md5':32s} AppA  archive  refs  durable"]
    for r in entries:
        if r["kind"] != "file":
            lines.append(f"{r['name']:78s} MISSING")
            continue
        arch = r["archive"]
        am = "-" if arch.get("match") is None else ("ok" if arch["match"] else
                                                   ("compact=" if arch.get("common_arrays_equal") else "DIFF"))
        aa = "-" if "appendix_a" not in r else ("ok" if r["appendix_a"]["match"] else "DIFF")
        rf_ = "-" if "refs" not in r else ("ok" if all(x["match"] for x in r["refs"]) else "DIFF")
        lines.append(f"{r['name']:78s} {r['bytes']:>14,d} {r['md5']} {aa:5s} {am:8s} {rf_:5s} "
                     f"{'yes' if r['durable'] else 'scratch'}")
    open(out.replace(".json", ".txt"), "w").write("\n".join(lines) + "\n\nsummary: " + str(
        {k: v for k, v in summary.items() if k != "problems"}) + "\nproblems:\n" + "\n".join(problems) + "\n")
    C.log(f"wrote {out} sha256 {h}")
    C.log("summary", {k: v for k, v in summary.items() if k != "problems"})
    for p in problems:
        C.log("PROBLEM", p)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
