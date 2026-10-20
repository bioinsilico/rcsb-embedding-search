#!/usr/bin/env python
"""P8 A9 (long sequences): the P0.7 statistics, i.e. the P0.7 analysis code itself, run on a set of score matrices.
NERSC CPU node (about 20-40 GB, minutes).

The P0.7 analysis is p0/long/analyze_long.py (the staged archive copy, md5 1c3905206b5c887d678772ee2be3d759, equal to
RS src/scripts/esm_codes/long_slice/analyze_long.py). It is imported unmodified and its main() is run as written. Only
its module-level paths are re-pointed:
  OUTD    the score directory: index.npz, manifest.json (md5 of every matrix, checked by main()), and the float32
          matrices lq_pan_{fwd,rev}, lq_ben_{fwd,rev}, nq_long_{fwd,rev}
  BSCORE  the benchmark directory: out/index.npz, out/S_D_{fwd,rev}.npy [1,465 x 50,000]; main() uses only the
          lq_ben columns (ben_cols) of the latter
  Q90N    Q90 of the 1,465 benchmark queries
  HERE    the working directory: long_q/long_t/long_r.fasta (read for the labels), and main()'s outputs
          long_slice_core.json/.txt, long_slice_partial.json, fp_pairs_*.npy
  SHIP, QFA, TFA  the shipped tables (ship_final.json) and the benchmark FASTAs (staged inputs, unchanged)
Its 'import rlib' resolves to the staged research rlib.py (md5 6bf2586b..., checked by acc_common._import_research()).

--source research    the P0.7 research scores (staged p0/long/out, evalue/score, bench_q90_shipped.npy). The
                     output long_slice_core.json is compared number by number with the stored P0.7
                     p0/long/long_slice_core.json (md5 53f3e368...): this shows that the code reproduces P0.7.
--source production  the FoldMatch production scores written by a9_merge.py in the same layout (A9/prod_out,
                     A9/prod_bscore): the A9 numbers.
Output: A9/core_<source>/ (main()'s files) and A9/core_<source>.json (sources, md5s, wall time, comparison, provenance).
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acc_common as C  # noqa: E402  (first: it sets the BLAS thread variables before numpy loads)

AL_MD5 = "1c3905206b5c887d678772ee2be3d759"          # p0/long/analyze_long.py (archive p0/nersc_long/)
CORE_MD5 = "53f3e368436989ad34c06a5a117d3762"        # p0/long/long_slice_core.json (the stored P0.7 output)


def import_analyze_long():
    C._import_research()                              # rlib (md5-checked) importable as 'rlib'
    path = C.inp("p0/long/analyze_long.py")
    got = C.md5_file(path)
    assert got == AL_MD5, f"analyze_long.py md5 {got} != {AL_MD5}"
    spec = importlib.util.spec_from_file_location("analyze_long", path)
    AL = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(AL)
    assert os.path.realpath(AL.R.__file__) == os.path.realpath(C.research_file("rlib.py"))
    return AL, path


def numbers_only_compare(a, b):
    """C._cmp_tree, with string mismatches (file paths: Mac SP vs NERSC) listed apart from the numeric ones."""
    out = C._cmp_tree(a, b)
    strings = [m for m in out["mismatches"] if m.split(": ", 1)[-1].startswith("'")]
    other = [m for m in out["mismatches"] if m not in strings]
    return dict(numbers=out["numbers"], max_abs=out["max_abs"], max_rel=out["max_rel"],
                numeric_or_structural_mismatches=other, string_mismatches=len(strings),
                string_mismatch_examples=strings[:12])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("research", "production"), required=True)
    a = ap.parse_args()
    A9 = C.result_dir("A9")
    here = C.result_dir("A9", f"core_{a.source}")
    for f in ("long_q.fasta", "long_t.fasta", "long_r.fasta"):
        dst = os.path.join(here, f)
        if not os.path.lexists(dst):
            os.symlink(os.path.realpath(C.inp(f"p0/long/{f}")), dst)
    if a.source == "research":
        paths = dict(OUTD=C.inp("p0/long/out"), BSCORE=C.inp("embcomp/evalue/score"),
                     Q90N=C.inp("embcomp/panel/fit2/results/bench_q90_shipped.npy"))
    else:
        paths = dict(OUTD=os.path.join(A9, "prod_out"), BSCORE=os.path.join(A9, "prod_bscore"),
                     Q90N=os.path.join(A9, "prod_out", "bench_q90_prod.npy"))
    paths.update(HERE=here, SHIP=C.inp("embcomp/panel/fit2/results/ship_final.json"),
                 QFA=C.inp("calib3di/results/queries.fasta"), TFA=C.inp("calib3di/results/targets.fasta"))
    AL, al_path = import_analyze_long()
    for k, v in paths.items():
        assert hasattr(AL, k), k
        setattr(AL, k, v)
    inputs = {}
    for k in ("index.npz", "manifest.json", "lq_pan_fwd.npy", "lq_pan_rev.npy", "lq_ben_fwd.npy", "lq_ben_rev.npy",
              "nq_long_fwd.npy", "nq_long_rev.npy"):
        p = os.path.join(paths["OUTD"], k)
        inputs[f"OUTD/{k}"] = dict(path=os.path.realpath(p), md5=C.md5_file(p))
    for k in ("out/index.npz", "out/S_D_fwd.npy", "out/S_D_rev.npy"):
        p = os.path.join(paths["BSCORE"], k)
        inputs[f"BSCORE/{k}"] = dict(path=os.path.realpath(p), md5=C.md5_file(p))
    for k in ("Q90N", "SHIP", "QFA", "TFA"):
        inputs[k] = dict(path=os.path.realpath(paths[k]), md5=C.md5_file(paths[k]))
    C.log("analyze_long.main() on", a.source, {k: v["md5"] for k, v in inputs.items()})
    t0 = time.time()
    AL.main()
    wall = time.time() - t0
    core_path = os.path.join(here, "long_slice_core.json")
    rec = {"title": f"P8 A9: P0.7 analysis (analyze_long.main, unmodified) on the {a.source} scores",
           "analyze_long": dict(path=al_path, md5=C.md5_file(al_path)), "repointed": paths, "inputs": inputs,
           "wall_s": round(wall, 1), "output": dict(path=core_path, md5=C.md5_file(core_path),
                                                    txt_md5=C.md5_file(os.path.join(here, "long_slice_core.txt")))}
    if a.source == "research":
        stored = C.inp("p0/long/long_slice_core.json")
        assert C.md5_file(stored) == CORE_MD5
        got, ref = C.read_json(core_path), C.read_json(stored)
        cmp = numbers_only_compare(got, ref)
        txt_same = open(os.path.join(here, "long_slice_core.txt")).read() == open(
            C.inp("p0/long/long_slice_core.txt")).read()
        rec["compare_with_stored_p07"] = dict(stored=stored, stored_md5=CORE_MD5, **cmp,
                                              text_report_identical=txt_same)
        rec["reproduces_p07"] = (not cmp["numeric_or_structural_mismatches"]) and cmp["max_rel"] <= 1e-9
        C.log("compare with stored P0.7:", {k: v for k, v in rec["compare_with_stored_p07"].items()
                                            if k != "string_mismatch_examples"})
    rec["provenance"] = C.provenance(kernel=False)
    sha = C.write_json(rec, os.path.join(A9, f"core_{a.source}.json"))
    C.log("wrote", os.path.join(A9, f"core_{a.source}.json"), sha)


if __name__ == "__main__":
    main()
