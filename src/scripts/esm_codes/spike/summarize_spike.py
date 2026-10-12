#!/usr/bin/env python
"""P0.4 spike summary: cross-build identity table (every verify_*.json) and the throughput / gate numbers.

  summarize_spike.py --verify DIR [DIR ...] --bench-epyc DIR --bench-m1 DIR --out spike_summary.json

Gate G-sim (EPYC 7763): full fwd+rev pair throughput of the new path >= 0.9 x the research path, single core,
Lq = 300, K = 1,000 chain-set candidates, each path at its own default block size (new 2^17 floats, research 2^20 =
score.py), plus the floor 2,500 pairs/s (P3 acceptance e). Also reported: best-vs-best over the swept block sizes and
against the buffer-reusing research variant.
"""
from __future__ import annotations

import argparse
import glob
import json
import os


def load(p):
    with open(p) as f:
        return json.load(f)


def cross_build(dirs):
    rows = {}
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "verify_*.json")) + glob.glob(os.path.join(d, "*.json"))):
            try:
                v = load(p)
            except Exception:
                continue
            if "golden" not in v:
                continue
            name = f"{v['host'].split('.')[0]}:{os.path.basename(v['build']['path'])}"
            s, f, g = v["F1_same_sim"], v["F1_full_path"], v["golden"]
            rows[name] = dict(
                isa=v["build"]["isa"], compiler=v["build"]["compiler"], sim_mode=v["build"].get("sim_mode"),
                sim_tile=[v["build"]["sim_tile_rows"], v["build"]["sim_tile_cols"]], machine=v["machine"],
                golden_sha256=g["sha256"], qp_sha256=g["sha256_qp"], sim_sha256=g["sha256_sim"],
                bench_sha256=v["invariance"]["sha256_bench_lq300_k1000"],
                invariance_ok=all(x for k, x in v["invariance"].items() if not k.startswith("sha")),
                golden_nan=g["nan"], golden_score_eq_tb=g["score_eq_tb"], golden_pairs=g["pairs"],
                f1_same_sim=[s["sf_bit_eq"], s["sr_bit_eq"], s["path_eq"], s["view_eq_copy"]],
                f1_same_sim_vs_this_host_research=[s["rhost_sf_bit_eq"], s["rhost_sr_bit_eq"], s["rhost_path_eq"]],
                f1_full=dict(path_eq=f["path_eq"], sf_bit_eq=f["sf_bit_eq"], sr_bit_eq=f["sr_bit_eq"],
                             max_rel_sf=f["max_rel_sf"], max_rel_sr=f["max_rel_sr"], qp_max_ulp=f["qp_max_ulp"],
                             qp_bit_eq=f["qp_bit_eq"], sim_bit_eq_frac=f["sim_bit_eq_cells"] / f["sim_cells"],
                             sim_max_abs=f["sim_max_abs"], rev_view_eq_qrev=f["rev_view_eq_qrev"],
                             rev_view_eq_alignrev=f["rev_view_eq_alignrev"], score_eq_tb=f["score_eq_tb"]),
                research_this_host_vs_fixture=v["F1_research_this_host"])
    goldens = sorted({r["golden_sha256"] for r in rows.values()})
    return dict(n_builds=len(rows), distinct_golden_sha256=goldens, all_identical=len(goldens) == 1,
                distinct_bench_sha256=sorted({r["bench_sha256"] for r in rows.values()}), builds=rows)


def gate(st, sweep=None, pool=None):
    out = {}
    for Lq, r in st["st"].items():
        g = dict(
            new_py_bf2_17=r["pair.new.py.bf131072"]["pairs_per_s"],
            new_c_bf2_17=r["pair.new.c.bf131072"]["pairs_per_s"],
            research_multi_bf2_20=r["pair.research.multi.bf1048576"]["pairs_per_s"],
            research_multi_bf2_17=r["pair.research.multi.bf131072"]["pairs_per_s"],
            research_multi_reuse_bf2_20=r.get("pair.research.multi_reuse.bf1048576", {}).get("pairs_per_s"),
            research_multi_reuse_bf2_17=r.get("pair.research.multi_reuse.bf131072", {}).get("pairs_per_s"),
            research_fwd_only_bf2_20=r["pair.research.fwd_only.bf1048576"]["pairs_per_s"],
            sim_new_ns_cell_bf2_17=r["sim.new.bf131072"]["ns_per_cell"],
            sim_research_decode_gemm_ns_cell_bf2_20=r["sim.research_decode_gemm.bf1048576"]["ns_per_cell"],
            sim_research_decode_gemm_ns_cell_bf2_17=r["sim.research_decode_gemm.bf131072"]["ns_per_cell"],
            sim_research_gemm_only_ns_cell_bf2_17=r["sim.research_gemm_only.bf131072"]["ns_per_cell"],
            dp_new_view_ns_cell_bf2_17=r["dp_fwd_rev.new_view.bf131072"]["ns_per_cell"],
            dp_research_copy_ns_cell_bf2_17=r["dp_fwd_rev.research_multi_copy.bf131072"]["ns_per_cell"])
        research_variants = [g[k] for k in g if k.startswith("research_multi") and g[k]]
        g["ratio_default_blocks"] = g["new_py_bf2_17"] / g["research_multi_bf2_20"]
        g["ratio_vs_best_research_st_blocks"] = g["new_py_bf2_17"] / max(research_variants)
        if sweep is not None and Lq in sweep["sweep"]:
            sw = sweep["sweep"][Lq]
            bn = max((v["pairs_per_s"], k) for k, v in sw.items() if k.startswith("new.py"))
            br = max((v["pairs_per_s"], k) for k, v in sw.items() if k.startswith("research"))
            g["sweep_best_new"] = dict(pairs_per_s=bn[0], variant=bn[1])
            g["sweep_best_research"] = dict(pairs_per_s=br[0], variant=br[1])
            g["ratio_best_vs_best_sweep"] = bn[0] / br[0]
        out[Lq] = g
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", nargs="+", required=True)
    ap.add_argument("--bench-epyc")
    ap.add_argument("--st-epyc", default="st_gcc_avx2hyb.json")
    ap.add_argument("--bench-m1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = dict(cross_build=cross_build(a.verify))
    for tag, d, stn in (("epyc", a.bench_epyc, a.st_epyc), ("m1", a.bench_m1, "st_clang_neon.json")):
        if not d:
            continue
        st = load(os.path.join(d, stn)) if os.path.exists(os.path.join(d, stn)) else None
        sw = load(os.path.join(d, "sweep1.json")) if os.path.exists(os.path.join(d, "sweep1.json")) else None
        if st:
            res[f"throughput_{tag}"] = dict(build=st["build"], host=st["host"], per_lq=gate(st, sw))
    if "throughput_epyc" in res:
        g = res["throughput_epyc"]["per_lq"]["300"]
        res["gate_G_sim"] = dict(
            criterion="EPYC 7763, 1 core, Lq 300, K 1000: new fwd+rev pairs/s >= 0.9 x research (and >= 2,500 pairs/s)",
            new_pairs_per_s=g["new_py_bf2_17"], research_pairs_per_s_default=g["research_multi_bf2_20"],
            ratio_default_blocks=g["ratio_default_blocks"],
            ratio_vs_best_research_variant=g["ratio_vs_best_research_st_blocks"],
            ratio_best_vs_best_sweep=g.get("ratio_best_vs_best_sweep"),
            passed=bool(min(g["ratio_default_blocks"], g["ratio_vs_best_research_st_blocks"],
                            g.get("ratio_best_vs_best_sweep") or 9) >= 0.9 and g["new_py_bf2_17"] >= 2500))
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    cb = res["cross_build"]
    print("builds", cb["n_builds"], "identical", cb["all_identical"], cb["distinct_golden_sha256"])
    if "gate_G_sim" in res:
        print(json.dumps(res["gate_G_sim"], indent=1))


if __name__ == "__main__":
    main()
