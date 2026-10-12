#!/usr/bin/env python
"""blas_div_summary.py -- one JSON table of every blas_div run of one host tag against its golden fixed_c run.

Reads, for every run dir RUNS/<bdir>/<work>/<label> (search-*/ and a1-*):
  meta.json, blas_probe.json, repeat.json, verify.json            (configuration, determinism, research agreement)
  ROOT/compare/<work>__<tag>__<ref label>__VS__<tag>__<label>.json (blas_div.py compare against the golden run)
and the envelope JSONs given with --envelope (blas_div_envelope.py), and writes --out with
  runs[]: configuration + headline divergence against golden + repeat + verify, per work set
  envelopes{}: the classes and the build-to-build envelope of each work set.

usage: blas_div_summary.py --runs ROOT/runs/epyc --tag epyc7763 --ref-label fixed_c-qpc-t1 --out OUT.json
                           [--envelope ENV.json ...]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import blas_div_envelope as EN  # noqa: E402
import blas_div as BD  # noqa: E402


def _g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def search_row(c):
    s = c["search"]
    row = dict(pairs=s["pairs"], S_f_frac_bit_equal=s["S_f"]["frac_bit_equal"], S_f_max_abs=s["S_f"]["abs"]["max"],
               S_f_max_rel_floor1=s["S_f"]["rel_floor1"]["max"], S_f_max_ulp=_g(s, "S_f", "ulp", "max"),
               S_r_frac_bit_equal=s["S_r"]["frac_bit_equal"], S_r_max_rel_floor1=s["S_r"]["rel_floor1"]["max"],
               S_max_abs=s["S"]["abs"]["max"], S_abs_p99=s["S"]["abs"]["p99"], S_frac_equal=s["S"]["abs"]["frac_zero"])
    if s.get("panel"):
        row.update(Q90_max_abs=s["Q90"]["max"], Q90_frac_equal=s["Q90"]["frac_equal"],
                   lnE_max_abs=s["lnE"]["abs"]["max"], lnE_abs_p99=s["lnE"]["abs"]["p99"],
                   lnE_hits_E_le_1=dict(n=s["lnE"]["E_ref<=1"]["n"], p50=s["lnE"]["E_ref<=1"].get("p50"),
                                        p95=s["lnE"]["E_ref<=1"].get("p95"), max=s["lnE"]["E_ref<=1"].get("max")),
                   E_flips={t: s["E_threshold_flips"][t]["flips"] for t in ("0.001", "0.01", "1", "10")},
                   E_ref_hits={t: s["E_threshold_flips"][t]["ref_hits"] for t in ("0.001", "0.01", "1", "10")},
                   E_flips_by_label_at_1=_g(s, "E_threshold_flips_by_label", "1"),
                   hitlist_E_le_1e3_set_diff=s["hit_lists"]["0.001"]["set_diff"],
                   hitlist_E_le_1_set_diff=s["hit_lists"]["1"]["set_diff"])
    row.update(top10_order_diff=s["ranking"]["top10"]["order_diff"], top10_set_diff=s["ranking"]["top10"]["set_diff"],
               top100_order_diff=s["ranking"]["top100"]["order_diff"])
    sm = c.get("sim")
    if sm and sm.get("n_blocks"):
        row["sim"] = dict(blocks=sm["n_blocks"], cells=sm["cells"]["n"], frac_bit_equal=sm["cells"]["frac_bit_equal"],
                          max_abs=sm["cells"]["abs"]["max"], max_ulp=_g(sm, "cells", "ulp", "max"),
                          ulp_frac_le1=_g(sm, "cells", "ulp", "frac_le1"),
                          frac_correctly_rounded_ref_test=[_g(sm, "oracle", "A_frac_correctly_rounded"),
                                                           _g(sm, "oracle", "B_frac_correctly_rounded")],
                          max_abs_err_vs_f64_ref_test=[_g(sm, "oracle", "A_abs_err", "max"),
                                                       _g(sm, "oracle", "B_abs_err", "max")])
    return row


def a1_row(c):
    a = c["a1"]
    t, dp = a["totals"], a["differing_paths"]
    return dict(pairs=t["n"], S_f_frac_bit_equal=a["S_f"]["frac_bit_equal"], S_f_max_abs=a["S_f"]["abs"]["max"],
                S_f_max_rel_floor1=a["S_f"]["rel_floor1"]["max"], S_r_frac_bit_equal=a["S_r"]["frac_bit_equal"],
                paths_identical=t["path_identical"], aligned_pairs_identical=t["aligned_pairs_identical"],
                test_eq_stored_research=t["B_eq_stored"], ref_eq_stored_research=t["A_eq_stored"],
                f1_diff_pairs=t["f1_diff_pairs"], max_abs_dF1=t["max_abs_dF1"],
                differing_paths=dp["n"], oracle_evaluated=dp["oracle_evaluated"],
                margin_lt_1e_4=dp["margin_lt_1e_4"], margin_ge_1e_4_unexplained=dp["margin_ge_1e_4_unexplained"],
                oracle_eq_ref=dp["oracle_eq_A"], oracle_eq_test=dp["oracle_eq_B"],
                margins=dp.get("margins"), n_cols_diff=dp.get("n_cols_diff"),
                per_set_f1_ref_test={k: [v.get("f1A"), v.get("f1B")] for k, v in a["per_set"].items() if v.get("f1A") is not None})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="ROOT/runs/<dir> holding <bdir>/<work>/<label>")
    ap.add_argument("--tag", required=True, help="host tag of the runs (compare file names)")
    ap.add_argument("--ref-label", default="fixed_c-qpc-t1")
    ap.add_argument("--compare-dir", default=os.path.join(BD.ROOT, "compare"))
    ap.add_argument("--envelope", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for d in sorted(glob.glob(os.path.join(a.runs, "*", "*", "*"))):
        work = os.path.basename(os.path.dirname(d))
        fin = "a1.npz" if work.startswith("a1-") else "search.npz"
        if not os.path.exists(os.path.join(d, fin)):
            continue
        desc = EN.describe(d)
        meta = json.load(open(os.path.join(d, "meta.json")))
        r = dict(work=work, label=desc["label"], dir=d, config=desc,
                 timing=dict(wall_s=_g(meta, "timing", "wall_s"), workers=_g(meta, "timing", "workers"),
                             t_sim_s=_g(meta, "timing", "t_sim_s"), t_score_s=_g(meta, "timing", "t_score_s")))
        rp = os.path.join(d, "repeat.json")
        if os.path.exists(rp):
            rj = json.load(open(rp))
            r["repeat"] = {v: {k: s[k] for k in ("blocks", "blocks_diff", "cells_diff", "max_ulp", "sf_diff", "sr_diff",
                                                 "paths_diff")} for v, s in rj["stats"].items()}
        vp = os.path.join(d, "verify.json")
        if os.path.exists(vp):
            v = json.load(open(vp))
            r["verify"] = dict(
                fixed_c_vs_spike_list_driver_pass=_g(v, "fixed_c_vs_spike_list_driver", "pass"),
                fixed_c_vs_spike_direct_pass=_g(v, "fixed_c_vs_spike_direct", "pass"),
                vs_research_rows=_g(v, "vs_research_stored_scores", "rows"),
                vs_research_max_rel_floor1=_g(v, "vs_research_stored_scores", "max_rel_floor1"),
                vs_research_pass_rel_1e_5=_g(v, "vs_research_stored_scores", "pass_rel_1e-5"),
                vs_research_S_f_frac_bit_equal=_g(v, "vs_research_stored_scores", "S_f", "frac_bit_equal"),
                vs_research_Q90_max_abs=_g(v, "vs_research_stored_scores", "Q90", "max"),
                a1_eq_stored=_g(v, "a1_vs_stored_setting_D_alignments", "eq_stored"),
                a1_n=_g(v, "a1_vs_stored_setting_D_alignments", "n"),
                a1_f1=_g(v, "a1_vs_stored_setting_D_alignments", "f1"),
                a1_f1_stored=_g(v, "a1_vs_stored_setting_D_alignments", "f1_stored"))
        if desc["label"] != a.ref_label:
            cp = os.path.join(a.compare_dir, f"{work}__{a.tag}__{a.ref_label}__VS__{a.tag}__{desc['label']}.json")
            if os.path.exists(cp):
                c = json.load(open(cp))
                r["vs_golden_file"] = cp
                if "search" in c:
                    r["vs_golden"] = search_row(c)
                elif "a1" in c:
                    r["vs_golden"] = a1_row(c)
            else:
                r["vs_golden_file"] = f"MISSING {cp}"
        rows.append(r)
    env = {}
    for p in a.envelope:
        if os.path.exists(p):
            e = json.load(open(p))
            env[os.path.basename(p)] = dict(file=p, classes=e["classes"], class_hash=e["result"]["class_hash"],
                                            envelope=e["envelope"], class_pairs=e["result"]["class_pairs"],
                                            per_class=e["result"].get("per_class"))
    out = dict(what="blas_div summary: every run vs the golden fixed_c run of the same work set, + build envelopes",
               created=time.strftime("%Y-%m-%d %H:%M:%S"), runs_dir=a.runs, tag=a.tag, ref_label=a.ref_label,
               runs=rows, envelopes=env)
    BD.jdump(out, a.out)
    for r in rows:
        g = r.get("vs_golden") or {}
        print(f"{r['work']:20s} {r['label']:52s} Sf_bit_eq={g.get('S_f_frac_bit_equal')} "
              f"Sf_max_rel={g.get('S_f_max_rel_floor1')} lnE_max={g.get('lnE_max_abs')} "
              f"paths_id={g.get('paths_identical')}")
    print("->", a.out)


if __name__ == "__main__":
    main()
