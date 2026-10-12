#!/usr/bin/env python
"""P0.7 final assembly (Mac): long_slice.json = sample + ESM3 length/memory probes + encoding checks + scoring checks +
FP-ratio results (long_slice_core.json from analyze_long.py) + FP-location diagnostic (fp_location.json) + the D15
decision evidence and the recommended rule. Every number carries its source file (sources: path + md5).

D15 criterion (PLAN.md section 9, D15): keep the high-m CLAMP only if the long-query FP ratio at E <= 1 is in
[0.7, 1.4]; otherwise adopt the capped ln-m EXTRAPOLATION. Applied here to each rule (A5 real null, shipped table,
long queries 1,000-3,000 aa pooled); the same band decides whether S13's 'target length <= 1,000' can be relaxed.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os

SP = "/private/tmp/claude-501/-Users-joan-devel-rcsb-embedding-search/d9266e69-77ea-4121-a428-2e6f3bcc4ca2/scratchpad"
HERE = f"{SP}/p0/long"
NS = "/pscratch/sd/j/jsegura/embcomp/p0/long"
BAND = (0.7, 1.4)
XS = ("1.0", "0.1", "0.01", "0.001")
RULES = ("clamp", "extrap966", "extrap3000")


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def load(p, srcs, key):
    srcs[key] = {"file": p.replace(HERE, NS) if "/results/" in p or "/enc/results/" in p else p, "local": p,
                 "md5": md5(p)}
    return json.load(open(p))


def r3(v):
    return None if v is None else round(float(v), 3)


def cell(e):
    return dict(ratio=r3(e["ratio"]), fp=e["fp"], nq=e["nq"], fq=r3(e["fq"]), ci_cluster=[r3(c) for c in e["ci_cluster"]],
                ci_query=[r3(c) for c in e["ci_query"]], ci_poisson=[r3(c) for c in e["ci_poisson"]])


def table(e, xs=XS):
    return {x: cell(e[x]) for x in xs}


def main():
    S = {}
    core = load(f"{HERE}/long_slice_core.json", S, "analysis_core")
    sel = load(f"{HERE}/select_long.json", S, "selection")
    pr = load(f"{HERE}/results/probe_research_cuda.58975892.json", S, "probe_research_a100_80gb")
    pf80 = load(f"{HERE}/results/probe_foldmatch_cuda.58975892.json", S, "probe_foldmatch_a100_80gb")
    pf40 = load(f"{HERE}/results/probe_foldmatch_cuda.58976720.json", S, "probe_foldmatch_a100_40gb")
    pfc = load(f"{HERE}/results/probe_foldmatch_cpu.58975998.json", S, "probe_foldmatch_cpu_epyc7763")
    gzg = load(f"{HERE}/results/geom_zero_cuda.58976720.json", S, "geom_zero_a100_40gb")
    gzc = load(f"{HERE}/results/geom_zero_cpu.58976720.json", S, "geom_zero_cpu")
    ver = load(f"{HERE}/enc/results/verify_long.58975892.json", S, "verify_long")
    bench = load(f"{HERE}/results/bench.login_n0.json", S, "score_bench")
    chk = load(f"{HERE}/results/check.login_n0.json", S, "score_check")
    st = [load(f, S, "selftest_" + os.path.basename(f)) for f in sorted(glob.glob(f"{HERE}/results/selftest.5897*_n*.json"))]
    tim = [load(f, S, "timing_" + os.path.basename(f)) for f in sorted(glob.glob(f"{HERE}/results/score_timing.*.json"))]
    loc = load(f"{HERE}/fp_location.json", S, "fp_location")
    SL, D = core["slices"], core["diagnostics"]

    out = {"title": "P0.7 long-sequence calibration slice (decides D15)", "date": "2026-09-27",
           "system": core["meta"]["table"]["version"], "score": "S = S_f - S_r setting D (offset 1.0, gaps 5/0.5), "
           "Q90 over the 1,000 shipped UniRef50 panel members, E = exp(lnK(m) + ln m + ln N - lambda(m)(S - Q90))",
           "sources": S}

    # ------------------------------------------------------------------ data
    out["data"] = {
        "why_not_uniref50": "UniRef50 representatives (panel/pool, uniref50.fasta) carry no SCOP labels, so they cannot "
                            "give the A5 real null (no shared SCOP fold). The long proteins come from the SCOP-annotated "
                            "MMseqs2-benchmark files A5 itself was drawn from.",
        "long_queries": dict(source="mmseqs-db/db/targetannotation.fasta (SCOP-annotated UniProt)", **sel["long_q"]),
        "long_targets": dict(source="mmseqs-db/db/targetannotation.fasta", **sel["long_t"]),
        "long_reversed_decoys": dict(source="mmseqs-db/db/targetdb-gt1000.fasta, non-'annotated_' entries (stored "
                                            "reversed; A5's 25,000 decoys came from targetdb-lt1000)", **sel["long_r"]),
        "normal_partners": dict(queries="the 1,465 A5 benchmark queries (evalue/encode queries.L2m_B8_s0.npz; Q90 = "
                                        "panel/fit2/results/bench_q90_shipped.npy)",
                                targets="the 25,000 A5 annotated targets (all) + 5,000 of the 25,000 A5 decoys "
                                        "(default_rng(20260927))", columns=core["meta"]["columns"]),
        "exclusion_rules": "not in the benchmark (accession, sequence either direction, benchmark query as substring) "
                           "nor the shipped panel (UniRef50 representative accession, sequence); <= 1% non-standard; "
                           "no class-e sccs; <= 2 proteins per SCOP fold-set key per role and length bin",
        "selection_counts": sel["counts"],
        "null": "A5 rules (evalue/score/plan.py): real null = annotated target, no shared SCOP fold (a.b), not "
                "(both b.67-70 propellers) and neither side class e; reversed null = every decoy. Null mask reproduces "
                "results/labels.npz FPA exactly on the A5 matrix (34,948,695 pairs).",
        "ci": "95% percentile, 2,000 cluster-bootstrap resamples of queries: long queries by SCOP fold-set key "
              "(fold components are degenerate: largest has 2,841 of 3,000), A5 queries by fold components (A5 rule); "
              "plus query bootstrap and exact Poisson (Garwood)",
        "clusters": core["meta"]["clusters"],
        "q90": core["meta"]["q90"],
    }

    # ------------------------------------------------------------------ ESM3
    def ladder(p, key):
        return [{k: v for k, v in x.items() if k in ("L", "seconds", "peak_alloc_gib", "peak_reserved_gib", "maxrss_gib",
                                                     "error")} for x in p[key]]
    gpu_stats = []
    for s_ in ver.get("rank_stats", []):
        gpu_stats.append(s_)
    out["esm3"] = {
        "encoding_path": "research store path (evalue/encode/encode.py imported: SequenceOnlyESM3 bf16 weights + bf16 "
                         "autocast, geometric attention off, batches <= 16,000 tokens / 40M pairs) -> zcw50 -> M8_s0 "
                         "f_q / f_t; job 58975892 (4 x A100-80GB): 12.32 M residues embedded in 270.9 s (4 ranks), "
                         "peak 7.9 GiB allocated per rank, max |embedding| 15,168 (< fp16 max), 0 unk tokens, 0 "
                         "token-length mismatches",
        "verify_long": dict(benchmark_targets_reencoded=ver["a_benchmark_targets"],
                            long_rerun_bit_agreement=round(ver["b_long"]["rerun_bit_agreement"], 5),
                            long_rerun_items_identical=ver["b_long"]["rerun_items_identical"],
                            long_items=ver["b_long"]["n"],
                            reference_sdk_vs_production_code_bit_agreement=round(ver["b_long"]["reference_code_bit_agreement"], 5),
                            reference_per_item=[dict(len=p["len"], bit_agreement=round(p["reference_code_bit_agreement"], 4),
                                                     cos_min=round(p["reference_vs_batched_cos_min"], 5))
                                                for p in ver["b_long"]["per_item"]]),
        "max_length": {
            "research_batched_geom_off_a100_80gb": ladder(pr, "research_batched_geom_off"),
            "sdk_reference_geom_on_bf16_weights_a100_80gb": ladder(pr, "research_reference_sdk_geom_on"),
            "foldmatch_forward_and_sample_a100_80gb": ladder(pf80, "foldmatch_forward_and_sample"),
            "foldmatch_forward_and_sample_a100_40gb": ladder(pf40, "foldmatch_forward_and_sample"),
            "foldmatch_forward_and_sample_cpu_fp32_128threads": ladder(pfc, "foldmatch_forward_and_sample"),
            "foldmatch_geometric_attention_off": dict(bit_identical_gpu=gzg["compare"], bit_identical_cpu=gzc["compare"],
                                                      ladder_a100_40gb=gzg["ladder_geom_off"],
                                                      ladder_cpu=gzc["ladder_geom_off"]),
            "installed_foldmatch": dict(env="NERSC conda env 'foldmatch'", version="0.10.0", installed_from=
                                        "file:///global/u2/j/jsegura/devel/rcsb-embedding-model (non-editable, RECORD "
                                        "dated 2026-09-23 09:45; the checkout is now at 6c1fda0 on js/dev-3di-alignments)",
                                        loaders_py_md5="b7c5790ffa6f3ba85f9d8e98429e69e9 (= the checkout; differs from "
                                        "FM js/dev-esm-code-alignments only by the pinned ESM revision 636acb42)"),
            "weights_blob_nersc": "readlink: rcsb/rcsb-esm snapshots 636acb42 and c9dc4f1e and EvolutionaryScale "
                                  "esm3-sm-open-v1 all resolve esm3_sm_open_v1.pth to blob 5ead5a135c65...acd9 (P0.2)",
            "summary": "FoldMatch's ESM3 pass (forward_and_sample, geometric attention on, fp32 weights + bf16 autocast) "
                       "is the binding length limit: A100-40GB OK at 2,500 aa (32.3 GiB allocated), OOM at 2,750; "
                       "A100-80GB OK at 3,500 (58.1 GiB), OOM at 4,000; CPU fp32 peak RSS 26.6 / 50.5 / 84.0 / 127.6 GB "
                       "at 2,000 / 3,000 / 4,000 / 5,000 aa (9.2 / 13.6 / 17.0 / 21.6 s, 128 threads). The cost is "
                       "block 0's geometric attention, a (1, 256, L, L, 3) tensor. Without coordinates it is an exact "
                       "no-op: per_residue_embedding with it switched off is bit-identical (GPU 300/1,000/2,000 aa, CPU "
                       "300/1,000 aa), and the pass then runs 16,000 aa in 10.4 GiB (0.86 s) on A100-40GB and 3,000 aa "
                       "on CPU in 4.6 s within 12.2 GB RSS. The research batched path (geometric attention off) ran "
                       "16,000 aa in 6.35 GiB. max_pair_cells = 2^28 (16,384^2) is not the binding limit."},
    }

    # ------------------------------------------------------------------ scoring
    out["scoring"] = {
        "kernel": "research kernel swmulti.c R = 2 (throughput/build swkern.c 438d5a98 included verbatim), gcc -O3 "
                  "-march=znver3 -ffp-contract=off, driver score_long.py (same arithmetic as the A5 and panel matrices)",
        "matrices": core["meta"]["score_manifest"]["files"],
        "selftest_multi_equals_batch": [list(x.values()) for x in st],
        "bench_vs_A5_matrices": {k: bench[k] for k in ("pairs", "fwd_bit_equal", "rev_bit_equal", "fwd_max_rel",
                                                       "rev_max_rel")},
        "harness_float64_check": {k: chk[k] for k in ("n_pairs", "fwd_max_abs", "rev_max_abs", "fwd_max_rel",
                                                      "rev_max_rel")},
        "timing": [dict(tag=t["tag"], wall_s=round(t["wall_s"]), cells=t["cells"],
                        ns_per_cell_run_worker=round(t["ns_per_cell_run_worker"], 2)) for t in tim],
        "jobs": "NERSC 58975892 (GPU encode+verify+probes, debug), 58975998 (CPU probe, debug), 58976720 (A100-40GB "
                "probes, debug), 58977531/33/34/36 (4 x 1-node CPU scoring, regular, 31-36 min each); merge/bench/check "
                "on the login node (39 s)",
    }

    # ------------------------------------------------------------------ results
    res = {"A5_reference_normal_queries_x_normal_targets": {r: table(SL["A5_NQxN_real"][r]) for r in RULES},
           "A5_reference_heldout_union": {r: table(SL["A5_NQxN_real_heldout_union"][r]) for r in RULES},
           "A5_reference_reversed_5000": {r: table(SL["A5_NQxN_rev5000"][r]) for r in RULES},
           "A5_queries_m_gt_594_918": {r: table(SL["A5_NQxN_real_m_gt_594"][r]) for r in RULES}}
    res["long_queries_x_normal_targets"] = {}
    for r in RULES:
        res["long_queries_x_normal_targets"][r] = {
            "pooled_1000_3000": table(SL["LQxN_real_all_long_1000_3000"][r]),
            "baseline_300_1000_same_population": table(SL["LQxN_real_all_base_300_1000"][r]),
            "by_query_bin": {g: {x: dict(ratio=r3(SL["LQxN_real_all"][r][x]["bins"][g]["ratio"]),
                                         fp=SL["LQxN_real_all"][r][x]["bins"][g]["fp"],
                                         fq=r3(SL["LQxN_real_all"][r][x]["bins"][g]["fq"]),
                                         ci_cluster=[r3(c) for c in SL["LQxN_real_all"][r][x]["bins"][g]["ci_cluster"]])
                                 for x in XS} for g in SL["LQxN_real_all"][r]["1.0"]["bins"]},
            "heldout_tables_main_swap_pooled": {t: table(SL[f"LQxN_real_{t}_long_1000_3000"][r]) for t in ("main", "swap")},
            "reversed_null_pooled_1000_3000": table(SL["LQxR_rev_all_long_1000_3000"][r]),
            "reversed_null_baseline_300_1000": table(SL["LQxR_rev_all_base_300_1000"][r]),
            "reversed_null_by_query_bin": {g: {x: dict(ratio=r3(SL["LQxR_rev_all"][r][x]["bins"][g]["ratio"]),
                                                       fp=SL["LQxR_rev_all"][r][x]["bins"][g]["fp"]) for x in XS}
                                           for g in SL["LQxR_rev_all"][r]["1.0"]["bins"]}}
    res["normal_queries_x_long_targets"] = {}
    for r in RULES:
        res["normal_queries_x_long_targets"][r] = {
            "real_all_bins": table(SL["NQxLT_real"]["all_bins"][r]),
            "real_all_bins_heldout_union": table(SL["NQxLT_real"]["all_bins_heldout_union"][r]),
            "real_by_target_bin": {g: table(SL["NQxLT_real"][g][r]) for g in ("1000-1500", "1500-2000", "2000-2500", "2500-3000")},
            "reversed_all_bins": table(SL["NQxLR_rev"]["all_bins"][r]),
            "reversed_by_target_bin": {g: table(SL["NQxLR_rev"][g][r]) for g in ("1000-1500", "1500-2000", "2000-2500", "2500-3000")}}
    out["results"] = res
    out["diagnostics"] = {
        "curves": D["curves"],
        "coverage_strata_E1": D["LQxN_real_by_coverage"],
        "fp_alignment_location_E1": {k: v for k, v in loc.items() if k != "definition"},
        "fp_alignment_location_definition": loc["definition"],
        "gumbel_refit_by_long_query_bin": D["gumbel_refit_by_lq_bin"],
        "sensitivity_propeller_exception_b66_b70": D["sensitivity_propeller_b66"],
        "exploratory_lambda_only_extrapolation": D["exploratory_lambda_only_extrapolation"],
        "extrapolation_parameters": core["meta"]["extrapolation"],
        "top_offenders_clamp": "concentration among the 2,000 long queries (A5 null, clamp): the top 10 carry 27% / "
                               "55% / 69% and the top 20 (1%) 37% / 60% / 78% of the 4,006 / 752 / 360 real-null FPs "
                               "at E <= 1 / 0.01 / 1e-3; the largest at E <= 0.01 (annotated_Q9JM99, 1,054 aa, b.66.1.1 "
                               "+ g.64.1.1) has 289, against b.67-70 propeller targets (A5 excepts b.67-70 x b.67-70 "
                               "but not b.66, the 4-bladed propeller)",
        "text": core["text"],
    }

    # ------------------------------------------------------------------ decision
    ev = {}
    for r in RULES:
        v = SL["LQxN_real_all_long_1000_3000"][r]
        ev[r] = dict(ratio_E1=r3(v["1.0"]["ratio"]), ci_cluster_E1=[r3(c) for c in v["1.0"]["ci_cluster"]],
                     in_band_E1=BAND[0] <= v["1.0"]["ratio"] <= BAND[1],
                     ratio_E01=r3(v["0.1"]["ratio"]), ratio_E001=r3(v["0.01"]["ratio"]), ratio_E0001=r3(v["0.001"]["ratio"]))
    lt = SL["NQxLT_real"]["all_bins"]["clamp"]
    out["decision"] = {
        "criterion": "D15: keep the high-m clamp only if the long-query FP ratio at E <= 1 is in [0.7, 1.4], otherwise "
                     "adopt the capped ln-m extrapolation (A5 real null, shipped 'all' table, queries 1,000-3,000 aa)",
        "long_queries_by_rule": ev,
        "long_queries_verdict": "FAIL for every rule: clamp 2.00 [1.55, 2.56], extrap(cap 966) 2.25 [1.76, 2.84], "
                                "extrap(cap 3,000) 2.76 [2.17, 3.46] at E <= 1. The pre-registered fallback fails its "
                                "own criterion and is FURTHER from 1 at E <= 1 than the clamp; it only trims the tail "
                                "(E <= 0.01: 37.6 -> 29.6; E <= 1e-3: 180 -> 114), which stays an order of magnitude "
                                "anti-conservative.",
        "length_effect_is_real": "reversed (homology-free) null: long queries 0.33 / 1.30 / 6.5 at E <= 0.1 / 0.01 / "
                                 "1e-3 vs 0.12 / 0.50 / 1.0 for the same-population 300-1,000 aa baseline and 0.05 / "
                                 "0.07 / 0 for the A5 queries; the fraction of queries with >= 1 real-null FP at "
                                 "E <= 0.01 rises with length: 0.028 / 0.014 / 0.040 / 0.066 / 0.078 / 0.092 (300-600 "
                                 "... 2,500-3,000) vs 0.0075 for A5; per-bin censored-Gumbel refits give lambda 0.028-0.033 "
                                 "for 1,500-3,000 aa (clamp 0.0416; the ln-m extrapolation 0.029-0.033) with lnK 0.9-1.2 "
                                 "above the extrapolated line. Not annotation gaps: 98-99% of real-null FP alignments "
                                 "at E <= 1 lie >= 50% inside SCOP-annotated ranges on both sides (>= 98.9% on the long "
                                 "side) although annotation covers only 25-37% of those long queries.",
        "label_artefact": "A5's propeller exception covers b.67-70 only; extending it to b.66 removes the "
                          "same-population baseline excess (300-1,000 aa: 1.78 -> 1.01 at E <= 1) and lowers long "
                          "queries to 1.54 [1.19, 1.92] (clamp) / 2.29 (extrap 3,000) at E <= 1 and 7.9 / 4.35 at "
                          "E <= 0.01; A5 itself barely moves (0.91 -> 0.90). Without the top 1% of long queries the "
                          "clamp ratio is 0.99 / 1.26 / 1.01 at E <= 1 / 0.1 / 0.01 (b.66-extended null): the excess "
                          "is carried by a few multi-domain proteins.",
        "in_domain_595_966": "queries 600-1,000 aa (500, same population): clamp 1.05 [0.68, 1.48] / 1.38 / 2.60 / 2.0 "
                             "at E <= 1 / 0.1 / 0.01 / 1e-3 (extrap 1.15 at E <= 1); A5's 58 queries above 594.9 aa: "
                             "clamp 1.50 [0.33, 3.62], extrap 1.67. The clamp is supported up to the calibrated-domain "
                             "limit 966; extrapolation is not better there.",
        "long_targets": dict(
            ratio_E1=r3(lt["1.0"]["ratio"]), ci_cluster_E1=[r3(c) for c in lt["1.0"]["ci_cluster"]],
            heldout_union_E1=r3(SL["NQxLT_real"]["all_bins_heldout_union"]["clamp"]["1.0"]["ratio"]),
            ratio_E01=r3(lt["0.1"]["ratio"]), ratio_E001=r3(lt["0.01"]["ratio"]), ratio_E0001=r3(lt["0.001"]["ratio"]),
            by_bin_E1={g: r3(SL["NQxLT_real"][g]["clamp"]["1.0"]["ratio"]) for g in ("1000-1500", "1500-2000", "2000-2500", "2500-3000")},
            a5_same_queries_normal_targets=[0.91, 1.03, 2.59, 6.14],
            verdict="PASS: 1.00 [0.70, 1.33] at E <= 1 (held-out tables 1.03), every 500-aa bin 0.78-0.99, and the "
                    "tail (1.41 / 1.43 / 3.41 at E <= 0.1 / 0.01 / 1e-3) is no worse than the same queries against "
                    "normal targets (1.03 / 2.59 / 6.14). Caveat: long reversed decoys give 0.20 / 0.69 / 2.46 vs "
                    "0.04 / 0.05 / 0.07 for normal decoys (heavier tail, homology-free population)."),
        "recommended_D15_rule": [
            "E arithmetic: KEEP the shipped clamp above the last centre (594.918). Do NOT adopt the ln-m extrapolation: "
            "it fails the same pre-registered test and is worse at E <= 1 (2.76 vs 2.00). This amends D15's "
            "'otherwise adopt the capped extrapolation' branch; the registry numbers stay as fitted (no new system).",
            "Calibrated domain (S13): query length 21-966 unchanged (600-1,000 aa validated under the clamp); target "
            "length raised from <= 1,000 to <= 3,000 aa (validated here). Queries > 966 aa and targets > 3,000 aa "
            "get ecal = 0 plus the once-per-run warning; E is still reported.",
            "Documentation (P9): for queries of 1,000-3,000 aa, E is anti-conservative: about 2x at E <= 1 on an "
            "8.6 M-residue database and 7x / 38x / 180x at E <= 0.1 / 0.01 / 1e-3 there; equivalently, at E <= 1 on "
            "~8.6e7 / 8.6e8 / 8.6e9-residue databases. Long queries' hits need E far below the threshold or "
            "independent confirmation.",
            "A9 then passes through its 'ecal = 0 on every out-of-domain hit' branch for queries > 966, and through the "
            "FP-ratio branch for targets 1,000-3,000 aa.",
            "Memory part of D15: keep max_pair_cells = 2^28 (not binding). The binding limit is FoldMatch's ESM3 "
            "pass (A100-40GB OOM at 2,750 aa, A100-80GB at 4,000, CPU 50 GB RSS at 3,000). Recommend a generic "
            "FoldMatch fix (P1-type, own commit): switch off block 0's geometric attention for sequence-only inputs "
            "(bit-identical embeddings measured at 300-2,000 aa on GPU and 300-1,000 aa on CPU; 16,000 aa in "
            "10.4 GiB), or at least a pre-flight per-device length refusal instead of a mid-pass OOM.",
            "Future system (v2, not v1): lambda-only ln-m extrapolation (lnK clamped) was explored post hoc; it brings "
            "the reversed null in line (0.04 / 0.14 / 0.45 / 0.50) and the real null to 0.70 at E <= 1 but leaves "
            "15x / 46x at E <= 0.01 / 1e-3. Long-query calibration needs its own fitted bins on held-out long "
            "queries (new system name)."],
    }
    json.dump(out, open(f"{HERE}/long_slice.json", "w"), indent=1, default=float)
    print(json.dumps(out["decision"]["long_queries_by_rule"], indent=1))
    print(json.dumps(out["decision"]["long_targets"], indent=1))


if __name__ == "__main__":
    main()
