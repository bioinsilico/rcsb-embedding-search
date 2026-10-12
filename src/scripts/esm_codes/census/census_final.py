#!/usr/bin/env python
"""P0.6 census, step 5 (local): assemble degenerate_census.json from the step outputs and derive the D14 evidence.

Inputs (TRACK dir): nersc/{seq_census.json, check.json, enc_meta.json, jobs.json}, census_report.json.
Output: TRACK/degenerate_census.json (the caller copies it to NERSC /pscratch/sd/j/jsegura/embcomp/p0/census/).
"""
from __future__ import annotations

import hashlib
import json
import os

SP = "/private/tmp/claude-501/-Users-joan-devel-rcsb-embedding-search/d9266e69-77ea-4121-a428-2e6f3bcc4ca2/scratchpad"
TRACK = os.environ.get("TRACK", f"{SP}/p0/census")
IN = f"{TRACK}/nersc"
RS = "/Users/joan/devel/rcsb-embedding-search/src/scripts/esm_codes/census"
XS = ("1", "0.1", "0.01", "0.001")


def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()


def rnd(v, k=3):
    return None if v is None else (round(v, k) if isinstance(v, float) else v)


def fp_row(e):
    if e is None:
        return None
    return {"hits": e["hits"], "expected": rnd(e["expected"], 4), "ratio": rnd(e["ratio"]),
            "boot_ci": [rnd(v) for v in e["boot_ci"]], "poisson_ci": [rnd(v) for v in e["poisson_ci"]]}


def table(block, tags=("filtered", "raw"), rev=False):
    out = {}
    for g, r in block.items():
        row = {"n_items": r["n_items"]}
        for x in XS:
            c = {t: fp_row(r.get(f"{t}@{x}")) for t in tags if f"{t}@{x}" in r}
            if rev and f"rev@{x}" in r:
                rv = r[f"rev@{x}"]
                c["reversed_target_null"] = {"hits": rv["hits"], "ratio": rnd(rv["ratio"]),
                                             "poisson_ci": [rnd(v) for v in rv["poisson_ci"]]}
            row[f"E<={x}"] = c
        out[g] = row
    return out


def pick_ratio(block, g, x, tag):
    e = block[g][f"{tag}@{x}"]
    return {"hits": e["hits"], "expected": rnd(e["expected"], 4), "ratio": rnd(e["ratio"]),
            "poisson_95ci": [rnd(v) for v in e["poisson_ci"]]}


def main():
    seqc = json.load(open(f"{IN}/seq_census.json"))
    chk = json.load(open(f"{IN}/check.json"))
    enc = json.load(open(f"{IN}/enc_meta.json"))
    jobs = json.load(open(f"{IN}/jobs.json"))
    rep = json.load(open(f"{TRACK}/census_report.json"))
    base = rep["baseline_benchmark"]
    A, T, TR = rep["fp_as_query"]["research"], rep["fp_as_target"]["research"], rep["fp_as_target_reversed_decoys"]["research"]
    degenerate_groups = ["both:Llt21", "both:allX", "both:homopoly", "both:histag", "code:distinct<10", "code:distinct<5",
                         "seq:allX|homopoly|histag (any source, sequence-defined)"]

    def zero_everywhere(g):
        return all(A[g][f"filtered@{x}"]["hits"] == 0 and A[g][f"rev@{x}"]["hits"] == 0 and T[g][f"filtered@{x}"]["hits"] == 0
                   and TR[g][f"raw@{x}"]["hits"] == 0 for x in XS)

    def zero_on_decoy_nulls(g):
        return all(A[g][f"rev@{x}"]["hits"] == 0 and TR[g][f"raw@{x}"]["hits"] == 0 for x in XS)

    d14 = {
        "question": "PLAN D14: keep S7, report the alignment, ecal=0 when the target has < 2 distinct codes or either "
                    "side is > 1% non-standard; escalate to 'report no alignment' if P0.6 shows an FP ratio > 2x at "
                    "E <= 1e-3 for such items.",
        "recommended_policy": [
            "ADOPT the D14 default unchanged: keep the S7 degenerate rule (a row/column with std < 1e-6 contributes 0), "
            "report the alignment, and set ecal=0 when the target has < 2 distinct codes or either side has > 1% "
            "non-standard residues.",
            "DO NOT adopt the 'report no alignment' escalation: no degenerate or low-information category shows an FP "
            "excess (see evidence); the degenerate categories are strongly conservative (0 hits at every threshold).",
            "ADD one rule for fully degenerate QUERIES (every panel S exactly 0, hence Q90 = 0: n = 1 or all q rows "
            "identical): report no hits (or E = +inf, ecal=0). Their E-value is K(m) m N for EVERY target (no "
            "information): 12.7 at the benchmark N_ann, and E <= 1 for every target when N < 6.76e5 residues. Such "
            "queries are already ecal=0 by the S13 length rule when L < 21, but --max-evalue could still pass them on "
            "tiny databases.",
            "In practice '< 2 distinct codes' is exactly 'target length 1': 0 of 12,672 real ESM3 items (L 2-1,000) and "
            "0 of their 12,672 reversed sequences have < 2 distinct codes; all 21 single-residue sequences have 1. PDB "
            "seqres has no L = 1 protein chain and UniRef50 none below 11, so the rule fires only on user input.",
        ],
        "evidence": {
            "escalation_trigger_met": False,
            "degenerate_categories_zero_hits_on_all_four_nulls_all_thresholds": {g: zero_everywhere(g)
                                                                                 for g in degenerate_groups},
            "degenerate_categories_zero_hits_on_both_homology_free_nulls": {g: zero_on_decoy_nulls(g)
                                                                            for g in degenerate_groups},
            "his_rich_target_hits_are_homologs": "the His-tag category's real-null target hits (39 at E <= 1, 1 at "
                                                 "E <= 1e-3; not removed by the BLOSUM filter) come from 6 long UniRef50 "
                                                 "His-rich proteins, each hitting a single SCOP family: 26 queries of "
                                                 "g.37.1 (C2H2 zinc finger) for one 440-aa protein, d.58.17/18 "
                                                 "(heavy-metal-associated) for a 297-aa CxxC protein, a.39.1 (EF-hand) "
                                                 "for two, d.17.1 and a.24.19 for one each; 0 hits on the reversed-"
                                                 "sequence decoys of the same items (19.4 expected at E <= 1).",
            "degenerate_power_note": "0 hits at E <= 1 where the expected null count is large (queries: L<21 4,000 "
                                     "expected; targets: all-X 17.6, His-rich 16.6 on the real null, 19-20 on the "
                                     "decoy null) bounds their FP ratio at E <= 1 far below 1 (Poisson 95% upper "
                                     "bounds in the tables). At E <= 1e-3 the expected counts are <= 4 (queries) and "
                                     "<= 0.02 (targets), so a 2x excess at 1e-3 cannot be bounded directly for these "
                                     "small categories; with 0 hits even at E <= 1 there is no tail to extrapolate.",
            "as_query_L<21": {x: pick_ratio(A, "both:Llt21", x, "filtered") for x in ("1", "0.001")},
            "as_query_allX_homopoly_histag": {x: pick_ratio(A, "seq:allX|homopoly|histag (any source, sequence-defined)",
                                                            x, "filtered") for x in ("1", "0.001")},
            "his_rich_query_hits_are_homologs": "the 253 filtered E <= 1e-3 query hits of the His-tag category come from "
                                                "3 long UniRef50 His-rich proteins: 248 from one 870-aa protein vs 246 "
                                                "SCOP f.38.1 (MFS transporter) targets, 24 from a 454-aa protein vs "
                                                "f.13.1 (GPCR) targets, 2 vs d.17.1; 0 hits on the reversed-target null; "
                                                "the 15 PDB His-tag peptides have 0 hits.",
            "nonstd_homology_free_nulls": {
                "as_query_vs_reversed_benchmark_targets(N_rev)": {
                    "nonstd>1%": {x: A["both:nonstd"][f"rev@{x}"]["hits"] for x in XS},
                    "nonstd>1%_ratio": {x: rnd(A["both:nonstd"][f"rev@{x}"]["ratio"]) for x in XS},
                    "random_real_proteins_ratio": {x: rnd(A["both:random"][f"rev@{x}"]["ratio"]) for x in XS},
                    "benchmark_queries_ratio(heldout_union)": {x: rnd(base["heldout_union"][str(float(x))]["rev_ratio"])
                                                               for x in XS}},
                "as_target_reversed_item_decoys(N_rev)": {
                    "nonstd>1%": {x: pick_ratio(TR, "both:nonstd", x, "raw") for x in XS},
                    "random_real_proteins": {x: pick_ratio(TR, "both:random", x, "raw") for x in XS},
                    "benchmark_reversed_targets_ratio(insample_all)": {
                        x: rnd(base["insample_all"][str(float(x))]["rev_ratio"]) for x in XS},
                    "note": "the single nonstd decoy hit at E <= 1e-3 is a 661-aa Ser/Thr low-complexity repeat protein "
                            "(reversed) vs a 770-aa b.69.4.1 query: a low-complexity/repeat effect, not a "
                            "non-standard-residue effect (1 hit vs 0.142 expected, Poisson 95% CI 0.18-39)."},
                "annotated_target_counts_are_homology_dominated": "on the real (annotated) targets, nonstd and random "
                    "real proteins both show large hit excesses (queries, filtered, E <= 1: nonstd 15.5 x, random 43.2 x; "
                    "targets: 15.3 x vs 26.5 x): true remote homologs of the SCOP benchmark that BLOSUM62 does not "
                    "detect, not false positives. nonstd is never above random."},
            "benchmark_real_null_baseline": {"heldout_union": {x: {"fp": base["heldout_union"][str(float(x))]["real_fp"],
                                                                   "ratio": rnd(base["heldout_union"][str(float(x))]["real_ratio"])}
                                                               for x in XS},
                                             "note": "the shipped table itself is ~5.5x optimistic at E <= 1e-3 on the "
                                                     "benchmark real null (8 FPs vs 1.465 expected) - any 2x criterion "
                                                     "at 1e-3 should be read against this baseline."},
            "S7_vs_research": "the S7 variant gives identical hit counts to the research kernel in every group, both "
                              "sides, all thresholds (degenerate rows in 25 of 18,564,480 target-side pairs, all L = 2 "
                              "targets; degenerate columns only for the 11 L = 2 queries).",
            "S7_needed_for_D2": "research kernel on degenerate blocks amplifies float noise by 1/(0 + 1e-8): synthetic "
                                "m = 1 target S_f = 1.0 (Mac NEON) vs 0.0 (EPYC AVX2), identical-row query S_f = 0.0 vs "
                                "1.25; S_f - S_r = 0 exactly in both; the S7 variant matches a float64 reference to "
                                "<= 1.7e-6 on both. Keeping S7 is required for the bit-identical cross-ISA contract.",
        },
    }
    single = rep["single_residue_real_esm3"]
    out = {
        "track": "P0.6 degenerate and low-information census (decides D14)",
        "date": "2026-09-27",
        "d14": d14,
        "meta": {
            "nersc_workdir": "/pscratch/sd/j/jsegura/embcomp/p0/census/",
            "local_trackdir": TRACK,
            "scripts": {f: md5(f"{RS}/{f}") for f in ("census_seq.py", "census_encode.py", "census_swk.c",
                                                         "census_score.py", "census_report.py", "census_final.py")},
            "scripts_note": "RS/src/scripts/esm_codes/census/ (new, uncommitted). The NERSC jobs ran earlier revisions "
                            "of census_score.py / census_encode.py that differ only in the PREDECODE option, the qmerge "
                            "E <= 10 reduction and the fixed encode merge; the scoring and encoding code paths are "
                            "unchanged.",
            "kernel": "census_swk.c #includes the research swkern.c verbatim (md5 438d5a9811153f61e66cee845f1ed037), "
                      "gcc 14.3.0 -O3 -march=znver3 -ffp-contract=off (libcensus.so md5 7d4e20808db6a27e6c6f45a216a3d0bb)",
            "pdb_seqres": {"url": "https://files.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz",
                           "bytes": 67128902, "expected_bytes": 67128902, "mtime": "2026-09-19 03:10:51 PDT",
                           "md5": "cc289979afd9958b095e4389ad4eb055", "downloaded_utc": "2026-09-27 22:57",
                           "path": "/pscratch/sd/j/jsegura/embcomp/p0/census/pdb_seqres.txt.gz"},
            "uniref50": {"path": "/pscratch/sd/j/jsegura/uniref50/uniref50.fasta", "bytes": 16956977214,
                         "release": "2026_03 (from uniref50.fasta.gz md5 0492e3cf4093276ae4319ba24d52514f, VERIFIED_OK)"},
            "research_inputs_md5": {
                "evalue/encode/encode.py": "d22a8e80b7d84039f67ded9eb5121ee4",
                "sw_mlp/models/L2m_B8_s0.pt": "85fdd985087d6813e9994274d4fca6ab",
                "sw_mlp/sw_common.py": "b8dab0a03907c9b7ed67663d96c55561",
                "clean_pairs/zcw50_clean.npz": "4c7692ed66e599aca906b41ec7c13f04",
                "src/scripts/esm3_sequence_embeddings.py (NERSC repo, imported)": "7e51a13b9c11fddc532b1ff072a9446c",
                "evalue/encode/out/queries.L2m_B8_s0.npz": "e821497f59dda06a3899c18dce37286c",
                "evalue/encode/out/targets.L2m_B8_s0.npz": "cfee06cd6f7c46080c4dc9082ad97eb8",
                "panel/fit2/results/shipped_panel_codes.npz": "15cc1c7b2d1dd488a5f9b742365ccb13",
                "panel/fit2/results/ship_final.json": "c51735f962c04bd8c441b9625c4cf30a",
                "panel/fit2/results/bench_q90_shipped.npy": "4969b813673919047431f7550c1fd82e",
                "evalue/score/results/labels.npz": "a8b36754e22753879b6bd5ed22f500f8",
                "evalue/score/out/index.npz": "ff0f64fc242625b6c7dfce5082a57abb"},
            "foldmatch": "not imported by this track (research pipeline only). NERSC 'foldmatch' conda env (used as the "
                         "numpy/scipy runtime): foldmatch 0.10.0, non-editable install from "
                         "file:///global/u2/j/jsegura/devel/rcsb-embedding-model (no commit recorded in direct_url.json; "
                         "that checkout is at 6c1fda0aaedcdf07c8f1128ba88f219700ca56f4, branch js/dev-3di-alignments); "
                         "biotite 1.7.1. Local report: /Users/joan/app/miniconda3/envs/foldmatch (biotite 1.7.1).",
            "scoring": "setting D (offset 1.0, gap open 5.0, extend 0.5), S = S_f - S_r (S_r = row-reversed query "
                       "block), Q90 = np.quantile over the 1,000 shipped panel members (0.9, linear), table "
                       "D_fmr/pnl_uniref50/M3t999 'all', lnE = lnK(m) + ln m + ln N - lambda(m)(S - Q90), m = query "
                       "length (clamped outside the centres).",
        },
        "sequence_level": {src: {k: v for k, v in seqc["sources"][src].items() if not k.startswith("examples")}
                           for src in seqc["sources"]},
        "sequence_level_definitions": seqc["meta"]["definition"],
        "sequence_level_examples": {src: {k: v for k, v in seqc["sources"][src].items() if k.startswith("examples")}
                                    for src in seqc["sources"]},
        "item_selection": {"cap_per_source_category_unique_sequences": seqc["meta"]["cap_per_category"],
                           "capped": {"pdb": ["Llt21 (9,009 -> 2,000)", "nonstd (4,391 with L <= 1,000 -> 2,000)"],
                                      "uniref50": ["Llt21 (25,373 -> 2,000)", "nonstd (96,124 with L <= 1,000 -> 2,000)"]},
                           "random_per_source": seqc["meta"]["n_random"],
                           "max_length_encoded": seqc["meta"]["lmax_encode"],
                           "excluded_L_gt_1000": {"pdb": {"allX": 9, "nonstd": 111},
                                                  "uniref50": {"homopoly": 2, "histag": 8, "nonstd": 5033}},
                           "L1_absent_in_sources": "PDB seqres protein min length 2, UniRef50 min 11 -> supplementary "
                                                   "set of 21 single-residue sequences (20 standard + X)",
                           "items": seqc["items"]},
        "encoding": {k: v for k, v in enc.items() if k != "rank_stats"} | {"rank_stats": enc.get("rank_stats")},
        "kernel_checks": chk,
        "code_level": {"groups": rep["code_level"],
                       "distinct_code_histogram_all_items_capped_at_20": rep[
                           "distinct_code_histogram_all_items(capped at 20)"],
                       "prevalence_lt2_distinct_codes": rep["prevalence_lt2_distinct_codes"],
                       "reversed_items_distinct_codes_lt2": rep["fp_as_target_reversed_decoys"][
                           "reversed_items_distinct_codes_lt2"]},
        "fully_degenerate_check": {"single_residue_real_esm3": single,
                                   "census_items": rep["fully_degenerate"],
                                   "synthetic": [c for c in chk["synthetic"]]},
        "fp_check": {
            "definition": "A5 protocol, research encodings: flagged/random items as QUERIES vs the 50,000 benchmark "
                          "targets (hits among the 25,000 annotated ones with N = N_ann; reversed targets with N = "
                          "N_rev = homology-free null) and as TARGETS for the 1,465 benchmark queries (expected = n_q x "
                          "N_items / N_ref; N_ref = N_ann on the real items, N_rev on the reversed-sequence item "
                          "decoys). 'filtered' removes sequence-detectable homologs (BLOSUM62 11/1 local, KA E <= 1e-3 "
                          "over m x N_ann). Ratios = hits / expected; CIs: item bootstrap (2,000) and exact Poisson.",
            "N_ann": rep["N_ann"], "N_rev": rep["N_rev"], "n_benchmark_queries": rep["n_benchmark_queries"],
            "homology_filter": rep["homology_filter"],
            "baseline_benchmark_real_null": base,
            "as_query": {"research": table(A, rev=True), "S7": table(rep["fp_as_query"]["S7"], rev=True)},
            "as_target": {"research": table(T), "S7": table(rep["fp_as_target"]["S7"])},
            "as_target_reversed_decoys": {"research": table(TR, tags=("raw",)),
                                          "S7": table(rep["fp_as_target_reversed_decoys"]["S7"], tags=("raw",)),
                                          "degenerate_pairs": rep["fp_as_target_reversed_decoys"]["degenerate_pairs"]},
            "sticky_targets": rep["sticky_targets"],
            "query_side_null_shape": rep["query_side_null_shape"]},
        "jobs": jobs,
        "inputs_md5": rep["inputs_md5"],
    }
    p = f"{TRACK}/degenerate_census.json"
    json.dump(out, open(p, "w"), indent=1)
    print(p, md5(p), os.path.getsize(p))


if __name__ == "__main__":
    main()
