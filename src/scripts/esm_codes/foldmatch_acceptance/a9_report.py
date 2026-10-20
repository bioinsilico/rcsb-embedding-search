#!/usr/bin/env python
"""P8 A9 (long sequences), step 6: the A9 verdict for Gate G8 and the record. NERSC CPU node (about 10 GB, minutes).

Plan section 8 P8, A9: "The P0.7 slice through FoldMatch (needs G13 for the 3,000-aa items on 40 GB GPUs).
Targets 1,000-3,000 aa with benchmark queries: FP ratio at E <= 1 in [0.7, 1.4] (clamp). Queries > 966 aa: ecal=0
on every hit, with the warning." Gate G8: a failure of (1) stops this line of work and is reported.

Criterion (1)  analyze_long.main() on the production scores (core_production): NQxLT_real / all_bins / clamp at
               E <= 1, the shipped 'all' table, the real null as in P0.7 (N = the annotated long-target residues; per
               500-aa bin N = that bin's residues), cluster-bootstrap 95% CI. Also E <= 0.1 / 0.01 / 1e-3, each bin,
               the held-out union, the reversed null and the A5 reference (same queries, normal targets).
Criterion (2)  every hit of a query > 966 aa has ecal=0, and the once-per-run calibrated-domain warning appears once
               with the right counts. Checked on every production run: the a9_stage2.py runs (every pair; hits at
               E <= 100), and the three 'fm-search query sequences' CLI runs of the GPU job. Their output files and
               logs are re-parsed here. The FP ratios of those queries are reported for the record (P0.7: optimistic,
               documented): analyze_long's pooled 1,000-3,000 aa slice and bins, plus the exact > 966 aa set. The
               latter is computed with analyze_long's own functions and seeds; the same code first reproduces main()'s
               pooled 1,000-3,000 slice bit for bit.
Also: the code reproduces P0.7 on the research scores (core_research); the FP counts at E <= 1 recomputed from the
public hits alone; production vs research (merge.json); the GPU facts (G13: A100-40GB, peak memory).
Output: A9/A9.json and A9/A9.txt.
Usage:  python a9_report.py --gpu-dirs A9/p8-a9-gpu1_<jobid> A9/p8-a9-gpu2_<jobid>
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acc_common as C  # noqa: E402  (first: it sets the BLAS thread variables before numpy loads)
import numpy as np  # noqa: E402
import a9_core  # noqa: E402

XS = ("1.0", "0.1", "0.01", "0.001")
TBL = ["1000-1500", "1500-2000", "2000-2500", "2500-3000"]
QBL = ["300-600", "600-1000", "1000-1500", "1500-2000", "2000-2500", "2500-3000"]
BAND = (0.7, 1.4)
DOMAIN_MSG = "E-values are calibrated only inside the calibrated domain"
INFO_RE = re.compile(r"queries: (\d+) aligned, (\d+) without candidates, (\d+) fully degenerate")


def cell(e):
    return {k: e.get(k) for k in ("ratio", "fp", "nq", "fq", "ci_cluster", "ci_query", "ci_poisson")}


def table(e):
    return {x: cell(e[x]) for x in XS}


def fmt(e):
    return " / ".join(f"{e[x]['ratio']:.2f} [{e[x]['ci_cluster'][0]:.2f}, {e[x]['ci_cluster'][1]:.2f}] ({e[x]['fp']})"
                      for x in XS)


def parse_tsv(path, fields):
    rows = [ln.rstrip("\n").split("\t") for ln in open(path) if ln.strip()]
    return {f: [r[k] for r in rows] for k, f in enumerate(fields)}, len(rows)


def cli_check(d, inp, stage2_emb, low_distinct_targets, gt966=966):
    """One 'fm-search query sequences --scoring esm-codes' run: the output rows and the log's warning."""
    cmd = open(os.path.join(d, "cmd.txt")).read().split()
    fields = cmd[cmd.index("--format-output") + 1].split(",")
    fasta = cmd[cmd.index("--fasta-file") + 1]
    thr = float(cmd[cmd.index("--threshold") + 1]) if "--threshold" in cmd else 0.8
    top_k = int(cmd[cmd.index("--top-k") + 1]) if "--top-k" in cmd else 100
    qkey = "lq" if fasta.endswith("a9_lq.fasta") else "nq"
    tkey = "db_ben" if "/db/ben/" in " ".join(cmd) else "db_long"
    T, n = parse_tsv(os.path.join(d, "out.tsv"), fields)
    log = open(os.path.join(d, "run.log")).read().splitlines()
    dom = [ln for ln in log if DOMAIN_MSG in ln]
    info = [ln for ln in log if "esm-codes Stage 2 done" in ln]
    qlen = np.array(T["qlen"], int)
    tlen = np.array(T["tlen"], int)
    ecal = np.array(T["ecal"], int)
    qid_len = dict(zip(inp[f"{qkey}_ids"].astype(str).tolist(), inp[f"{qkey}_len"].tolist()))
    qid_ns = dict(zip(inp[f"{qkey}_ids"].astype(str).tolist(), inp[f"{qkey}_nonstd"].tolist()))
    tid_ns = dict(zip(inp[f"{tkey}_ids"].astype(str).tolist(), inp[f"{tkey}_nonstd"].tolist()))
    qns = np.array([qid_ns[q] for q in T["query"]]) / np.maximum(qlen, 1)
    tns = np.array([tid_ns[t] for t in T["target"]]) / np.maximum(tlen, 1)
    q_in = (qlen >= 21) & (qlen <= gt966) & (qns <= 0.01)
    t_out_long, t_out_ns = tlen > 3000, tns > 0.01
    exp_ecal = q_in & ~t_out_long & ~t_out_ns
    e0 = ~exp_ecal
    # the queries Stage 1 hands to Stage 2: >= 1 of its top_k with embscore >= threshold (production qpass
    # embeddings; exact if the CLI's ESM3 pass gave the same chain embeddings)
    ids_all = inp[f"{qkey}_ids"].astype(str)
    L_all = inp[f"{qkey}_len"]
    ns_all = inp[f"{qkey}_nonstd"] / L_all
    aligned = np.nanmax(stage2_emb, axis=1) >= thr if stage2_emb is not None else None
    out = dict(dir=d, command=" ".join(cmd), threshold=thr, top_k=top_k, rows=n, hits_ecal0=int((ecal == 0).sum()),
               rows_query_gt_966=int((qlen > gt966).sum()),
               every_hit_of_a_query_gt_966_has_ecal0=bool(np.all(ecal[qlen > gt966] == 0)),
               queries_gt_966_with_hits=len({q for q, m in zip(T["query"], qlen) if m > gt966}),
               ecal_matches_s13_from_lengths_and_residues=bool(np.array_equal(ecal == 1, exp_ecal)),
               domain_warning_lines=len(dom), warning_text=dom[0][:2000] if dom else None,
               stage2_info=info[0][:1500] if info else None,
               tmp_left=sorted(os.listdir(os.path.join(d, "tmp"))) if os.path.isdir(os.path.join(d, "tmp")) else None)
    mt = re.search(r"This run: (\d+) queries shorter than \d+ aa, (\d+) longer than \d+ aa, (\d+) with too many "
                   r"non-standard residues; (\d+) of (\d+) hits with ecal=0 \((\d+) for the query, (\d+) target too "
                   r"long, (\d+) target non-standard, (\d+) target with too few distinct codes\); (\d+) fully "
                   r"degenerate", dom[0]) if dom else None
    keys = ("short", "long", "nonstandard", "ecal0", "hits", "query_outside", "target_long", "target_nonstandard",
            "target_low_distinct", "degenerate")
    got = dict(zip(keys, map(int, mt.groups()))) if mt else None
    # the database's targets with < 2 distinct codes (a9_stage2.py counts them from the code store): none expected
    assert low_distinct_targets == 0, "targets with < 2 distinct codes: the expected counts need their ids"
    exp_hits = dict(ecal0=int(e0.sum()), hits=n, query_outside=int((e0 & ~q_in).sum()),
                    target_long=int((e0 & t_out_long).sum()), target_nonstandard=int((e0 & t_out_ns).sum()),
                    target_low_distinct=0, degenerate=0)
    mi = INFO_RE.search(info[0]) if info else None
    out["info_counts"] = dict(zip(("aligned", "without_candidates", "degenerate"), map(int, mi.groups()))) if mi else None
    if aligned is not None:
        exp_q = dict(short=int((aligned & (L_all < 21)).sum()), long=int((aligned & (L_all > gt966)).sum()),
                     nonstandard=int((aligned & (ns_all > 0.01)).sum()))
        out["stage1_aligned_from_qpass_embeddings"] = int(aligned.sum())
    else:
        exp_q = None
    out["warning_counts"] = got
    out["expected_hit_counts_from_output"] = exp_hits
    out["expected_query_counts_from_qpass_stage1"] = exp_q
    want = bool(exp_hits["ecal0"] > 0 or (exp_q and (exp_q["short"] + exp_q["long"] + exp_q["nonstandard"]) > 0))
    out["warning_expected"] = want
    out["warning_once"] = (len(dom) == 1) if want else (len(dom) == 0)
    if got is not None:
        out["hit_counts_match"] = all(got[k] == v for k, v in exp_hits.items())
        out["query_counts_match_qpass_stage1"] = (all(got[k] == v for k, v in exp_q.items()) if exp_q else None)
        out["long_count_bounds_ok"] = out["queries_gt_966_with_hits"] <= got["long"] <= int((L_all > gt966).sum())
    else:
        out["hit_counts_match"] = None if not want else False
    out["ok"] = bool(out["every_hit_of_a_query_gt_966_has_ecal0"] and out["ecal_matches_s13_from_lengths_and_residues"]
                     and out["warning_once"] and (out["hit_counts_match"] in (True, None))
                     and (out.get("long_count_bounds_ok", True)) and out["tmp_left"] == [])
    return out


def gt966_slice(AL, src_out, q90n_unused, label):
    """analyze_long's LQxN pooled-slice computation (main(): 'long_1000_3000') for a query mask, with its functions,
    its labels and its seeds (boot_idx(groups, NB, 21), default_rng(22)); returns {mask name: {rule: table}}."""
    ix = np.load(os.path.join(src_out, "index.npz"))
    lq_recs = AL.read_fasta(C.inp("p0/long/long_q.fasta"))
    th = [h for h, _ in AL.read_fasta(C.inp("calib3di/results/targets.fasta"))]
    ben_cols = ix["ben_cols"]
    lq_len = ix["lq_len"].astype(np.float64)
    ben_len = ix["ben_len"].astype(np.float64)
    lq_sccs = [AL.sccs_of(h) for h, _ in lq_recs]
    bt_sccs = [AL.sccs_of(h) for h in th]
    ben_rev = np.array([th[j].startswith("rev_") for j in ben_cols])
    ld = lambda n: np.load(os.path.join(src_out, f"{n}.npy")).astype(np.float64)  # noqa: E731
    q90_lq = np.quantile(ld("lq_pan_fwd") - ld("lq_pan_rev"), 0.9, axis=1)
    S_lb = ld("lq_ben_fwd") - ld("lq_ben_rev")
    NULL_lb, _ = AL.null_mask(lq_sccs, [bt_sccs[j] for j in ben_cols], ~ben_rev)
    REV_lb = np.broadcast_to(ben_rev[None, :], S_lb.shape)
    N_ann, N_rev = float(ben_len[~ben_rev].sum()), float(ben_len[ben_rev].sum())
    lq_folds = [set(AL.lvl(s, 2) for s in l_) for l_ in lq_sccs]
    lq_comp = AL.components(lq_folds)
    lq_key = np.unique(np.array([";".join(sorted(f)) for f in lq_folds]), return_inverse=True)[1]
    lq_groups = lq_key if np.bincount(lq_comp).max() / len(lq_comp) > 0.2 else lq_comp
    ship = C.read_json(C.inp("embcomp/panel/fit2/results/ship_final.json"))["tables"][AL.KEY]
    out = {}
    for nm, msk in (("long_1000_3000", lq_len >= 1000), ("gt_966", lq_len > 966)):
        ii = np.flatnonzero(msk)
        bci = AL.boot_idx(lq_groups[ii], AL.NB, 21)
        rng = np.random.default_rng(22)
        bqi = [rng.integers(0, len(ii), len(ii)) for _ in range(AL.NB)]
        real, _ = AL.slice_eval(S_lb[ii], q90_lq[ii], lq_len[ii], N_ann, ship["all"], NULL_lb[ii], bci, bqi)
        rev, _ = AL.slice_eval(S_lb[ii], q90_lq[ii], lq_len[ii], N_rev, ship["all"], REV_lb[ii], bci, bqi)
        out[nm] = dict(nq=int(len(ii)), real=real, rev=rev)
    C.log("gt966 slice", label, {k: v["real"]["clamp"]["1.0"]["ratio"] for k, v in out.items()})
    return out


def hits_crosscheck(A9, inp, core):
    """FP counts at E <= 1 of NQxLT (all bins and each bin) and of the pooled long-query slice, from the public hits
    of a9_stage2.py only: ln E(slice N) = log_evalue (N_db) - ln N_db + ln N, over the null pairs among the hits."""
    AL, _ = a9_core.import_analyze_long()
    out = {}
    lt = AL.read_fasta(C.inp("p0/long/long_t.fasta"))
    lr = AL.read_fasta(C.inp("p0/long/long_r.fasta"))
    qh = [h for h, _ in AL.read_fasta(C.inp("calib3di/results/queries.fasta"))]
    nq_sccs = [AL.sccs_of(h) for h in qh]
    lt_sccs = [AL.sccs_of(h) for h, _ in lt] + [[] for _ in lr]
    long_len = inp["db_long_len"].astype(np.float64)
    long_is_rev = np.array([x.startswith("lrev_") for x in inp["db_long_ids"].astype(str)])
    NULL_nl, _ = AL.null_mask(nq_sccs, lt_sccs, ~long_is_rev)
    d = glob.glob(os.path.join(A9, "stage2", "nq_long_p0of1"))[0]
    z = np.load(os.path.join(d, "scores.npz"))
    r = C.read_json(os.path.join(d, "stage2.json"))
    hq, hc, hl = z["hit_q"].astype(int), z["hit_c"].astype(int), z["hit_lne"].astype(np.float64)
    qrows = z["qrows"]
    N_db = float(r["N_db"])
    tb = np.digitize(long_len, [1500, 2000, 2500])
    res = {}
    for k, g in enumerate(["all_bins"] + TBL):
        cols = (~long_is_rev) if g == "all_bins" else ((~long_is_rev) & (tb == k - 1))
        N = float(long_len[cols].sum())
        complete = math.log(r["max_evalue"]) >= math.log(N_db / N)
        m = cols[hc] & NULL_nl[qrows[hq], hc]
        fp = int(((hl[m] - math.log(N_db) + math.log(N)) <= 0.0).sum())
        res[g] = dict(fp_from_hits=fp, fp_core=core["slices"]["NQxLT_real"][g]["clamp"]["1.0"]["fp"], N=N,
                      max_evalue_covers=complete, equal=fp == core["slices"]["NQxLT_real"][g]["clamp"]["1.0"]["fp"])
    out["NQxLT_real_E1"] = res
    # the pooled long-query slice (LQxN_real_all_long_1000_3000: N = the annotated benchmark-target residues)
    lq_recs = AL.read_fasta(C.inp("p0/long/long_q.fasta"))
    th = [h for h, _ in AL.read_fasta(C.inp("calib3di/results/targets.fasta"))]
    ben_cols = inp["ben_cols"]
    ben_len = inp["db_ben_len"].astype(np.float64)
    ben_rev = np.array([th[j].startswith("rev_") for j in ben_cols])
    NULL_lb, _ = AL.null_mask([AL.sccs_of(h) for h, _ in lq_recs], [AL.sccs_of(th[j]) for j in ben_cols], ~ben_rev)
    N_ann = float(ben_len[~ben_rev].sum())
    fp, covers = 0, True
    for d in sorted(glob.glob(os.path.join(A9, "stage2", "lq_ben_p*"))):
        z = np.load(os.path.join(d, "scores.npz"))
        r = C.read_json(os.path.join(d, "stage2.json"))
        hq, hc, hl = z["hit_q"].astype(int), z["hit_c"].astype(int), z["hit_lne"].astype(np.float64)
        rows = z["qrows"][hq]
        m = NULL_lb[rows, hc] & (inp["lq_len"][rows] >= 1000)
        fp += int(((hl[m] - math.log(float(r["N_db"])) + math.log(N_ann)) <= 0.0).sum())
        covers = covers and math.log(r["max_evalue"]) >= math.log(float(r["N_db"]) / N_ann)
    ref = core["slices"]["LQxN_real_all_long_1000_3000"]["clamp"]["1.0"]["fp"]
    out["LQxN_real_long_1000_3000_E1"] = dict(fp_from_hits=fp, fp_core=ref, N=N_ann, max_evalue_covers=covers,
                                              equal=fp == ref)
    return out


def g13_statement(smi, nv, qp):
    """What the GPU jobs show about G13 (geometric attention off for sequence input): the device of each job (from its
    nvidia-smi listing), its peak memory.used per GPU (nvidia-smi every 5 s) and the query passes' torch peaks."""
    parts = []
    for job, lines in smi.items():
        devs = sorted({ln.split(",")[1].strip() for ln in lines[1:] if "," in ln})
        peak = max(nv.get(job, {}).values()) if nv.get(job) else None
        parts.append(f"{job}: {', '.join(devs)}, peak memory.used {peak} MiB per GPU")
    q = "; ".join(f"query pass {k} on {v['gpu'].get('device')}: max length {v['max_len']} aa, torch peak allocated "
                  f"{v['gpu'].get('peak_allocated_gib', float('nan')):.2f} GiB" for k, v in qp.items())
    return ("Sequence input: FoldMatch's ESM3 pass runs with geometric attention off (G13), batch size 1. "
            + " | ".join(parts) + " | " + q)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-dirs", nargs="+", required=True)
    a = ap.parse_args()
    A9 = C.result_dir("A9")
    gdirs = [g if os.path.isabs(g) else os.path.join(A9, os.path.basename(g)) for g in a.gpu_dirs]
    inp = np.load(os.path.join(A9, "inputs", "a9_inputs.npz"))
    stored = C.read_json(C.inp("p0/long/long_slice_core.json"))
    cres = C.read_json(os.path.join(A9, "core_research.json"))
    cprod = C.read_json(os.path.join(A9, "core_production.json"))
    P = C.read_json(cprod["output"]["path"])
    R = stored
    merge = C.read_json(os.path.join(A9, "merge.json"))
    rec = {"title": "P8 A9: long sequences (the P0.7 slice through FoldMatch cc55c01's production path)",
           "plan": "PLAN.md section 8 P8 A9 + Gate G8; S11 (clamp above 594.918), S13 (queries 21-966 aa, targets "
                   "<= 3,000 aa), D15 (decided 2026-10-12)"}

    # ------------------------------------------------------------------ the analysis code reproduces P0.7
    rec["reproduces_p07_on_research_scores"] = dict(
        ok=cres["reproduces_p07"], numbers=cres["compare_with_stored_p07"]["numbers"],
        max_rel=cres["compare_with_stored_p07"]["max_rel"],
        text_report_identical=cres["compare_with_stored_p07"]["text_report_identical"],
        string_mismatches=cres["compare_with_stored_p07"]["string_mismatches"],
        analyze_long_md5=cres["analyze_long"]["md5"], core_research_json_sha256=C.sha256_file(
            os.path.join(A9, "core_research.json")))

    # ------------------------------------------------------------------ criterion (1)
    SLp, SLr = P["slices"], R["slices"]
    c1 = SLp["NQxLT_real"]["all_bins"]["clamp"]["1.0"]
    pass1 = BAND[0] <= c1["ratio"] <= BAND[1]
    rec["criterion_1"] = dict(
        statement="targets 1,000-3,000 aa with the 1,465 benchmark queries: FP ratio at E <= 1 in [0.7, 1.4] (clamp)",
        ratio=c1["ratio"], ci_cluster=c1["ci_cluster"], fp=c1["fp"], nq=c1["nq"], verdict="PASS" if pass1 else "FAIL",
        p07_research=cell(SLr["NQxLT_real"]["all_bins"]["clamp"]["1.0"]),
        production=dict(all_bins=table(SLp["NQxLT_real"]["all_bins"]["clamp"]),
                        bins={b: table(SLp["NQxLT_real"][b]["clamp"]) for b in TBL},
                        heldout_union={g: {x: cell(SLp["NQxLT_real"][f"{g}_heldout_union"]["clamp"][x]) for x in XS}
                                       for g in ["all_bins"] + TBL},
                        reversed_null={g: table(SLp["NQxLR_rev"][g]["clamp"]) for g in ["all_bins"] + TBL},
                        a5_reference_normal_targets=table(SLp["A5_NQxN_real"]["clamp"]),
                        a5_reference_heldout_union={x: cell(SLp["A5_NQxN_real_heldout_union"]["clamp"][x]) for x in XS},
                        extrap966=table(SLp["NQxLT_real"]["all_bins"]["extrap966"]),
                        extrap3000=table(SLp["NQxLT_real"]["all_bins"]["extrap3000"])),
        research=dict(all_bins=table(SLr["NQxLT_real"]["all_bins"]["clamp"]),
                      bins={b: table(SLr["NQxLT_real"][b]["clamp"]) for b in TBL},
                      reversed_null={g: table(SLr["NQxLR_rev"][g]["clamp"]) for g in ["all_bins"] + TBL},
                      a5_reference_normal_targets=table(SLr["A5_NQxN_real"]["clamp"])),
        null_pairs=P["meta"]["columns"]["null_pairs"], columns=P["meta"]["columns"],
        source=dict(core_production_json=os.path.join(A9, "core_production.json"),
                    long_slice_core_json=cprod["output"]))

    # ------------------------------------------------------------------ criterion (2)
    s2 = {}
    for f in sorted(glob.glob(os.path.join(A9, "stage2", "*", "stage2.json"))):
        r = C.read_json(f)
        s = r["checks"]["s13"]
        s2[os.path.basename(os.path.dirname(f))] = dict(
            ok=r["ok"], queries=r["queries"], targets=r["targets"], N_db=r["N_db"], max_evalue=r["max_evalue"],
            hits=r["checks"]["hits"]["n"], queries_gt_966=s["queries_gt_966"],
            hits_of_queries_gt_966=s["hits_of_queries_gt_966"], queries_gt_966_with_hits=s["queries_gt_966_with_hits"],
            every_hit_of_a_query_gt_966_has_ecal0=s["every_hit_of_a_query_gt_966_has_ecal0"],
            ecal_matches_expected=s["ecal_matches_expected"], hits_ecal1=s["hits_ecal1"],
            domain_warnings=s["domain_warnings"], warning_once=s["warning_once"],
            warning_counts=s["warning_counts"], expected_counts=s["expected_counts"],
            warning_counts_match=s["warning_counts_match"], warnings=[w["message"][:1500] for w in r["warnings"]],
            stage2_info=r["stage2_info"], wall_s=r["wall_s"], workers=r["workers"],
            stage2_json_sha256=C.sha256_file(f))
    lq_parts = {k: v for k, v in s2.items() if k.startswith("lq_ben")}
    low_distinct = {}
    for f in sorted(glob.glob(os.path.join(A9, "stage2", "*", "stage2.json"))):
        r = C.read_json(f)
        low_distinct.setdefault("long" if r["task"] == "nq_long" else "ben", set()).add(
            r["checks"]["s13"]["targets_with_lt2_distinct_codes"])
    low_distinct = {k: max(v) for k, v in low_distinct.items()}
    nqlong_emb = None
    lq_emb = np.concatenate([np.load(os.path.join(A9, "stage2", k, "scores.npz"))["emb"] for k in sorted(lq_parts)])
    lq_rows = np.concatenate([np.load(os.path.join(A9, "stage2", k, "scores.npz"))["qrows"] for k in sorted(lq_parts)])
    lq_emb = lq_emb[np.argsort(lq_rows)]
    nqlong_emb = np.load(os.path.join(A9, "stage2", "nq_long_p0of1", "scores.npz"))["emb"]
    cli = {}
    for d in sorted(glob.glob(os.path.join(A9, "cli", "*"))):
        name = os.path.basename(d)
        if not os.path.exists(os.path.join(d, "out.tsv")):
            cli[name] = dict(ok=False, error="no out.tsv")
            continue
        emb = lq_emb if "lq_" in name else nqlong_emb
        low = low_distinct["ben" if "_ben" in name else "long"]
        cli[name] = cli_check(d, inp, emb, low)
    agg = dict(stage2_lq_hits_of_queries_gt_966=sum(v["hits_of_queries_gt_966"] for v in lq_parts.values()),
               stage2_lq_queries_gt_966=sum(v["queries_gt_966"] for v in lq_parts.values()),
               stage2_lq_queries_gt_966_with_hits=sum(v["queries_gt_966_with_hits"] for v in lq_parts.values()),
               cli_rows_query_gt_966={k: v.get("rows_query_gt_966") for k, v in cli.items()})
    pass2 = (all(v["ok"] and v["every_hit_of_a_query_gt_966_has_ecal0"] and v["warning_once"]
                 and v["warning_counts_match"] in (True, None) for v in s2.values())
             and all(v.get("ok") for v in cli.values()) and len(lq_parts) > 0 and len(cli) == 3)
    AL, _ = a9_core.import_analyze_long()
    g_prod = gt966_slice(AL, os.path.join(A9, "prod_out"), None, "production")
    g_res = gt966_slice(AL, C.inp("p0/long/out"), None, "research")
    same_as_main = dict(
        production=C._cmp_tree(g_prod["long_1000_3000"]["real"], SLp["LQxN_real_all_long_1000_3000"]),
        research=C._cmp_tree(g_res["long_1000_3000"]["real"], SLr["LQxN_real_all_long_1000_3000"]))
    rec["criterion_2"] = dict(
        statement="queries > 966 aa: every hit has ecal=0, and the once-per-run calibrated-domain warning appears with "
                  "the right counts; their FP ratios reported for the record (expected optimistic, as documented)",
        verdict="PASS" if pass2 else "FAIL", aggregate=agg, stage2_runs=s2, cli_runs=cli,
        fp_ratios_for_the_record=dict(
            production=dict(gt_966=dict(nq=g_prod["gt_966"]["nq"], real=table(g_prod["gt_966"]["real"]["clamp"]),
                                        reversed=table(g_prod["gt_966"]["rev"]["clamp"])),
                            pooled_1000_3000=table(SLp["LQxN_real_all_long_1000_3000"]["clamp"]),
                            pooled_1000_3000_reversed=table(SLp["LQxR_rev_all_long_1000_3000"]["clamp"]),
                            base_300_1000=table(SLp["LQxN_real_all_base_300_1000"]["clamp"]),
                            bins={b: {x: cell(SLp["LQxN_real_all"]["clamp"][x]["bins"][b]) for x in XS} for b in QBL},
                            extrap966_pooled_1000_3000=table(SLp["LQxN_real_all_long_1000_3000"]["extrap966"])),
            research=dict(gt_966=dict(nq=g_res["gt_966"]["nq"], real=table(g_res["gt_966"]["real"]["clamp"]),
                                      reversed=table(g_res["gt_966"]["rev"]["clamp"])),
                          pooled_1000_3000=table(SLr["LQxN_real_all_long_1000_3000"]["clamp"]),
                          pooled_1000_3000_reversed=table(SLr["LQxR_rev_all_long_1000_3000"]["clamp"]),
                          base_300_1000=table(SLr["LQxN_real_all_base_300_1000"]["clamp"]),
                          bins={b: {x: cell(SLr["LQxN_real_all"]["clamp"][x]["bins"][b]) for x in XS} for b in QBL}),
            gt_966_code_check=dict(note="the > 966 aa slice uses analyze_long's functions and seeds; the same code "
                                        "on the pooled 1,000-3,000 mask reproduces main()'s slice",
                                   production=dict(numbers=same_as_main["production"]["numbers"],
                                                   max_abs=same_as_main["production"]["max_abs"],
                                                   mismatches=same_as_main["production"]["mismatches"][:5]),
                                   research=dict(numbers=same_as_main["research"]["numbers"],
                                                 max_abs=same_as_main["research"]["max_abs"],
                                                 mismatches=same_as_main["research"]["mismatches"][:5]))))

    # ------------------------------------------------------------------ cross-checks and context
    rec["fp_counts_from_public_hits"] = hits_crosscheck(A9, inp, P)
    rec["production_vs_research"] = merge["production_vs_research"]
    rec["merge_checks"] = merge["checks"]
    gpu, nv, smi = {}, {}, {}
    for gdir in gdirs:
        job = os.path.basename(gdir)
        for f in sorted(glob.glob(os.path.join(gdir, "*.done"))):
            gpu[f"{job}/{os.path.basename(f)[:-5]}"] = C.read_json(f)
        smi[job] = open(os.path.join(gdir, "nvidia_smi.txt")).read().strip().splitlines()
        nv[job] = {}
        if os.path.exists(os.path.join(gdir, "nvsmi.csv")):
            for ln in open(os.path.join(gdir, "nvsmi.csv")):
                p = [x.strip() for x in ln.split(",")]
                if len(p) >= 4 and p[2].endswith("MiB"):
                    g, mu = int(p[1]), int(p[2].split()[0])
                    nv[job][g] = max(nv[job].get(g, 0), mu)
    qp = {k: C.read_json(os.path.join(A9, "qpass", k, "qpass.json")) for k in ("nq", "lq")}
    rec["gpu"] = dict(
        dirs=gdirs, nvidia_smi=smi, steps=gpu, peak_memory_used_mib_per_gpu=nv,
        query_pass={k: dict(n=v["n"], residues=v["residues"], max_len=v["max_len"], gpu=v["gpu"], wall_s=v["wall_s"],
                            residues_per_s=v["residues_per_s"], checks=v["checks"], ok=v["ok"])
                    for k, v in qp.items()},
        g13=g13_statement(smi, nv, qp))
    rec["verdict"] = dict(criterion_1=rec["criterion_1"]["verdict"], criterion_2=rec["criterion_2"]["verdict"],
                          A9="PASS" if (pass1 and pass2 and rec["reproduces_p07_on_research_scores"]["ok"]) else "FAIL",
                          gate_g8=("A9 passes: no stop" if pass1 else
                                   "A9 criterion (1) FAILS: Gate G8 stops this line of work (remedy: v2 refit or a "
                                   "D15 change with a new system name, never a parameter tweak)"))
    rec["provenance"] = C.provenance()
    sha = C.write_json(rec, os.path.join(A9, "A9.json"))

    # ------------------------------------------------------------------ text summary
    L = [f"P8 A9 long sequences: {rec['verdict']['A9']}  (criterion 1 {rec['verdict']['criterion_1']}, criterion 2 "
         f"{rec['verdict']['criterion_2']}; {rec['verdict']['gate_g8']})", ""]
    rp = rec["reproduces_p07_on_research_scores"]
    L.append(f"P0.7 analysis code on the research scores: reproduces the stored P0.7 core = {rp['ok']} "
             f"({rp['numbers']} numbers, max rel diff {rp['max_rel']:.1e}, text report identical "
             f"{rp['text_report_identical']})")
    L.append("")
    L.append("(1) 1,465 benchmark queries x 2,000 long targets (1,000-3,000 aa), real null, shipped table, clamp; "
             "FP ratio [cluster CI] (FPs) at E <= 1 / 0.1 / 0.01 / 1e-3:")
    L.append(f"  production  {fmt(SLp['NQxLT_real']['all_bins']['clamp'])}")
    L.append(f"  P0.7        {fmt(SLr['NQxLT_real']['all_bins']['clamp'])}")
    for b in TBL:
        L.append(f"  bin {b:9s} prod {fmt(SLp['NQxLT_real'][b]['clamp'])}")
        L.append(f"  {'':13s} P0.7 {fmt(SLr['NQxLT_real'][b]['clamp'])}")
    L.append(f"  held-out union at E<=1: prod {SLp['NQxLT_real']['all_bins_heldout_union']['clamp']['1.0']['ratio']:.3f}"
             f"  P0.7 {SLr['NQxLT_real']['all_bins_heldout_union']['clamp']['1.0']['ratio']:.3f}")
    L.append(f"  reversed null (2,000 long decoys): prod {fmt(SLp['NQxLR_rev']['all_bins']['clamp'])}")
    L.append(f"  A5 reference (same queries, 25,000 normal targets): prod {fmt(SLp['A5_NQxN_real']['clamp'])}")
    L.append(f"  -> E<=1 ratio {c1['ratio']:.3f} [{c1['ci_cluster'][0]:.2f}, {c1['ci_cluster'][1]:.2f}] in [0.7, 1.4]: "
             f"{rec['criterion_1']['verdict']}")
    L.append("")
    L.append("(2) queries > 966 aa (2,019 of the 3,000 long queries):")
    for k, v in s2.items():
        L.append(f"  stage2 {k:16s} hits {v['hits']:8d}  of >966 queries {v['hits_of_queries_gt_966']:7d} all ecal=0 "
                 f"{v['every_hit_of_a_query_gt_966_has_ecal0']}; warning once {v['warning_once']}, counts match "
                 f"{v['warning_counts_match']}: {v['warning_counts']}")
    for k, v in cli.items():
        L.append(f"  CLI {k:19s} rows {v.get('rows')}; >966 rows {v.get('rows_query_gt_966')} all ecal=0 "
                 f"{v.get('every_hit_of_a_query_gt_966_has_ecal0')}; warning lines {v.get('domain_warning_lines')}, "
                 f"counts {v.get('warning_counts')}, hit counts match {v.get('hit_counts_match')}, query counts match "
                 f"qpass Stage 1 {v.get('query_counts_match_qpass_stage1')}; ok {v.get('ok')}")
    fr = rec["criterion_2"]["fp_ratios_for_the_record"]
    L.append(f"  FP ratio, > 966 aa ({fr['production']['gt_966']['nq']} queries) x 25,000 annotated, real null, clamp:")
    L.append(f"    production {fmt(fr['production']['gt_966']['real'])}")
    L.append(f"    P0.7 data  {fmt(fr['research']['gt_966']['real'])}")
    L.append(f"  pooled 1,000-3,000 aa: prod {fmt(fr['production']['pooled_1000_3000'])}")
    L.append(f"                         P0.7 {fmt(fr['research']['pooled_1000_3000'])}")
    L.append(f"  baseline 300-1,000 aa: prod {fmt(fr['production']['base_300_1000'])}")
    L.append(f"  reversed null 1,000-3,000: prod {fmt(fr['production']['pooled_1000_3000_reversed'])}")
    L.append("")
    hx = rec["fp_counts_from_public_hits"]
    L.append("FP counts at E <= 1 from the public hits only vs the full matrices: NQxLT " + ", ".join(
        f"{g} {v['fp_from_hits']}/{v['fp_core']} ({'=' if v['equal'] else 'DIFF'})"
        for g, v in hx["NQxLT_real_E1"].items()) + f"; LQ 1,000-3,000 {hx['LQxN_real_long_1000_3000_E1']['fp_from_hits']}"
        f"/{hx['LQxN_real_long_1000_3000_E1']['fp_core']}")
    pv = rec["production_vs_research"]
    L.append("Production vs research (informative): code bits " + ", ".join(
        f"{k[10:]} {v['agreement']:.5f} (worst item {v['worst_item']:.4f})" for k, v in pv.items()
        if k.startswith("code_bits")) + "; q cosine median " + ", ".join(
        f"{k[2:]} {v['cosine_median']:.5f}" for k, v in pv.items() if k.startswith("q_")) + "; |dlnE| median/P95 " +
        ", ".join(f"{k[7:]} {v['dlnE_median']:.3f}/{v['dlnE_p95']:.3f}" for k, v in pv.items()
                  if k.startswith("scores_") and v.get("dlnE_median") is not None))
    L.append(f"G13: {rec['gpu']['g13']}")
    L.append(f"GPU: peak memory.used per GPU "
             f"(MiB) {nv}; query pass peak allocated (GiB): " + ", ".join(
                 f"{k} {v['gpu']['peak_allocated_gib']:.2f} (max len {v['max_len']})" for k, v in qp.items()))
    open(os.path.join(A9, "A9.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    C.log("wrote", os.path.join(A9, "A9.json"), sha)


if __name__ == "__main__":
    main()
