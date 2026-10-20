#!/usr/bin/env python
"""P8 A8 (plan FINAL v1, section 8 P8): the A5 benchmark through the real pipeline.

The runs (GPU job, a8_pipeline.py around the real CLI; the Stage-1 threshold and top_k at their CLI defaults):
  main     fm-search query sequences --scoring esm-codes --max-evalue 10 --format-output <every scalar column>
           of the 1,465 benchmark queries against the production database of the 50,000 A5 targets;
  default  the same with every default (max_evalue 1e-3, the 13 default columns): its rows must be the main run's rows
           with E <= 1e-3, in the same order, with the same text;
  self     1,000 targets drawn from the database (seed 8) as queries, --max-evalue inf: self-hit expectations (P6 note:
           weak self E-values for short / low-complexity queries, Stage-1 HNSW misses; 'self ranked first' is the
           meaningful measure for esm-codes, the self-hit study).

Pass criterion (plan A8 row): the E <= 1 FP ratio is <= the exhaustive A5 value (the prefilter may only make E more
conservative). Checked under the pipeline's own E (shipped table, N = the database's total residues) and under the
research protocol's E (N = annotated residues; main / swap / heldout_union / all_insample), with the per-pair check
that every pipeline hit carries exactly the exhaustive run's S_f, S_r, Q90 and ln E. Also reported: the TPs / FPs the
prefilter removed, by reason (below the threshold, beyond top_k, HNSW recall miss), HNSW recall against the exact
embedding ranking, sensitivity at 1e-3, and the self-hit run.

  python a8_analysis.py make-self-fasta --n 1000 --seed 8 --out FILE
  python a8_analysis.py analyze --a5-dir DIR --db DB --main CAP:OUT --default CAP:OUT [--self CAP:OUT:FASTA] --out J
"""
from __future__ import annotations

import argparse
import inspect
import math
import os
import sys
import time

import acc_common as AC  # noqa: E402  (first: BLAS threads = 1)
import numpy as np  # noqa: E402

XS = (1e-3, 0.01, 0.1, 1.0, 10.0)


def make_self_fasta(a) -> int:
    recs = AC.read_fasta(AC.inp(AC.BENCH_IN["tfasta"]))
    rng = np.random.default_rng(a.seed)
    pick = np.sort(rng.choice(len(recs), a.n, replace=False))
    with open(a.out, "w") as fh:
        for k in pick.tolist():
            h, s = recs[k]
            fh.write(f">{h}\n{s}\n")
    print(a.out, AC.md5_file(a.out), len(pick))
    return 0


def read_tsv(path, fields):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            p = line.split("\t")
            assert len(p) == len(fields), (path, len(p), len(fields), line[:200])
            rows.append(p)
    return rows


def load_capture(cap, B):
    run = AC.read_json(os.path.join(cap, "run.json"))
    z = np.load(os.path.join(cap, "hits.npz"))
    qids, tids = z["qids"].astype(str), z["tids"].astype(str)
    qpos = {q: i for i, q in enumerate(B.qids)}
    tpos = {t: i for i, t in enumerate(B.tids)}
    H = {k: z[k] for k in z.files if k not in ("qids", "tids")}
    H["bq"] = np.array([qpos[q] for q in qids], np.int64)[H["qi"]] if len(H["qi"]) else np.zeros(0, np.int64)
    H["bt"] = np.array([tpos[t] for t in tids], np.int64)[H["ti"]] if len(H["ti"]) else np.zeros(0, np.int64)
    H["qid"] = qids[H["qi"]] if len(H["qi"]) else np.zeros(0, str)
    H["tid"] = tids[H["ti"]] if len(H["ti"]) else np.zeros(0, str)
    s1 = np.load(os.path.join(cap, "stage1_raw.npz"))
    return run, H, s1


def fields_of(run):
    argv = run["fm_search_argv"]
    if "--format-output" in argv:
        return argv[argv.index("--format-output") + 1].split(",")
    from foldmatch.search.output import DEFAULT_OUTPUT_FIELDS
    return list(DEFAULT_OUTPUT_FIELDS)


def out_of(run):
    argv = run["fm_search_argv"]
    return argv[argv.index("--output-file") + 1]


def tsv_matches_hits(run, H) -> dict:
    """The CLI's own output file against the captured hits: same rows, same (query, target) order, evalue / raw
    columns rendered from the captured values."""
    fields = fields_of(run)
    rows = read_tsv(out_of(run), fields)
    out = dict(rows=len(rows), hits=int(len(H["qi"])), fields=fields)
    iq, it = fields.index("query"), fields.index("target")
    pairs_ok = len(rows) == len(H["qi"]) and all(r[iq] == q and r[it] == t for r, q, t in zip(rows, H["qid"],
                                                                                           H["tid"]))
    out["pairs_in_order"] = bool(pairs_ok)
    if pairs_ok:
        # the captured values rendered with FoldMatch's own renderers (output.py) give the file's text
        from foldmatch.search import output as O
        checks = {"raw": ("score", O._format_3f), "rawrev": ("score_rev", O._format_3f), "q90": ("q90", O._format_3f),
                  "rawadj": ("score_adj", O._format_3f), "evalue": ("log_evalue", O.format_evalue_from_log)}
        for f, (k, fn) in checks.items():
            if f in fields:
                i = fields.index(f)
                out[f"{f}_text_equal_captured"] = bool(all(r[i] == fn(float(v)) for r, v in zip(rows, H[k])))
    return out, rows


def counts_at(L, mask, xs):
    return {str(x): int(((L <= math.log(x)) & mask).sum()) for x in xs}


def analyze(a) -> int:
    t0 = time.time()
    import foldmatch.cli.search as cli
    B = AC.Bench()
    tabs = AC.ship_tables()
    res = {"what": "P8 A8: the A5 benchmark through the real pipeline (fm-search query sequences --scoring esm-codes, "
                   "default Stage-1 threshold and top_k) against the exhaustive A5 run",
           "criterion": "the E <= 1 FP ratio is <= the exhaustive A5 value (the prefilter may only make E more "
                        "conservative)"}
    sig = inspect.signature(cli.query_database_from_fasta).parameters
    res["cli_defaults"] = {k: sig[k].default for k in ("top_k", "threshold", "max_evalue", "min_seq_identity",
                                                         "min_coverage", "align_workers", "batch_size")}
    res["cli_esm_codes_defaults"] = dict(cli.ESM_CODES_QUERY_DEFAULTS)
    st5 = AC.read_json(os.path.join(a.a5_dir, "stage2.json"))
    res["index"] = st5.get("db", {}).get("faiss")
    res["faiss_metadata"] = st5.get("db", {}).get("faiss_metadata")
    res["db_manifest_total_residues"] = st5.get("db", {}).get("manifest", {}).get("total_residues")
    z5 = np.load(os.path.join(a.a5_dir, "matrices.npz"))
    assert list(z5["qids"].astype(str)) == list(B.qids) and list(z5["tids"].astype(str)) == list(B.tids)
    S_f5, S_r5, q905 = z5["S_f"], z5["S_r"], z5["q90"]
    S5 = S_f5.astype(np.float64) - S_r5.astype(np.float64)
    L5_prod = AC.ln_evalue(tabs["all"], B.qlen, S5, q905, B.N_all)
    cos = np.load(os.path.join(a.a5_dir, "cos.npz"))["cos"]

    # ------------------------------------------------------------------------------------------------ main run
    cap, out_main = a.main
    run, H, s1 = load_capture(cap, B)
    res["main_run"] = {k: run.get(k) for k in ("fm_search_argv", "rc", "cli_wall_s", "stage1", "label")}
    res["main_run"]["stage2"] = {k: v for k, v in run["stage2"].items() if k != "stats"}
    st = dict(run["stage2"]["stats"])
    st.pop("q90", None)
    res["main_run"]["stage2_stats"] = st
    res["main_run"]["provenance_foldmatch"] = run.get("provenance", {}).get("foldmatch", {}).get("commit")
    res["main_run"]["kernel"] = run.get("provenance", {}).get("kernel")
    tm, rows_main = tsv_matches_hits(run, H)
    res["main_run"]["output_file_vs_hits"] = tm
    me = float(run["stage2"]["options"]["max_evalue"])
    bq, bt = H["bq"], H["bt"]
    # per-pair: the pipeline's values are the exhaustive run's
    eq = dict(
        S_f=bool(np.array_equal(H["score"], S_f5[bq, bt].astype(np.float64))),
        S_r=bool(np.array_equal(H["score_rev"], S_r5[bq, bt].astype(np.float64))),
        q90=bool(np.array_equal(H["q90"], q905[bq])),
        lnE_max_abs=float(np.abs(H["log_evalue"] - L5_prod[bq, bt]).max()) if len(bq) else 0.0)
    eq["all_equal"] = bool(eq["S_f"] and eq["S_r"] and eq["q90"] and eq["lnE_max_abs"] <= 1e-9)
    res["per_pair_pipeline_vs_exhaustive"] = eq
    # A8 matrices (non-hits = -inf)
    S8 = np.full_like(S5, -np.inf)
    S8[bq, bt] = H["score"] - H["score_rev"]
    L8_prod = np.full_like(S5, np.inf)
    L8_prod[bq, bt] = H["log_evalue"]
    # (P) the pipeline's own E
    P = {}
    for name, L in (("pipeline", L8_prod), ("exhaustive_A5", L5_prod)):
        P[name] = {str(x): dict(real_fp=int(((L <= math.log(x)) & B.REAL).sum()),
                                ratio=float(((L <= math.log(x)) & B.REAL).sum() / (B.nq * x)),
                                fq=float(((L <= math.log(x)) & B.REAL).any(1).mean()),
                                rev_fp=int(((L <= math.log(x)) & B.REVM).sum()),
                                tp=int(((L <= math.log(x)) & B.TP).sum())) for x in XS}
    res["fp_ratio_pipeline_E"] = dict(definition="E as the pipeline reports it: shipped table, N = the database's "
                                                 f"total residues {B.N_all:.0f} (annotated + reversed)", **P)
    # (R) the research protocol's E (N annotated residues) on the pipeline's hits
    cov = None
    try:
        import fm_stage2 as F
        cov = F.coverage_max_evalue(B.qlen, tabs["all"], B.N_all, {k: tabs[k] for k in ("all", "main", "swap")},
                                    {"N_ann": B.N_ann}, 1.0)
        cov["pipeline_max_evalue"] = me
        cov["covered"] = bool(cov["required_max_evalue"] <= me)
    except Exception as e:  # noqa: BLE001
        cov = {"error": repr(e)}
    res["research_E_coverage_of_pipeline_hits"] = cov
    p8 = AC.bench_protocol(B, S8, tabs, q905)
    p5 = AC.bench_protocol(B, S5, tabs, q905)
    R = {}
    for e in ("main", "swap", "all_insample", "heldout_union"):
        R[e] = {x: dict(pipeline=dict(fp=p8[e][x]["fp"], ratio=p8[e][x]["ratio"], ci_fold=p8[e][x]["ci_fold"],
                                      fq=p8[e][x]["fq"]),
                        exhaustive_A5=dict(fp=p5[e][x]["fp"], ratio=p5[e][x]["ratio"], ci_fold=p5[e][x]["ci_fold"],
                                           fq=p5[e][x]["fq"]),
                        removed_fp=p5[e][x]["fp"] - p8[e][x]["fp"])
                for x in ("0.01", "0.1", "1.0")}
        R[e]["per_query_pipeline_le_exhaustive"] = bool(all(
            (np.asarray(p8["per_query"][e][x]) <= np.asarray(p5["per_query"][e][x])).all()
            for x in p8["per_query"][e]))
        R[e]["bins_E1"] = {g: dict(pipeline=p8[e]["1.0"]["bins"][g]["ratio"], exhaustive=p5[e]["1.0"]["bins"][g]["ratio"])
                           for g in p8[e]["1.0"]["bins"]}
    R["sensitivity_1e3"] = {sp: dict(pipeline=p8["sensitivity"][sp]["sens_1e3"],
                                     exhaustive=p5["sensitivity"][sp]["sens_1e3"]) for sp in ("main", "swap")}
    res["fp_ratio_research_E"] = R
    res["lines_pipeline"] = [AC.protocol_line(e, p8[e]) for e in ("main", "swap", "all_insample", "heldout_union")]
    res["lines_exhaustive"] = [AC.protocol_line(e, p5[e]) for e in ("main", "swap", "all_insample", "heldout_union")]
    crit = {"pipeline_E": P["pipeline"]["1.0"]["ratio"] <= P["exhaustive_A5"]["1.0"]["ratio"]}
    for e in ("main", "swap", "all_insample", "heldout_union"):
        crit[f"research_E_{e}"] = R[e]["1.0"]["pipeline"]["ratio"] <= R[e]["1.0"]["exhaustive_A5"]["ratio"]
    crit["per_pair_values_equal_exhaustive"] = eq["all_equal"]
    crit["no_pipeline_pair_more_significant"] = bool((L8_prod >= L5_prod - 1e-9).all())
    res["criterion_checks"] = {k: bool(v) for k, v in crit.items()}
    res["A8"] = "PASS" if all(crit[k] for k in crit if k.startswith(("pipeline_E", "research_E"))) else "FAIL"

    # ---------------------------------------------------------------------- what the prefilter removed, and why
    thr = float(run["stage1"]["threshold"]) if run["stage1"]["threshold"] is not None else -np.inf
    top_k = int(res["cli_defaults"]["top_k"])
    raw_q, raw_c, raw_t, raw_s = (s1["qids"].astype(str), s1["counts"], s1["tids"].astype(str), s1["scores"])
    qpos = {q: i for i, q in enumerate(B.qids)}
    tpos = {t: i for i, t in enumerate(B.tids)}
    RAW = np.zeros(S5.shape, bool)
    RAWS = np.full(S5.shape, np.nan, np.float32)
    o = 0
    for q, c in zip(raw_q, raw_c.tolist()):
        i = qpos[q]
        js = np.array([tpos[t] for t in raw_t[o:o + c]], np.int64)
        RAW[i, js] = True
        RAWS[i, js] = raw_s[o:o + c]
        o += c
    CAND = RAW & (RAWS >= thr)
    # exact ranking by the exact cosine (descending; rank 1 = best)
    order = np.argsort(-cos, axis=1, kind="stable")
    rank = np.empty_like(order)
    rank[np.arange(B.nq)[:, None], order] = np.arange(1, B.nt + 1)[None, :]
    exact_top = rank <= top_k
    res["stage1"] = dict(threshold=thr, top_k=top_k, raw_candidates=int(RAW.sum()),
                         candidates_after_threshold=int(CAND.sum()),
                         stage2_candidates_reported=run["stage2"]["candidates_in"],
                         hnsw_vs_exact_cos_max_abs=float(np.nanmax(np.abs(RAWS[RAW] - cos[RAW]))),
                         recall_at_top_k=float((RAW & exact_top).sum() / exact_top.sum()),
                         recall_at_top_k_above_threshold=float((RAW & exact_top & (cos >= thr)).sum()
                                                               / max((exact_top & (cos >= thr)).sum(), 1)),
                         queries_with_full_recall=int(((RAW & exact_top).sum(1) == exact_top.sum(1)).sum()),
                         pairs_above_threshold_exact=int((cos >= thr).sum()),
                         queries_without_candidates=int((CAND.sum(1) == 0).sum()))
    removal = {}
    labels = {"TP": B.TP, "FP_real": B.REAL, "FP_reversed": B.REVM, "ignored": ~(B.TP | B.FP)}
    for defn, L5, L8 in (("pipeline_E", L5_prod, L8_prod),
                         ("research_E_all_table_N_ann", AC.ln_evalue(tabs["all"], B.qlen, S5, q905, B.N_ann),
                          AC.ln_evalue(tabs["all"], B.qlen, S8, q905, B.N_ann))):
        removal[defn] = {}
        for x in (1e-3, 1.0):
            A5set = L5 <= math.log(x)
            A8set = L8 <= math.log(x)
            rem = A5set & ~A8set
            add = A8set & ~A5set
            d = {}
            for lab, m in labels.items():
                r = rem & m
                d[lab] = dict(exhaustive=int((A5set & m).sum()), pipeline=int((A8set & m).sum()), removed=int(r.sum()),
                              added=int((add & m).sum()),
                              removed_below_threshold=int((r & (cos < thr)).sum()),
                              removed_beyond_top_k=int((r & (cos >= thr) & ~exact_top).sum()),
                              removed_hnsw_miss=int((r & (cos >= thr) & exact_top & ~RAW).sum()),
                              removed_other=int((r & (cos >= thr) & exact_top & RAW).sum()))
            removal[defn][str(x)] = d
    res["prefilter_removed"] = removal

    # ---------------------------------------------------------------------------------------- default run
    if a.default:
        capd, _ = a.default
        rund, Hd, _ = load_capture(capd, B)
        tmd, rows_def = tsv_matches_hits(rund, Hd)
        fd = fields_of(rund)
        fm = fields_of(run)
        sel = [k for k in range(len(rows_main)) if H["log_evalue"][k] <= math.log(1e-3)]
        proj = [[rows_main[k][fm.index(f)] for f in fd] for k in sel]
        res["default_run"] = dict(fm_search_argv=rund["fm_search_argv"], rc=rund["rc"], cli_wall_s=rund["cli_wall_s"],
                                  stage1=rund["stage1"], fields=fd, output_file_vs_hits=tmd, rows=len(rows_def),
                                  main_rows_E_le_1e3=len(sel), rows_equal_main_subset=bool(proj == rows_def),
                                  max_evalue=rund["stage2"]["options"]["max_evalue"],
                                  stage2_stats={k: v for k, v in rund["stage2"]["stats"].items() if k != "q90"})
        if proj != rows_def:
            diff = [(i, p, r) for i, (p, r) in enumerate(zip(proj, rows_def)) if p != r][:5]
            res["default_run"]["first_differences"] = diff

    # ------------------------------------------------------------------------------------------ self hits
    if a.self:
        res["self_hits"] = self_hits(a.self, B)
    res["provenance"] = AC.provenance()
    res["wall_s"] = round(time.time() - t0, 1)
    h = AC.write_json(res, a.out)
    with open(a.out.replace(".json", ".txt"), "w") as fh:
        fh.write(f"A8: {res['A8']}  checks {res['criterion_checks']}\n")
        fh.write(f"stage1 {res['stage1']}\n")
        for x in ("0.001", "0.01", "0.1", "1.0", "10.0"):
            fh.write(f"pipeline E <= {x}: pipeline {P['pipeline'][x]}  exhaustive {P['exhaustive_A5'][x]}\n")
        fh.write("research-protocol E, pipeline hits:\n" + "\n".join(res["lines_pipeline"]) + "\n")
        fh.write("research-protocol E, exhaustive:\n" + "\n".join(res["lines_exhaustive"]) + "\n")
        fh.write(f"json {a.out} sha256 {h}\n")
    AC.log(f"wrote {a.out} sha256 {h}: A8 {res['A8']} {res['criterion_checks']}")
    return 0


def self_hits(spec, B) -> dict:
    cap, out_file, fasta = spec
    run = AC.read_json(os.path.join(cap, "run.json"))
    z = np.load(os.path.join(cap, "hits.npz"))
    qids, tids = z["qids"].astype(str), z["tids"].astype(str)
    s1 = np.load(os.path.join(cap, "stage1_raw.npz"))
    recs = AC.read_fasta(fasta)
    qseq = {h.split()[0]: s for h, s in recs}
    # identical-sequence twins in the database
    by_seq = {}
    for h, s in AC.read_fasta(AC.inp(AC.BENCH_IN["tfasta"])):
        by_seq.setdefault(s, []).append(h.split()[0])
    raw = {}
    o = 0
    st_q, st_c, st_t, st_s = s1["qids"].astype(str), s1["counts"], s1["tids"].astype(str), s1["scores"]
    for q, c in zip(st_q, st_c.tolist()):
        raw[q] = (list(st_t[o:o + c]), st_s[o:o + c])
        o += c
    hits = {}
    for k in range(len(z["qi"])):
        hits.setdefault(qids[z["qi"][k]], []).append((tids[z["ti"][k]], float(z["log_evalue"][k]), int(z["rank"][k]),
                                                     float(z["score"][k]), float(z["score_rev"][k])))
    rows = []
    for q in qids:
        s = qseq[q]
        twins = set(by_seq[s])
        hl = sorted(hits.get(q, []), key=lambda r: r[2])
        self_rank = next((r[2] for r in hl if r[0] == q), None)
        self_lne = next((r[1] for r in hl if r[0] == q), None)
        top = hl[0][0] if hl else None
        rids, rsc = raw.get(q, ([], []))
        rows.append(dict(q=q, L=len(s), rev=q.startswith("rev_"), twins=len(twins) - 1, in_stage1=q in rids,
                         self_rank=self_rank, self_lnE=self_lne, top_is_self=top == q,
                         top_is_self_or_twin=top in twins if top else False,
                         self_emb=float(rsc[rids.index(q)]) if q in rids else None))
    def summ(sel):
        n = len(sel)
        if not n:
            return dict(n=0)
        le = np.array([r["self_lnE"] for r in sel if r["self_lnE"] is not None])
        return dict(n=n, self_in_stage1=sum(r["in_stage1"] for r in sel) / n,
                    self_ranked_first=sum(r["top_is_self"] for r in sel) / n,
                    self_or_twin_ranked_first=sum(r["top_is_self_or_twin"] for r in sel) / n,
                    self_reported=len(le) / n,
                    self_E_le_1e3=float((le <= math.log(1e-3)).sum() / n),
                    self_E_le_1=float((le <= 0).sum() / n),
                    self_E_median=float(np.exp(np.median(le))) if len(le) else None,
                    self_E_p90=float(np.exp(np.percentile(le, 90))) if len(le) else None)
    bins = {"21-100": (21, 100), "101-300": (101, 300), "301-966": (301, 966), "967+": (967, 10 ** 9),
            "<21": (0, 20)}
    out = dict(fasta=fasta, fasta_md5=AC.md5_file(fasta), fm_search_argv=run["fm_search_argv"], rc=run["rc"],
               cli_wall_s=run["cli_wall_s"], stage1=run["stage1"],
               stage2_stats={k: v for k, v in run["stage2"]["stats"].items() if k != "q90"},
               all=summ(rows), annotated=summ([r for r in rows if not r["rev"]]),
               reversed=summ([r for r in rows if r["rev"]]),
               bins={g: summ([r for r in rows if lo <= r["L"] <= hi]) for g, (lo, hi) in bins.items()},
               misses=[r for r in rows if not r["top_is_self_or_twin"]][:40],
               note="'self ranked first' (the top hit is the query itself, or an identical-sequence twin) is the "
                    "measure for esm-codes (self-hit study, P6 note); self E <= 1e-3 is not expected for short or "
                    "homogeneous sequences")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-self-fasta")
    m.add_argument("--n", type=int, default=1000)
    m.add_argument("--seed", type=int, default=8)
    m.add_argument("--out", required=True)
    an = sub.add_parser("analyze")
    an.add_argument("--a5-dir", required=True)
    an.add_argument("--main", required=True, type=lambda s: s.split(":"), help="CAPTURE_DIR:OUT_TSV")
    an.add_argument("--default", type=lambda s: s.split(":"), default=None)
    an.add_argument("--self", type=lambda s: s.split(":"), default=None, help="CAPTURE_DIR:OUT_TSV:FASTA")
    an.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "make-self-fasta":
        return make_self_fasta(a)
    return analyze(a)


if __name__ == "__main__":
    sys.exit(main())
