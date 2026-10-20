#!/usr/bin/env python
"""P8 A9 (long sequences), step 3: production Stage 1 + Stage 2 of one query set against one A9 database, over EVERY
pair (the P0.7 slice is exhaustive). NERSC CPU node (EPYC 7763, 128 cores).

Tasks (query set x database; matrices in the research column order of p0/long/out/index.npz):
  nq_long  the 1,465 benchmark queries x DB_LONG (long_t 2,000 annotated + long_r 2,000 long decoys, 1,000-3,000 aa)
           = P0.7 nq_long -> A9 criterion (1)
  lq_ben   the 3,000 long queries (300-3,000 aa) x DB_BEN (the 30,000 lq_ben columns) = P0.7 lq_ben -> criterion (2)
  nq_ben   the 1,465 benchmark queries x DB_BEN = the P0.7 'A5 reference' (normal targets) columns of S_D
  --part k --nparts K: the queries split by LPT over their cells (length x (panel + database residues)).

Everything that computes is FoldMatch cc55c01, called as 'fm-search query sequences --scoring esm-codes' calls it:
  * S.resolve_esm_codes_database(db, max_evalue): the read-only resolution of P6.2 (store, manifest, registry,
    check_compatible, kernel, bundle). Then the FAISS count check.
  * query encodings: the production query pass of a9_qpass.py (its memmap and entries, reopened as a
    QueryEncodingIndex). The database must name the system that pass used.
  * Stage 1: EmbeddingDatabase.search_by_embeddings on the database's flat (exact) index with top_k = every database
    entry and no --threshold, so every pair is a candidate, in FoldMatch's own order and with its embedding scores.
  * Stage 2: code_alignment.align_candidates_esm_codes on a CodeAlignmentPool of --workers spawn workers, at the
    esm-codes defaults (identity 0, coverage 0, block / task / batch sizes, max_pair_cells, panel 'auto') except
    max_evalue (default 100, at N = the database residues). The hits are written with output.write_aligned_results.
  * The once-per-run calibrated-domain warning and the INFO summary are captured from FoldMatch's logging.
The full matrices: the hits cover only E <= max_evalue, so the round-1 values of every pair are recorded from inside
the run. code_alignment._statistics (S9-S11 in the parent, once per query) is wrapped: the original runs unchanged,
then the wrapper copies the query's S_f, S_r, ln E, Q90 and panel S_f / S_r arrays. This does not touch FoldMatch's
files, and nothing in the run changes. The recording is checked against the public outputs:
  * every hit's raw / rawrev / log_evalue / q90 equals the recorded value bit for bit;
  * the hits are exactly the recorded pairs with ln E <= ln(max_evalue), less only the empty alignments;
  * Q90 = code_statistics.q90(panel S) = np.quantile(f64 S, 0.9), the fit_final2.py rule;
  * ln E = code_statistics.log_evalue(S - Q90, m, N) bit for bit, and the research arithmetic (bench_validate L,
    shipped 'all' table) within float rounding.
Checks of S13 and the warning (A9 criterion 2): the expected ecal of every hit is computed independently: query
21-966 aa with <= 1% non-standard residues; target <= 3,000 aa with <= 1% non-standard residues; target with >= 2
distinct codes, read from the code store. Every hit of a query > 966 aa must have ecal=0. Exactly one domain warning
must appear, with the counts recomputed here.
Outputs: A9/stage2/<task>_p<k>of<K>/{scores.npz, hits.tsv, stage2.json, stage2.log}.
Usage:  python a9_stage2.py --task nq_long [--part 0 --nparts 1] [--workers 128] [--max-evalue 100]
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acc_common as C  # noqa: E402  (first: it sets the BLAS thread variables before numpy loads)
import numpy as np  # noqa: E402

TASKS = {"nq_long": ("nq", "long", "db_long"), "nq_ben": ("nq", "ben", "db_ben"), "lq_ben": ("lq", "ben", "db_ben")}
DBP = {"long": "db/long/L", "ben": "db/ben/B"}
FIELDS = ("query,target,embscore,evalue,bits,raw,rawrev,q90,rawadj,ecal,qlen,tlen,fident,alnlen,mismatch,gapopen,"
          "qstart,qend,tstart,tend,qcov,tcov")
DOMAIN_MSG = "E-values are calibrated only inside the calibrated domain"
WARN_RE = re.compile(
    r"This run: (\d+) queries shorter than \d+ aa, (\d+) longer than \d+ aa, (\d+) with too many non-standard "
    r"residues; (\d+) of (\d+) hits with ecal=0 \((\d+) for the query, (\d+) target too long, (\d+) target "
    r"non-standard, (\d+) target with too few distinct codes\); (\d+) fully degenerate queries")
WARN_KEYS = ("short", "long", "nonstandard", "ecal0", "hits", "query_outside", "target_long", "target_nonstandard",
             "target_low_distinct", "degenerate")


class Capture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.INFO)
        self.records = []

    def emit(self, record):
        self.records.append(dict(t=record.created, level=record.levelname, logger=record.name,
                                 message=record.getMessage()))


def lpt_parts(cells, nparts):
    part = np.empty(len(cells), np.int64)
    load = np.zeros(nparts)
    for i in np.argsort(-np.asarray(cells, np.float64), kind="stable"):
        k = int(np.argmin(load))
        part[i] = k
        load[k] += cells[i]
    return part, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=sorted(TASKS), required=True)
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--max-evalue", type=float, default=100.0)
    a = ap.parse_args()
    qkey, dkey, ckey = TASKS[a.task]
    A9 = C.result_dir("A9")
    out = C.result_dir("A9", "stage2", f"{a.task}_p{a.part}of{a.nparts}")
    from foldmatch.cli import search as S
    from foldmatch.cli.args_utils import set_log_level
    from foldmatch.search import code_alignment as CA
    from foldmatch.search import code_statistics as cs
    from foldmatch.search import output
    from foldmatch.search.embedding_database import EmbeddingDatabase
    from foldmatch.search.query_encodings import QueryEncodingEntry, QueryEncodingIndex
    from foldmatch.types.api_types import FormatMode, LogLevel

    set_log_level(LogLevel.info)
    cap = Capture()
    logging.getLogger().addHandler(cap)
    prov = C.provenance(kernel=True)
    T = {"start": time.time()}
    inp = np.load(os.path.join(A9, "inputs", "a9_inputs.npz"))
    q_ids_all = inp[f"{qkey}_ids"].astype(str)
    q_len_all, q_ns_all = inp[f"{qkey}_len"], inp[f"{qkey}_nonstd"]
    col_ids = inp[f"{ckey}_ids"].astype(str)
    col_len, col_ns = inp[f"{ckey}_len"], inp[f"{ckey}_nonstd"]
    db_path = os.path.join(A9, DBP[dkey])

    # ---------------------------------------------------------------- production resolution (P6.2), FAISS count
    database = S.resolve_esm_codes_database(db_path, a.max_evalue)
    system = database.system
    N_db = database.total_residues
    edb = EmbeddingDatabase(db_path=db_path)
    edb.load()
    database.code_store.check_faiss_count(len(edb.db.chain_ids))
    n_db = len(edb.db.chain_ids)
    assert n_db == len(col_ids) and set(edb.db.chain_ids) == set(col_ids.tolist()), "database entries != columns"
    assert N_db == int(col_len.sum()), (N_db, int(col_len.sum()))
    T["resolve"] = time.time()

    # ---------------------------------------------------------------- the production query pass (a9_qpass.py)
    qdir = os.path.join(A9, "qpass", qkey)
    ent = C.read_json(os.path.join(qdir, "entries.json"))
    assert ent["system_name"] == system.name, (ent["system_name"], system.name)
    enc_sha = C.sha256_file(ent["path"])
    assert enc_sha == ent["sha256"], "query encodings changed since the query pass"
    entries = {q: QueryEncodingEntry(int(o), int(n), bytes.fromhex(h)) for q, (o, n, h) in ent["entries"].items()}
    qindex = QueryEncodingIndex(ent["path"], ent["rows"], ent["width"], entries, ent["system_name"])
    seqs = C.read_json(os.path.join(qdir, "sequences.json"))
    embz = np.load(os.path.join(qdir, "embeddings.npz"))
    assert embz["ids"].astype(str).tolist() == q_ids_all.tolist() == list(seqs), "query order"
    assert np.array_equal(np.array([len(seqs[q]) for q in q_ids_all]), q_len_all)

    # ---------------------------------------------------------------- this part's queries
    cells = q_len_all.astype(np.float64) * (system.panel_residues + N_db)
    part_of, load = lpt_parts(cells, a.nparts)
    sel = np.flatnonzero(part_of == a.part)
    qids = q_ids_all[sel].tolist()
    q_len, q_ns = q_len_all[sel], q_ns_all[sel]
    C.log(f"{a.task} part {a.part}/{a.nparts}: {len(qids)} queries ({int(q_len.sum()):,} residues) x {n_db} targets "
          f"(N = {N_db:,}); part loads {[f'{x:.3g}' for x in load]}")

    # ---------------------------------------------------------------- Stage 1 (exact, every database entry)
    res = edb.search_by_embeddings([(qids, embz["emb"][sel])], top_k=n_db)
    prefilter = {q: res[q] for q in qids}
    assert all(len(prefilter[q][0]) == n_db == len(set(prefilter[q][0])) for q in qids), "Stage 1 not exhaustive"
    query_sequences = {q: seqs[q] for q in qids}
    T["stage1"] = time.time()

    # ---------------------------------------------------------------- Stage 2 with the round-1 recorder
    REC = {}
    order_ok = []
    orig = CA._statistics

    def recording_statistics(q, run):
        orig(q, run)
        order_ok.append(q.cand_ids == list(prefilter[q.qid][0]))
        REC[q.qid] = (q.s_f.copy(), q.s_r.copy(), q.skipped.copy(), None if q.lne is None else q.lne.copy(),
                      q.q90, None if q.panel_f is None else q.panel_f.copy(),
                      None if q.panel_r is None else q.panel_r.copy(), int(q.n))

    fields = S._output_fields(FIELDS)
    stats = {}
    CA._statistics = recording_statistics
    try:
        with CA.CodeAlignmentPool(a.workers) as pool:
            aligned = CA.align_candidates_esm_codes(
                qindex, prefilter, database.code_store, database.sequence_store.fetch, database.prepared, N_db, pool,
                query_sequences=query_sequences, min_seq_identity=0.0, min_coverage=0.0, max_evalue=a.max_evalue,
                output_needs=fields, stats=stats)
    finally:
        CA._statistics = orig
    T["stage2"] = time.time()
    tsv = os.path.join(out, "hits.tsv")
    output.write_aligned_results(aligned, fields, tsv, FormatMode.headless_tsv)
    T["write"] = time.time()

    # ---------------------------------------------------------------- matrices in the research column order
    col_index = {t: j for j, t in enumerate(col_ids.tolist())}
    nq, nc = len(qids), len(col_ids)
    SF = np.full((nq, nc), np.nan, np.float32)
    SR = np.full((nq, nc), np.nan, np.float32)
    LNE = np.full((nq, nc), np.nan, np.float64)
    EMB = np.full((nq, nc), np.nan, np.float32)
    SKIP = np.zeros((nq, nc), bool)
    Q90 = np.full(nq, np.nan)
    PF = np.full((nq, system.panel_size), np.nan, np.float32)
    PR = np.full((nq, system.panel_size), np.nan, np.float32)
    for i, q in enumerate(qids):
        ids, sc = prefilter[q]
        idx = np.fromiter((col_index[t] for t in ids), np.int64, len(ids))
        s_f, s_r, skipped, lne, q90, pf, pr, n = REC[q]
        assert n == q_len[i]
        SF[i, idx], SR[i, idx], SKIP[i, idx], LNE[i, idx] = s_f, s_r, skipped, lne
        EMB[i, idx] = np.asarray(sc, np.float32)
        Q90[i], PF[i], PR[i] = q90, pf, pr

    # ---------------------------------------------------------------- checks of the recording
    chk = {}
    chk["every_query_recorded"] = len(REC) == nq
    chk["candidate_order_is_stage1_order"] = bool(all(order_ok)) and len(order_ok) == nq
    chk["all_pairs_scored"] = bool(np.isfinite(SF).all() and np.isfinite(SR).all() and np.isfinite(LNE).all())
    chk["no_skipped_pairs"] = int(SKIP.sum()) == 0 and stats.get("skipped_pairs") == 0
    chk["stats_counts"] = dict(candidate_pairs=stats.get("candidate_pairs") == nq * nc,
                               panel_pairs=stats.get("panel_pairs") == nq * system.panel_size,
                               queries=stats.get("queries") == nq, degenerate=stats.get("degenerate_queries") == [],
                               no_candidates=stats.get("queries_without_candidates") == 0)
    q90_stats = np.array([stats["q90"][q] for q in qids])
    q90_cs = np.array([cs.q90(PF[i].astype(np.float64) - PR[i].astype(np.float64), system.panel_quantile,
                              system.quantile_method) for i in range(nq)])
    q90_np = C.q90_of(PF.astype(np.float64) - PR.astype(np.float64))
    chk["q90"] = dict(recorded_eq_stats=bool(np.array_equal(Q90, q90_stats)),
                      eq_code_statistics_q90=bool(np.array_equal(Q90, q90_cs)),
                      eq_fit_final2_np_quantile=bool(np.array_equal(Q90, q90_np)))
    Sd = SF.astype(np.float64) - SR.astype(np.float64)
    # as _statistics calls it: per query, S' = S - Q90, the query length as an int, N = the database residues
    lne_cs = np.stack([cs.log_evalue(Sd[i] - Q90[i], int(q_len[i]), N_db, system) for i in range(nq)])
    tab = C.ship_tables()["all"]
    lne_res = C.ln_evalue(tab, q_len, Sd, Q90, float(N_db))
    reg_eq_ship = (tuple(float(x) for x in tab["center_len"]) == tuple(system.bin_centres)
                   and tuple(float(x) for x in tab["lam"]) == tuple(system.lam)
                   and tuple(float(x) for x in tab["lnK"]) == tuple(system.lnK))
    chk["lnE"] = dict(eq_code_statistics_log_evalue=bool(np.array_equal(LNE, lne_cs)),
                      vs_research_arithmetic_max_abs=float(np.max(np.abs(LNE - lne_res))),
                      registry_table_eq_ship_final_all=bool(reg_eq_ship))

    # ---------------------------------------------------------------- the hits (public output)
    ln_max = math.log(a.max_evalue)
    surv = LNE <= ln_max
    q_row = {q: i for i, q in enumerate(qids)}
    H = {k: [] for k in ("q", "c", "sf", "sr", "q90", "lne", "bits", "ecal", "qlen", "tlen", "alnlen", "fident",
                         "evalue")}
    for q, hits in aligned.items():
        i = q_row[q]
        for h in hits:
            m = h.metrics
            H["q"].append(i)
            H["c"].append(col_index[h.subject_id])
            H["sf"].append(m.score)
            H["sr"].append(m.score_rev)
            H["q90"].append(m.q90)
            H["lne"].append(m.log_evalue)
            H["bits"].append(m.bit_score)
            H["ecal"].append(m.ecal)
            H["qlen"].append(m.query_len)
            H["tlen"].append(m.subject_len)
            H["alnlen"].append(m.aln_len)
            H["fident"].append(m.identity_aln)
            H["evalue"].append(m.evalue)
    H = {k: np.asarray(v) for k, v in H.items()}
    hq, hc = H["q"].astype(np.int64), H["c"].astype(np.int64)
    nh = len(hq)
    hit_mask = np.zeros((nq, nc), bool)
    hit_mask[hq, hc] = True
    chk["hits"] = dict(
        n=nh, unique_pairs=int(hit_mask.sum()) == nh,
        raw_eq_recorded=bool(np.array_equal(H["sf"].astype(np.float32).view(np.uint32), SF[hq, hc].view(np.uint32))),
        rawrev_eq_recorded=bool(np.array_equal(H["sr"].astype(np.float32).view(np.uint32),
                                               SR[hq, hc].view(np.uint32))),
        lne_eq_recorded=bool(np.array_equal(H["lne"].astype(np.float64), LNE[hq, hc])),
        q90_eq_recorded=bool(np.array_equal(H["q90"].astype(np.float64), Q90[hq])),
        qlen_ok=bool(np.array_equal(H["qlen"], q_len[hq])), tlen_ok=bool(np.array_equal(H["tlen"], col_len[hc])),
        hits_subset_of_survivors=bool(np.all(surv[hq, hc])), survivors=int(surv.sum()),
        survivors_without_hit=int((surv & ~hit_mask).sum()), stats_survivors=stats.get("survivors"),
        stats_hits=stats.get("hits"), tsv_rows=sum(1 for _ in open(tsv)))
    chk["hits"]["survivors_minus_hits_are_empty_alignments_only"] = (
        chk["hits"]["survivors_without_hit"] == stats.get("survivors", -1) - nh)

    # ---------------------------------------------------------------- S13 (ecal) and the once-per-run warning
    d = system.calibrated_domain
    codes = database.code_store.open_memmap()
    offs, lens = database.code_store.lookup_rows(col_ids.tolist())
    assert np.array_equal(lens, col_len)
    t_distinct2 = np.array([bool(np.any(codes[o:o + m] != codes[o])) if m else False
                            for o, m in zip(offs.tolist(), lens.tolist())])
    q_frac = q_ns / q_len
    q_inside = (q_len >= d.query_min_length) & (q_len <= d.query_max_length) & (q_frac <= d.max_nonstandard_fraction)
    t_long = col_len > d.target_max_length
    t_nonstd = (col_ns / col_len) > d.max_nonstandard_fraction
    t_low = ~t_distinct2
    exp_ecal = q_inside[hq] & ~t_long[hc] & ~t_nonstd[hc] & ~t_low[hc]
    got_ecal = H["ecal"].astype(bool)
    e0 = ~exp_ecal
    expected = dict(short=int((q_len < d.query_min_length).sum()), long=int((q_len > d.query_max_length).sum()),
                    nonstandard=int((q_frac > d.max_nonstandard_fraction).sum()), ecal0=int(e0.sum()), hits=nh,
                    query_outside=int((e0 & ~q_inside[hq]).sum()), target_long=int((e0 & t_long[hc]).sum()),
                    target_nonstandard=int((e0 & t_nonstd[hc]).sum()), target_low_distinct=int((e0 & t_low[hc]).sum()),
                    degenerate=0)
    dom = [r for r in cap.records if r["level"] == "WARNING" and DOMAIN_MSG in r["message"]]
    parsed = None
    if len(dom) == 1:
        mt = WARN_RE.search(dom[0]["message"])
        parsed = dict(zip(WARN_KEYS, map(int, mt.groups()))) if mt else "unparsed"
    want_warning = (expected["short"] + expected["long"] + expected["nonstandard"] + expected["ecal0"]) > 0
    long_q = q_len[hq] > d.query_max_length
    chk["s13"] = dict(
        ecal_matches_expected=bool(np.array_equal(got_ecal, exp_ecal)), ecal_mismatches=int((got_ecal != exp_ecal).sum()),
        hits_of_queries_gt_966=int(long_q.sum()), queries_gt_966_with_hits=int(len(np.unique(hq[long_q]))),
        queries_gt_966=int((q_len > d.query_max_length).sum()),
        every_hit_of_a_query_gt_966_has_ecal0=bool(not np.any(got_ecal[long_q])),
        hits_ecal1=int(got_ecal.sum()), targets_with_lt2_distinct_codes=int(t_low.sum()),
        targets_gt_3000=int(t_long.sum()), targets_nonstandard_gt_1pct=int(t_nonstd.sum()),
        domain_warnings=len(dom), warning_expected=bool(want_warning),
        warning_once=(len(dom) == 1) if want_warning else (len(dom) == 0),
        warning_counts=parsed, expected_counts=expected,
        warning_counts_match=(parsed == expected) if want_warning else None,
        stats_query_domain=stats.get("query_domain"), stats_hit_domain=stats.get("hit_domain"))
    chk["s13"]["stats_match_expected"] = (
        stats.get("query_domain", {}).get("long") == expected["long"]
        and stats.get("hit_domain", {}).get("ecal0") == expected["ecal0"]
        and stats.get("hit_domain", {}).get("hits") == nh)

    def allbool(x):
        if isinstance(x, dict):
            return all(allbool(v) for k, v in x.items() if isinstance(v, (bool, dict)))
        return bool(x)
    ok = (allbool({k: chk[k] for k in ("every_query_recorded", "candidate_order_is_stage1_order", "all_pairs_scored",
                                        "no_skipped_pairs", "stats_counts", "q90")})
          and chk["lnE"]["eq_code_statistics_log_evalue"] and chk["lnE"]["registry_table_eq_ship_final_all"]
          and chk["lnE"]["vs_research_arithmetic_max_abs"] <= 1e-9
          and all(chk["hits"][k] for k in ("unique_pairs", "raw_eq_recorded", "rawrev_eq_recorded", "lne_eq_recorded",
                                           "q90_eq_recorded", "qlen_ok", "tlen_ok", "hits_subset_of_survivors",
                                           "survivors_minus_hits_are_empty_alignments_only"))
          and chk["hits"]["tsv_rows"] == nh
          and chk["s13"]["ecal_matches_expected"] and chk["s13"]["every_hit_of_a_query_gt_966_has_ecal0"]
          and chk["s13"]["warning_once"] and chk["s13"]["warning_counts_match"] in (True, None)
          and chk["s13"]["stats_match_expected"])

    np.savez(os.path.join(out, "scores.npz"), qids=np.array(qids), qrows=sel, col_ids=col_ids, S_f=SF, S_r=SR,
             lnE=LNE, emb=EMB, q90=Q90, panel_f=PF, panel_r=PR,
             **{f"hit_{k}": v for k, v in H.items() if k != "ecal"}, hit_ecal=got_ecal)
    T["end"] = time.time()
    rec = dict(
        title=f"P8 A9 production Stage 1 + Stage 2: {a.task} part {a.part} of {a.nparts}", task=a.task,
        part=a.part, nparts=a.nparts, queries=nq, targets=nc, database=db_path, N_db=N_db,
        manifest=dict(path=str(database.code_store.manifest_path), sha256=C.sha256_file(str(database.code_store.manifest_path))),
        system=dict(name=system.name, digest=system.digest(), calibration_version=system.calibration_version),
        max_evalue=a.max_evalue, workers=a.workers, fields=FIELDS,
        query_pass=dict(dir=qdir, encodings=ent["path"], encodings_sha256=enc_sha,
                        qpass_json_sha256=C.sha256_file(os.path.join(qdir, "qpass.json"))),
        part_loads=load.tolist(), stats={k: v for k, v in stats.items() if k != "q90"},
        warnings=[r for r in cap.records if r["level"] == "WARNING"],
        stage2_info=[r["message"] for r in cap.records if r["message"].startswith("esm-codes Stage 2")],
        checks=chk, ok=bool(ok),
        wall_s=dict(resolve=T["resolve"] - T["start"], stage1=T["stage1"] - T["resolve"],
                    stage2=T["stage2"] - T["stage1"], write=T["write"] - T["stage2"], total=T["end"] - T["start"]),
        files=dict(scores_npz=C.sha256_file(os.path.join(out, "scores.npz")), hits_tsv=C.sha256_file(tsv)),
        provenance=prov)
    sha = C.write_json(rec, os.path.join(out, "stage2.json"))
    C.log("stage2", a.task, a.part, "ok" if ok else "NOT OK", {k: rec[k] for k in ("wall_s",)},
          {k: chk[k] for k in ("hits", "s13")}, "sha256", sha)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
