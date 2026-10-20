#!/usr/bin/env python
"""P8 A5 (plan FINAL v1, section 8 P8): calibration on FoldMatch encodings, exhaustive (no prefilter).

Data: the P0.5(d) benchmark, 1,465 queries x 50,000 targets (calib3di/results/{queries,targets}.fasta: 25,000 SCOP-
annotated targets and 25,000 reversed decoys), labels by the MMseqs2 benchmark rules (plan.py; acc_common.Bench).

  stage2   --encodings foldmatch: the database built by the production pipeline at cc55c01
           ('fm-search build sequences --residue-codes' of targets.fasta) and the production query encodings (the query
           pass of 'fm-search query sequences --scoring esm-codes', captured by a8_pipeline.py); every query against
           all 50,000 targets through the production Stage 2 (code_alignment.align_candidates_esm_codes, the CLI's
           function) with an explicit worker pool and a max_evalue that provably covers E <= 10 under every table /
           N the protocol uses (fm_stage2.coverage_max_evalue); the round-1 S_f / S_r / Q90 of every pair recorded by
           the read-only observer (fm_stage2.Observer), checked against the returned hits.
           --encodings research: the same production Stage 2 on the research encodings (evalue/encode/out/
           {queries,targets}.L2m_B8_s0.npz) - the control that the harness reproduces the research numbers.
  protocol the research FP-ratio protocol (bench_validate.py 'own' NEW table, reproduced in acc_common and checked
           there against the research code and stored results: acc_common selftest) on the recorded matrices: main
           (table 'main' on odd rows) and swap ('swap' on even rows) halves, all_insample, heldout_union; fold-
           cluster and query bootstrap CIs, per query-length bin, the fraction of queries with a false hit (fq);
           sensitivity at E <= 1e-3 (N = all residues); FP-ratio curves up to E <= 10; the A5 criteria
           (acc_common.A5_CRITERIA, copied verbatim from the plan's A5 row) -> PASS / FAIL (gate G8).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import acc_common as AC  # noqa: E402  (first: BLAS threads = 1)
import numpy as np  # noqa: E402

import fm_stage2 as F  # noqa: E402

XS_CURVE = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0)
RQ = "embcomp/evalue/encode/out/queries.L2m_B8_s0.npz"
RT = "embcomp/evalue/encode/out/targets.L2m_B8_s0.npz"
P05_ENC_Q = "p0/parity/enc/gpu.a5q.npz"
P05_ENC_T = "p0/parity/enc/gpu.a5t.npz"
P05_PARITY = "p0/parity/esm3_parity.json"


# =====================================================================================================================
# stage 2
# =====================================================================================================================
def fasta_order(path):
    recs = AC.read_fasta(path)
    return [h.split()[0] for h, _ in recs], [s for _, s in recs], [h for h, _ in recs]


def stage2(a) -> int:
    from foldmatch.search import scoring_system as SS
    out_dir = a.out_dir
    os.makedirs(out_dir, exist_ok=True)
    qfa = a.queries_fasta or AC.inp(AC.BENCH_IN["qfasta"])
    tfa = a.targets_fasta or AC.inp(AC.BENCH_IN["tfasta"])
    qids, qseqs_l, _ = fasta_order(qfa)
    tids, tseqs_l, theads = fasta_order(tfa)
    qlen = np.array([len(s) for s in qseqs_l], np.float64)
    tlen = np.array([len(s) for s in tseqs_l], np.float64)
    rev = np.array([t.startswith("rev_") for t in tids])
    N_all, N_ann = float(tlen.sum()), float(tlen[~rev].sum())
    tpos = {t: i for i, t in enumerate(tids)}
    rec = {"what": f"P8 A5 stage 2 ({a.encodings} encodings): production esm-codes Stage 2, every query x every "
                   "target, observed", "encodings": a.encodings, "queries_fasta": qfa, "targets_fasta": tfa,
           "queries_fasta_md5": AC.md5_file(qfa), "targets_fasta_md5": AC.md5_file(tfa),
           "nq": len(qids), "nt": len(tids), "N_all": N_all, "N_ann": N_ann, "n_rev": int(rev.sum())}
    tabs = AC.ship_tables()
    tab_ship = tabs["all"]
    cov = F.coverage_max_evalue(qlen, tab_ship, N_all, {k: tabs[k] for k in ("all", "main", "swap")},
                                {"N_ann": N_ann, "N_all": N_all}, 10.0)
    me = a.max_evalue if a.max_evalue else float(max(10.0, math.ceil(cov["required_max_evalue"] * 1.05)))
    rec["coverage"] = dict(cov, chosen_max_evalue=me)
    AC.log(f"max_evalue {me} (coverage of E <= 10 under every table / N needs {cov['required_max_evalue']:.3f}, "
           f"at {cov['at']})")
    emb = None
    if a.encodings == "foldmatch":
        from foldmatch.cli.search import resolve_esm_codes_database
        database = resolve_esm_codes_database(a.db, me)
        code_store, fetch, prepared = database.code_store, database.sequence_store.fetch, database.prepared
        N = database.total_residues
        rec["db"] = F.db_info(a.db)
        query_index, cap = F.query_index_from_capture(a.capture)
        qseq_cap = AC.read_json(os.path.join(a.capture, "query_sequences.json"))
        from foldmatch.utils.sequence import normalize_sequence
        same = [normalize_sequence(s) == qseq_cap.get(q) for q, s in zip(qids, qseqs_l)]
        rec["capture"] = dict(dir=a.capture, queries_f16_sha256=cap["sha256"], entries=len(cap["entries"]),
                              rows=cap["rows"], system_name=cap["system_name"],
                              sequences_equal_fasta=int(sum(same)))
        assert all(same) and set(cap["entries"]) == set(qids), "capture queries differ from the FASTA"
        qseqs = {q: qseq_cap[q] for q in qids}
        # exact embedding cosine of every pair (the database's normalised vectors x the run's query embeddings)
        qe = os.path.join(a.capture, "query_embeddings.npz")
        if os.path.exists(qe):
            z = np.load(qe)
            qpos = {q: i for i, q in enumerate(z["ids"].astype(str))}
            Q = np.ascontiguousarray(z["emb"][[qpos[q] for q in qids]], np.float32)
            Q /= np.linalg.norm(Q, axis=1, keepdims=True)
            vids, V = F.index_vectors(a.db)
            vpos = {t: i for i, t in enumerate(vids)}
            V = V[[vpos[t] for t in tids]]
            emb = Q @ V.T
            rec["exact_cos"] = dict(sha256=F.save_npz(os.path.join(out_dir, "cos.npz"), cos=emb.astype(np.float32)),
                                    note="cosine of the normalised query chain embedding (this run) and the "
                                         "database's normalised vector (FAISS reconstruct_n), float32 matmul")
    else:
        system = SS.get(SS.DEFAULT_ESM_CODES)
        prepared = SS.prepare(system, local_files_only=True)
        zq, zt = np.load(AC.inp(RQ)), np.load(AC.inp(RT))
        assert list(zq["ids"].astype(str)) == qids and list(zt["ids"].astype(str)) == tids
        tmp = a.tmp or os.path.join(out_dir, "tmp")
        os.makedirs(tmp, exist_ok=True)
        bin_p, q_p = os.path.join(tmp, "research_codes.bin"), os.path.join(tmp, "research_queries.f16")
        rec["research_encodings"] = dict(
            queries=AC.inp(RQ), targets=AC.inp(RT), queries_md5=AC.md5_file(AC.inp(RQ)),
            targets_md5=AC.md5_file(AC.inp(RT)),
            codes_bin_sha256=F.write_raw(bin_p, np.ascontiguousarray(zt["codes"], np.uint8)),
            queries_f16_sha256=F.write_raw(q_p, np.ascontiguousarray(zq["q"], np.float16)))
        toff, tl = zt["offsets"][:-1], zt["lengths"]
        assert np.array_equal(tl, tlen.astype(np.int64))
        code_store = F.ArrayCodeStore(bin_p, tids, toff, tl)
        query_index = F.query_index_from_arrays(q_p, qids, zq["offsets"][:-1], zq["lengths"], qseqs_l, system.name)
        qseqs = {q: s for q, s in zip(qids, qseqs_l)}
        tseq = {t: s for t, s in zip(tids, tseqs_l)}

        def fetch(ids):
            return {t: tseq[t] for t in ids if t in tseq}
        N = int(N_all)
    rec["db_residues"] = int(N)
    assert int(N) == int(N_all), (N, N_all)
    AC.log(f"stage 2: {len(qids)} x {len(tids)} pairs, workers {a.workers}, max_evalue {me}, N {N}")
    if a.limit_queries:
        qids = qids[:a.limit_queries]
        if emb is not None:
            emb = emb[:a.limit_queries]
        qlen = qlen[:a.limit_queries]
    R = F.run_stage2(query_index, qids, qseqs, tids, code_store, fetch, prepared, N, workers=a.workers,
                     max_evalue=me, emb=emb)
    AC.log(f"stage 2 done in {R['wall_s']:.1f}s: {R['stats'].get('hits')} hits, workers {R['stats'].get('workers')}, "
           f"pairs {R['stats'].get('candidate_pairs')}")
    M = F.assemble(R["obs"], qids, len(tids))
    H = F.hits_arrays(R["hits"], qids, tpos)
    chk = F.check_run(M, H, qlen, tab_ship, N, me)
    AC.log("run checks", {k: v for k, v in chk.items()})
    st = dict(R["stats"])
    q90d = st.pop("q90", {})
    rec.update(stage2=dict(wall_s=R["wall_s"], workers_requested=a.workers, pool=R["pool"], max_evalue=me, stats=st,
                           q90_in_stats=len(q90d)), checks=chk)
    rec["q90_stats_equal_observed"] = bool(all(q90d.get(q) == M["q90"][i] for i, q in enumerate(qids)
                                               if np.isfinite(M["q90"][i])))
    rec["matrices_sha256"] = F.save_npz(os.path.join(out_dir, "matrices.npz"), qids=np.array(qids),
                                        tids=np.array(tids), S_f=M["S_f"], S_r=M["S_r"], q90=M["q90"],
                                        panel_f=M["panel_f"], panel_r=M["panel_r"], n_surv=M["n_surv"],
                                        observed=M["observed"], skipped=M["skipped"])
    rec["hits_sha256"] = F.save_npz(os.path.join(out_dir, "hits.npz"), qids=np.array(qids), tids=np.array(tids), **H)
    if a.encodings == "foldmatch":
        rec["encoding_diagnostics"] = encoding_diagnostics(code_store, query_index, qids, tids)
    rec["provenance"] = AC.provenance()
    h = AC.write_json(rec, os.path.join(out_dir, "stage2.json"))
    AC.log(f"wrote {out_dir}/stage2.json sha256 {h}; checks ok={chk['ok']}")
    return 0 if chk["ok"] else 1


def encoding_diagnostics(code_store, query_index, qids, tids) -> dict:
    """Code-bit agreement of the production target codes with the research codes and with the P0.5 GPU codes, and the
    per-residue cosine of the production query encodings with the research and P0.5 GPU q (diagnostics, not gates;
    P0.5: GPU vs research code bits 0.99273 on the A5 targets)."""
    out = {}
    offs, lens = code_store.lookup_rows(list(tids))
    C = code_store.open_memmap()
    order = np.concatenate([np.arange(o, o + n) for o, n in zip(offs.tolist(), lens.tolist())])
    prod = np.asarray(C[order])
    Qm = query_index.memmap()
    qorder = np.concatenate([np.arange(query_index.entries[q].offset,
                                       query_index.entries[q].offset + query_index.entries[q].length) for q in qids])
    qprod = np.asarray(Qm[qorder], np.float32)
    for name, tp, qp in (("research", RT, RQ), ("p05_gpu", P05_ENC_T, P05_ENC_Q)):
        try:
            zt = np.load(AC.inp(tp))
            # both files store the records in the benchmark order (ids, offsets, lengths): checked, not assumed
            assert list(zt["ids"].astype(str)) == list(tids), f"{tp}: target ids / order differ"
            assert np.array_equal(zt["lengths"], lens), f"{tp}: target lengths differ"
            ref = zt["codes"]
            assert ref.shape == prod.shape, (ref.shape, prod.shape)
            x = np.unpackbits(np.bitwise_xor(prod, ref), axis=1)
            per_res_identical = float((x.sum(1) == 0).mean())
            agree = 1.0 - float(x.sum()) / x.size
            # per item
            item = np.add.reduceat(x.sum(1), np.concatenate([[0], np.cumsum(lens)[:-1]])) / (64.0 * lens)
            out[f"target_code_bits_vs_{name}"] = dict(
                agreement=agree, bits=int(x.size), residues_identical=per_res_identical,
                items_identical=int((item == 0).sum()), item_min=float(1 - item.max()),
                item_median=float(1 - np.median(item)), source=AC.inp(tp))
            del x, ref
            zq = np.load(AC.inp(qp))
            assert list(zq["ids"].astype(str)) == list(qids), f"{qp}: query ids / order differ"
            rq = np.asarray(zq["q"], np.float32)
            assert rq.shape == qprod.shape, (rq.shape, qprod.shape)
            cos = (rq * qprod).sum(1) / (np.linalg.norm(rq, axis=1) * np.linalg.norm(qprod, axis=1) + 1e-30)
            out[f"query_q_cosine_vs_{name}"] = dict(
                residues=int(len(cos)), min=float(cos.min()), p1=float(np.percentile(cos, 1)),
                median=float(np.median(cos)), mean=float(cos.mean()), frac_ge_0_999=float((cos >= 0.999).mean()),
                rows_bitwise_equal=float(np.all(np.asarray(zq["q"]).view(np.uint16)
                                                == np.asarray(Qm[qorder]).view(np.uint16), axis=1).mean()),
                source=AC.inp(qp))
        except Exception as e:  # noqa: BLE001
            out[f"vs_{name}_error"] = repr(e)
    return out


# =====================================================================================================================
# protocol
# =====================================================================================================================
def load_run(d):
    z = np.load(os.path.join(d, "matrices.npz"))
    h = np.load(os.path.join(d, "hits.npz"))
    return z, h


def curves(B, S, tabs, q90) -> dict:
    """FP ratio (real null, N = annotated residues) at XS_CURVE for main / swap / all_insample / heldout_union;
    production-E counts (shipped table, N = all residues = the database's total_residues) of real-null FPs, reversed
    decoy hits and TPs, all queries."""
    idx = np.arange(B.nq)
    ev, od = idx[idx % 2 == 0], idx[idx % 2 == 1]
    cols = np.arange(B.nt)
    out = {"xs": list(XS_CURVE)}
    per = {}
    for e, (sp, rows) in {"main": ("main", od), "swap": ("swap", ev), "all_insample": ("all", idx)}.items():
        per[e] = AC.fp_counts(B, S, tabs[sp], rows, cols, q90, XS_CURVE)
        out[e] = {str(x): dict(fp=int(per[e][x].sum()), ratio=float(per[e][x].sum() / (len(rows) * x)),
                               fq=float((per[e][x] > 0).mean())) for x in XS_CURVE}
    out["heldout_union"] = {}
    for x in XS_CURVE:
        c = int(per["main"][x].sum() + per["swap"][x].sum())
        n = np.zeros(B.nq, bool)
        n[od] = per["main"][x] > 0
        n[ev] = per["swap"][x] > 0
        out["heldout_union"][str(x)] = dict(fp=c, ratio=c / (B.nq * x), fq=float(n.mean()))
    L = AC.ln_evalue(tabs["all"], B.qlen, S, q90, B.N_all)
    out["production_E"] = {"N": B.N_all, "table": "all (shipped)"}
    for x in XS_CURVE:
        m = L <= math.log(x)
        out["production_E"][str(x)] = dict(real_fp=int((m & B.REAL).sum()), real_ratio=float((m & B.REAL).sum()
                                                                                            / (B.nq * x)),
                                           rev_fp=int((m & B.REVM).sum()), tp=int((m & B.TP).sum()),
                                           tp_frac=float((m & B.TP).sum() / B.TP.sum()))
    return out


def reversed_null(B, S, tabs, q90) -> dict:
    """rlib.evaluate on each held-out half and in-sample: the reversed-decoy null FP ratios (N = reversed residues),
    top-1 query, ratio without the top-1 query, top-5% share (research diagnostics)."""
    out = {}
    for name, tb, rows in (("main", tabs["main"], B.splits["main"][1]), ("swap", tabs["swap"], B.splits["swap"][1]),
                           ("all", tabs["all"], np.arange(B.nq))):
        o, _ = AC.evaluate(B, S, AC.tab_lnc(tb), rows, row_shift=q90)
        out[name] = {"real": o["real"], "rev": o["rev"], "sens_1e3": o["sens"], "sens_bins": o["sens_bins"],
                     "budget": o["budget"]}
    return out


def boundary_check(B, S_new, S_ref, q90_new, q90_ref, tabs) -> dict:
    """Per-query FP counts (protocol definitions) of a run against the research scores: identical except pairs at a
    threshold boundary (|ln E_ref - ln x| < 1e-4, S16); also TP decisions at 1e-3 (N = all residues)."""
    idx = np.arange(B.nq)
    ev, od = idx[idx % 2 == 0], idx[idx % 2 == 1]
    out = {}
    for e, (sp, rows, N, mask, xs) in {
            "main": ("main", od, B.N_ann, B.REAL, AC.XS), "swap": ("swap", ev, B.N_ann, B.REAL, AC.XS),
            "all_insample": ("all", idx, B.N_ann, B.REAL, AC.XS),
            "sens_main": ("main", od, B.N_all, B.TP, (1e-3,)), "sens_swap": ("swap", ev, B.N_all, B.TP, (1e-3,))
    }.items():
        Ln = AC.ln_evalue(tabs[sp], B.qlen[rows], S_new[rows], q90_new[rows], N)
        Lr = AC.ln_evalue(tabs[sp], B.qlen[rows], S_ref[rows], q90_ref[rows], N)
        mk = mask[rows]
        o = {}
        for x in xs:
            a_ = (Ln <= math.log(x)) & mk
            b_ = (Lr <= math.log(x)) & mk
            diff = a_ ^ b_
            bnd = np.abs(Lr - math.log(x)) < 1e-4
            o[str(x)] = dict(new=int(a_.sum()), ref=int(b_.sum()), differing_pairs=int(diff.sum()),
                             differing_not_at_boundary=int((diff & ~bnd).sum()),
                             max_abs_dlnE_differing=float(np.abs(Ln - Lr)[diff].max()) if diff.any() else 0.0)
        out[e] = o
    return out


def research_vs_run(B, S, q90, S_ref, q90_ref, tabs, ns=200_000) -> dict:
    """Diagnostics of a run against the research scores (A4-style; not gates): dQ90, dS, d ln E on research hits
    (E <= 1, shipped table, N = annotated residues), hit-set agreement at E <= 1 and 1e-3."""
    out = {}
    d = q90 - q90_ref
    out["dQ90"] = dict(n=int(np.isfinite(d).sum()), max_abs=float(np.nanmax(np.abs(d))),
                       median_abs=float(np.nanmedian(np.abs(d))), p95_abs=float(np.nanpercentile(np.abs(d), 95)),
                       signed_median=float(np.nanmedian(d)))
    Ln = AC.ln_evalue(tabs["all"], B.qlen, S, q90, B.N_ann)
    Lr = AC.ln_evalue(tabs["all"], B.qlen, S_ref, q90_ref, B.N_ann)
    rh = Lr <= 0.0
    dl = np.abs(Ln - Lr)[rh]
    out["dlnE_research_hits_E1"] = dict(n=int(rh.sum()), median=float(np.median(dl)), p95=float(np.percentile(dl, 95)),
                                        max=float(dl.max()))
    dS = np.abs(S - S_ref)
    rng = np.random.default_rng(0)
    pick = rng.integers(0, dS.size, ns)
    out["dS_abs"] = dict(sample_pairs=ns, median=float(np.median(dS.ravel()[pick])),
                         p99=float(np.percentile(dS.ravel()[pick], 99)), research_hits_E1_median=float(np.median(dS[rh])),
                         research_hits_E1_p95=float(np.percentile(dS[rh], 95)))
    for x in (1.0, 1e-3):
        a_ = Ln <= math.log(x)
        b_ = Lr <= math.log(x)
        out[f"hit_agreement_E{x}"] = dict(run=int(a_.sum()), research=int(b_.sum()), both=int((a_ & b_).sum()),
                                          jaccard=float((a_ & b_).sum() / max((a_ | b_).sum(), 1)))
    return out


def s16_vs_research(M, B, S_ref_f, S_ref_r, q90_ref, tabs) -> dict:
    """The research-encodings control against the stored research matrices (S16 bounds; A3 extended to all 1,465
    queries): S_f / S_r |d| / max(|ref|, 1), |dQ90|, |d ln E| (shipped table, N annotated) over all pairs and over
    pairs with E <= 10."""
    out = {}
    for k, ref in (("S_f", S_ref_f), ("S_r", S_ref_r)):
        new = M[k].astype(np.float64)
        rel = np.abs(new - ref) / np.maximum(np.abs(ref), 1.0)
        out[k] = dict(max_rel=float(rel.max()), p99999_rel=float(np.quantile(rel, 0.99999)),
                      n_gt_5em5=int((rel > 5e-5).sum()), bitwise_equal_frac=float((new == ref).mean()))
        del rel
    d = np.abs(M["q90"] - q90_ref)
    out["Q90"] = dict(max_abs=float(d.max()), n_gt_1em4=int((d > 1e-4).sum()))
    S = M["S_f"].astype(np.float64) - M["S_r"].astype(np.float64)
    Sr = S_ref_f - S_ref_r
    Ln = AC.ln_evalue(tabs["all"], B.qlen, S, M["q90"], B.N_ann)
    Lr = AC.ln_evalue(tabs["all"], B.qlen, Sr, q90_ref, B.N_ann)
    dl = np.abs(Ln - Lr)
    near = Lr <= math.log(10.0)
    out["lnE"] = dict(max_abs_all=float(dl.max()), max_abs_E_le_10=float(dl[near].max()), n_E_le_10=int(near.sum()),
                      n_gt_1em4_E_le_10=int((dl[near] > 1e-4).sum()))
    out["s16_pass"] = bool(out["S_f"]["max_rel"] <= 5e-5 and out["S_r"]["max_rel"] <= 5e-5
                           and out["Q90"]["max_abs"] <= 1e-4 and out["lnE"]["max_abs_E_le_10"] <= 1e-4)
    return out


def protocol(a) -> int:
    t0 = time.time()
    B = AC.Bench()
    tabs = AC.ship_tables()
    z, h = load_run(a.run_dir)
    st = AC.read_json(os.path.join(a.run_dir, "stage2.json"))
    assert list(z["qids"].astype(str)) == list(B.qids) and list(z["tids"].astype(str)) == list(B.tids)
    S = z["S_f"].astype(np.float64) - z["S_r"].astype(np.float64)
    q90 = z["q90"].astype(np.float64)
    assert np.isfinite(S).all() and np.isfinite(q90).all(), "unobserved queries (S13b?)"
    res = {"what": f"P8 A5: calibration on {st['encodings']} encodings (exhaustive: 1,465 x 50,000 through the "
                   "production Stage 2; research FP-ratio protocol)",
           "encodings": st["encodings"], "run_dir": a.run_dir,
           "inputs": {"stage2_json_sha256": AC.sha256_file(os.path.join(a.run_dir, "stage2.json")),
                      "matrices_sha256": AC.sha256_file(os.path.join(a.run_dir, "matrices.npz")),
                      "hits_sha256": AC.sha256_file(os.path.join(a.run_dir, "hits.npz"))},
           "stage2_checks": st["checks"], "stage2": st["stage2"], "coverage": st["coverage"],
           "criteria": AC.A5_CRITERIA,
           "protocol_source": {k: AC.RESEARCH_SOURCES[k] for k in ("bench_validate.py", "rlib.py", "plan.py",
                                                                   "fit_final2.py")}}
    AC.log("protocol on the full matrices")
    proto = AC.bench_protocol(B, S, tabs, q90)
    verdict = AC.a5_verdict(proto)
    res["verdict"] = verdict
    res["A5"] = "PASS" if verdict["pass"] else "FAIL"
    for e in ("main", "swap", "all_insample", "heldout_union"):
        res[e] = proto[e]
    res["sensitivity"] = proto["sensitivity"]
    res["meta"] = proto["meta"]
    res["lines"] = [AC.protocol_line(e, proto[e]) for e in ("main", "swap", "all_insample", "heldout_union")]
    for line in res["lines"]:
        AC.log(line)
    AC.log(f"sensitivity @1e-3 main {proto['sensitivity']['main']['sens_1e3']:.5f} swap "
           f"{proto['sensitivity']['swap']['sens_1e3']:.5f}; A5 {res['A5']}")
    # hits-only protocol (coverage): the aligner's returned hits alone give the same per-query counts
    AC.log("protocol on the returned hits only (coverage check)")
    Sh = np.full_like(S, -np.inf)
    Sh[h["qi"], h["ti"]] = h["score"] - h["score_rev"]
    ph = AC.bench_protocol(B, Sh, tabs, q90, nboot=10)
    same = all(np.array_equal(ph["per_query"][e][x], proto["per_query"][e][x]) for e in proto["per_query"]
               for x in proto["per_query"][e])
    same_sens = all(ph["sensitivity"][sp] == proto["sensitivity"][sp] for sp in ("main", "swap"))
    res["hits_only_protocol_equal"] = dict(per_query_counts_equal=bool(same), sensitivity_and_budgets_equal=same_sens)
    del Sh
    res["curves"] = curves(B, S, tabs, q90)
    res["reversed_null_and_rlib"] = reversed_null(B, S, tabs, q90)
    # research reference numbers (bench_validate.json own NEW) and P0.5(d) preview
    bv = AC.read_json(AC.research_file("bench_validate.json"))
    ref = {e: {x: {k: bv["protocols"]["own"][e]["new"][x][k] for k in ("fp", "ratio", "ci_fold", "fq")}
               for x in ("0.01", "0.1", "1.0")} for e in ("main", "swap", "all_insample", "heldout_union")}
    ref["sensitivity"] = {sp: bv["sensitivity_heldout_own"][f"new/{sp}"] for sp in ("main", "swap")}
    res["research_reference"] = ref
    try:
        p05 = AC.read_json(AC.inp(P05_PARITY))["a5_preview"]
        res["p05_preview"] = p05
    except Exception as e:  # noqa: BLE001
        res["p05_preview_error"] = repr(e)
    # vs the research scores
    AC.log("diagnostics against the research scores")
    S_ref_f = np.load(AC.inp("embcomp/evalue/score/out/S_D_fwd.npy")).astype(np.float64)
    S_ref_r = np.load(AC.inp("embcomp/evalue/score/out/S_D_rev.npy")).astype(np.float64)
    q90_ref = np.load(AC.inp(AC.BENCH_IN["q90"]))
    S_ref = S_ref_f - S_ref_r
    res["vs_research"] = research_vs_run(B, S, q90, S_ref, q90_ref, tabs)
    res["vs_research_counts"] = boundary_check(B, S, S_ref, q90, q90_ref, tabs)
    if st["encodings"] == "research":
        res["s16_vs_research"] = s16_vs_research(z, B, S_ref_f, S_ref_r, q90_ref, tabs)
        exp = AC.SELFTEST_EXPECT["all_insample_fp"]
        got = {x: proto["all_insample"][x]["fp"] for x in ("0.01", "0.1", "1.0")}
        bc = res["vs_research_counts"]
        no_unexplained = all(v["differing_not_at_boundary"] == 0 for e in bc.values() for v in e.values())
        res["control"] = dict(all_insample_fp=got, research_fp=exp, equal=got == exp,
                              every_difference_at_a_boundary=bool(no_unexplained),
                              ok=bool(no_unexplained and res["s16_vs_research"]["s16_pass"]))
        AC.log("research control", res["control"])
    if st["encodings"] == "foldmatch":
        res["encoding_diagnostics"] = st.get("encoding_diagnostics")
    res["provenance"] = AC.provenance()
    res["provenance_stage2"] = st.get("provenance")
    res["wall_s"] = round(time.time() - t0, 1)
    out = a.out or os.path.join(a.run_dir, f"A5_{st['encodings']}.json")
    hsh = AC.write_json(res, out)
    with open(out.replace(".json", ".txt"), "w") as fh:
        fh.write(f"A5 ({st['encodings']} encodings): {res['A5']}\n")
        for line in res["lines"]:
            fh.write(line + "\n")
        fh.write(f"sensitivity@1e-3 main {proto['sensitivity']['main']['sens_1e3']:.5f} swap "
                 f"{proto['sensitivity']['swap']['sens_1e3']:.5f}\n")
        for sp in ("main", "swap"):
            fh.write(f"{sp}: {verdict[sp]}\n")
        fh.write(f"json {out} sha256 {hsh}\n")
    AC.log(f"wrote {out} sha256 {hsh}: A5 {res['A5']}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage2")
    s.add_argument("--encodings", choices=("foldmatch", "research"), required=True)
    s.add_argument("--db", help="FoldMatch database (foldmatch encodings)")
    s.add_argument("--capture", help="a8_pipeline.py capture dir with the production query encodings")
    s.add_argument("--workers", type=int, required=True)
    s.add_argument("--max-evalue", type=float, default=None, help="default: the coverage bound for E <= 10")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--tmp", default=None, help="node-local dir for the research-encoding raw files")
    s.add_argument("--queries-fasta", default=None)
    s.add_argument("--targets-fasta", default=None)
    s.add_argument("--limit-queries", type=int, default=0, help="smoke tests only")
    p = sub.add_parser("protocol")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "stage2":
        if a.encodings == "foldmatch" and not (a.db and a.capture):
            ap.error("--encodings foldmatch needs --db and --capture")
        return stage2(a)
    return protocol(a)


if __name__ == "__main__":
    sys.exit(main())
