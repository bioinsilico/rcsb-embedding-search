#!/usr/bin/env python
"""Validation of auc1.py on the research 50k sample (1,465 queries x 50,000 targets) with the stored research scores
(NERSC CPU node; run as 'python auc1.py validate [--out DIR] [--quick]').

Research references (each md5-checked):
  R1  embcomp/evalue/score/summarize.py T2 'AUC1 per query (ties 0.5)' - its auc1() run now on the stored
      S_<A..D>_<fwd|rev>.npy (fwd-rev formed in float32, as summarize.py does) with labels.npz (plan.py rules), and the
      stored numbers in embcomp/evalue/score/results/summary.json (FP_all and FP_real means) - exact equality;
  R2  calib3di/auc1/reconcile/auc_final.py: its run() / aucs() on the stored S_D (S_f - S_r in float64 = the shipped
      score, and S_f), with its labels re-derived by the main() code transcribed below (FAM, FOLD, EXC, FP0 / FP1, the
      6 query-source columns, fam_den with CAP 4000), for the eight Table-6 protocols (P0 spec ... P1 paper and its
      FP-first / TP-first bounds): per-query AUC1 and AUC5 - exact equality;
  R3  calib3di/fix_eval/reconcile/recon.py T8 (tables_final.txt 'all 959 scorable queries'): the 3Di rescoring
      variants cb / cb3 / qd10 / cb_qd10 from fm-db/calib3di/rescore (the 'base' scores are not on NERSC any more),
      recon.auc1 (textually identical to summarize.auc1, checked) - per-query exact, mean to the 4 printed decimals.
auc1.py runs on the same scores two ways: dense rows (every query x target pair) and the hit-table path (a TSV of
every pair scoring at or above the query's 5th-FP score under every protocol, E written as '1E<-S>' so that
log10 E = -S exactly, bits = S), through read_hits / rows_from_table / evaluate_protocol.
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
import time

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import auc1 as A  # noqa: E402
import bench_common as BC  # noqa: E402
from bench_common import AC, log  # noqa: E402

ARCH = "/global/homes/j/jsegura/esm_codes_archive/research_files"
SUMMARIZE_PY = (f"{ARCH}/embcomp/evalue/score/summarize.py", "558f0aa282e7214874b9adbbe6d31cc4")
RECON_PY = (f"{ARCH}/calib3di/fix_eval/reconcile/recon.py", "c7410a44958d2d28626fa95d99adacf3")
SUMMARY_JSON = "/pscratch/sd/j/jsegura/embcomp/evalue/score/results/summary.json"
SCORE_OUT = "/pscratch/sd/j/jsegura/embcomp/evalue/score/out"
RESCORE = "/pscratch/sd/j/jsegura/fm-db/calib3di/rescore"
T8_ALL = {"base": 0.6935, "cb": 0.7259, "cb3": 0.7263, "qd10": 0.7233, "cb_qd10": 0.7262}   # tables_final.txt T8
TABLE6 = [("P0 spec", "L0", "ntp", "half"), ("+exceptions", "Lexc", "ntp", "half"), ("+self removed", "Lself", "ntp",
          "half"), ("+famsize den", "L0", "code", "half"), ("+random ties", "L0", "ntp", "rnd"),
          ("P1 paper", "L1", "code", "rnd"), ("P1 FP-first", "L1", "code", "pes"), ("P1 TP-first", "L1", "code", "opt")]
TIE_MAP = {"half": "half", "rnd": "expect", "pes": "pes", "opt": "opt"}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def func_source(path, name):
    """The text of a top-level 'def name' block."""
    txt = open(path).read()
    m = re.search(rf"^def {name}\(.*?(?=^\S)", txt, re.S | re.M)
    return m.group(0).strip()


def research_labels(AF, q_headers, t_headers, t_ids):
    """auc_final.main() lines 'labels' .. 'fam_den', transcribed (sccs_of / lvl / PROP / CLASS_E / CAP from AF)."""
    q_sccs = [AF.sccs_of(h) for h in q_headers]
    T_sccs = [AF.sccs_of(h) for h in t_headers]
    q_ids = [h.split()[0] for h in q_headers]
    q_src = np.array(["MMseqs2" if " |" in h else "SCOP25" for h in q_headers])
    nq, nt = len(q_sccs), len(T_sccs)
    is_rev = np.array([t.startswith("rev_") for t in t_ids])
    is_ann = ~is_rev
    vocab = {}

    def inc(lists, n):
        r, c = [], []
        for i, labs in enumerate(lists):
            for l_ in sorted(set(AF.lvl(s, n) for s in labs)):
                r.append(i)
                c.append(vocab.setdefault((n, l_), len(vocab)))
        return r, c

    def shared(n):
        qr, qc = inc(q_sccs, n)
        tr, tc = inc(T_sccs, n)
        m = len(vocab)
        Q = sp.csr_matrix((np.ones(len(qr), np.float32), (qr, qc)), shape=(nq, m))
        Tm = sp.csr_matrix((np.ones(len(tr), np.float32), (tr, tc)), shape=(nt, m))
        return Q, Tm
    Qf, Tf = shared(4)
    FAM = (Qf @ Tf.T).toarray() > 0
    Qs, Ts = shared(3)
    Qd, Td = shared(2)
    FOLD = (Qd @ Td.T).toarray() > 0
    q_prop = np.array([any(AF.PROP.match(s) for s in l_) for l_ in q_sccs])
    q_e = np.array([any(AF.CLASS_E.match(s) for s in l_) for l_ in q_sccs])
    t_prop = np.array([any(AF.PROP.match(s) for s in l_) for l_ in T_sccs])
    t_e = np.array([any(AF.CLASS_E.match(s) for s in l_) for l_ in T_sccs])
    EXC = (q_prop[:, None] & t_prop[None, :]) | q_e[:, None] | t_e[None, :]
    FP0 = is_rev[None, :] | (is_ann[None, :] & ~FOLD)
    FP1 = is_rev[None, :] | (is_ann[None, :] & ~FOLD & ~EXC)
    q_acc = {q.rsplit("_", 1)[0]: i for i, q in enumerate(q_ids) if q_src[i] == "MMseqs2"}
    self_cols = [j for j, t in enumerate(t_ids) if t.startswith("annotated_") and t[10:] in q_acc]
    db_all = np.ones(nt, bool)
    db_noself = db_all.copy()
    db_noself[self_cols] = False
    fam_cols = [v for (n, l_), v in vocab.items() if n == 4]
    Qf_c, Tf_c = Qf.tocsc()[:, fam_cols], Tf.tocsc()[:, fam_cols]

    def fam_den(dbm):
        size = np.asarray(Tf_c[dbm].sum(0)).ravel()
        return np.minimum(np.asarray(Qf_c @ size).ravel(), AF.CAP)
    LS = {"L0": (FP0, db_all), "Lexc": (FP1, db_all), "Lself": (FP0, db_noself), "L1": (FP1, db_noself)}
    return dict(FAM=FAM, LS=LS, fam_den=fam_den, is_rev=is_rev, self_cols=[t_ids[j] for j in self_cols],
                q_src=q_src)


def dense_rows(LS, kind, ann, trank, nq, nt):
    return dict(qi=np.repeat(np.arange(nq, dtype=np.int64), nt), tcode=np.tile(np.arange(nt, dtype=np.int64), nq),
                kind=np.tile(kind, nq), ann=np.tile(ann, nq), trank=np.tile(trank, nq),
                file_pos=np.arange(nq * nt, dtype=np.int64))


def compare(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    nan_eq = bool(np.array_equal(np.isnan(a), np.isnan(b)))
    m = ~np.isnan(a) & ~np.isnan(b)
    d = np.abs(a[m] - b[m])
    return dict(nan_pattern_equal=nan_eq, n=int(m.sum()), n_bit_equal=int((a[m] == b[m]).sum()),
                max_abs_diff=float(d.max()) if d.size else 0.0, exact=bool(nan_eq and (a[m] == b[m]).all()))


def my_protocol(name, ls, den, tie):
    exc = ls in ("Lexc", "L1")
    excl = "global" if ls in ("Lself", "L1") else "none"
    return A.Protocol(name, sort="evalue:asc", den="ntp" if den == "ntp" else "famsize", ties=(TIE_MAP[tie],),
                      exclude_self=excl, exceptions=exc, drop_gt1=False)


def run(args):
    t_start = time.time()
    os.makedirs(args.out, exist_ok=True)
    R = {"checks": {}}
    # ------------------------------------------------------------------ research code, md5-checked
    af_path = AC.research_file("auc_final.py")
    src = {"auc_final.py": (af_path, AC.RESEARCH_SOURCES["auc_final.py"][1]),
           "summarize.py": SUMMARIZE_PY, "recon.py": RECON_PY}
    R["research_sources"] = {}
    for k, (p, want) in src.items():
        got = AC.md5_file(p)
        R["research_sources"][k] = dict(path=p, md5=got, expected=want, ok=(want is None or got == want))
        assert want is None or got == want, (k, got, want)
    AF = load_module("auc_final_research", af_path)
    SU = load_module("summarize_research", SUMMARIZE_PY[0])
    R["research_sources"]["recon.auc1 == summarize.auc1 (source text)"] = (
        func_source(RECON_PY[0], "auc1") == func_source(SUMMARIZE_PY[0], "auc1"))
    stored = BC.read_json(SUMMARY_JSON)["T2"]
    R["research_sources"]["summary.json"] = dict(path=SUMMARY_JSON, md5=AC.md5_file(SUMMARY_JSON))

    B = AC.Bench()
    nq, nt = B.nq, B.nt
    t_ids = list(B.tids)
    # my label store for the sample (research tokenisation; the evaluator regex parses the same, checked)
    LS, tids_b, info = BC.labels_from_headers(B.q_headers, B.qlen.astype(int), B.t_headers, B.tlen.astype(int),
                                              query_rule="research", target_rule="research")
    LSe, _, info_e = BC.labels_from_headers(B.q_headers, B.qlen.astype(int), B.t_headers, B.tlen.astype(int))
    R["labels"] = dict(info=info, evaluator_rule_info=info_e)
    kind, ann = LS.classify(tids_b)
    trank = np.empty(nt, np.int64)
    trank[np.argsort(tids_b, kind="stable")] = np.arange(nt)
    rows0 = dense_rows(LS, kind, ann, trank, nq, nt)
    st = LS.pair_status(rows0["qi"], rows0["kind"], rows0["ann"]).reshape(nq, nt)
    R["labels"]["equal_to_labels_npz"] = dict(TP=bool(np.array_equal(st == BC.ST_TP, B.TP)),
                                              FP=bool(np.array_equal(st == BC.ST_FP, B.FP)),
                                              FPA=bool(np.array_equal((st == BC.ST_FP) & ~B.rev[None, :], B.REAL)))
    del st
    RL = research_labels(AF, B.q_headers, B.t_headers, t_ids)
    R["labels"]["research_self_cols"] = RL["self_cols"]
    R["labels"]["my_global_self"] = [LS.ann_ids[a].decode() for a in LS.q_self[LS.q_self >= 0]]
    R["labels"]["research_FAM_equals_my_TP"] = bool(np.array_equal(RL["FAM"], B.TP))
    log("labels ready", R["labels"]["equal_to_labels_npz"])

    # ------------------------------------------------------------------ R1: summarize.py T2
    sets = ["D"] if args.quick else ["A", "B", "C", "D"]
    R["R1_T2"] = {}
    hasTP = B.TP.any(1)
    dense_cache = {}
    stc = {}                        # pair statuses depend on the labels only: shared by every score matrix
    for s in sets:
        f = np.load(f"{SCORE_OUT}/S_{s}_fwd.npy")
        r = np.load(f"{SCORE_OUT}/S_{s}_rev.npy")
        for name, M32 in ((f"{s}_fwd", f), (f"{s}_fwd-rev", f - r)):
            M = M32.astype(np.float64)
            rows = dict(rows0, log10e=-M.ravel(), bits=M.ravel())
            rec = {}
            for fpm, FPm in (("fp_all", B.FP), ("fp_real", B.REAL)):
                ref = SU.auc1(M32, B.TP, FPm)
                ref_mean = float(np.nanmean(ref[hasTP]))
                P = A.Protocol(f"T2_{fpm}", sort="evalue:asc", ties=("half",), fp_mode="all" if fpm == "fp_all"
                               else "real", drop_gt1=False)
                res = A.evaluate_protocol(LS, rows, P, stc, dense_cache.setdefault(name, {}))
                mine = res["half"]["auc1"]
                my_mean = float(np.nanmean(mine[res["evaluated"]]))
                rec[fpm] = dict(stored=stored[name][fpm], research_code_now=ref_mean, mine=my_mean,
                                stored_eq_research=ref_mean == stored[name][fpm], mine_eq_stored=my_mean ==
                                stored[name][fpm], per_query=compare(mine, ref), n=int(res["evaluated"].sum()))
            R["R1_T2"][name] = rec
            log("R1", name, {k: (v["stored"], v["mine"], v["mine_eq_stored"], v["per_query"]["exact"])
                             for k, v in rec.items()})
            if name != "D_fwd" and name != "D_fwd-rev":
                dense_cache.pop(name, None)
        del f, r
    # ------------------------------------------------------------------ R2: auc_final Table-6 protocols
    R["R2_auc_final"] = {}
    S_fmr = B.research_scores("D", "fmr")                     # float64(S_f) - float64(S_r): the shipped score
    S_fwd = B.research_scores("D", "fwd")
    allrows = np.arange(nq)
    tsv_rows = {}
    for sname, S in (("D_fmr_f64", S_fmr), ("D_fwd", S_fwd)):
        RES = {ls: AF.run(S, allrows, RL["FAM"], FPM, RL["is_rev"], dbm) for ls, (FPM, dbm) in RL["LS"].items()}
        DEN = {ls: RL["fam_den"](dbm) for ls, (FPM, dbm) in RL["LS"].items()}
        rows = dict(rows0, log10e=-S.ravel(), bits=S.ravel())
        sc, oc = stc, {}
        rec = {}
        thr = np.full(nq, np.inf)
        for label, ls, den, tie in TABLE6:
            d = DEN[ls] if den == "code" else None
            a1, a5 = AF.aucs(RES[ls], d, tie)
            P = my_protocol(label, ls, den, tie)
            dov = LS.denominators("famsize", P.exclude_self, cap=AF.CAP) if den == "code" else None
            res = A.evaluate_protocol(LS, rows, P, sc, oc, den_override=dov)
            m1, m5 = res[P.ties[0]]["auc1"], res[P.ties[0]]["auc5"]
            if den == "code":
                rec_den = compare(dov, d)
            else:
                rec_den = None
            rec[label] = dict(research_auc1=float(np.nanmean(a1)), research_auc5=float(np.nanmean(a5)),
                              mine_auc1=float(np.nanmean(m1)), mine_auc5=float(np.nanmean(m5)),
                              auc1=compare(m1, a1), auc5=compare(m5, a5), den=rec_den,
                              n=int((~np.isnan(a1)).sum()))
            # 5th-FP score per query under this protocol (for the hit-table round trip)
            FPM, dbm = RL["LS"][ls]
            for i in range(nq):
                fps = S[i][FPM[i] & dbm]
                if fps.size >= 5:
                    thr[i] = min(thr[i], np.partition(fps, fps.size - 5)[fps.size - 5])
                else:
                    thr[i] = -np.inf
        rec["max_rank_of_5th_fp_P1"] = int(max(x["rank5"] for x in RES["L1"] if x))
        R["R2_auc_final"][sname] = rec
        log("R2", sname, {k: (v["research_auc1"], v["mine_auc1"], v["auc1"]["exact"], v["auc5"]["exact"])
                          for k, v in rec.items() if isinstance(v, dict)})
        tsv_rows[sname] = (S, thr, RES, DEN)
        del rows, oc
    # ------------------------------------------------------------------ hit-table path (TSV round trip)
    R["tsv_round_trip"] = {}
    for sname in ("D_fmr_f64",):
        S, thr, RES, DEN = tsv_rows[sname]
        path = os.path.join(args.out, f"sample_{sname}.tsv")
        n_rows = 0
        with open(path, "w") as fh:
            fh.write("query\ttarget\tevalue\tbits\n")
            for i in range(nq):
                js = np.flatnonzero(S[i] >= thr[i])
                for j in js:
                    v = float(S[i, j])
                    fh.write(f"{B.qids[i]}\t{t_ids[j]}\t1E{-v!r}\t{v!r}\n")
                n_rows += js.size
        H = A.read_hits(path)
        rows, info_t = A.rows_from_table(LS, H)
        sc, oc = {}, {}
        rec = dict(n_rows=n_rows, table=info_t, log10e_exact=bool(np.array_equal(
            rows["log10e"], -S[rows["qi"], _col_of(rows, H, t_ids)])))
        for label, ls, den, tie in TABLE6:
            d = DEN[ls] if den == "code" else None
            a1, a5 = AF.aucs(RES[ls], d, tie)
            P = my_protocol(label, ls, den, tie)
            dov = LS.denominators("famsize", P.exclude_self, cap=AF.CAP) if den == "code" else None
            res = A.evaluate_protocol(LS, rows, P, sc, oc, den_override=dov)
            rec[label] = dict(auc1=compare(res[P.ties[0]]["auc1"], a1), auc5=compare(res[P.ties[0]]["auc5"], a5))
        R["tsv_round_trip"][sname] = rec
        log("TSV", sname, {k: (v["auc1"]["exact"], v["auc5"]["exact"]) for k, v in rec.items() if isinstance(v, dict)
                           and "auc1" in v})
        os.remove(path)
    # ------------------------------------------------------------------ R3: recon T8 3Di variants
    if not args.quick:
        R["R3_recon_T8"] = {}
        qpos = {q: i for i, q in enumerate(B.qids)}
        for v in ("cb", "cb3", "qd10", "cb_qd10"):
            M = np.full((nq, nt), np.nan, np.float32)
            for k in range(4):
                z = np.load(f"{RESCORE}/scores.{k}.{v}.npz")
                assert np.array_equal(z["target_ids"].astype(str), B.tids)
                sc_ = z["score"]                      # an npz member is decompressed on every access: once
                for r_, q in enumerate(z["query_ids"].astype(str)):
                    M[qpos[q]] = sc_[r_]
            assert not np.isnan(M).any()
            ref = SU.auc1(M, B.TP, B.FP)
            M64 = M.astype(np.float64)
            rows = dict(rows0, log10e=-M64.ravel(), bits=M64.ravel())
            P = A.Protocol("T8", sort="evalue:asc", ties=("half",), drop_gt1=False)
            res = A.evaluate_protocol(LS, rows, P, stc, {})
            mine = res["half"]["auc1"]
            R["R3_recon_T8"][v] = dict(stored_4dp=T8_ALL[v], research_code_now=float(np.nanmean(ref)),
                                       mine=float(np.nanmean(mine)), per_query=compare(mine, ref),
                                       mine_rounds_to_stored=round(float(np.nanmean(mine)), 4) == T8_ALL[v])
            log("R3", v, R["R3_recon_T8"][v]["stored_4dp"], R["R3_recon_T8"][v]["mine"],
                R["R3_recon_T8"][v]["per_query"]["exact"])
            del M, M64, rows
    # ------------------------------------------------------------------ verdict
    ok = all(R["labels"]["equal_to_labels_npz"].values())
    ok &= all(v[f]["mine_eq_stored"] and v[f]["stored_eq_research"] and v[f]["per_query"]["exact"]
              for v in R["R1_T2"].values() for f in ("fp_all", "fp_real"))
    ok &= all(v["auc1"]["exact"] and v["auc5"]["exact"] for s in R["R2_auc_final"].values() for v in s.values()
              if isinstance(v, dict))
    ok &= all(v["auc1"]["exact"] and v["auc5"]["exact"] for s in R["tsv_round_trip"].values() for v in s.values()
              if isinstance(v, dict) and "auc1" in v)
    if "R3_recon_T8" in R:
        ok &= all(v["per_query"]["exact"] and v["mine_rounds_to_stored"] for v in R["R3_recon_T8"].values())
    R["pass"] = bool(ok)
    R["wall_s"] = round(time.time() - t_start, 1)
    R["provenance"] = AC.provenance(fm=False, kernel=False)
    BC.write_json(R, os.path.join(args.out, "validate.json"))
    log("VALIDATION", "PASS" if ok else "FAIL", os.path.join(args.out, "validate.json"))


def _col_of(rows, H, t_ids):
    """Target column (sample order) of each table row."""
    pos = {t: j for j, t in enumerate(t_ids)}
    col_u = np.array([pos[t] for t in H["target_uniq"]], np.int64)
    return col_u[rows["tcode"]]
