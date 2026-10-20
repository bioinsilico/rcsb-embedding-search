#!/usr/bin/env python
"""P8 A6 (optional; plan FINAL v1, section 8 P8): DB-independence of the esm-codes-v1 E-values on the CATH set, with
the CATH-fold null and the SCOP-superfamily null. Pass criterion (plan A6 row): E <= 1 FP ratio <= 1.5.

Data: panel/fit/data/cath_set.npz (5,795 CATH domains = the research CATH val + test domains; superfamily C.A.T.H,
fold C.A.T); VT = the 2,528 domains of split val / test, ALLF = all 5,795, each query against the other domains of its
set (research cath_validate.py, md5 below; its own functions are imported and used: tabp, boot_idx, fold_overlap,
scop_map, evaluate, ci).

  make-fasta  the CATH set as a FASTA (ids and sequences of cath_set.npz, in order) for the production build / query
  stage2      every domain against every domain through the production Stage 2 (fm_stage2.run_stage2: the CLI's
              align_candidates_esm_codes, explicit workers, observed), on
                --encodings foldmatch: the production re-encoding ('fm-search build sequences --residue-codes' of the
                  FASTA, and the query pass of 'fm-search query sequences --scoring esm-codes' captured by
                  a8_pipeline.py), or
                --encodings research: cath_set.npz's own q / codes (the research encodings);
  protocol    cath_validate.py's evaluation for the NEW (shipped) table on the recorded S and Q90: sets VT / ALLF,
              nulls cath / cath_noRB / scop_dom / scop_sf, tables all / main / swap; E with N_q = the residues of the
              query's targets; then the A6 criterion. --reference: the same loop on the research scores
              (score2/out/cath_cath_S_D_*, Q90 from the score2 shipped-panel columns) against the stored
              cath_validate.json (the reproduction check).
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

CATH_SET = "embcomp/panel/fit/data/cath_set.npz"
CV_FILE = "research_files/embcomp/panel/fit2/scripts/cath_validate.py"
CV_JSON = "embcomp/panel/fit2/results/cath_validate.json"
SCOP_LINKS_DIR = f"{AC.ARCHIVE}/research_files/embcomp/panel/fit2"      # data/scop_links.tsv (md5 7549152c...)
SCOP_LINKS_MD5 = "7549152c3319e6f134239deb6fe644f5"
S2 = "embcomp/panel/score2/out"
XS_CURVE = (1e-3, 1e-2, 0.1, 1.0, 10.0)


def import_cv():
    p = AC.inp(CV_FILE)
    d = os.path.dirname(p)
    if d not in sys.path:
        sys.path.append(d)
    import cath_validate as CV  # noqa: E402
    assert os.path.realpath(CV.__file__) == os.path.realpath(p)
    CV.FIT2 = SCOP_LINKS_DIR            # scop_map() reads f"{FIT2}/data/scop_links.tsv"
    got = AC.md5_file(f"{SCOP_LINKS_DIR}/data/scop_links.tsv")
    assert got == SCOP_LINKS_MD5, got
    return CV, AC.md5_file(p)


def make_fasta(a) -> int:
    z = np.load(AC.inp(CATH_SET))
    with open(a.out, "w") as fh:
        for i, s, sf, sp in zip(z["ids"].astype(str), z["sequences"].astype(str), z["superfamily"].astype(str),
                                z["split"].astype(str)):
            fh.write(f">{i} {sf} {sp}\n{s}\n")
    print(a.out, AC.md5_file(a.out), len(z["ids"]))
    return 0


def stage2(a) -> int:
    from foldmatch.search import scoring_system as SS
    os.makedirs(a.out_dir, exist_ok=True)
    z = np.load(AC.inp(CATH_SET))
    ids = list(z["ids"].astype(str))
    seqs = list(z["sequences"].astype(str))
    L = z["lengths"].astype(np.float64)
    assert all(len(s) == n for s, n in zip(seqs, L.astype(int)))
    N_all = float(L.sum())
    Nq = N_all - L
    tabs = AC.ship_tables()
    cov = F.coverage_max_evalue(L, tabs["all"], N_all, {k: tabs[k] for k in ("all", "main", "swap")},
                                {"Nq_min": float(Nq.min()), "Nq_max": float(Nq.max())}, 10.0)
    me = a.max_evalue if a.max_evalue else float(max(10.0, math.ceil(cov["required_max_evalue"] * 1.05)))
    rec = {"what": f"P8 A6 stage 2 ({a.encodings} encodings): the CATH set all-vs-all through the production Stage 2",
           "encodings": a.encodings, "cath_set": AC.inp(CATH_SET), "cath_set_md5": AC.md5_file(AC.inp(CATH_SET)),
           "n": len(ids), "N_all": N_all, "coverage": dict(cov, chosen_max_evalue=me)}
    tpos = {t: i for i, t in enumerate(ids)}
    emb = None
    if a.encodings == "foldmatch":
        from foldmatch.cli.search import resolve_esm_codes_database
        from foldmatch.utils.sequence import normalize_sequence
        database = resolve_esm_codes_database(a.db, me)
        code_store, fetch, prepared = database.code_store, database.sequence_store.fetch, database.prepared
        N = database.total_residues
        rec["db"] = F.db_info(a.db)
        query_index, cap = F.query_index_from_capture(a.capture)
        qseq = AC.read_json(os.path.join(a.capture, "query_sequences.json"))
        same = [normalize_sequence(s) == qseq.get(q) for q, s in zip(ids, seqs)]
        assert all(same) and set(cap["entries"]) == set(ids), "capture queries differ from the CATH set"
        rec["capture"] = dict(dir=a.capture, queries_f16_sha256=cap["sha256"], entries=len(cap["entries"]))
        qseqs = {q: qseq[q] for q in ids}
        # code / q agreement with the research encodings of the same domains (diagnostic)
        offs, lens = code_store.lookup_rows(ids)
        C = code_store.open_memmap()
        order = np.concatenate([np.arange(o, o + n) for o, n in zip(offs.tolist(), lens.tolist())])
        x = np.unpackbits(np.bitwise_xor(np.asarray(C[order]), z["codes"]), axis=1)
        Qm = query_index.memmap()
        qorder = np.concatenate([np.arange(query_index.entries[q].offset,
                                           query_index.entries[q].offset + query_index.entries[q].length) for q in ids])
        qp, rq = np.asarray(Qm[qorder], np.float32), z["q"].astype(np.float32)
        cos = (qp * rq).sum(1) / (np.linalg.norm(qp, axis=1) * np.linalg.norm(rq, axis=1) + 1e-30)
        rec["encoding_vs_research"] = dict(code_bits_agreement=1 - float(x.sum()) / x.size, bits=int(x.size),
                                           q_cos_median=float(np.median(cos)), q_cos_min=float(cos.min()),
                                           q_cos_frac_ge_0_999=float((cos >= 0.999).mean()))
        del x
    else:
        system = SS.get(SS.DEFAULT_ESM_CODES)
        prepared = SS.prepare(system, local_files_only=True)
        tmp = a.tmp or os.path.join(a.out_dir, "tmp")
        os.makedirs(tmp, exist_ok=True)
        bin_p, q_p = os.path.join(tmp, "cath_codes.bin"), os.path.join(tmp, "cath_q.f16")
        rec["research_encodings"] = dict(codes_bin_sha256=F.write_raw(bin_p, np.ascontiguousarray(z["codes"])),
                                         q_f16_sha256=F.write_raw(q_p, np.ascontiguousarray(z["q"], np.float16)))
        code_store = F.ArrayCodeStore(bin_p, ids, z["offsets"][:-1], z["lengths"])
        query_index = F.query_index_from_arrays(q_p, ids, z["offsets"][:-1], z["lengths"], seqs, system.name)
        qseqs = dict(zip(ids, seqs))
        tseq = dict(zip(ids, seqs))

        def fetch(want):
            return {t: tseq[t] for t in want if t in tseq}
        N = int(N_all)
    assert int(N) == int(N_all), (N, N_all)
    rec["db_residues"] = int(N)
    AC.log(f"A6 stage 2 ({a.encodings}): {len(ids)} x {len(ids)}, workers {a.workers}, max_evalue {me}, N {N}")
    R = F.run_stage2(query_index, ids, qseqs, ids, code_store, fetch, prepared, N, workers=a.workers, max_evalue=me,
                     emb=emb)
    M = F.assemble(R["obs"], ids, len(ids))
    H = F.hits_arrays(R["hits"], ids, tpos)
    chk = F.check_run(M, H, L, tabs["all"], N, me)
    AC.log(f"done in {R['wall_s']:.1f}s; checks ok={chk['ok']}")
    st = dict(R["stats"])
    st.pop("q90", None)
    rec.update(stage2=dict(wall_s=R["wall_s"], workers_requested=a.workers, pool=R["pool"], max_evalue=me, stats=st),
               checks=chk)
    rec["matrices_sha256"] = F.save_npz(os.path.join(a.out_dir, "matrices.npz"), ids=np.array(ids), S_f=M["S_f"],
                                        S_r=M["S_r"], q90=M["q90"], panel_f=M["panel_f"], panel_r=M["panel_r"],
                                        observed=M["observed"])
    rec["hits_sha256"] = F.save_npz(os.path.join(a.out_dir, "hits.npz"), ids=np.array(ids), **H)
    rec["provenance"] = AC.provenance()
    h = AC.write_json(rec, os.path.join(a.out_dir, "stage2.json"))
    AC.log(f"wrote {a.out_dir}/stage2.json sha256 {h}")
    return 0 if chk["ok"] else 1


def cv_new_table(CV, z, S, q90, tabs, *, nboot=None):
    """cath_validate.main() restricted to the NEW table (the shipped one): per set (VT, ALLF), per sub-table (all,
    main, swap) and null (cath, cath_noRB, scop_dom, scop_sf): CV.evaluate (ratio, fold / query CIs, fq, bins,
    top topology pairs, Rieske-Bet v1 FPs, sensitivity at 1e-3). Same seeds (fold bootstrap 11, query 12)."""
    nb = nboot or CV.NB
    ids = z["ids"].astype(str)
    d_folds_all, s_folds_all, sinfo, _sf_links = CV.scop_map(ids, z["superfamily"].astype(str))
    split = z["split"].astype(str)
    out = {"scop_map": {k: v for k, v in sinfo.items() if k != "rieske_betv1"}, "sets": {}}
    for sname, sel in (("VT", np.flatnonzero(np.isin(split, ["val", "test"]))), ("ALLF", np.arange(len(ids)))):
        n = len(sel)
        L = z["lengths"][sel].astype(np.float64)
        fold, sf = z["fold"][sel].astype(str), z["superfamily"][sel].astype(str)
        notself = ~np.eye(n, dtype=bool)
        same_sf = sf[:, None] == sf[None, :]
        same_fold = fold[:, None] == fold[None, :]
        TP = same_sf & notself
        FPc = ~same_fold
        dom_rel = CV.fold_overlap([d_folds_all[i] for i in sel])
        sf_rel = CV.fold_overlap([s_folds_all[i] for i in sel])
        RB = (((fold == CV.RIESKE)[:, None] & (fold == CV.BETV1)[None, :])
              | ((fold == CV.BETV1)[:, None] & (fold == CV.RIESKE)[None, :]))
        nulls = {"cath": FPc, "cath_noRB": FPc & ~RB, "scop_dom": FPc & ~dom_rel, "scop_sf": FPc & ~sf_rel}
        Nq = L.sum() - L
        qbin = np.digitize(L, CV.EDGES[1:-1], right=False)
        grp = np.unique(fold, return_inverse=True)[1]
        bc = CV.boot_idx(grp, nb, 11)
        rq = np.random.default_rng(12)
        bq = [rq.integers(0, n, n) for _ in range(nb)]
        R = out["sets"].setdefault(sname, dict(n_queries=int(n), residues=float(L.sum()),
                                               n_folds=int(len(set(fold))), pairs={k: int(v.sum()) for k, v in nulls.items()},
                                               tp_pairs=int(TP.sum()), tables={}, curves={}))
        Sp = S[np.ix_(sel, sel)] - q90[sel][:, None]
        for sp in ("all", "main", "swap"):
            lam, lnK = CV.tabp(tabs[sp], L)
            Lm = (lnK + np.log(L) + np.log(Nq))[:, None] - lam[:, None] * Sp
            Lm[~notself] = np.inf
            for nname, FP in nulls.items():
                fnull = (FP * L[None, :]).sum(1) / Nq
                o = CV.evaluate(Lm, FP, TP, L, fold, bc, bq, qbin, fnull)
                for x in CV.XS:
                    o["x"][str(x)].pop("counts")
                R["tables"].setdefault(sp, {})[nname] = o
                if sp == "all":
                    R["curves"][nname] = {str(x): dict(fp=int(((Lm <= math.log(x)) & FP).sum()),
                                                       ratio=float(((Lm <= math.log(x)) & FP).sum() / (n * x)))
                                          for x in XS_CURVE}
        R["q90"] = dict(mean=float(q90[sel].mean()), pct=[float(v) for v in np.percentile(q90[sel], [5, 25, 50, 75, 95])])
    return out


def research_inputs(z):
    """The research S (score2/out/cath_cath_S_D_*, = panel/fit/data/cath_S_D_* within 8.8e-6 rel per the research)
    and the research NEW Q90 (score2/out/cath_strat_* columns of the shipped draw S1000:0, cath_validate.main)."""
    S = np.load(AC.inp(f"{S2}/cath_cath_S_D_fwd.npy")).astype(np.float64)
    S -= np.load(AC.inp(f"{S2}/cath_cath_S_D_rev.npy")).astype(np.float64)
    ix = np.load(AC.inp(f"{S2}/cath_strat_index.npz"))
    assert np.array_equal(ix["query_ids"].astype(str), z["ids"].astype(str))
    ship = AC.read_json(AC.inp("embcomp/panel/fit2/results/ship_final.json"))
    k = int(np.flatnonzero((ix["draw_size"] == 1000) & (ix["draw_index"] == 0))[0])
    pc = ix["draw_cols"][ix["draw_offsets"][k]:ix["draw_offsets"][k + 1]]
    assert list(ix["member_ids"].astype(str)[pc]) == ship["meta"]["panel_ids"]
    P = np.load(AC.inp(f"{S2}/cath_strat_S_D_fwd.npy"), mmap_mode="r")[:, pc].astype(np.float64)
    P -= np.load(AC.inp(f"{S2}/cath_strat_S_D_rev.npy"), mmap_mode="r")[:, pc].astype(np.float64)
    return S, np.quantile(P, 0.9, axis=1)


def compare_to_stored(mine, stored) -> dict:
    """Ratios / FPs / sens of this run's NEW-table evaluation against cath_validate.json tables.new."""
    rows, worst = [], 0.0
    for sname in ("VT", "ALLF"):
        for sp in ("all", "main", "swap"):
            for nname in ("cath", "cath_noRB", "scop_dom", "scop_sf"):
                a = mine["sets"][sname]["tables"][sp][nname]
                b = stored["sets"][sname]["tables"]["new"][sp][nname]
                for x in ("0.01", "0.1", "1.0"):
                    d = a["x"][x]["fp"] - b["x"][x]["fp"]
                    worst = max(worst, abs(d))
                    rows.append(dict(set=sname, table=sp, null=nname, x=x, fp=a["x"][x]["fp"], fp_ref=b["x"][x]["fp"],
                                     ratio=a["x"][x]["ratio"], ratio_ref=b["x"][x]["ratio"]))
                rows.append(dict(set=sname, table=sp, null=nname, x="sens_1e-3", value=a["sens_1e-3"],
                                 ref=b["sens_1e-3"]))
    return dict(max_abs_fp_difference=int(worst), rows=rows)


def verdict(mine) -> dict:
    v = {}
    for sname in ("VT", "ALLF"):
        for nname in ("cath", "scop_sf"):
            r = mine["sets"][sname]["tables"]["all"][nname]["x"]["1.0"]["ratio"]
            v[f"{sname}/{nname}"] = dict(ratio_e1=r, ci_fold=mine["sets"][sname]["tables"]["all"][nname]["x"]["1.0"]
                                         ["ci_fold"], pass_le_1_5=bool(r <= 1.5))
    return v


def protocol(a) -> int:
    t0 = time.time()
    CV, cv_md5 = import_cv()
    z = np.load(AC.inp(CATH_SET))
    tabs = AC.ship_tables()
    stored = AC.read_json(AC.inp(CV_JSON))
    res = {"what": "P8 A6 (optional): DB-independence on the CATH set (cath_validate.py protocol, NEW table)",
           "criterion": "E <= 1 FP ratio <= 1.5 (plan A6 row), CATH-fold null and SCOP-superfamily null",
           "cath_validate_py": dict(path=AC.inp(CV_FILE), md5=cv_md5),
           "scop_links": dict(path=f"{SCOP_LINKS_DIR}/data/scop_links.tsv", md5=SCOP_LINKS_MD5),
           "research_reference_e1": {f"{s}/{n}": stored["sets"][s]["tables"]["new"]["all"][n]["x"]["1.0"]["ratio"]
                                     for s in ("VT", "ALLF") for n in ("cath", "cath_noRB", "scop_dom", "scop_sf")}}
    if a.reference:
        S, q90 = research_inputs(z)
        mine = cv_new_table(CV, z, S, q90, tabs)
        res["reference_reproduction"] = compare_to_stored(mine, stored)
        res["reference_reproduction"]["q90_vs_stored_local_max_abs"] = float(np.abs(
            np.load(AC.inp("embcomp/panel/fit2/results/cath_q90_shipped_local.npy")) - q90).max())
        AC.log("reference reproduction: max |dFP|", res["reference_reproduction"]["max_abs_fp_difference"])
    for d in a.run_dir or []:
        st = AC.read_json(os.path.join(d, "stage2.json"))
        m = np.load(os.path.join(d, "matrices.npz"))
        assert list(m["ids"].astype(str)) == list(z["ids"].astype(str))
        S = m["S_f"].astype(np.float64) - m["S_r"].astype(np.float64)
        q90 = m["q90"]
        mine = cv_new_table(CV, z, S, q90, tabs)
        enc = st["encodings"]
        res[enc] = dict(run_dir=d, stage2_checks=st["checks"], stage2=st["stage2"], coverage=st["coverage"],
                        encoding_vs_research=st.get("encoding_vs_research"), evaluation=mine,
                        vs_stored_research=compare_to_stored(mine, stored), verdict=verdict(mine),
                        provenance_stage2=st.get("provenance"))
        AC.log(enc, res[enc]["verdict"])
    res["provenance"] = AC.provenance()
    res["wall_s"] = round(time.time() - t0, 1)
    h = AC.write_json(res, a.out)
    with open(a.out.replace(".json", ".txt"), "w") as fh:
        fh.write(f"A6 (optional): {res['criterion']}\nresearch reference (NEW, all, E<=1): {res['research_reference_e1']}\n")
        if "reference_reproduction" in res:
            fh.write(f"reference reproduction max |dFP| {res['reference_reproduction']['max_abs_fp_difference']}\n")
        for enc in ("research", "foldmatch"):
            if enc in res:
                fh.write(f"{enc}: {res[enc]['verdict']}\n")
                for sname in ("VT", "ALLF"):
                    for nname in ("cath", "cath_noRB", "scop_dom", "scop_sf"):
                        o = res[enc]["evaluation"]["sets"][sname]["tables"]["all"][nname]
                        fh.write(f"  [{sname} {nname:9s}] FP/q/x .01/.1/1 " + " / ".join(
                            f"{o['x'][x]['ratio']:.2f} [{o['x'][x]['ci_fold'][0]:.2f},{o['x'][x]['ci_fold'][1]:.2f}]"
                            for x in ("0.01", "0.1", "1.0")) + "  FPs " + "/".join(
                            str(o['x'][x]['fp']) for x in ("0.01", "0.1", "1.0")) + "  fq " + "/".join(
                            f"{o['x'][x]['fq']:.3f}" for x in ("0.01", "0.1", "1.0"))
                            + f"  sens@1e-3 {o['sens_1e-3']:.3f}\n")
        fh.write(f"json {a.out} sha256 {h}\n")
    AC.log(f"wrote {a.out} sha256 {h}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make-fasta")
    m.add_argument("--out", required=True)
    s = sub.add_parser("stage2")
    s.add_argument("--encodings", choices=("foldmatch", "research"), required=True)
    s.add_argument("--db")
    s.add_argument("--capture")
    s.add_argument("--workers", type=int, required=True)
    s.add_argument("--max-evalue", type=float, default=None)
    s.add_argument("--out-dir", required=True)
    s.add_argument("--tmp", default=None)
    p = sub.add_parser("protocol")
    p.add_argument("--run-dir", action="append")
    p.add_argument("--reference", action="store_true")
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "make-fasta":
        return make_fasta(a)
    if a.cmd == "stage2":
        if a.encodings == "foldmatch" and not (a.db and a.capture):
            ap.error("--encodings foldmatch needs --db and --capture")
        return stage2(a)
    return protocol(a)


if __name__ == "__main__":
    sys.exit(main())
