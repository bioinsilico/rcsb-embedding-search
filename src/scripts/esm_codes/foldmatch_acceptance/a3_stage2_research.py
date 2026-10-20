#!/usr/bin/env python
"""P8 A3: Stage 2 on the research encodings (plan FINAL v1, section 8 P8, A3 row; method S9-S11, S16).

FoldMatch's production esm-codes aligner (foldmatch.search.code_alignment.align_candidates_esm_codes, the function
`fm-search query sequences --scoring esm-codes` calls) on the research encodings of the benchmark:
  queries  the P0.5 q200 set (p0/parity/inputs/sets.json 'q200': 200 of the 1,465 benchmark queries, stratified by
           length), q fp16 from evalue/encode/out/queries.L2m_B8_s0.npz
  targets  all 50,000 benchmark targets (incl. the 25,000 reversed decoys), codes from
           evalue/encode/out/targets.L2m_B8_s0.npz, written into a real database: SequenceStore + ResidueCodeStore
           (create + validate, the build's store code), N = the store's total_residues (15,889,271)
  Stage 1  replaced by every target as a candidate of every query (200 x 50,000 = 10,000,000 pairs)
  pool     CodeAlignmentPool(--workers), an explicit worker count (P6 / P7 notes for P8)
run at max_evalue = inf (every candidate traced back and reported: S_f, S_r, Q90, ln E for every pair) and at the
production E filters 10 and 1e-3 (the survivors path); output fields = the default 13 + raw + rawrev / q90 / rawadj /
ecal (no strings).

Reference = the research rows of the same queries: S_D_fwd / S_D_rev (evalue/score/out, float32), Q90 =
panel/fit2/results/bench_q90_shipped.npy (fit_final2.py: np.quantile(float64 S over the shipped panel, 0.9)), ln E with
the shipped table (ship_final.json tables D_fmr/pnl_uniref50/M3t999 'all') through bench_validate.counts' formula
(acc_common.ln_evalue), with the same N as FoldMatch. Labels: plan.py rules (acc_common.Bench = rlib.Bench).

Pass (plan A3, S16): S_f and S_r within 5e-5 max(|ref|, 1) (every pair); |dQ90| <= 1e-4 (max over the 200); |d ln E|
<= 1e-4 (max over every pair); benchmark TP / FP counts at E <= 1e-3, 1e-2, 0.1, 1, 10 identical except
threshold-boundary pairs (|ln E_ref - ln t| < 1e-4: every pair whose decision differs must be one; how many there are
is reported). Also: the filtered runs report exactly the inf run's pairs with ln E <= ln t, bit for bit.

Usage: python a3_stage2_research.py --workers N [--tag TAG] [--queries K] [--max-evalues inf 10 0.001]
       -> <P8_ROOT>/A3/<TAG>/a3_result.json, a3_foldmatch_scores.npz
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import acc_common as C  # noqa: E402
import numpy as np  # noqa: E402

XS = (1e-3, 1e-2, 0.1, 1.0, 10.0)
S16_SCORE, S16_Q90, S16_LNE, S16_BOUNDARY = 5e-5, 1e-4, 1e-4, 1e-4
INP = {
    "queries_enc": ("embcomp/evalue/encode/out/queries.L2m_B8_s0.npz", "e821497f59dda06a3899c18dce37286c"),
    "targets_enc": ("embcomp/evalue/encode/out/targets.L2m_B8_s0.npz", None),
    "S_fwd": ("embcomp/evalue/score/out/S_D_fwd.npy", "1d015c63cb870524664442bd3784384e"),
    "S_rev": ("embcomp/evalue/score/out/S_D_rev.npy", "70f3a0f75bf58243a9697da73d449cbd"),
    "index": ("embcomp/evalue/score/out/index.npz", "ff0f64fc242625b6c7dfce5082a57abb"),
    "q90": ("embcomp/panel/fit2/results/bench_q90_shipped.npy", None),
    "ship_final": ("embcomp/panel/fit2/results/ship_final.json", "c51735f962c04bd8c441b9625c4cf30a"),
    "sets": ("p0/parity/inputs/sets.json", None),
    "qfasta": ("calib3di/results/queries.fasta", "cf6bb37859a48052875b4a2912080b1b"),
    "tfasta": ("calib3di/results/targets.fasta", "a34ca17a20cfba6994a33012e064f4a5"),
    "labels": ("embcomp/evalue/score/results/labels.npz", None),
}
TARGETS_MD5_PREFIX = "cfee06cd"                      # evalue/score/README.txt 'targets md5 cfee06cd...'
log = C.log


def _seqs(z):
    sc, off = z["seq_concat"], z["offsets"]
    return [sc[off[k]:off[k + 1]].tobytes().decode("ascii") for k in range(len(off) - 1)]


def build_db(folder, system, tids, tcodes, toff, tseqs):
    """A real esm-codes database of the 50,000 targets: SequenceStore (FASTA) + ResidueCodeStore (create + validate)."""
    from foldmatch.search.residue_code_store import ResidueCodeStore
    from foldmatch.search.sequence_store import SequenceStore
    from foldmatch.utils.sequence import normalize_sequence, seq_sha1
    fasta = os.path.join(folder, "db.fasta")
    with open(fasta, "w") as f:
        for tid, s in zip(tids, tseqs):
            f.write(f">{tid}\n{s}\n")
    seq_store = SequenceStore(folder, "db")
    seq_store.create(fasta)
    store = ResidueCodeStore(folder, "db")
    norm = [normalize_sequence(s) for s in tseqs]
    assert all(len(s) == toff[k + 1] - toff[k] for k, s in enumerate(norm))
    created = store.create([(list(tids), [tcodes[toff[k]:toff[k + 1]] for k in range(len(tids))],
                             [seq_sha1(s) for s in norm])], system)
    validated = store.validate(seq_store, len(tids), preflight=None)
    return seq_store, store, created, validated


def hits_to_arrays(results, qids, col, nt):
    """{qid: [Hit]} -> per-pair arrays [nq, nt] (NaN / 0 where a pair has no hit)."""
    nq = len(qids)
    A = dict(s_f=np.full((nq, nt), np.nan, np.float32), s_r=np.full((nq, nt), np.nan, np.float32),
             lne=np.full((nq, nt), np.nan), s_adj=np.full((nq, nt), np.nan), bits=np.full((nq, nt), np.nan),
             ecal=np.zeros((nq, nt), np.int8), hit=np.zeros((nq, nt), bool))
    q90 = np.full(nq, np.nan)
    order_ok = True
    for i, qid in enumerate(qids):
        hs = results[qid]
        if not hs:
            continue
        j = np.fromiter((col[h.subject_id] for h in hs), np.int64, len(hs))
        if len(np.unique(j)) != len(j):
            raise RuntimeError(f"{qid}: a target is reported twice")
        m = [h.metrics for h in hs]
        A["s_f"][i, j] = np.fromiter((x.score for x in m), np.float32, len(m))
        A["s_r"][i, j] = np.fromiter((x.score_rev for x in m), np.float32, len(m))
        A["lne"][i, j] = np.fromiter((x.log_evalue for x in m), np.float64, len(m))
        A["s_adj"][i, j] = np.fromiter((x.score_adj for x in m), np.float64, len(m))
        A["bits"][i, j] = np.fromiter((x.bit_score for x in m), np.float64, len(m))
        A["ecal"][i, j] = np.fromiter((x.ecal for x in m), np.int8, len(m))
        A["hit"][i, j] = True
        qs = {x.q90 for x in m}
        if len(qs) != 1:
            raise RuntimeError(f"{qid}: hits carry {len(qs)} Q90 values")
        q90[i] = qs.pop()
        S = np.fromiter((x.score - x.score_rev for x in m), np.float64, len(m))
        order_ok &= bool(np.all(np.diff(S) <= 0))           # S14: S descending
    A["q90"] = q90
    A["s14_order_ok"] = order_ok
    return A


def run(args) -> int:
    t_start = time.time()
    tag = args.tag or C.HOST
    rdir = C.result_dir("A3", tag)
    from foldmatch.search import code_alignment as ca
    from foldmatch.search import code_statistics as cs
    from foldmatch.search import output as fo
    from foldmatch.search import scoring_system as ss
    from foldmatch.search.query_encodings import QueryEncodingEntry, QueryEncodingIndex
    from foldmatch.utils.sequence import normalize_sequence, seq_sha1
    res = {"check": "A3", "what": __doc__.split("\n\n")[0], "tag": tag, "workers": args.workers,
           "criteria": dict(s16_score=f"|S - S_ref| <= {S16_SCORE} max(|S_ref|, 1)", s16_q90=S16_Q90,
                            s16_lne_max=S16_LNE, thresholds=XS, boundary_tol=S16_BOUNDARY),
           "inputs": {}}
    res["provenance"] = C.provenance(kernel_selftest=False)
    for k, (name, md5) in INP.items():
        p = C.inp(name)
        got = C.md5_file(p)
        ok = (got == md5) if md5 else (got.startswith(TARGETS_MD5_PREFIX) if k == "targets_enc" else None)
        res["inputs"][k] = dict(name=name, realpath=os.path.realpath(p), md5=got, expected=md5, match=ok)
        if ok is False:
            raise RuntimeError(f"{name}: md5 {got} does not match {md5 or TARGETS_MD5_PREFIX}")
    B = C.Bench()
    sets = C.read_json(C.inp(INP["sets"][0]))
    rows = np.asarray(sets["q200"]["rows"], np.int64)
    assert len(rows) == 200 and len(np.unique(rows)) == 200
    if args.queries:
        rows = rows[np.linspace(0, len(rows) - 1, args.queries).round().astype(int)]
    if args.positions:
        rows = rows[np.asarray(args.positions, np.int64)]
    nq, nt = len(rows), B.nt
    res["queries"] = dict(set="p0/parity/inputs/sets.json q200", n=nq, rows=rows, seed=sets["q200"].get("seed"),
                          lengths=B.qlen[rows].astype(int))
    # ------------------------------------------------------------------ research encodings and references
    qz, tz = np.load(C.inp(INP["queries_enc"][0])), np.load(C.inp(INP["targets_enc"][0]))
    assert np.array_equal(qz["ids"].astype(str), B.qids) and np.array_equal(tz["ids"].astype(str), B.tids)
    qoff, toff = qz["offsets"], tz["offsets"]
    assert np.array_equal(np.diff(qoff), B.qlen.astype(np.int64)) and np.array_equal(np.diff(toff), B.tlen.astype(np.int64))
    qseqs, tseqs = _seqs(qz), _seqs(tz)
    fa_q = {h.split()[0]: s for h, s in C.read_fasta(C.inp(INP["qfasta"][0]))}
    fa_t = {h.split()[0]: s for h, s in C.read_fasta(C.inp(INP["tfasta"][0]))}
    res["sequence_checks"] = dict(
        queries_equal_fasta=int(sum(fa_q[q] == s for q, s in zip(B.qids, qseqs))),
        targets_equal_fasta=int(sum(fa_t[t] == s for t, s in zip(B.tids, tseqs))),
        queries_normalised_unchanged=int(sum(normalize_sequence(s) == s for s in qseqs)),
        targets_normalised_unchanged=int(sum(normalize_sequence(s) == s for s in tseqs)))
    Sf_ref = np.load(C.inp(INP["S_fwd"][0]), mmap_mode="r")[rows].astype(np.float32)
    Sr_ref = np.load(C.inp(INP["S_rev"][0]), mmap_mode="r")[rows].astype(np.float32)
    S_ref = Sf_ref.astype(np.float64) - Sr_ref.astype(np.float64)
    q90_ref = np.load(C.inp(INP["q90"][0]))[rows].astype(np.float64)
    tabs = C.ship_tables(C.inp(INP["ship_final"][0]))
    tab = tabs["all"]
    system = ss.get(ss.DEFAULT_ESM_CODES)
    res["table_equals_registry"] = bool(
        np.array_equal(np.asarray(tab["center_len"], np.float64), np.asarray(system.bin_centres, np.float64))
        and np.array_equal(np.asarray(tab["lam"], np.float64), np.asarray(system.lam, np.float64))
        and np.array_equal(np.asarray(tab["lnK"], np.float64), np.asarray(system.lnK, np.float64)))
    if not res["table_equals_registry"]:
        raise RuntimeError("ship_final.json 'all' table differs from the registry entry")
    # ------------------------------------------------------------------ the database and the query encodings
    tmp_root = args.work or os.environ.get("TMPDIR") or rdir
    folder = tempfile.mkdtemp(prefix="a3db_", dir=tmp_root)
    try:
        prepared = ss.prepare(system, local_files_only=True)
        t = time.perf_counter()
        seq_store, store, created, validated = build_db(folder, system, B.tids.tolist(), tz["codes"], toff, tseqs)
        man = store.manifest()
        N = int(man["total_residues"])
        assert N == int(B.tlen.sum()) == int(toff[-1])
        res["database"] = dict(folder=folder, build_s=round(time.perf_counter() - t, 1), entries=man["entries"],
                               total_residues=N, bin_sha256=man["bin_sha256"], entries_digest=man["entries_digest"],
                               system_digest=man["system_digest"], nonstandard_entries=man.get("nonstandard_entries"))
        qids = B.qids[rows].tolist()
        qpath = os.path.join(folder, "queries.f16")
        entries, r0 = {}, 0
        with open(qpath, "wb") as f:
            for i, k in zip(range(nq), rows.tolist()):
                q16 = qz["q"][qoff[k]:qoff[k + 1]]
                assert q16.dtype == np.float16 and q16.shape == (B.qlen[k], 64)
                f.write(np.ascontiguousarray(q16, "<f2").tobytes())
                entries[qids[i]] = QueryEncodingEntry(r0, int(len(q16)), seq_sha1(normalize_sequence(qseqs[k])))
                r0 += len(q16)
        qindex = QueryEncodingIndex(qpath, r0, 64, entries, system.name)
        qseq_map = {qids[i]: qseqs[k] for i, k in enumerate(rows.tolist())}
        tids = B.tids.tolist()
        col = {t: j for j, t in enumerate(tids)}
        prefilter = {q: (tids, [0.0] * nt) for q in qids}
        fields = list(fo.DEFAULT_OUTPUT_FIELDS) + ["raw"] + list(fo.ESM_CODES_OUTPUT_FIELDS)
        res["aligner"] = dict(function="foldmatch.search.code_alignment.align_candidates_esm_codes",
                              output_fields=fields, db_residues=N, pool=f"CodeAlignmentPool({args.workers})",
                              panel_policy="auto", candidates_per_query=nt)
        runs = {}
        with ca.CodeAlignmentPool(args.workers) as pool:
            for mx in args.max_evalues:
                me = None if math.isinf(mx) else mx
                stats = {}
                t = time.perf_counter()
                out = ca.align_candidates_esm_codes(qindex, prefilter, store, seq_store.fetch, prepared, N, pool,
                                                    query_sequences=qseq_map, max_evalue=me, output_needs=fields,
                                                    stats=stats)
                wall = time.perf_counter() - t
                A = hits_to_arrays(out, qids, col, nt)
                del out
                st = {k: v for k, v in stats.items() if k != "q90"}
                runs[str(mx)] = dict(A=A, stats=st, wall_s=round(wall, 2), q90_stats=stats.get("q90"))
                log(f"max_evalue {mx}: {int(A['hit'].sum())} hits in {wall:.1f} s ({json.dumps(C.jsonable(st))[:400]})")
            res["pool"] = dict(workers=pool.workers, starts=pool.starts)
        # pairs the inf run reports no hit for (an empty alignment, S_f = 0: the aligner keeps alnlen > 0 only):
        # their S_f and S_r from the production round-1 task, run in this process on the same store and query memmap
        # (BLAS 1 thread, as in the workers), so that the S16 score comparison covers every one of the pairs
        inf_run = runs[str(float("inf"))]["A"]
        noh = np.argwhere(~inf_run["hit"])
        res["no_hit_pairs"] = dict(count=int(len(noh)), how="code_alignment._round1_task in-process "
                                   "(_init_worker on the A3 store .bin and query memmap)")
        if len(noh):
            from foldmatch.search import swkern
            init = ca._WorkerInit(isa=swkern.load().isa, spec=prepared.for_worker(), codes=(str(store.bin_path), N, 8),
                                  queries=tuple(qindex.memmap_args()), block_floats=swkern.DEFAULT_BLOCK_FLOATS,
                                  max_pair_cells=swkern.DEFAULT_MAX_PAIR_CELLS)
            ca._init_worker(init, raise_errors=True)
            spec_w = prepared.for_worker()
            try:
                for i in np.unique(noh[:, 0]).tolist():
                    js = noh[noh[:, 0] == i, 1]
                    off, ln = store.lookup_rows([tids[j] for j in js.tolist()])
                    e = entries[qids[i]]
                    _tid, sf, sr, sk = ca._round1_task((0, ca._CODES, e.offset, e.length,
                                                       np.column_stack((off, ln)).astype(np.int64)))
                    assert not sk.any()
                    inf_run["s_f"][i, js], inf_run["s_r"][i, js] = sf, sr
                    sp = sf.astype(np.float64) - sr.astype(np.float64) - inf_run["q90"][i]
                    inf_run["lne"][i, js] = cs.log_evalue(sp, float(e.length), float(N), spec_w)
                    inf_run["s_adj"][i, js] = sp
            finally:
                ca._reset_worker_state()
            res["no_hit_pairs"].update(s_f_all_zero=bool(np.all(inf_run["s_f"][~inf_run["hit"]] == 0.0)),
                                       min_lne=float(np.min(inf_run["lne"][~inf_run["hit"]])),
                                       pairs=[[int(a), int(b)] for a, b in noh[:20]])
    finally:
        if not args.keep_db:
            shutil.rmtree(folder, ignore_errors=True)
    # ------------------------------------------------------------------ comparisons (the inf run: every pair)
    lnE_ref = C.ln_evalue(tab, B.qlen[rows], S_ref, q90_ref, float(N))
    inf = runs[str(float("inf"))]["A"] if str(float("inf")) in runs else None
    if inf is None:
        raise SystemExit("the inf run is required for the full comparison")
    have = inf["hit"]
    cmp_ = {"pairs": int(nq * nt), "pairs_with_hit": int(have.sum()), "pairs_without_hit": int((~have).sum())}
    if (~have).any():
        cmp_["no_hit_research_S_f_max"] = float(np.max(Sf_ref[~have]))
    cmp_["pairs_compared_s_f_s_r_lne"] = int(np.isfinite(inf["s_f"]).sum())
    sf_ratio = np.abs(inf["s_f"].astype(np.float64) - Sf_ref) / (S16_SCORE * np.maximum(np.abs(Sf_ref), 1.0))
    sr_ratio = np.abs(inf["s_r"].astype(np.float64) - Sr_ref) / (S16_SCORE * np.maximum(np.abs(Sr_ref), 1.0))
    dq90 = np.abs(inf["q90"] - q90_ref)
    dlne = np.abs(inf["lne"] - lnE_ref)
    k_sf = np.unravel_index(np.nanargmax(sf_ratio), sf_ratio.shape)
    k_sr = np.unravel_index(np.nanargmax(sr_ratio), sr_ratio.shape)
    k_le = np.unravel_index(np.nanargmax(dlne), dlne.shape)
    cmp_.update(
        s_f=dict(max_ratio_to_s16=float(np.nanmax(sf_ratio)), at=[int(k_sf[0]), int(k_sf[1])],
                 fm=float(inf["s_f"][k_sf]), ref=float(Sf_ref[k_sf]),
                 max_rel=float(np.nanmax(np.abs(inf["s_f"] - Sf_ref) / np.maximum(np.abs(Sf_ref), 1.0))),
                 bit_equal=int(np.sum(inf["s_f"].view(np.uint32) == Sf_ref.view(np.uint32)))),
        s_r=dict(max_ratio_to_s16=float(np.nanmax(sr_ratio)), at=[int(k_sr[0]), int(k_sr[1])],
                 max_rel=float(np.nanmax(np.abs(inf["s_r"] - Sr_ref) / np.maximum(np.abs(Sr_ref), 1.0))),
                 bit_equal=int(np.sum(inf["s_r"].view(np.uint32) == Sr_ref.view(np.uint32)))),
        q90=dict(max_abs=float(np.nanmax(dq90)), median_abs=float(np.nanmedian(dq90)),
                 at=int(np.nanargmax(dq90)), bit_equal=int(np.sum(inf["q90"] == q90_ref))),
        ln_e=dict(max_abs=float(np.nanmax(dlne)), p99_abs=float(np.nanpercentile(dlne, 99)),
                  median_abs=float(np.nanmedian(dlne)), at=[int(k_le[0]), int(k_le[1])],
                  at_lne_ref=float(lnE_ref[k_le])),
        s14_order_ok=bool(inf["s14_order_ok"]))
    # ------------------------------------------------------------------ benchmark counts at the thresholds
    TP, FP, FPA, REV = (M[rows] for M in (B.TP, B.FP, B.REAL, B.REVM))
    masks = {"TP": TP, "FP": FP, "FP_real_null": FPA, "FP_reversed": REV, "all_pairs": np.ones_like(TP)}
    L_fm = np.where(have, inf["lne"], np.inf)
    counts = {}
    for x in XS:
        lx = math.log(x)
        dec_fm, dec_ref = L_fm <= lx, lnE_ref <= lx
        diff = dec_fm != dec_ref
        bnd = np.abs(lnE_ref - lx) < S16_BOUNDARY
        c = {}
        for name, M in masks.items():
            c[name] = dict(foldmatch=int((dec_fm & M).sum()), research=int((dec_ref & M).sum()),
                           decisions_differ=int((diff & M).sum()), boundary_pairs=int((bnd & M).sum()),
                           differ_not_boundary=int((diff & ~bnd & M).sum()))
            c[name]["identical"] = c[name]["foldmatch"] == c[name]["research"]
        counts[str(x)] = c
    counts_ok = all(counts[str(x)][k]["differ_not_boundary"] == 0 for x in XS for k in masks)
    # the research bench_validate convention (N = annotated residues for the real null, reversed residues for the
    # reversed null, all residues for TPs), FoldMatch side through code_statistics.log_evalue on its own S'
    conv = {}
    N_ann, N_rev = float(B.tlen[~B.rev].sum()), float(B.tlen[B.rev].sum())
    spec = prepared.for_worker()
    m_rows = B.qlen[rows]
    for name, Nx, M in (("FP_real_null@N_ann", N_ann, FPA), ("FP_reversed@N_rev", N_rev, REV), ("TP@N_all", float(N), TP)):
        Lr = C.ln_evalue(tab, m_rows, S_ref, q90_ref, Nx)
        sp = np.where(have, inf["s_adj"], 0.0)
        Lf = np.empty_like(Lr)
        for i in range(nq):
            Lf[i] = cs.log_evalue(sp[i], float(m_rows[i]), Nx, spec)
        Lf = np.where(have, Lf, np.inf)
        conv[name] = {}
        for x in XS:
            lx = math.log(x)
            diff = (Lf <= lx) != (Lr <= lx)
            bnd = np.abs(Lr - lx) < S16_BOUNDARY
            conv[name][str(x)] = dict(foldmatch=int(((Lf <= lx) & M).sum()), research=int(((Lr <= lx) & M).sum()),
                                      boundary_pairs=int((bnd & M).sum()), differ_not_boundary=int((diff & ~bnd & M).sum()))
        conv[name]["max_abs_dlnE"] = float(np.max(np.abs(np.where(have, Lf - Lr, 0.0))))
    # ------------------------------------------------------------------ the filtered runs vs the inf run
    filt = {}
    for key, rr in runs.items():
        mx = float(key)
        if math.isinf(mx):
            continue
        A = rr["A"]
        want = have & (inf["lne"] <= math.log(mx))
        same_set = bool(np.array_equal(A["hit"], want))
        h = A["hit"]
        filt[key] = dict(hits=int(h.sum()), inf_hits_within=int(want.sum()), same_pairs=same_set,
                         s_f_bit_equal=bool(np.array_equal(A["s_f"][h].view(np.uint32), inf["s_f"][h].view(np.uint32))),
                         s_r_bit_equal=bool(np.array_equal(A["s_r"][h].view(np.uint32), inf["s_r"][h].view(np.uint32))),
                         lne_bit_equal=bool(np.array_equal(A["lne"][h], inf["lne"][h])),
                         q90_bit_equal=bool(np.array_equal(A["q90"][~np.isnan(A["q90"])],
                                                           inf["q90"][~np.isnan(A["q90"])])),
                         counts={str(x): {n: int(((A["lne"] <= math.log(x)) & h & M).sum()) for n, M in masks.items()}
                                 for x in XS if x <= mx})
        filt[key]["counts_equal_inf_run"] = all(filt[key]["counts"][str(x)][n] == counts[str(x)][n]["foldmatch"]
                                               for x in XS if x <= mx for n in masks)
    verdict = dict(
        s_f_ok=bool(cmp_["s_f"]["max_ratio_to_s16"] <= 1.0), s_r_ok=bool(cmp_["s_r"]["max_ratio_to_s16"] <= 1.0),
        q90_ok=bool(cmp_["q90"]["max_abs"] <= S16_Q90), lne_ok=bool(cmp_["ln_e"]["max_abs"] <= S16_LNE),
        counts_ok=bool(counts_ok), every_pair_compared=bool(cmp_["pairs_compared_s_f_s_r_lne"] == nq * nt),
        filtered_runs_consistent=bool(all(v["same_pairs"] and v["s_f_bit_equal"] and v["s_r_bit_equal"]
                                          and v["lne_bit_equal"] and v["counts_equal_inf_run"] for v in filt.values())))
    verdict["pass"] = bool(verdict["s_f_ok"] and verdict["s_r_ok"] and verdict["q90_ok"] and verdict["lne_ok"]
                           and verdict["counts_ok"] and verdict["filtered_runs_consistent"]
                           and verdict["every_pair_compared"])
    res.update(comparison=cmp_, counts=counts, counts_research_convention=conv, filtered_runs=filt,
               runs={k: dict(stats=v["stats"], wall_s=v["wall_s"], hits=int(v["A"]["hit"].sum())) for k, v in runs.items()},
               verdict=verdict, N=dict(N_all=float(N), N_ann=N_ann, N_rev=N_rev))
    res["wall_s"] = round(time.time() - t_start, 1)
    np.savez_compressed(os.path.join(rdir, "a3_foldmatch_scores.npz"), rows=rows, query_ids=np.array(qids),
                        s_f=inf["s_f"], s_r=inf["s_r"], q90=inf["q90"], lne=inf["lne"], hit=inf["hit"],
                        ecal=inf["ecal"])
    out = os.path.join(rdir, "a3_result.json")
    h = C.write_json(res, out)
    log(f"A3 {tag}: S_f max |d|/S16 {cmp_['s_f']['max_ratio_to_s16']:.4g} (max rel {cmp_['s_f']['max_rel']:.3g}), S_r "
        f"{cmp_['s_r']['max_ratio_to_s16']:.4g}; |dQ90| max {cmp_['q90']['max_abs']:.3g}; |d lnE| max "
        f"{cmp_['ln_e']['max_abs']:.3g}; pairs without hit {cmp_['pairs_without_hit']}")
    for x in XS:
        c = counts[str(x)]
        log(f"E <= {x:g}: " + "; ".join(f"{n} {v['foldmatch']}/{v['research']} (differ {v['decisions_differ']}, "
                                         f"boundary {v['boundary_pairs']})" for n, v in c.items()))
    log(f"VERDICT A3 {tag}: {'PASS' if verdict['pass'] else 'FAIL'} {json.dumps(verdict)}")
    log(f"wrote {out} sha256 {h}")
    return 0 if verdict["pass"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workers", type=int, required=True, help="explicit Stage-2 worker count")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--queries", type=int, default=None, help="trial: a subset of the 200 queries")
    ap.add_argument("--positions", type=int, nargs="+", default=None, help="trial: these positions of the q200 set")
    ap.add_argument("--max-evalues", type=float, nargs="+", default=[float("inf"), 10.0, 1e-3])
    ap.add_argument("--work", default=None, help="directory for the temporary database (default $TMPDIR)")
    ap.add_argument("--keep-db", action="store_true")
    a = ap.parse_args(argv)
    if not any(math.isinf(x) for x in a.max_evalues):
        ap.error("--max-evalues must include inf")
    a.max_evalues = sorted(a.max_evalues, key=lambda x: -x)          # inf first
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
