#!/usr/bin/env python
"""P8 A1: kernel full differential (plan FINAL v1, section 8 P8, A1 row; method S5-S9, S16).

FoldMatch's production esm-codes Stage-2 path, setting D, BLAS per-pair similarity, against the research setting-D
alignments of evalue/f1_settings (f1_eval.py = sw_eval/eval_all.py + xharness._run + harness.smith_waterman, float64):

  domains  12,228 CATH pairs: val related 2,198 / remote 3,409 / unrelated 1,600, test remote 3,421 / unrelated 1,600
           (encodings throughput/build/data/L2m_B8_s0.{val,test}.npz; stored alignments
           evalue/f1_settings/results/dom/M8_s0.{val,test}.npz, aln_D_*; pair 'a' = query, 'b' = target)
  chains   732 chain pairs (chains/data/pairs.json; encodings throughput/bench/data/L2m_B8_s0.chains.npz; stored
           alignments evalue/f1_settings/results/chains/M8_s0_D.npz)

FoldMatch side (the code under test, commit cc55c01): the query encodings and the target codes are written as the raw
per-run query memmap and code-store memmap the Stage-2 workers map, and every pair goes through the production worker
task functions of foldmatch.search.code_alignment on a CodeAlignmentPool with an explicit worker count:
  round 1  code_alignment._round1_task  -> swkern.score_fwd_rev (S_f, S_r; one fp32 GEMM per pair, S_r on its view)
  round 2  code_alignment._round2_task  -> swkern.align with traceback, the S12 column tallies and the CIGAR
the aligned residue pairs are rebuilt from the reported CIGAR and 1-based bounds (what the user sees). Cross-check: the
swkern API itself (swkern.query_prep + swkern.align, in this process) gives the same paths and scores bit for bit.

Pass (plan A1): >= 99.9% identical alignments (aligned residue pairs) over the 12,228 + 732 pairs; EVERY mismatch has
a float64-oracle margin < 1e-4 + 2e-6 |S_f| (S16: the float64 scores of the two alignments on the research float64 z
differ by less than that, i.e. a best-cell or traceback near-tie); mean SW F1 vs US-align within 1e-4 of 0.6396 /
0.2664 / 0.2620 / 0.6134 (val related / val remote / test remote / the 582 similar chain pairs). Also reported: S_f vs
the research float64 DP score within S16 (5e-5 max(|ref|, 1)); round-2 S_f == round-1 S_f bit for bit.

The float64 oracle is the research arithmetic: cosine of float64(q) / ||q|| and the decoded +-1 code / 8 (eval_all.
learned_sim_fn / chain_eval._cos), harness.normalize_scores(., 'rowcol') (imported from the staged harness.py), and
the setting-D score of an aligned-pair list (sum of z - 1 over the pairs, minus 5 + (L - 1) 0.5 per gap run in each
sequence; harness._align's affine convention, X <-> Y adjacency allowed). It reproduces the stored research scores
(checked, max |difference| reported).

Usage:
  python a1_kernel_differential.py extract
      NERSC: the reference extract from the research harness itself (harness.load_set / load_pairs: the 12,228 pair
      lists with their US-align references and the domain sequences) -> <P8_ROOT>/A1/reference_extract/ (the Mac,
      which has neither val.npz nor test.npz, uses a copy; its sha256 is checked).
  python a1_kernel_differential.py run --workers N [--tag TAG]
      -> <P8_ROOT>/A1/<TAG>/a1_result.json (+ a1_foldmatch_alignments.npz, a1_mismatches.json).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import acc_common as C  # noqa: E402  (first: BLAS thread variables = 1 before numpy loads)
import numpy as np  # noqa: E402

SETTING_D = (1.0, 5.0, 0.5)                    # (offset, gap open, gap extend): f1_settings 'D' = esm-codes-v1 S8
D_INDEX = 3                                    # f1_settings TAGS A, B, C, D -> sw[:, 3, :]
SETS = {"val": ("related", "remote", "unrelated"), "test": ("remote", "unrelated")}
F1_PLAN = {"val/related": 0.6396, "val/remote": 0.2664, "test/remote": 0.2620, "chains/similar": 0.6134}
F1_TOL = 1e-4
IDENTICAL_MIN = 0.999
EXPECT_PAIRS = {"val/related": 2198, "val/remote": 3409, "val/unrelated": 1600, "test/remote": 3421,
                "test/unrelated": 1600, "chains/all": 732}
N_DOMAIN, N_CHAIN, N_CHAIN_SIMILAR = 12228, 732, 582
HARNESS_MD5 = "05f387cb400cc85dbacaef771a275eeb"          # plan Appendix A (oracle copy)
F1_NERSC_MD5 = {"dom/M8_s0.val.npz": "67a6ee1d9d9825978c8cd7a25869c8a9",
                "dom/M8_s0.test.npz": "f4225dd075ef82288748749dcbb016ef",
                "chains/M8_s0_D.npz": "9931e3c82bcde55dc0bde12fd75fcad5"}  # f1_settings results/nersc_md5.txt
INP = {
    "enc/val": ("embcomp/throughput/build/data/L2m_B8_s0.val.npz", "5aebc14942665748313f22ca737ca26d"),
    "enc/test": ("embcomp/throughput/build/data/L2m_B8_s0.test.npz", "e570bc26e1a142183aaf85e9a7a57df7"),
    "enc/chains": ("embcomp/throughput/bench/data/L2m_B8_s0.chains.npz", "5519866044ab0f348c74b9319125acd2"),
    "dom/val": ("embcomp/evalue/f1_settings/results/dom/M8_s0.val.npz", F1_NERSC_MD5["dom/M8_s0.val.npz"]),
    "dom/test": ("embcomp/evalue/f1_settings/results/dom/M8_s0.test.npz", F1_NERSC_MD5["dom/M8_s0.test.npz"]),
    "chains/result": ("embcomp/evalue/f1_settings/results/chains/M8_s0_D.npz", F1_NERSC_MD5["chains/M8_s0_D.npz"]),
    "chains/pairs": ("embcomp/chains/data/pairs.json", None),
    "chains/meta": ("embcomp/chains/data/chains_meta.json", None),
    "harness": ("embcomp/harness/harness.py", HARNESS_MD5),
}
EXTRACT_DIR = os.path.join(C.P8_ROOT, "A1", "reference_extract")
EXTRACT = os.path.join(EXTRACT_DIR, "a1_reference_extract.npz")
EXTRACT_RECORD = os.path.join(EXTRACT_DIR, "a1_reference_extract.json")
log = C.log


# =====================================================================================================================
# research side
# =====================================================================================================================
def _harness():
    """The research harness (harness.py, plan Appendix A md5 05f387cb...), imported from the staged copy."""
    p = C.inp(INP["harness"][0])
    got = C.md5_file(p)
    if got != HARNESS_MD5:
        raise RuntimeError(f"harness.py md5 {got} != {HARNESS_MD5}")
    d = os.path.dirname(p)
    if d not in sys.path:
        sys.path.append(d)
    import harness as H  # noqa: E402
    assert os.path.realpath(H.__file__) == os.path.realpath(p), H.__file__
    return H


def _flatten(arrs):
    arrs = [np.asarray(a, np.int32).reshape(-1, 2) for a in arrs]
    off = np.zeros(len(arrs) + 1, np.int64)
    off[1:] = np.cumsum([len(a) for a in arrs])
    return (np.concatenate(arrs) if arrs else np.zeros((0, 2), np.int32)), off


def extract(args) -> int:
    """NERSC: the pair lists, US-align references and domain sequences straight from the research harness."""
    os.environ["EMBCOMP_DATA"] = os.path.join(C.INPUTS, "embcomp")
    H = _harness()
    t0 = time.time()
    blob, meta = {}, {"what": "A1 reference extract: research harness.load_set / load_pairs output (pair lists, "
                              "US-align reference alignments, domain ids and sequences)",
                      "embcomp_data": os.environ["EMBCOMP_DATA"], "harness": H.__file__, "harness_md5": HARNESS_MD5,
                      "load_pairs_stats": {}, "source_md5": {}}
    for split in ("val", "test"):
        es = H.load_set(split)
        blob[f"{split}/domain_ids"] = np.array(es.ids)
        blob[f"{split}/sequences"] = np.array(es.seqs)
        blob[f"{split}/lengths"] = (es.ends - es.starts).astype(np.int64)
        pairs, stats = H.load_pairs(split)
        meta["load_pairs_stats"][split] = stats
        for s in SETS[split]:
            P = pairs[s]
            blob[f"{split}/{s}/a"] = np.array([p.a for p in P])
            blob[f"{split}/{s}/b"] = np.array([p.b for p in P])
            blob[f"{split}/{s}/tm"] = np.array([p.tm for p in P], np.float64)
            blob[f"{split}/{s}/has_ref"] = np.array([p.ref is not None for p in P])
            blob[f"{split}/{s}/ref_flat"], blob[f"{split}/{s}/ref_off"] = _flatten(
                [p.ref if p.ref is not None else np.zeros((0, 2), np.int32) for p in P])
        meta["source_md5"][f"embcomp/{split}.npz"] = C.md5_file(C.inp(f"embcomp/{split}.npz"))
    for sp in ("val", "test"):
        for table, refs in H.PAIR_TABLES[sp].values():
            for x in (table, refs):
                if x:
                    meta["source_md5"][f"embcomp/{x}"] = C.md5_file(C.inp(f"embcomp/{x}"))
    blob["meta_json"] = np.array(json.dumps(meta))
    os.makedirs(EXTRACT_DIR, exist_ok=True)
    tmp = EXTRACT + f".tmp{os.getpid()}.npz"
    np.savez(tmp, **blob)
    os.replace(tmp, EXTRACT)
    rec = dict(meta, extract=EXTRACT, sha256=C.sha256_file(EXTRACT), md5=C.md5_file(EXTRACT),
               wall_s=round(time.time() - t0, 1),
               provenance=C.provenance(kernel=False, require_commit=C.EXPECTED_FM_COMMIT))
    h = C.write_json(rec, EXTRACT_RECORD)
    log(f"wrote {EXTRACT} sha256 {rec['sha256']}; record {EXTRACT_RECORD} sha256 {h}")
    return 0


def load_extract() -> tuple:
    rec = C.read_json(EXTRACT_RECORD)
    got = C.sha256_file(EXTRACT)
    if got != rec["sha256"]:
        raise RuntimeError(f"{EXTRACT}: sha256 {got} != the extract record's {rec['sha256']}")
    return np.load(EXTRACT), rec


def research_z(q16: np.ndarray, codes: np.ndarray, H) -> np.ndarray:
    """float64 rowcol z of a pair, the research arithmetic: cosine(float64 q / ||q||, (+-1 code) / ||code||)
    (eval_all.learned_sim_fn; chain_eval._cos is the same for nonzero rows), then harness.normalize_scores."""
    Q = q16.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
        T = np.unpackbits(codes, axis=1)[:, :64].astype(np.float64) * 2 - 1
        T = T / np.linalg.norm(T, axis=1, keepdims=True)
        sim = Q @ T.T
    nonfinite = int(np.count_nonzero(~np.isfinite(sim)))
    if nonfinite:
        sim = np.nan_to_num(sim, nan=0.0, posinf=1.0, neginf=-1.0)
    return H.normalize_scores(sim, "rowcol"), nonfinite


def pairs_score(Z: np.ndarray, P: np.ndarray, sw=SETTING_D) -> float:
    """Setting-D score of an aligned-pair list on a float64 z (harness._align convention: cell z - offset; a gap run
    of length L in either sequence costs open + (L - 1) extend; X and Y runs between two pairs counted separately)."""
    off, go, ge = sw
    P = np.asarray(P, np.int64).reshape(-1, 2)
    if not len(P):
        return 0.0
    s = float(np.sum(Z[P[:, 0], P[:, 1]] - off))
    for d in (np.diff(P[:, 0]) - 1, np.diff(P[:, 1]) - 1):
        g = d[d > 0]
        s -= g.size * go + float(g.sum() - g.size) * ge
    return s


def prf(H, pred, ref):
    return H.prf(np.asarray(pred, np.int32).reshape(-1, 2), np.asarray(ref, np.int32).reshape(-1, 2))


# =====================================================================================================================
# FoldMatch side
# =====================================================================================================================
def cigar_pairs(cigar: str, q_start: int, t_start: int) -> np.ndarray:
    """Aligned (query, target) residue pairs, 0-based, from a reported CIGAR (M both residues, I query residue vs a
    target gap, D a gap in the query; swkern.path_ops) and the 1-based start bounds."""
    from foldmatch.search import swkern
    ops = swkern.expand_cigar(cigar)
    i, j = q_start - 1, t_start - 1
    out = []
    for o in ops:
        if o == "M":
            out.append((i, j))
            i += 1
            j += 1
        elif o == "I":
            i += 1
        else:
            j += 1
    return np.asarray(out, np.int32).reshape(-1, 2)


def path_pairs(path: np.ndarray) -> np.ndarray:
    p = np.asarray(path).reshape(-1, 2)
    return p[(p[:, 0] >= 0) & (p[:, 1] >= 0)].astype(np.int32)


# =====================================================================================================================
# run
# =====================================================================================================================
def _load_inputs(rec_inputs: dict):
    out = {}
    for key, (name, md5) in INP.items():
        p = C.inp(name)
        got = C.md5_file(p)
        rec_inputs[key] = dict(name=name, path=p, realpath=os.path.realpath(p), md5=got, expected_md5=md5,
                               match=None if md5 is None else got == md5)
        if md5 is not None and got != md5:
            raise RuntimeError(f"{name}: md5 {got} != {md5}")
        out[key] = p
    return out


def run(args) -> int:
    t_start = time.time()
    tag = args.tag or C.HOST
    rdir = C.result_dir("A1", tag)
    work = C.result_dir("A1", tag, "work")
    res = {"check": "A1", "what": __doc__.split("\n\n")[0], "tag": tag, "workers": args.workers,
           "criteria": {"identical_fraction_min": IDENTICAL_MIN, "mismatch_margin": "< 1e-4 + 2e-6 |S_f| (float64 "
                        "oracle: |score64(FoldMatch alignment) - score64(research alignment)| on the research z)",
                        "f1_plan": F1_PLAN, "f1_tol": F1_TOL, "s16_score_bound": "5e-5 max(|ref|, 1)"},
           "inputs": {}}
    res["provenance"] = C.provenance(kernel_selftest=False)
    paths = _load_inputs(res["inputs"])
    H = _harness()
    ext, ext_rec = load_extract()
    res["inputs"]["reference_extract"] = dict(path=EXTRACT, sha256=ext_rec["sha256"], record=EXTRACT_RECORD,
                                              record_sha256=C.sha256_file(EXTRACT_RECORD),
                                              made_on=ext_rec["provenance"]["host"]["hostname"])
    from foldmatch.search import code_alignment as ca
    from foldmatch.search import scoring_system as ss
    from foldmatch.search import swkern
    # ------------------------------------------------------------------ encodings -> the workers' memmaps
    enc = {g: np.load(paths[f"enc/{g}"]) for g in ("val", "test", "chains")}
    q_parts, c_parts, base, groups = [], [], {}, {}
    row = 0
    for g in ("val", "test", "chains"):
        z = enc[g]
        q, codes, off, ids = z["q"], z["codes"], z["offsets"], z["ids"].astype(str)
        assert q.dtype == np.float16 and q.shape[1] == 64 and codes.dtype == np.uint8 and codes.shape[1] == 8
        assert len(q) == len(codes) == off[-1] and len(off) == len(ids) + 1
        base[g] = row
        groups[g] = dict(index={d: k for k, d in enumerate(ids)}, off=off, ids=ids)
        q_parts.append(q)
        c_parts.append(codes)
        row += len(q)
    Q16 = np.ascontiguousarray(np.concatenate(q_parts))
    CODES = np.ascontiguousarray(np.concatenate(c_parts))
    q_path, c_path = os.path.join(work, "queries.f16"), os.path.join(work, "codes.u8")
    Q16.astype("<f2").tofile(q_path)
    CODES.tofile(c_path)
    res["memmaps"] = dict(queries=q_path, codes=c_path, rows=int(len(Q16)), queries_sha256=C.sha256_file(q_path),
                          codes_sha256=C.sha256_file(c_path))

    def rows_of(g, d):
        G = groups[g]
        k = G["index"][d]
        return base[g] + int(G["off"][k]), int(G["off"][k + 1] - G["off"][k])

    # ------------------------------------------------------------------ the pairs and the research references
    pairs = []          # dict per pair
    chk = {}
    for split in ("val", "test"):
        z = np.load(paths[f"dom/{split}"])
        meta = json.loads(str(z["meta_json"]))
        assert meta["tags"][D_INDEX] == "D" and tuple(meta["sw_params"][D_INDEX]) == SETTING_D, meta
        seqs = dict(zip(ext[f"{split}/domain_ids"].astype(str).tolist(), ext[f"{split}/sequences"].astype(str).tolist()))
        for s in SETS[split]:
            a, b = z[f"{s}/a"].astype(str), z[f"{s}/b"].astype(str)
            same = (np.array_equal(a, ext[f"{split}/{s}/a"].astype(str)) and
                    np.array_equal(b, ext[f"{split}/{s}/b"].astype(str)) and
                    np.array_equal(z[f"{s}/idx"], np.arange(len(a))))
            chk[f"{split}/{s}/stored_pairs_equal_harness_load_pairs"] = bool(same)
            if not same:
                raise RuntimeError(f"{split}/{s}: stored pair ids differ from harness.load_pairs")
            sw = z[f"{s}/sw"][:, D_INDEX, :]
            af, ao = z[f"{s}/aln_D_flat"], z[f"{s}/aln_D_off"]
            hr = ext[f"{split}/{s}/has_ref"]
            rf, ro = ext[f"{split}/{s}/ref_flat"], ext[f"{split}/{s}/ref_off"]
            assert int(z[f"{s}/nonfinite"].sum()) == 0
            for k in range(len(a)):
                q0, n = rows_of(split, a[k])
                t0, m = rows_of(split, b[k])
                assert (n, m) == (int(z[f"{s}/len_a"][k]), int(z[f"{s}/len_b"][k])), (split, s, k)
                assert len(seqs[a[k]]) == n and len(seqs[b[k]]) == m
                pairs.append(dict(set=f"{split}/{s}", k=k, a=a[k], b=b[k], q0=q0, n=n, t0=t0, m=m,
                                  qseq=seqs[a[k]], tseq=seqs[b[k]], ref_aln=af[ao[k]:ao[k + 1]],
                                  ref_score=float(sw[k, 0]), ref_n=float(sw[k, 1]), ref_f1=float(sw[k, 4]),
                                  usref=rf[ro[k]:ro[k + 1]] if hr[k] else None))
    cp = json.load(open(paths["chains/pairs"]))
    cm = json.load(open(paths["chains/meta"]))
    cz = np.load(paths["chains/result"])
    cmeta = json.loads(str(cz["meta_json"]))
    assert tuple(cmeta["sw"]) == SETTING_D and cmeta["normalize"] is True and bool(cz["evaluated"].all()), cmeta
    coff, cflat = cz["aln_off"], cz["aln_flat"]
    for k, r in enumerate(cp):
        q0, n = rows_of("chains", r["a"])
        t0, m = rows_of("chains", r["b"])
        assert (n, m) == (int(cz["f/len_a"][k]), int(cz["f/len_b"][k])), k
        qs, ts = cm[r["a"]]["sequence"], cm[r["b"]]["sequence"]
        assert len(qs) == n and len(ts) == m
        ref = r.get("ref_primary")
        ref = None if ref is None else np.asarray(ref, np.int64).reshape(-1, 2)
        pairs.append(dict(set="chains/all", k=k, a=r["a"], b=r["b"], q0=q0, n=n, t0=t0, m=m, qseq=qs, tseq=ts,
                          kind=r["kind"], ref_aln=cflat[coff[k]:coff[k + 1]], ref_score=float(cz["f/sw_score"][k]),
                          ref_n=float(cz["f/sw_n"][k]), ref_f1=float(cz["f/f1"][k]),
                          usref=ref if ref is not None and len(ref) else None))
    counts = {s: sum(1 for p in pairs if p["set"] == s) for s in EXPECT_PAIRS}
    chk["pair_counts"] = counts
    if counts != EXPECT_PAIRS:
        raise RuntimeError(f"pair counts {counts} != {EXPECT_PAIRS}")
    log(f"{len(pairs)} pairs ({N_DOMAIN} domain + {N_CHAIN} chain)")

    # ------------------------------------------------------------------ FoldMatch: production worker tasks on a pool
    system = ss.get(ss.DEFAULT_ESM_CODES)
    prepared = ss.prepare(system, local_files_only=True)
    spec = prepared.for_worker()
    dp = spec.dp_params()
    if (dp.offset, dp.gap_open, dp.gap_extend) != SETTING_D or dp != swkern.SETTING_D:
        raise RuntimeError(f"registry DP parameters {dp} != setting D {SETTING_D}")
    kernel = swkern.load()
    init = ca._WorkerInit(isa=kernel.isa, spec=spec, codes=(c_path, int(len(CODES)), 8),
                          queries=(q_path, int(len(Q16)), 64), block_floats=swkern.DEFAULT_BLOCK_FLOATS,
                          max_pair_cells=swkern.DEFAULT_MAX_PAIR_CELLS)
    res["foldmatch_setup"] = dict(system=system.name, digest=system.digest(), dp=[dp.offset, dp.gap_open, dp.gap_extend],
                                  kernel=kernel.info(), bundle_files={k: str(v) for k, v in prepared.files.items()},
                                  block_floats=swkern.DEFAULT_BLOCK_FLOATS,
                                  max_pair_cells=swkern.DEFAULT_MAX_PAIR_CELLS, task_api="code_alignment._round1_task "
                                  "/ _round2_task on CodeAlignmentPool(%d) (spawn)" % args.workers)
    # one task per query (and per pair set group of it), the targets in the stored order
    by_q = {}
    for i, p in enumerate(pairs):
        by_q.setdefault((p["q0"], p["n"]), []).append(i)
    keys = sorted(by_q, key=lambda key: -key[1] * sum(pairs[i]["m"] for i in by_q[key]))   # LPT
    task_pairs = {tid: np.array([[pairs[i]["t0"], pairs[i]["m"]] for i in by_q[key]], np.int64)
                  for tid, key in enumerate(keys)}
    r1_tasks = [(tid, ca._CODES, key[0], key[1], task_pairs[tid]) for tid, key in enumerate(keys)]
    r2_tasks = [(tid, key[0], key[1], task_pairs[tid], pairs[by_q[key][0]]["qseq"],
                 [pairs[i]["tseq"] for i in by_q[key]], (True, False, False, False)) for tid, key in enumerate(keys)]
    S_f1 = np.full(len(pairs), np.nan, np.float32)
    S_r1 = np.full(len(pairs), np.nan, np.float32)
    S_f2 = np.full(len(pairs), np.nan, np.float32)
    skipped = np.zeros(len(pairs), bool)
    cols = np.zeros((len(pairs), len(ca._COLUMNS)), np.int64)
    ends = np.zeros((len(pairs), 2), np.int64)
    end_ok = np.zeros(len(pairs), bool)
    cigars = [None] * len(pairs)
    t = time.perf_counter()
    with ca.CodeAlignmentPool(args.workers) as pool:
        for tid, sf, sr, sk in pool.run(ca._round1_task, r1_tasks, init):
            ii = by_q[keys[tid]]
            S_f1[ii], S_r1[ii], skipped[ii] = sf, sr, sk
        t1 = time.perf_counter()
        for out in pool.run(ca._round2_task, r2_tasks, init):
            tid, sc, en, co, eo, _ns, _dist, sk, strings = out
            ii = by_q[keys[tid]]
            S_f2[ii], ends[ii], cols[ii], end_ok[ii] = sc, en, co, eo
            skipped[ii] |= sk
            for j, i in enumerate(ii):
                cigars[i] = strings[0][j]
        t2 = time.perf_counter()
        res["pool"] = dict(workers=pool.workers, processes=pool.processes, starts=pool.starts)
    res["timing"] = dict(round1_s=round(t1 - t, 3), round2_s=round(t2 - t1, 3), tasks=len(r1_tasks))
    log(f"FoldMatch rounds 1 + 2: {len(r1_tasks)} tasks each, {t1 - t:.2f} s + {t2 - t1:.2f} s on {args.workers} workers")
    ci = {c: j for j, c in enumerate(ca._COLUMNS)}
    fm_aln = [cigar_pairs(cigars[i], int(cols[i, ci["q_start"]]), int(cols[i, ci["t_start"]])) if cigars[i]
              else np.zeros((0, 2), np.int32) for i in range(len(pairs))]

    # ------------------------------------------------------------------ cross-check: the swkern API in this process
    t = time.perf_counter()
    api_same_pairs = api_same_score = 0
    api_paths = [None] * len(pairs)
    for tid, key in enumerate(keys):
        qp = swkern.query_prep(Q16[key[0]:key[0] + key[1]], kernel)
        tp = task_pairs[tid]
        a = swkern.align(qp, CODES, tp[:, 0], swkern.SETTING_D, lengths=tp[:, 1], traceback=True, kernel=kernel)
        for j, i in enumerate(by_q[key]):
            api_paths[i] = a.path_of(j).copy()
            api_same_pairs += int(np.array_equal(path_pairs(api_paths[i]), fm_aln[i]))
            api_same_score += int(np.float32(a.scores[j]).view(np.uint32) == np.float32(S_f2[i]).view(np.uint32))
    res["timing"]["swkern_api_s"] = round(time.perf_counter() - t, 3)

    # ------------------------------------------------------------------ comparisons
    t = time.perf_counter()
    per = []
    for i, p in enumerate(pairs):
        q16 = Q16[p["q0"]:p["q0"] + p["n"]]
        codes = CODES[p["t0"]:p["t0"] + p["m"]]
        same = bool(np.array_equal(fm_aln[i], np.asarray(p["ref_aln"], np.int32).reshape(-1, 2)))
        Z, nonfinite = research_z(q16, codes, H)
        s64_ref = pairs_score(Z, p["ref_aln"])
        s64_fm = pairs_score(Z, fm_aln[i])
        s64_fm_path = swkern.path_score(Z, api_paths[i], swkern.SETTING_D) if len(api_paths[i]) else 0.0
        sf = float(S_f1[i])
        f1 = None
        if p["usref"] is not None:
            f1 = prf(H, fm_aln[i], p["usref"])[2]
        per.append(dict(same=same, s64_ref=s64_ref, s64_fm=s64_fm, s64_fm_path=s64_fm_path, nonfinite=nonfinite,
                        oracle_vs_stored=abs(s64_ref - p["ref_score"]),
                        s_ratio=abs(sf - p["ref_score"]) / (5e-5 * max(abs(p["ref_score"]), 1.0)),
                        margin=abs(s64_fm - s64_ref), bound=1e-4 + 2e-6 * abs(sf), f1=f1,
                        f1_ref_recomputed=(prf(H, p["ref_aln"], p["usref"])[2] if p["usref"] is not None else None)))
    res["timing"]["oracle_s"] = round(time.perf_counter() - t, 3)

    # ------------------------------------------------------------------ summaries
    def summary(idx):
        idx = list(idx)
        n_same = sum(per[i]["same"] for i in idx)
        mis = [i for i in idx if not per[i]["same"]]
        return dict(pairs=len(idx), identical=n_same, identical_fraction=n_same / max(len(idx), 1),
                    mismatches=len(mis), mismatches_within_margin=sum(per[i]["margin"] < per[i]["bound"] for i in mis),
                    max_score_ratio_s16=max(per[i]["s_ratio"] for i in idx),
                    max_abs_sf_minus_research=max(abs(float(S_f1[i]) - pairs[i]["ref_score"]) for i in idx),
                    max_oracle_vs_stored_score=max(per[i]["oracle_vs_stored"] for i in idx),
                    aligned_pairs_fm=int(sum(len(fm_aln[i]) for i in idx)),
                    aligned_pairs_research=int(sum(len(pairs[i]["ref_aln"]) for i in idx)))
    sets = {}
    for s in EXPECT_PAIRS:
        idx = [i for i, p in enumerate(pairs) if p["set"] == s]
        sets[s] = summary(idx)
    chain_sim = [i for i, p in enumerate(pairs) if p["set"] == "chains/all" and p["kind"] != "unrelated"]
    sets["chains/similar"] = summary(chain_sim)
    assert len(chain_sim) == N_CHAIN_SIMILAR
    # F1 (research means: nanmean of sw[:, D, 4] (domains); chain_eval summarize: nan_to_num over the similar pairs)
    f1 = {}
    for s in ("val/related", "val/remote", "test/remote", "chains/similar"):
        idx = chain_sim if s == "chains/similar" else [i for i, p in enumerate(pairs) if p["set"] == s]
        fm = np.array([np.nan if per[i]["f1"] is None else per[i]["f1"] for i in idx], np.float64)
        stored = np.array([pairs[i]["ref_f1"] for i in idx], np.float64)
        rec = np.array([np.nan if per[i]["f1_ref_recomputed"] is None else per[i]["f1_ref_recomputed"] for i in idx])
        if s == "chains/similar":
            fm_mean, st_mean, rc_mean = (float(np.mean(np.nan_to_num(x))) for x in (fm, stored, rec))
        else:
            fm_mean, st_mean, rc_mean = (float(np.nanmean(x)) for x in (fm, stored, rec))
        f1[s] = dict(n=len(idx), n_with_reference=int(np.isfinite(fm).sum()), foldmatch=fm_mean,
                     research_stored=st_mean, research_recomputed_with_harness_prf=rc_mean, plan=F1_PLAN[s],
                     abs_diff_vs_plan=abs(fm_mean - F1_PLAN[s]), abs_diff_vs_stored=abs(fm_mean - st_mean),
                     pass_=bool(abs(fm_mean - F1_PLAN[s]) <= F1_TOL),
                     per_pair_f1_equal_stored=int(np.sum((np.nan_to_num(fm, nan=-1) == np.nan_to_num(stored, nan=-1)))))
    dom_idx = [i for i, p in enumerate(pairs) if p["set"] != "chains/all"]
    all_idx = list(range(len(pairs)))
    totals = dict(domains=summary(dom_idx), chains=summary([i for i, p in enumerate(pairs) if p["set"] == "chains/all"]),
                  all=summary(all_idx))
    mism = []
    for i in all_idx:
        if per[i]["same"]:
            continue
        p = pairs[i]
        mism.append(dict(set=p["set"], k=p["k"], a=p["a"], b=p["b"], n=p["n"], m=p["m"], S_f=float(S_f1[i]),
                         research_score=p["ref_score"], score64_research_alignment=per[i]["s64_ref"],
                         score64_foldmatch_alignment=per[i]["s64_fm"], margin=per[i]["margin"], bound=per[i]["bound"],
                         within_margin=bool(per[i]["margin"] < per[i]["bound"]), aligned_fm=int(len(fm_aln[i])),
                         aligned_research=int(len(p["ref_aln"])),
                         end_fm=[int(ends[i, 0]), int(ends[i, 1])],
                         end_research=(np.asarray(p["ref_aln"])[-1] + 1).tolist() if len(p["ref_aln"]) else [0, 0],
                         f1_fm=per[i]["f1"], f1_research=p["ref_f1"]))
    internal = dict(
        skipped_pairs=int(skipped.sum()),
        round2_sf_bit_equal_round1=int(np.sum(S_f1.view(np.uint32) == S_f2.view(np.uint32))),
        paths_end_at_best_cell=int(end_ok.sum()),
        swkern_api_aligned_pairs_equal_cigar=api_same_pairs, swkern_api_scores_bit_equal=api_same_score,
        oracle_pairs_score_vs_swkern_path_score_max=max(abs(per[i]["s64_fm"] - per[i]["s64_fm_path"]) for i in all_idx),
        nonfinite_similarities=int(sum(x["nonfinite"] for x in per)),
        aligned_count_equal_research_n=int(sum(len(fm_aln[i]) == int(pairs[i]["ref_n"]) for i in all_idx)),
        n_pairs=len(pairs), checks=chk)
    verdict = dict(
        identical_fraction_all=totals["all"]["identical_fraction"],
        identical_ok=bool(totals["all"]["identical_fraction"] >= IDENTICAL_MIN),
        identical_ok_domains=bool(totals["domains"]["identical_fraction"] >= IDENTICAL_MIN),
        identical_ok_chains=bool(totals["chains"]["identical_fraction"] >= IDENTICAL_MIN),
        every_mismatch_within_margin=bool(all(x["within_margin"] for x in mism)),
        f1_ok=bool(all(v["pass_"] for v in f1.values())),
        s16_scores_ok=bool(totals["all"]["max_score_ratio_s16"] <= 1.0),
        internal_ok=bool(internal["skipped_pairs"] == 0 and internal["round2_sf_bit_equal_round1"] == len(pairs)
                         and internal["paths_end_at_best_cell"] == len(pairs)
                         and api_same_pairs == len(pairs) and api_same_score == len(pairs)))
    verdict["pass"] = bool(verdict["identical_ok"] and verdict["identical_ok_domains"] and verdict["identical_ok_chains"]
                           and verdict["every_mismatch_within_margin"] and verdict["f1_ok"])
    res.update(sets=sets, totals=totals, f1=f1, mismatches=mism, internal=internal, verdict=verdict)
    res["timing"]["total_s"] = round(time.time() - t_start, 1)
    np.savez_compressed(os.path.join(rdir, "a1_foldmatch_alignments.npz"),
                        set=np.array([p["set"] for p in pairs]), k=np.array([p["k"] for p in pairs]),
                        a=np.array([p["a"] for p in pairs]), b=np.array([p["b"] for p in pairs]),
                        s_f=S_f1, s_r=S_r1, s_f_round2=S_f2, ends=ends, columns=cols,
                        cigar=np.array([c or "" for c in cigars]),
                        aln_flat=_flatten(fm_aln)[0], aln_off=_flatten(fm_aln)[1])
    out = os.path.join(rdir, "a1_result.json")
    h = C.write_json(res, out)
    log(f"A1 {tag}: identical {totals['all']['identical']}/{totals['all']['pairs']} "
        f"({totals['all']['identical_fraction']:.6f}); domains {totals['domains']['identical']}/{totals['domains']['pairs']}, "
        f"chains {totals['chains']['identical']}/{totals['chains']['pairs']}; mismatches {len(mism)} "
        f"(within margin {sum(x['within_margin'] for x in mism)}); max S16 score ratio "
        f"{totals['all']['max_score_ratio_s16']:.4g}")
    for s, v in f1.items():
        log(f"F1 {s}: FoldMatch {v['foldmatch']:.6f}  research {v['research_stored']:.6f}  plan {v['plan']}  "
            f"|d| {v['abs_diff_vs_plan']:.2e}  {'PASS' if v['pass_'] else 'FAIL'}")
    log(f"internal {json.dumps(C.jsonable({k: v for k, v in internal.items() if k != 'checks'}))}")
    log(f"VERDICT A1 {tag}: {'PASS' if verdict['pass'] else 'FAIL'} {json.dumps(verdict)}")
    log(f"wrote {out} sha256 {h}")
    return 0 if verdict["pass"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("extract")
    r = sub.add_parser("run")
    r.add_argument("--workers", type=int, required=True, help="explicit Stage-2 worker count (> 1: spawn pool)")
    r.add_argument("--tag", default=None, help="result subdirectory (default: the host kind, mac / nersc)")
    a = ap.parse_args(argv)
    return extract(a) if a.cmd == "extract" else run(a)


if __name__ == "__main__":
    sys.exit(main())
