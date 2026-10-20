#!/usr/bin/env python
"""P8 A5 / A6 / A8 helpers (plan FINAL v1, section 8 P8): FoldMatch's production esm-codes Stage 2 (commit cc55c01)
driven exhaustively, with a read-only observer of its per-query statistics.

* ``run_stage2``: ``foldmatch.search.code_alignment.align_candidates_esm_codes`` - the function that
  ``fm-search query sequences --scoring esm-codes`` calls - with a given candidate list for every query (A5: all 50,000
  targets), an explicit worker pool (``CodeAlignmentPool(workers)``) and a finite ``max_evalue``.  While it runs,
  ``Observer`` wraps the module's ``_statistics`` (S9-S11 in the parent: called once per query after its round-1
  tasks are back) and copies the query's round-1 ``S_f`` / ``S_r`` (float32, candidate order), its skipped mask, its
  panel ``S_f`` / ``S_r``, Q90 and ln E: the very arrays the aligner computes its survivors from.  The wrapper calls
  the original first and only copies; the hits are what the aligner returns.
* ``hits_arrays``: the returned hits as flat arrays (full precision: score / score_rev are the exact float32 values,
  q90 / score_adj / log_evalue float64).
* ``ArrayCodeStore``: the four members of ``ResidueCodeStore`` the aligner uses (``bin_path``, ``manifest()``,
  ``open_memmap()``, ``lookup_rows()``), over a raw ``uint8 [N_res, 8]`` file: the production Stage 2 on research
  encodings (A5 control, A6).
* ``query_index_from_arrays`` / ``query_index_from_capture``: a ``QueryEncodingIndex`` over a raw fp16 file.
* ``coverage_max_evalue``: the smallest production ``max_evalue`` (shipped table, N of the database) whose survivors
  provably include every pair with E <= x under any other table / N (E decreases in S' under every table).

Import ``acc_common`` first (it sets the BLAS thread variables to 1 before numpy is imported).
"""
from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import acc_common as AC  # noqa: F401  (first: BLAS threads = 1)
import numpy as np

# every scalar column of an esm-codes hit (no strings: they cost time and are not needed here)
STAGE2_FIELDS = ["query", "target", "embscore", "fident", "alnlen", "mismatch", "gapopen", "qstart", "qend", "tstart",
                 "tend", "evalue", "bits", "raw", "rawrev", "q90", "rawadj", "ecal", "qlen", "tlen", "qcov", "tcov",
                 "nident"]


# =====================================================================================================================
# code stores and query encodings
# =====================================================================================================================
class ArrayCodeStore:
    """Read-only stand-in for ``foldmatch.search.residue_code_store.ResidueCodeStore`` over a raw ``uint8 [N_res,
    code_bytes]`` file, with exactly the members ``align_candidates_esm_codes`` uses: ``bin_path``, ``manifest()``
    (total_residues, code_bytes), ``open_memmap()`` and ``lookup_rows(ids)`` (int64 offsets / lengths aligned with
    ids, repeats allowed; an unknown id raises KeyError)."""

    def __init__(self, bin_path, ids: Sequence[str], offsets, lengths, code_bytes: int = 8):
        self.bin_path = Path(bin_path)
        self.ids = list(ids)
        self.offsets = np.ascontiguousarray(offsets, np.int64)
        self.lengths = np.ascontiguousarray(lengths, np.int64)
        assert len(self.ids) == len(self.offsets) == len(self.lengths)
        self.code_bytes = int(code_bytes)
        self.total = int(self.lengths.sum())
        self._pos = {t: k for k, t in enumerate(self.ids)}
        assert len(self._pos) == len(self.ids), "duplicate ids"

    def manifest(self) -> dict:
        return {"total_residues": self.total, "code_bytes": self.code_bytes, "kind": "ArrayCodeStore (P8 harness)",
                "bin": str(self.bin_path)}

    def open_memmap(self) -> np.ndarray:
        size = self.bin_path.stat().st_size
        if size != self.total * self.code_bytes:
            raise ValueError(f"{self.bin_path}: {size} bytes, expected {self.total * self.code_bytes}")
        return np.memmap(self.bin_path, dtype=np.uint8, mode="r", shape=(self.total, self.code_bytes))

    def lookup_rows(self, ids):
        ids = ids if isinstance(ids, list) else list(ids)
        pick = np.fromiter((self._pos[t] for t in ids), np.int64, len(ids))
        return np.ascontiguousarray(self.offsets[pick]), np.ascontiguousarray(self.lengths[pick])


def write_raw(path, arr) -> str:
    """Write a C-contiguous array's bytes (little-endian host) to path; returns the sha256 of the file."""
    arr = np.ascontiguousarray(arr)
    assert arr.dtype.byteorder in ("=", "<", "|")
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as fh:
        fh.write(memoryview(arr).cast("B"))
    os.replace(tmp, path)
    return AC.sha256_file(path)


def query_index_from_arrays(path, ids: Sequence[str], offsets, lengths, seqs: Sequence[str], system_name: str,
                            width: int = 64):
    """A QueryEncodingIndex over a raw fp16 [rows, width] file (entries: offset, length, seq_sha1 of the normalised
    sequence, S15) - what the CLI's query pass builds (QueryEncodingIndex.build) for its own run."""
    from foldmatch.search.query_encodings import QueryEncodingEntry, QueryEncodingIndex
    from foldmatch.utils.sequence import normalize_sequence, seq_sha1
    offsets = np.asarray(offsets, np.int64)
    lengths = np.asarray(lengths, np.int64)
    rows = int(os.path.getsize(path) // (2 * width))
    entries = {}
    for q, o, n, s in zip(ids, offsets.tolist(), lengths.tolist(), seqs):
        s = normalize_sequence(s)
        assert len(s) == n, (q, len(s), n)
        entries[q] = QueryEncodingEntry(int(o), int(n), seq_sha1(s))
    assert int((offsets + lengths).max()) <= rows
    return QueryEncodingIndex(path, rows, width, entries, system_name)


def query_index_from_capture(capdir):
    """The QueryEncodingIndex of a pipeline run, from a8_pipeline.py's capture (queries.f16 + query_encodings.json)."""
    from foldmatch.search.query_encodings import QueryEncodingEntry, QueryEncodingIndex
    cap = AC.read_json(os.path.join(capdir, "query_encodings.json"))
    path = os.path.join(capdir, "queries.f16")
    got = AC.sha256_file(path)
    if got != cap["sha256"]:
        raise RuntimeError(f"{path}: sha256 {got} != captured {cap['sha256']}")
    entries = {q: QueryEncodingEntry(int(o), int(n), bytes.fromhex(h)) for q, (o, n, h) in cap["entries"].items()}
    return QueryEncodingIndex(path, int(cap["rows"]), int(cap["width"]), entries, cap.get("system_name", "")), cap


# =====================================================================================================================
# the observer and the exhaustive run
# =====================================================================================================================
class Observer:
    """Context manager: wraps ``foldmatch.search.code_alignment._statistics`` (module global, looked up at each call)
    and copies each query's round-1 arrays after the original has run. ``rec[qid]`` = dict(s_f, s_r, skipped, panel_f,
    panel_r, q90, lne, n_surv)."""

    def __init__(self, keep_lne: bool = True):
        self.rec: Dict[str, dict] = {}
        self.keep_lne = keep_lne

    def __enter__(self):
        import foldmatch.search.code_alignment as ca
        self._ca = ca
        self._orig = ca._statistics
        rec, keep = self.rec, self.keep_lne

        def observed(q, run):
            self._orig(q, run)
            rec[q.qid] = dict(
                s_f=np.array(q.s_f, np.float32, copy=True), s_r=np.array(q.s_r, np.float32, copy=True),
                skipped=np.array(q.skipped, bool, copy=True),
                panel_f=None if q.panel_f is None else np.array(q.panel_f, np.float32, copy=True),
                panel_r=None if q.panel_r is None else np.array(q.panel_r, np.float32, copy=True),
                q90=None if q.q90 is None else float(q.q90),
                lne=None if (q.lne is None or not keep) else np.array(q.lne, np.float64, copy=True),
                n_surv=int(q.surv.size))
        ca._statistics = observed
        return self

    def __exit__(self, *exc):
        self._ca._statistics = self._orig
        return False


def run_stage2(query_index, query_ids: Sequence[str], query_seqs: Dict[str, str], target_ids: Sequence[str],
               code_store, fetch, prepared, db_residues: int, *, workers: int, max_evalue: float,
               emb: Optional[np.ndarray] = None, fields: Sequence[str] = STAGE2_FIELDS, observe: bool = True,
               keep_lne: bool = True) -> dict:
    """The production Stage 2 with ``target_ids`` as the candidates of every query (one shared list object), an
    explicit pool of ``workers`` processes and ``max_evalue``; ``emb`` [nq, nt] gives the embedding scores (default 0,
    which only affects the S14 tie-break and the embscore column). Returns hits, stats, the observer's record, the
    wall time and the pool's planned / used workers."""
    from foldmatch.search.code_alignment import CodeAlignmentPool, align_candidates_esm_codes
    tids = list(target_ids)
    zeros = [0.0] * len(tids)
    pre = {}
    for i, q in enumerate(query_ids):
        pre[q] = (tids, zeros if emb is None else emb[i].astype(np.float64).tolist())
    stats: dict = {}
    obs = Observer(keep_lne=keep_lne)
    t0 = time.perf_counter()
    with CodeAlignmentPool(workers) as pool:
        if observe:
            with obs:
                hits = align_candidates_esm_codes(
                    query_index, pre, code_store, fetch, prepared, int(db_residues), pool,
                    query_sequences={q: query_seqs[q] for q in query_ids}, min_seq_identity=0.0, min_coverage=0.0,
                    max_evalue=max_evalue, output_needs=list(fields), stats=stats)
        else:
            hits = align_candidates_esm_codes(
                query_index, pre, code_store, fetch, prepared, int(db_residues), pool,
                query_sequences={q: query_seqs[q] for q in query_ids}, min_seq_identity=0.0, min_coverage=0.0,
                max_evalue=max_evalue, output_needs=list(fields), stats=stats)
        pool_repr = repr(pool)
    wall = time.perf_counter() - t0
    return dict(hits=hits, stats=stats, obs=obs.rec, wall_s=wall, pool=pool_repr, workers=workers,
                max_evalue=max_evalue, db_residues=int(db_residues))


HIT_FIELDS_F = ("emb_score", "score", "score_rev", "q90", "score_adj", "log_evalue", "bit_score", "evalue",
                "identity_aln", "query_coverage", "subject_coverage")
HIT_FIELDS_I = ("aln_len", "n_ident", "mismatch", "gap_open", "q_start", "q_end", "t_start", "t_end", "query_len",
                "subject_len")


def hits_arrays(hits: Dict[str, list], query_ids: Sequence[str], target_pos: Dict[str, int]) -> Dict[str, np.ndarray]:
    """Flat per-hit arrays in query order then the aligner's (S14) order: qi, ti, rank, ecal and the metric fields."""
    qpos = {q: i for i, q in enumerate(query_ids)}
    cols = {k: [] for k in ("qi", "ti", "rank", "ecal") + HIT_FIELDS_F + HIT_FIELDS_I}
    for q, hl in hits.items():
        qi = qpos[q]
        for r, h in enumerate(hl):
            m = h.metrics
            cols["qi"].append(qi)
            cols["ti"].append(target_pos[h.subject_id])
            cols["rank"].append(r)
            cols["ecal"].append(bool(m.ecal))
            cols["emb_score"].append(float(h.emb_score))
            for k in HIT_FIELDS_F[1:]:
                v = getattr(m, k)
                cols[k].append(np.nan if v is None else float(v))
            for k in HIT_FIELDS_I:
                cols[k].append(int(getattr(m, k)))
    out = {}
    for k, v in cols.items():
        if k in ("qi", "ti", "rank") + HIT_FIELDS_I:
            out[k] = np.asarray(v, np.int64)
        elif k == "ecal":
            out[k] = np.asarray(v, bool)
        else:
            out[k] = np.asarray(v, np.float64)
    return out


def assemble(obs: Dict[str, dict], query_ids: Sequence[str], nt: int, panel_n: int = 1000) -> dict:
    """The observer's record as matrices in query order: S_f, S_r float32 [nq, nt] (NaN for a query not observed:
    S13b degenerate), skipped [nq, nt], q90 [nq] (NaN), lne [nq, nt] float64 (if kept), panel_f / panel_r
    float32 [nq, panel_n], n_surv [nq]."""
    nq = len(query_ids)
    S_f = np.full((nq, nt), np.nan, np.float32)
    S_r = np.full((nq, nt), np.nan, np.float32)
    skipped = np.zeros((nq, nt), bool)
    q90 = np.full(nq, np.nan)
    keep_lne = any(r["lne"] is not None for r in obs.values())
    lne = np.full((nq, nt), np.nan) if keep_lne else None
    pf = np.full((nq, panel_n), np.nan, np.float32)
    pr = np.full((nq, panel_n), np.nan, np.float32)
    n_surv = np.zeros(nq, np.int64)
    observed = np.zeros(nq, bool)
    for i, q in enumerate(query_ids):
        r = obs.get(q)
        if r is None:
            continue
        observed[i] = True
        S_f[i], S_r[i], skipped[i] = r["s_f"], r["s_r"], r["skipped"]
        if r["q90"] is not None:
            q90[i] = r["q90"]
        if keep_lne and r["lne"] is not None:
            lne[i] = r["lne"]
        if r["panel_f"] is not None:
            pf[i], pr[i] = r["panel_f"], r["panel_r"]
        n_surv[i] = r["n_surv"]
    return dict(S_f=S_f, S_r=S_r, skipped=skipped, q90=q90, lne=lne, panel_f=pf, panel_r=pr, n_surv=n_surv,
                observed=observed)


# =====================================================================================================================
# consistency checks of one run
# =====================================================================================================================
def check_run(M: dict, H: Dict[str, np.ndarray], qlen: np.ndarray, tab_ship: dict, N: float, max_evalue: float,
              spec=None) -> dict:
    """Self-consistency of an observed run: (a) every hit's score / score_rev / q90 / log_evalue equal the observed
    round-1 arrays bit for bit; (b) the hits are exactly the observed pairs with ln E <= ln(max_evalue) and a
    non-empty alignment; (c) Q90 recomputed from the observed panel S (fit_final2.py: np.quantile(float64 S, 0.9))
    equals the aligner's Q90 bit for bit; (d) ln E recomputed with the shipped table (acc_common.ln_evalue,
    bench_validate arithmetic) vs the aligner's ln E (max abs difference); (e) no skipped pair."""
    qi, ti = H["qi"], H["ti"]
    out = {}
    sf, sr = M["S_f"][qi, ti], M["S_r"][qi, ti]
    out["hits"] = int(len(qi))
    out["hit_score_equals_round1"] = bool(np.array_equal(H["score"].astype(np.float32).view(np.uint32),
                                                         sf.view(np.uint32))
                                          and np.array_equal(H["score"], sf.astype(np.float64)))
    out["hit_score_rev_equals_round1"] = bool(np.array_equal(H["score_rev"], sr.astype(np.float64)))
    out["hit_q90_equals_observed"] = bool(np.array_equal(H["q90"], M["q90"][qi]))
    if M["lne"] is not None:
        out["hit_lne_equals_observed"] = bool(np.array_equal(H["log_evalue"], M["lne"][qi, ti]))
    out["hit_score_adj_equals"] = bool(np.array_equal(
        H["score_adj"], (sf.astype(np.float64) - sr.astype(np.float64)) - M["q90"][qi]))
    # (b) the survivor set
    if M["lne"] is not None:
        lnmax = math.log(max_evalue)
        surv = np.nan_to_num(M["lne"], nan=np.inf) <= lnmax
        n_surv = int(surv.sum())
        hitmask = np.zeros_like(surv)
        hitmask[qi, ti] = True
        out["survivors"] = n_surv
        missing = surv & ~hitmask
        out["survivors_not_hits"] = int(missing.sum())
        # the aligner keeps a survivor only with a non-empty alignment (P6.4 filter alnlen > 0); an empty alignment
        # scores S_f = 0 exactly (S8), so a survivor without a hit must have S_f == 0
        out["survivors_not_hits_with_S_f_0"] = int((missing & (M["S_f"] == 0)).sum())
        if missing.any():
            qi_m, ti_m = np.nonzero(missing)
            out["survivors_not_hits_detail"] = dict(
                queries=int(len(set(qi_m.tolist()))), max_query_length=float(np.asarray(qlen)[qi_m].max()),
                min_lnE=float(M["lne"][qi_m, ti_m].min()))
        out["hits_not_survivors"] = int((hitmask & ~surv).sum())
        out["n_surv_per_query_equal"] = bool(np.array_equal(surv.sum(1), M["n_surv"]))
        out["hits_equal_survivors_with_alignment"] = bool(
            out["hits_not_survivors"] == 0 and out["survivors_not_hits"] == out["survivors_not_hits_with_S_f_0"])
        # (d)
        S = M["S_f"].astype(np.float64) - M["S_r"].astype(np.float64)
        ok = np.isfinite(M["q90"])
        L = AC.ln_evalue(tab_ship, qlen[ok], S[ok], M["q90"][ok], N)
        d = np.abs(L - M["lne"][ok])
        out["lne_recomputed_max_abs_diff"] = float(np.nanmax(d))
        out["lne_recomputed_max_abs_diff_le_1e-9"] = bool(np.nanmax(d) <= 1e-9)
    # (c)
    ok = np.isfinite(M["q90"])
    P = M["panel_f"][ok].astype(np.float64) - M["panel_r"][ok].astype(np.float64)
    q90_re = AC.q90_of(P)
    out["q90_from_panel_equal"] = bool(np.array_equal(q90_re, M["q90"][ok]))
    out["q90_from_panel_max_abs"] = float(np.abs(q90_re - M["q90"][ok]).max()) if ok.any() else None
    out["queries_observed"] = int(M["observed"].sum())
    out["skipped_pairs"] = int(M["skipped"].sum())
    out["nan_scores_observed_rows"] = int(np.isnan(M["S_f"][M["observed"]]).sum())
    out["ok"] = bool(out["hit_score_equals_round1"] and out["hit_score_rev_equals_round1"]
                     and out["hit_q90_equals_observed"] and out["hit_score_adj_equals"]
                     and out.get("hit_lne_equals_observed", True) and out.get("hits_equal_survivors_with_alignment", True)
                     and out.get("lne_recomputed_max_abs_diff_le_1e-9", True) and out["q90_from_panel_equal"]
                     and out["skipped_pairs"] == 0 and out["nan_scores_observed_rows"] == 0)
    return out


# =====================================================================================================================
# coverage
# =====================================================================================================================
def coverage_max_evalue(qlen, tab_prod: dict, N_prod: float, tables: Dict[str, dict], Ns: Dict[str, float],
                        x: float) -> dict:
    """The production max_evalue (shipped table tab_prod, N_prod) needed so that every pair with E <= x under any
    table in ``tables`` and any N in ``Ns`` is a survivor. For each query length m and each (table T, N): the S' at
    which E_T,N = x is s_T = (lnK_T(m) + ln m + ln N - ln x) / lam_T(m); the production ln E at s_T is
    lnK_P(m) + ln m + ln N_P - lam_P(m) s_T; E falls with S' under every table (lam > 0), so a pair with E_T,N <= x
    has S' >= s_T and production E <= exp(that). Returns the max over m, T, N and where it is attained."""
    m = np.asarray(qlen, np.float64)
    lamP, lnKP = AC.tabp(tab_prod, m)
    best, arg = -np.inf, None
    per = {}
    for tn, tab in tables.items():
        lam, lnK = AC.tabp(tab, m)
        assert (lam > 0).all() and (lamP > 0).all()
        for nn, N in Ns.items():
            s = (lnK + np.log(m) + math.log(N) - math.log(x)) / lam
            lp = lnKP + np.log(m) + math.log(N_prod) - lamP * s
            k = int(np.argmax(lp))
            per[f"{tn}/{nn}"] = float(np.exp(lp[k]))
            if lp[k] > best:
                best, arg = float(lp[k]), dict(table=tn, N=nn, m=float(m[k]))
    return dict(x=x, required_max_evalue=float(math.exp(best)), at=arg, per=per)


def bin_index(qlen) -> np.ndarray:
    return np.digitize(np.asarray(qlen, np.float64), AC.EDGES[1:-1], right=False)


def save_npz(path, **arrays) -> str:
    tmp = f"{path}.tmp{os.getpid()}.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    return AC.sha256_file(path)


def db_info(db_path: str) -> dict:
    """The database under test: code-store manifest, sequence-store totals and the FAISS index type / parameters."""
    from foldmatch.search.residue_code_store import ResidueCodeStore
    cs = ResidueCodeStore.from_db_path(db_path)
    man = cs.manifest()
    out = {"db_path": db_path, "manifest": man}
    ss = cs.sequence_store()
    out["sequence_store"] = {"path": str(ss.path), "total_residues": int(ss.total_residues())}
    try:
        out["check_compatible"] = bool(cs.check_compatible() is not None)
    except Exception as e:  # noqa: BLE001
        out["check_compatible"] = repr(e)
    files = {}
    for suf in (".index", ".metadata", ".sequences", ".rescodes.bin", ".rescodes.idx", ".rescodes.json"):
        p = Path(f"{cs.base}{suf}")
        if p.exists():
            files[suf] = {"bytes": p.stat().st_size}
    files[".rescodes.bin"]["sha256"] = AC.sha256_file(str(cs.bin_path))
    out["files"] = files
    import faiss
    from foldmatch.search.db_path import parse_db_path
    folder, name, base = parse_db_path(db_path)
    idx = faiss.read_index(base + ".index")
    out["faiss"] = {"class": type(idx).__name__, "ntotal": int(idx.ntotal), "d": int(idx.d),
                    "metric_type": int(idx.metric_type)}
    try:
        h = faiss.downcast_index(idx)
        out["faiss"]["downcast"] = type(h).__name__
        if hasattr(h, "hnsw"):
            out["faiss"]["hnsw"] = {"M": int(h.hnsw.nb_neighbors(1)), "efSearch": int(h.hnsw.efSearch),
                                    "efConstruction": int(h.hnsw.efConstruction), "max_level": int(h.hnsw.max_level)}
    except Exception as e:  # noqa: BLE001
        out["faiss"]["downcast_error"] = repr(e)
    meta_p = base + ".metadata"
    if os.path.exists(meta_p):
        try:
            import pickle  # the FoldMatch metadata file (FaissEmbeddingDatabase.load_database reads it the same way)
            with open(meta_p, "rb") as fh:
                meta = pickle.load(fh)
            out["faiss_metadata"] = {k: (v if isinstance(v, (int, float, str, bool)) or v is None else
                                         (f"<{type(v).__name__} len {len(v)}>" if isinstance(v, (list, tuple, dict))
                                          else str(v)))
                                     for k, v in meta.items()}
        except Exception as e:  # noqa: BLE001
            out["faiss_metadata_error"] = repr(e)
    return out


def index_vectors(db_path: str):
    """(chain_ids, normalised float32 [n, d]) of the database's FAISS index (reconstruct_n; flat / HNSW-flat)."""
    from foldmatch.search.embedding_database import EmbeddingDatabase
    db = EmbeddingDatabase(db_path=db_path)
    db.load()
    idx = db.db.index
    V = idx.reconstruct_n(0, idx.ntotal)
    return list(db.db.chain_ids), np.ascontiguousarray(V, np.float32)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="fm_stage2 helpers")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("db-info")
    d.add_argument("--db", required=True)
    d.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "db-info":
        rec = {"what": "P8 database under test", "provenance": AC.provenance(), "db": db_info(a.db)}
        print(a.out, AC.write_json(rec, a.out))
        print(json.dumps(AC.jsonable(rec["db"]), indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
