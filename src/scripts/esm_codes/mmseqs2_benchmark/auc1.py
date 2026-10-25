#!/usr/bin/env python
"""AUC1 evaluator of the full MMseqs2 benchmark (nbt.3988; EvaluateResults.cpp semantics, research rules).

  python auc1.py evaluate --hits FILE --method NAME [--columns c1,c2,..] [--protocols spec,paper] [--out DIR]
                          [--labels DIR] [--query-subset FILE] [--sort SPEC]
  python auc1.py validate [--out DIR]          (NERSC: the research 50k sample and its stored scores)
  python auc1.py compare --runs DIR [DIR ...] [--protocol spec] [--out FILE]

Input: a hit table per method, tab-separated, with at least query, target, evalue, bits (a header row naming the
columns, or --columns for a headless file; other columns are ignored unless named in --sort). Labels: the store
written by build_labels.py (bench_common.Labels).

Per query (EvaluateResults.cpp evaluateResult / main, rocx = 1..5): the hits are ranked, duplicate targets dropped
(first kept), each hit labelled TP / FP / IGN / NONE (bench_common), and
  num_k = the number of TPs ranked before the k-th FP (k = 1..5); a list with fewer than k FPs is credited every TP
          it holds (EvaluateResults: all_auc = auc + (rocx - fp_cnt) * tp_cnt);
  AUC1  = num_1 / den,  AUC5 = (num_1 + ... + num_5) / (5 den),  sens_k = num_k / den;
  den   = the query's TP count; a query with den = 0 is not evaluated, a query without hits has AUC 0.
Summary: mean AUC1 / AUC5 / sens_1..5 over the evaluated queries; the cumulative distribution (fraction of queries
with AUC1 > x and >= x on a 0.01 grid; the paper's figure plots the evaluator's '>= x' curve, writeRoc5Data); the
evaluator's printed 'ROC5 AUC' (area under that curve, C++ arithmetic); breakdowns by query length, SCOP class and TP
count; list-length and tie diagnostics.

Protocols (--protocols, comma-separated; the first is the headline):
  spec    the agreed benchmark definition: rank by E-value ascending, ties by bit score descending, then target id
          (bytes order); the whole list; den = the query's distinct TP targets in targetdb-lt2000; ties as ranked
          ('order'), with the research expectation over random orders of exact (E, bits) ties ('expect', research P1
          'rnd') and the FP-first / TP-first bounds ('pes' / 'opt') reported alongside.
  paper   EvaluateResults as called in the paper (scripts/run_mmseqs_blast.sh): the m8 sorted with
          'sort -k1,1 -k11,11g' (E-value; equal E-values in line order = target id), the first 4,000 hits per query
          (resSize; truncation before de-duplication, as parseM8), den = min(sum over the query's families of the
          family sizes, 4,000) (qFamSize), queries with AUC > 1 dropped (none expected), ties as ranked (the
          tie modes of exact E-value ties alongside).
  spec_noself / paper_noself   the same without each query's own source sequence 'annotated_<acc>' (removed from
          its list and from its denominator; EvaluateResults keeps it).
  spec_res4000   spec with lists cut at 4,000 hits and den capped at 4,000 (equal list lengths for every method).
  spec_res500    spec with lists cut at the first 500 distinct targets and den capped at 500: BLAST's list length
          (blastp's default -max_target_seqs 500, a cap on targets, not HSP lines), so the cut counts targets after
          de-duplication and BLAST's extra HSP lines do not use up the list.
A run with other --protocols on the same table adds them to the method's summary_all.json (earlier entries kept).
E-values are parsed from their text as log10 (mantissa, exponent), so FoldMatch's sub-1e-308 values ('5.076E-435')
keep their order.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from typing import List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_common as BC  # noqa: E402
from bench_common import AC, log  # noqa: E402

K = 5
GRID = np.round(np.arange(0, 101) / 100.0, 2)
NTP_EDGES = [1, 11, 101, 1001, 4001, 10 ** 12]
NTP_LABELS = ["1-10", "11-100", "101-1000", "1001-4000", ">4000"]


@dataclasses.dataclass(frozen=True)
class Protocol:
    name: str
    sort: str = "evalue:asc,bits:desc,target:asc"   # 'file' = the table's own row order within each query
    res_size: Optional[int] = None                   # EvaluateResults resSize: list cut and denominator cap
    res_unit: str = "rows"                           # the cut counts 'rows' (before de-duplication, parseM8) |
                                                     # 'targets' (distinct targets, after de-duplication)
    den: str = "ntp"                                 # 'ntp' | 'famsize'
    ties: Tuple[str, ...] = ("order",)               # order | expect | half | pes | opt; the first is the headline
    exclude_self: str = "none"                       # none | own | global
    fp_mode: str = "all"                             # all | real | decoy
    exceptions: bool = True                          # b.67-70 / class-e rule
    drop_gt1: bool = True                            # EvaluateResults drops queries with AUC > 1


# 'order' (headline) and, for exact ties of the score columns, the expectation over random orders and the FP-first /
# TP-first bounds (how much the target-id tie break can matter)
ALL_TIES = ("order", "expect", "pes", "opt")
PROTOCOLS = {
    "spec": Protocol("spec", ties=ALL_TIES),
    "spec_noself": Protocol("spec_noself", ties=ALL_TIES, exclude_self="own"),
    "spec_res4000": Protocol("spec_res4000", res_size=4000, ties=ALL_TIES),
    "spec_res500": Protocol("spec_res500", res_size=500, res_unit="targets", ties=ALL_TIES),
    "paper": Protocol("paper", sort="evalue:asc,target:asc", res_size=4000, den="famsize", ties=ALL_TIES),
    "paper_noself": Protocol("paper_noself", sort="evalue:asc,target:asc", res_size=4000, den="famsize",
                             ties=ALL_TIES, exclude_self="own"),
}


# =====================================================================================================================
# reading a hit table
# =====================================================================================================================
def log10_evalue_from_text(col) -> np.ndarray:
    """log10 of E-value strings ('1.469E-162', '5.076E-435', '0.000E+00', '1e-5', 'inf') without underflow."""
    import pyarrow as pa
    import pyarrow.compute as pc
    out = []
    for ch in (col.chunks if hasattr(col, "chunks") else [col]):
        s = pc.utf8_upper(pc.cast(ch, pa.string()))
        has_e = pc.match_substring(s, "E")
        s = pc.if_else(has_e, s, pc.binary_join_element_wise(s, pa.scalar("E0"), ""))
        parts = pc.split_pattern(s, "E", max_splits=1)
        mant = pc.cast(pc.list_element(parts, 0), pa.float64())
        expo = pc.cast(pc.list_element(parts, 1), pa.float64())
        v = pc.add(pc.log10(mant), expo)
        out.append(v.to_numpy(zero_copy_only=False))
    return np.concatenate(out) if out else np.zeros(0)


def read_hits(path: str, columns: Optional[Sequence[str]] = None, need: Sequence[str] = ("query", "target", "evalue",
                                                                                          "bits"),
              target_col: str = "target", first_token: bool = False) -> dict:
    """Read a TSV hit table with pyarrow; returns query / target dictionary codes, the unique ids and the numeric
    columns needed for ranking (evalue as log10). target_col / first_token: take the target id from another column,
    optionally its first whitespace token (MMseqs2: 'theader', because convertalis 'target' prints the UniProt
    accession 'D9YDE7' of 'tr|D9YDE7|D9YDE7_9DELT', Util::parseFastaHeader, while the DB id is the first token)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pcsv
    with open(path) as fh:
        first = fh.readline().rstrip("\n").split("\t")
    header = columns is None
    if header:
        if first[:1] != ["query"]:
            raise ValueError(f"{path}: no header row (first field {first[:1]}); pass --columns")
        names = first
    else:
        names = list(columns)
        if len(names) != len(first):
            raise ValueError(f"{path}: {len(first)} fields per row, --columns names {len(names)}")
    need = list(dict.fromkeys(target_col if c == "target" else c for c in need))
    miss = [c for c in need if c not in names]
    if miss:
        raise ValueError(f"{path}: missing columns {miss} (has {names})")
    types = {c: pa.string() for c in names}
    for c in need:
        if c not in ("query", target_col, "evalue"):
            types[c] = pa.float64()
    t0 = time.time()
    tab = pcsv.read_csv(path, read_options=pcsv.ReadOptions(block_size=1 << 26, column_names=names,
                                                             skip_rows=1 if header else 0),
                        parse_options=pcsv.ParseOptions(delimiter="\t", quote_char=False, escape_char=False),
                        convert_options=pcsv.ConvertOptions(include_columns=need, column_types=types,
                                                            strings_can_be_null=False))
    log(f"read {tab.num_rows} rows from {path} in {time.time() - t0:.1f}s")
    out = {"n_rows": tab.num_rows, "columns": names, "target_col": target_col, "first_token": first_token}
    for c in ("query", "target"):
        col = tab[target_col if c == "target" else c]
        if c == "target" and first_token:
            col = pa.chunked_array([pc.list_element(pc.utf8_split_whitespace(ch, max_splits=1), 0)
                                    for ch in col.chunks], type=pa.string())
        d = pc.dictionary_encode(col).unify_dictionaries()
        dic = d.chunk(0).dictionary if d.num_chunks else pa.array([], pa.string())
        idx = (np.concatenate([ch.indices.to_numpy(zero_copy_only=False) for ch in d.chunks]) if d.num_chunks
               else np.zeros(0, np.int64))
        out[c + "_code"] = idx.astype(np.int64)
        out[c + "_uniq"] = dic.to_pylist()
    out["log10e"] = log10_evalue_from_text(tab["evalue"])
    for c in need:
        if c not in ("query", target_col, "evalue"):
            out[c] = tab[c].to_numpy().astype(np.float64)
    out["file_pos"] = np.arange(tab.num_rows, dtype=np.int64)
    return out


def rows_from_table(L: BC.Labels, H: dict, query_subset: Optional[set] = None) -> Tuple[dict, dict]:
    """Map a read_hits() table onto the label store: per row qi (label query index), kind, ann, target code /
    lexicographic rank, plus the numeric sort columns."""
    qmap = np.array([L.qpos.get(q, -1) for q in H["query_uniq"]], np.int64)
    qi = qmap[H["query_code"]] if len(qmap) else np.zeros(0, np.int64)
    unknown_q = sorted({q for q, m in zip(H["query_uniq"], qmap) if m < 0})
    tu = np.array(H["target_uniq"], dtype="S") if H["target_uniq"] else np.zeros(0, "S1")
    t0 = time.time()
    kind_u, ann_u = L.classify(tu)
    rank_u = np.empty(len(tu), np.int64)
    rank_u[np.argsort(tu, kind="stable")] = np.arange(len(tu))
    log(f"classified {len(tu)} unique targets in {time.time() - t0:.1f}s: "
        + ", ".join(f"{BC.KIND_NAMES[k]} {int((kind_u == k).sum())}" for k in np.unique(kind_u)))
    keep = qi >= 0
    if query_subset is not None:
        sub = np.array([q in query_subset for q in L.q_ids])
        keep &= sub[np.maximum(qi, 0)]
    tc = H["target_code"]
    rows = dict(qi=qi[keep], tcode=tc[keep], kind=kind_u[tc[keep]], ann=ann_u[tc[keep]], trank=rank_u[tc[keep]],
                log10e=H["log10e"][keep], file_pos=H["file_pos"][keep])
    for c, v in H.items():
        if c not in ("query_code", "target_code", "query_uniq", "target_uniq", "log10e", "file_pos", "n_rows",
                     "columns", "target_col", "first_token"):
            rows[c] = v[keep]
    info = dict(n_rows=int(H["n_rows"]), n_rows_used=int(keep.sum()), unknown_queries=unknown_q[:20],
                n_unknown_query_rows=int((qi < 0).sum()), n_unique_targets=int(len(tu)),
                rows_by_kind={BC.KIND_NAMES[k]: int((rows["kind"] == k).sum()) for k in np.unique(rows["kind"])},
                unique_targets_by_kind={BC.KIND_NAMES[k]: int((kind_u == k).sum()) for k in np.unique(kind_u)},
                nan_evalue_rows=int(np.isnan(rows["log10e"]).sum()))
    if (rows["kind"] == BC.KIND_UNKNOWN).any():
        bad = np.unique(rows["tcode"][rows["kind"] == BC.KIND_UNKNOWN])[:10]
        raise ValueError(f"{int((rows['kind'] == BC.KIND_UNKNOWN).sum())} rows have targets that are not in the "
                         f"benchmark DB, e.g. {[H['target_uniq'][i] for i in bad]}")
    return rows, info


# =====================================================================================================================
# ranking and the walk
# =====================================================================================================================
def parse_sort(spec: str) -> List[Tuple[str, bool]]:
    if spec == "file":
        return [("file", False)]
    out = []
    for part in spec.split(","):
        c, _, d = part.partition(":")
        d = d or "asc"
        assert d in ("asc", "desc"), part
        out.append((c.strip(), d == "desc"))
    return out


def sort_column(rows: dict, c: str) -> np.ndarray:
    if c == "evalue":
        return rows["log10e"]
    if c == "target":
        return rows["trank"]
    if c == "file":
        return rows["file_pos"]
    return rows[c]


def rank_rows(rows: dict, spec: str, sel: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Order of the rows (query, then the sort spec; the file position breaks any remaining tie) and, for the tie
    modes, a tie-group key per ranked row (equal consecutive keys within a query = exact ties of the score columns:
    every sort column except 'target' and 'file')."""
    keys = parse_sort(spec)
    idx = np.arange(len(rows["qi"])) if sel is None else np.asarray(sel)
    lk = [rows["file_pos"][idx]]
    for c, desc in reversed(keys):
        v = sort_column(rows, c)[idx]
        if desc:
            v = -v
        lk.append(v)
    lk.append(rows["qi"][idx])
    o = idx[np.lexsort(lk)]
    score_cols = [c for c, _ in keys if c not in ("target", "file")]
    if not score_cols:
        return o, None
    q = rows["qi"][o]
    new = np.zeros(len(o), bool)
    if len(o):
        new[0] = True
        new[1:] = q[1:] != q[:-1]
        for c in score_cols:
            v = sort_column(rows, c)[o]
            ne = v[1:] != v[:-1]
            both_nan = np.isnan(v[1:]) & np.isnan(v[:-1]) if v.dtype.kind == "f" else np.zeros(len(v) - 1, bool)
            new[1:] |= ne & ~both_nan
    return o, np.cumsum(new)


def walk(q: np.ndarray, st: np.ndarray, nq: int, group: Optional[np.ndarray], tie: str, k_max: int = K) -> dict:
    """num [nq, k_max]: TPs ranked before the k-th FP of each query (tie rule for exact ties, see module doc), the fill
    rule for lists with fewer than k FPs, and per-query list counts. q must be non-decreasing."""
    n = len(q)
    is_tp = (st == BC.ST_TP).astype(np.int64)
    is_fp = (st == BC.ST_FP).astype(np.int64)
    starts = np.searchsorted(q, np.arange(nq + 1))
    row_start = starts[q] if n else np.zeros(0, np.int64)
    ctp = np.cumsum(is_tp) - is_tp
    cfp = np.cumsum(is_fp) - is_fp
    tp_ex = ctp - ctp[row_start] if n else ctp          # TPs before the row, within its query
    fp_ex = cfp - cfp[row_start] if n else cfp
    n_tp = np.bincount(q, weights=is_tp, minlength=nq).astype(np.int64)
    n_fp = np.bincount(q, weights=is_fp, minlength=nq).astype(np.int64)
    num = np.zeros((nq, k_max), np.float64)
    r = np.flatnonzero(is_fp)
    k = fp_ex[r] + 1
    m = k <= k_max
    r, k = r[m], k[m]
    if tie == "order" or group is None:
        val = tp_ex[r].astype(np.float64)
    else:
        newg = np.ones(n, bool)
        newg[1:] = group[1:] != group[:-1]
        gstart = np.flatnonzero(newg)
        gid = np.cumsum(newg) - 1
        t_gt = tp_ex[gstart][gid[r]]
        f_gt = fp_ex[gstart][gid[r]]
        t_eq = np.add.reduceat(is_tp, gstart)[gid[r]]
        f_eq = np.add.reduceat(is_fp, gstart)[gid[r]]
        i = k - f_gt
        if tie == "half":
            val = t_gt + 0.5 * t_eq
        elif tie == "expect":
            val = t_gt + (i * t_eq) / (f_eq + 1)
        elif tie == "pes":
            val = t_gt.astype(np.float64)
        elif tie == "opt":
            val = (t_gt + t_eq).astype(np.float64)
        else:
            raise ValueError(tie)
    num[q[r], k - 1] = val
    for kk in range(1, k_max + 1):
        f = n_fp < kk
        num[f, kk - 1] = n_tp[f]
    first_pos = np.full(nq, -1, np.int64)
    r1 = np.flatnonzero(is_fp.astype(bool) & (fp_ex == 0))
    first_pos[q[r1]] = r1 - row_start[r1]
    return dict(num=num, n_tp=n_tp, n_fp=n_fp, n_rows=np.diff(starts), first_fp_pos=first_pos,
                n_ign=np.bincount(q, weights=(st == BC.ST_IGN), minlength=nq).astype(np.int64),
                n_none=np.bincount(q, weights=(st == BC.ST_NONE), minlength=nq).astype(np.int64))


def evaluate_protocol(L: BC.Labels, rows: dict, P: Protocol, status_cache: Optional[dict] = None,
                      order_cache: Optional[dict] = None, query_subset: Optional[np.ndarray] = None,
                      den_override: Optional[np.ndarray] = None) -> dict:
    """One protocol on a row set (rows_from_table, or the dense validation rows). Returns per-query arrays + info."""
    status_cache = {} if status_cache is None else status_cache
    order_cache = {} if order_cache is None else order_cache
    skey = (P.fp_mode, P.exceptions)
    if skey not in status_cache:
        status_cache[skey] = L.pair_status(rows["qi"], rows["kind"], rows["ann"], fp_mode=P.fp_mode,
                                           exceptions=P.exceptions)
    st_all = status_cache[skey]
    # removed targets (as if the DB did not hold them)
    sel = None
    is_ann = rows["kind"] == BC.KIND_ANN
    if P.exclude_self == "own":
        drop = is_ann & (rows["ann"] == L.q_self[rows["qi"]])
        sel = np.flatnonzero(~drop)
    elif P.exclude_self == "global":
        ex = L.excluded_ann("global")
        drop = is_ann & ex[np.maximum(rows["ann"], 0)]
        sel = np.flatnonzero(~drop)
    okey = (P.sort, P.exclude_self)
    if okey not in order_cache:
        order_cache[okey] = rank_rows(rows, P.sort, sel)
    o, grp = order_cache[okey]
    q = rows["qi"][o]
    n_removed = 0 if sel is None else int(len(rows["qi"]) - len(sel))
    # resSize cut (before de-duplication, as parseM8), then duplicate targets (first kept); res_unit 'targets': the
    # cut after de-duplication (the first res_size distinct targets)
    assert P.res_unit in ("rows", "targets"), P.res_unit
    keep = np.ones(len(o), bool)
    if P.res_size is not None and P.res_unit == "rows" and len(o):
        starts = np.searchsorted(q, q)
        keep &= (np.arange(len(o)) - starts) < P.res_size
    n_cut = int((~keep).sum())
    key = q.astype(np.int64) * (int(rows["tcode"].max()) + 1 if len(o) else 1) + rows["tcode"][o]
    kk = np.flatnonzero(keep)
    _, first = np.unique(key[kk], return_index=True)
    dup = np.ones(len(kk), bool)
    dup[first] = False
    keep[kk[dup]] = False
    n_dup = int(dup.sum())
    if P.res_size is not None and P.res_unit == "targets" and len(o):
        kk = np.flatnonzero(keep)
        qk = q[kk]
        cut = kk[(np.arange(len(kk)) - np.searchsorted(qk, qk)) >= P.res_size]
        keep[cut] = False
        n_cut = int(cut.size)
    o2 = o[keep]
    q2 = rows["qi"][o2]
    st = st_all[o2]
    g2 = grp[keep] if grp is not None else None
    den = (L.denominators(P.den, P.exclude_self, cap=P.res_size) if den_override is None
           else np.asarray(den_override, np.float64))
    res = {"den": den, "protocol": dataclasses.asdict(P),
           "list": dict(n_rows_ranked=int(len(o)), n_removed_self=n_removed, n_cut_res_size=n_cut,
                        n_duplicate_targets=n_dup, n_rows=int(len(o2)))}
    evaluated = den > 0
    if query_subset is not None:
        evaluated &= query_subset
    for tie in P.ties:
        w = walk(q2, st, L.nq, g2, tie)
        with np.errstate(divide="ignore", invalid="ignore"):
            a1 = w["num"][:, 0] / den
            a5 = w["num"].sum(1) / (K * den)
            sens = w["num"] / den[:, None]
        ok = evaluated.copy()
        gt1 = ok & ((a1 > 1.0) | (a5 > 1.0))
        if P.drop_gt1:
            ok &= ~gt1
        res[tie] = dict(num=w["num"], auc1=np.where(ok, a1, np.nan), auc5=np.where(ok, a5, np.nan),
                        sens=np.where(ok[:, None], sens, np.nan), n_gt1=int(gt1.sum()))
        if tie == P.ties[0]:
            res["walk"] = {k: v for k, v in w.items() if k != "num"}
            res["evaluated"] = ok
    return res


# =====================================================================================================================
# summaries
# =====================================================================================================================
def evaluator_roc_auc(v: np.ndarray, step: float = 0.01) -> float:
    """writeRoc5Data's printed 'ROC5 AUC': area under the fraction-of-queries-with-value->=-step curve, stepping
    from 1.0 down by `step` in double precision (C++: auc += stepSize*((float)i)/((float)n))."""
    v = np.sort(np.asarray(v, np.float64))[::-1]
    n = v.size
    i, area, s = 0, 0.0, 1.0
    while s >= 0.0 - step:
        while i < n and v[i] >= s:
            i += 1
        area = area + step * float(np.float32(i)) / float(np.float32(n))
        s -= step
    return area


def summarize(L: BC.Labels, res: dict, method: str, tie: Optional[str] = None) -> dict:
    tie = tie or res["protocol"]["ties"][0]
    r = res[tie]
    ok = res["evaluated"] if tie == res["protocol"]["ties"][0] else ~np.isnan(r["auc1"])
    a1, a5, sens = r["auc1"][ok], r["auc5"][ok], r["sens"][ok]
    n = int(ok.sum())

    def block(m):
        x1, x5 = r["auc1"][m & ok], r["auc5"][m & ok]
        if not x1.size:
            return dict(n=0)
        return dict(n=int(x1.size), auc1=float(x1.mean()), auc5=float(x5.mean()),
                    frac_auc1_gt0=float((x1 > 0).mean()), frac_auc1_ge_half=float((x1 >= 0.5).mean()),
                    frac_auc1_eq1=float((x1 >= 1.0).mean()))
    ntp_bin = np.digitize(res["den"], NTP_EDGES[1:-1])
    out = dict(method=method, protocol=res["protocol"], tie=tie, n_queries_evaluated=n,
               n_queries_with_den=int((res["den"] > 0).sum()), n_dropped_auc_gt1=r["n_gt1"],
               mean_auc1=float(a1.mean()) if n else None, mean_auc5=float(a5.mean()) if n else None,
               mean_sens_up_to_kth_fp={str(k + 1): float(sens[:, k].mean()) for k in range(K)} if n else None,
               mean_tp_before_first_fp=float(r["num"][ok, 0].mean()) if n else None,
               cumulative=dict(x=GRID.tolist(), frac_gt=[float((a1 > x).mean()) for x in GRID],
                               frac_ge=[float((a1 >= x - 1e-12).mean()) for x in GRID]) if n else None,
               evaluator_roc_auc=evaluator_roc_auc(a1) if n else None,
               by_len_bin={g: block(L.q_lenbin == b) for b, g in enumerate(BC.LEN_LABELS)},
               by_scop_class={c: block(L.q_class == c) for c in sorted(set(L.q_class.tolist()))},
               by_den={g: block(ntp_bin == b) for b, g in enumerate(NTP_LABELS)},
               list=res["list"])
    for t in res["protocol"]["ties"]:
        x = res[t]["auc1"]
        out.setdefault("mean_auc1_by_tie", {})[t] = float(np.nanmean(x)) if np.isfinite(x).any() else None
        out.setdefault("mean_auc5_by_tie", {})[t] = float(np.nanmean(res[t]["auc5"])) if np.isfinite(x).any() else None
    w = res["walk"]
    out["walk"] = dict(queries_with_hits=int((w["n_rows"] > 0).sum()), queries_evaluated_without_hits=int(
        (ok & (w["n_rows"] == 0)).sum()), hits_per_query_mean=float(w["n_rows"].mean()),
        hits_per_query_median=float(np.median(w["n_rows"])), hits_per_query_max=int(w["n_rows"].max()),
        queries_with_fewer_than_1_fp=int((ok & (w["n_fp"] < 1)).sum()),
        queries_with_fewer_than_5_fp=int((ok & (w["n_fp"] < 5)).sum()),
        tp_rows=int(w["n_tp"].sum()), fp_rows=int(w["n_fp"].sum()), ign_rows=int(w["n_ign"].sum()),
        none_rows=int(w["n_none"].sum()))
    return out


def per_query_tsv(L: BC.Labels, res: dict, path: str) -> None:
    ties = res["protocol"]["ties"]
    t0 = ties[0]
    w = res["walk"]
    with open(path, "w") as fh:
        cols = (["query", "length", "len_bin", "scop_class", "n_tp_db", "famsize_sum", "den", "evaluated", "n_hits",
                 "n_tp_list", "n_fp_list", "n_ign_list", "first_fp_rank"]
                + [f"tp_before_fp{k + 1}" for k in range(K)] + ["auc1", "auc5"]
                + [f"auc1_{t}" for t in ties[1:]] + [f"auc5_{t}" for t in ties[1:]])
        fh.write("\t".join(cols) + "\n")
        for i in range(L.nq):
            r = res[t0]
            vals = [L.q_ids[i], L.q_len[i], BC.LEN_LABELS[L.q_lenbin[i]], L.q_class[i], L.n_tp[i], L.famsize_sum[i],
                    f"{res['den'][i]:.0f}", int(res["evaluated"][i]), w["n_rows"][i], w["n_tp"][i], w["n_fp"][i],
                    w["n_ign"][i], w["first_fp_pos"][i] + 1 if w["first_fp_pos"][i] >= 0 else ""]
            vals += [f"{x:.17g}" for x in r["num"][i]]             # full precision (repr-exact float64)
            vals += [f"{r['auc1'][i]:.17g}", f"{r['auc5'][i]:.17g}"]
            vals += [f"{res[t]['auc1'][i]:.17g}" for t in ties[1:]] + [f"{res[t]['auc5'][i]:.17g}" for t in ties[1:]]
            fh.write("\t".join(str(v) for v in vals) + "\n")


# =====================================================================================================================
# commands
# =====================================================================================================================
def cmd_evaluate(args):
    L = BC.Labels.load(args.labels)
    sub = None
    if args.query_subset:
        ids = set()
        for line in open(args.query_subset):
            line = line.strip()
            if line.startswith(">"):
                ids.add(line[1:].split()[0])
            elif line and not args.query_subset.endswith((".fasta", ".fa")):
                ids.add(line.split()[0])
        sub = ids
    protos = [PROTOCOLS[p] for p in args.protocols.split(",")]
    if args.sort:
        protos = [dataclasses.replace(p, sort=args.sort) for p in protos]
    need = {"query", "target", "evalue"}
    for p in protos:
        for c, _ in parse_sort(p.sort):
            if c not in ("target", "file"):
                need.add(c)
    H = read_hits(args.hits, args.columns.split(",") if args.columns else None,
                  need=["query", "target", "evalue"] + sorted(need - {"query", "target", "evalue"}),
                  target_col=args.target_column, first_token=args.target_first_token)
    rows, info = rows_from_table(L, H, sub)
    del H
    log("rows", info)
    out_dir = os.path.join(args.out, args.method)
    os.makedirs(out_dir, exist_ok=True)
    sub_mask = None if sub is None else np.array([q in sub for q in L.q_ids])
    allsum = dict(method=args.method, hits=os.path.abspath(args.hits), hits_bytes=os.path.getsize(args.hits),
                  labels=L.source, table=info, protocols={})
    sc, oc = {}, {}
    for P in protos:
        t0 = time.time()
        res = evaluate_protocol(L, rows, P, sc, oc, query_subset=sub_mask)
        s = summarize(L, res, args.method)
        s["wall_s"] = round(time.time() - t0, 1)
        d = os.path.join(out_dir, P.name)
        os.makedirs(d, exist_ok=True)
        per_query_tsv(L, res, os.path.join(d, "per_query.tsv"))
        BC.write_json(s, os.path.join(d, "summary.json"))
        allsum["protocols"][P.name] = {k: s[k] for k in ("n_queries_evaluated", "mean_auc1", "mean_auc5",
                                                         "mean_auc1_by_tie", "mean_sens_up_to_kth_fp",
                                                         "evaluator_roc_auc", "walk", "list")}
        log(f"{args.method} / {P.name}: n {s['n_queries_evaluated']}, AUC1 {s['mean_auc1']:.4f}, AUC5 "
            f"{s['mean_auc5']:.4f}, by tie {s['mean_auc1_by_tie']}, ROC-AUC(evaluator) {s['evaluator_roc_auc']:.4f}")
    allsum["provenance"] = AC.provenance(fm=False, kernel=False)
    p_all = os.path.join(out_dir, "summary_all.json")
    if os.path.exists(p_all):
        # an earlier run on the same table (same rows used): keep its protocols that this run did not redo, so the
        # file matches the protocol directories; its provenance moves to 'provenance_earlier'
        old = json.load(open(p_all))
        if (old.get("hits"), old.get("hits_bytes")) == (allsum["hits"], allsum["hits_bytes"]) and \
                old.get("table") == AC.jsonable(info):
            allsum["protocols"] = {**{k: v for k, v in old["protocols"].items() if k not in allsum["protocols"]},
                                   **allsum["protocols"]}
            allsum["provenance_earlier"] = old.get("provenance_earlier", []) + [old["provenance"]]
    BC.write_json(allsum, p_all)


def boot_ci(d: np.ndarray, nb: int = 10000, seed: int = 0) -> List[float]:
    """auc_final.boot_ci: plain query bootstrap of the mean (default_rng(0), 10,000 resamples)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(nb, d.size))
    return [float(x) for x in np.percentile(d[idx].mean(1), [2.5, 97.5])]


def cmd_compare(args):
    runs = {}
    for d in args.runs:
        p = os.path.join(d, args.protocol, "per_query.tsv")
        import csv
        rows = list(csv.DictReader(open(p), delimiter="\t"))
        runs[os.path.basename(os.path.normpath(d))] = {r["query"]: r for r in rows}
    names = list(runs)
    common = sorted(set.intersection(*[{q for q, r in v.items() if r["evaluated"] == "1"} for v in runs.values()]))
    out = dict(protocol=args.protocol, n_common=len(common), methods={}, pairs={})
    A = {m: np.array([float(runs[m][q]["auc1"]) for q in common]) for m in names}
    for m in names:
        out["methods"][m] = dict(auc1=float(A[m].mean()), ci=boot_ci(A[m]),
                                 frac_gt0=float((A[m] > 0).mean()), frac_eq1=float((A[m] >= 1).mean()))
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            d = A[a] - A[b]
            out["pairs"][f"{a} - {b}"] = dict(diff=float(d.mean()), ci=boot_ci(d), win=int((d > 1e-12).sum()),
                                             tie=int((np.abs(d) <= 1e-12).sum()), loss=int((d < -1e-12).sum()))
    BC.write_json(out, args.out)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("evaluate")
    e.add_argument("--hits", required=True)
    e.add_argument("--method", required=True)
    e.add_argument("--columns", default=None, help="column names of a headless TSV, comma-separated")
    e.add_argument("--protocols", default="spec,paper")
    e.add_argument("--sort", default=None, help="override the protocols' ranking, e.g. evalue:asc,rawadj:desc,target")
    e.add_argument("--labels", default=BC.LABELS_DIR)
    e.add_argument("--out", default=BC.EVAL_ROOT)
    e.add_argument("--query-subset", default=None, help="FASTA or id list: evaluate only these queries (dry runs)")
    e.add_argument("--target-column", default="target", help="column holding the target id (MMseqs2: theader)")
    e.add_argument("--target-first-token", action="store_true", help="the target id is that column's first token")
    v = sp.add_parser("validate")
    v.add_argument("--out", default=os.path.join(BC.EVAL_ROOT, "validate"))
    v.add_argument("--quick", action="store_true", help="setting D only")
    c = sp.add_parser("compare")
    c.add_argument("--runs", nargs="+", required=True)
    c.add_argument("--protocol", default="spec")
    c.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "evaluate":
        cmd_evaluate(args)
    elif args.cmd == "validate":
        import auc1_validate
        auc1_validate.run(args)
    else:
        cmd_compare(args)


if __name__ == "__main__":
    main()
