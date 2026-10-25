#!/usr/bin/env python
"""Labels of the full MMseqs2 benchmark (query-lt2000 x targetdb-lt2000) from the FASTA headers - NERSC CPU node.

  python build_labels.py [--out DIR] [--md5] [--no-crosscheck] [--query-fasta F] [--target-fasta F] [--annot-fasta F]

Full set (sequences longer than 2,000 aa included): --query-fasta .../db/query.fasta --target-fasta
.../db/targetdb.fasta --out /pscratch/sd/j/jsegura/embcomp/benchfull/eval/labels (same rules, same annotation file).

Reads query-lt2000.fasta (6,294 queries), targetannotation.fasta (the annotated sequences = the paper's db.fas, the
evaluator's target FASTA) and streams targetdb-lt2000.fasta (30,308,746 entries). Rules: bench_common docstring
(EvaluateResults.cpp). Choices:
  * a target is annotated when its id starts with 'annotated_' (mmseqs-benchmark addSuffixToFasta.py); its families
    are read from its targetannotation.fasta header (the file the evaluator reads); the targetdb header is compared
    and every difference counted;
  * every other target is a decoy (the reversed UniProt sequences, invertFastaSequences.py; checked by the share of
    decoy sequences ending in M against the share starting with M);
  * the TP set of a query = all annotated DB targets carrying any of its families (its own source sequence
    'annotated_<acc>' included when the DB has it - EvaluateResults has no self exclusion; counted);
  * fam_size_db (the evaluator's family sizes) are counted over the annotated targets of targetdb-lt2000 (what the
    searched DB holds); fam_size_file over every targetannotation.fasta entry (the paper's db.fas, > 2,000 aa
    included) is stored for reference.
Writes <out>/ (default bench_common.LABELS_DIR):
  labels.npz         the label store (bench_common.build_label_arrays; loaded by auc1.Labels)
  targets.parquet    every DB target in DB order: target_id, kind (0 decoy / 1 annotated / 2 no family / 3 missing
                     from targetannotation), length, families (space-separated, annotated only)
  tp_pairs.parquet   the per-query TP sets: query, target (dictionary-encoded)
  queries.tsv        per query: id, length, length bin, SCOP class, families, n_tp, famsize_sum, self flags
  labels_summary.json  all counts, checks and the 50k-sample cross-check
Cross-check (research 50k sample, the P8 Bench = rlib.Bench / plan.py labels.npz: TP 31,099, real null 34,948,695,
959 queries with a TP): (a) these rules applied to the sample's own headers reproduce labels.npz TP / FP / FPA
exactly; (b) the full-DB label store restricted to the sample (sample annotated targets by id, sample decoys 'rev_*'
by their original UniProt id, the sample's MMseqs2-benchmark queries by id) gives the same TP / FP / FPA rows.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_common as BC  # noqa: E402
from bench_common import AC, log  # noqa: E402


def scan_fasta(path: str):
    """Yield (header bytes without '>', sequence length, first residue, last residue) of every record."""
    with open(path, "rb", buffering=1 << 24) as f:
        hdr, n, first, last = None, 0, b"", b""
        for line in f:
            if line[:1] == b">":
                if hdr is not None:
                    yield hdr, n, first, last
                hdr, n, first, last = line[1:].rstrip(b"\r\n"), 0, b"", b""
            else:
                s = line.rstrip(b"\r\n")
                if s:
                    if not n:
                        first = s[:1]
                    n += len(s)
                    last = s[-1:]
        if hdr is not None:
            yield hdr, n, first, last


def file_info(path: str, md5: bool) -> dict:
    st = os.stat(path)
    d = dict(path=path, size=st.st_size, mtime=int(st.st_mtime))
    if md5:
        h = hashlib.md5()
        with open(path, "rb") as f:
            for b in iter(lambda: f.read(1 << 26), b""):
                h.update(b)
        d["md5"] = h.hexdigest()
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=BC.LABELS_DIR)
    ap.add_argument("--md5", action="store_true", help="md5 of the three input FASTA files (adds ~1 min)")
    ap.add_argument("--no-crosscheck", action="store_true")
    ap.add_argument("--query-fasta", default=BC.QUERY_FASTA)
    ap.add_argument("--target-fasta", default=BC.TARGET_FASTA)
    ap.add_argument("--annot-fasta", default=BC.ANNOT_FASTA)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    S = {"inputs": {k: file_info(p, args.md5) for k, p in (("query_fasta", args.query_fasta),
                                                           ("target_fasta", args.target_fasta),
                                                           ("annot_fasta", args.annot_fasta))},
         "evaluator_source": BC.EVALUATOR_SOURCE}

    # ------------------------------------------------------------------ queries
    qrecs = AC.read_fasta(args.query_fasta)
    q_hdr = [h for h, _ in qrecs]
    q_ids = [h.split()[0] for h in q_hdr]
    assert len(set(q_ids)) == len(q_ids), "duplicate query ids"
    q_len = [len(s) for _, s in qrecs]
    q_fe = [BC.families_evaluator(BC.comment_of(h)) for h in q_hdr]
    q_fr = [BC.families_research(h) for h in q_hdr]
    S["queries"] = dict(n=len(q_ids), max_len=max(q_len), min_len=min(q_len),
                        n_without_family=int(sum(not f for f in q_fe)),
                        evaluator_vs_research_family_parse_mismatch=int(sum(a != b for a, b in zip(q_fe, q_fr))),
                        n_with_bar=int(sum(" |" in h for h in q_hdr)),
                        n_domains_dist={str(k): int(v) for k, v in zip(*np.unique([len(f) for f in q_fe],
                                                                                    return_counts=True))})
    log("queries", S["queries"])

    # ------------------------------------------------------------------ targetannotation.fasta (= db.fas)
    annot = {}
    annot_len = {}
    n_dup = 0
    fam_size_file = {}
    for hdr, n, _, _ in scan_fasta(args.annot_fasta):
        p = hdr.split(None, 1)
        tid = p[0]
        if tid in annot:
            n_dup += 1
        com = p[1] if len(p) > 1 else b""
        annot[tid] = com
        annot_len[tid] = n
    for com in annot.values():
        for f in BC.families_evaluator(com.decode("ascii")):
            fam_size_file[f] = fam_size_file.get(f, 0) + 1
    S["annotation_file"] = dict(n=len(annot), duplicate_ids=n_dup,
                                n_prefixed=int(sum(k.startswith(BC.ANN_PREFIX) for k in annot)),
                                n_len_ge_2000=int(sum(v >= 2000 for v in annot_len.values())))
    log("targetannotation", S["annotation_file"])

    # ------------------------------------------------------------------ targetdb-lt2000.fasta (streamed)
    all_ids, all_len, all_kind = [], [], []
    ann_db = {}                      # id -> (db comment, length)
    dec_ids = []
    dec_first_m = dec_last_m = 0
    ann_first_m = ann_last_m = 0
    max_len = 0
    for k, (hdr, n, first, last) in enumerate(scan_fasta(args.target_fasta)):
        sp = hdr.find(b" ")
        tid = hdr if sp < 0 else hdr[:sp]
        all_ids.append(tid)
        all_len.append(n)
        if n > max_len:
            max_len = n
        if tid.startswith(BC.ANN_PREFIX):
            ann_db[tid] = (b"" if sp < 0 else hdr[sp + 1:].strip(), n)
            ann_first_m += first == b"M"
            ann_last_m += last == b"M"
            all_kind.append(BC.KIND_ANN)
        else:
            dec_ids.append(tid)
            dec_first_m += first == b"M"
            dec_last_m += last == b"M"
            all_kind.append(BC.KIND_DECOY)
        if (k + 1) % 5_000_000 == 0:
            log(f"  targetdb: {k + 1} records")
    n_t = len(all_ids)
    log(f"{os.path.basename(args.target_fasta)}: {n_t} records, {len(ann_db)} 'annotated_', {len(dec_ids)} other; "
        f"max length {max_len}")
    all_kind = np.array(all_kind, np.int8)

    # ------------------------------------------------------------------ classify the annotated DB entries
    ann_sorted = sorted(ann_db)
    ann_ok, ann_ok_len, ann_ok_fams = [], [], []
    other_ids, other_kind = [], []
    n_hdr_diff = n_missing = n_empty = 0
    n_db_vs_file_fam_diff = 0
    for tid in ann_sorted:
        com_db, n = ann_db[tid]
        if tid not in annot:
            n_missing += 1
            other_ids.append(tid)
            other_kind.append(BC.KIND_MISSING)
            continue
        com = annot[tid]
        if com.strip() != com_db:
            n_hdr_diff += 1
        fams = BC.families_evaluator(com.decode("ascii"))
        if fams != BC.families_evaluator(com_db.decode("ascii")):
            n_db_vs_file_fam_diff += 1
        if not fams:
            n_empty += 1
            other_ids.append(tid)
            other_kind.append(BC.KIND_EMPTY)
            continue
        ann_ok.append(tid)
        ann_ok_len.append(n)
        ann_ok_fams.append(fams)
    kind_of = {t: k for t, k in zip(other_ids, other_kind)}
    for j in np.flatnonzero(all_kind == BC.KIND_ANN):
        k = kind_of.get(all_ids[j])
        if k is not None:
            all_kind[j] = k
    research_parse_diff = 0
    for tid in ann_ok[:: max(1, len(ann_ok) // 200000)]:      # research tokenisation on a 1-in-k subsample
        h = (tid + b" " + annot[tid]).decode("ascii")
        research_parse_diff += BC.families_research(h) != BC.families_evaluator(annot[tid].decode("ascii"))
    db_ids_set_ann = set(ann_db)
    S["targets"] = dict(
        n=n_t, max_len=max_len, n_annotated=len(ann_ok), n_decoy=len(dec_ids),
        n_unannotated_no_family=n_empty, n_annotated_id_missing_from_annotation=n_missing,
        n_unannotated_total=n_empty + n_missing,
        annotated_header_differs_from_annotation_file=n_hdr_diff,
        annotated_families_differ_db_vs_annotation_file=n_db_vs_file_fam_diff,
        annotation_file_entries_not_in_db=int(sum(1 for k in annot if k not in db_ids_set_ann)),
        annotation_file_entries_not_in_db_len_ge_2000=int(sum(1 for k, v in annot_len.items()
                                                               if k not in db_ids_set_ann and v >= 2000)),
        research_vs_evaluator_family_parse_mismatch_subsample=int(research_parse_diff),
        decoy_first_residue_M=dec_first_m, decoy_last_residue_M=dec_last_m,
        annotated_first_residue_M=ann_first_m, annotated_last_residue_M=ann_last_m)
    log("targets", S["targets"])

    # ------------------------------------------------------------------ the label store
    dh = BC.id_hashes(dec_ids)
    n_uh = int(np.unique(dh).size)
    S["targets"]["decoy_hash_unique"] = n_uh
    assert n_uh == len(dec_ids), "64-bit id hash collision among decoys (or duplicate decoy ids)"
    arr = BC.build_label_arrays(q_ids, q_len, q_fe, ann_ok, ann_ok_len, ann_ok_fams, dh, other_ids=other_ids,
                                other_kind=other_kind, fam_size_file=fam_size_file)
    np.savez(os.path.join(args.out, "labels.npz"), **arr)
    L = BC.Labels(arr, source=args.out)
    log("wrote labels.npz")

    # ------------------------------------------------------------------ statistics
    n_tp = L.n_tp
    has = n_tp > 0
    selfm = L.q_self >= 0
    self_tp = L._self_is_tp()
    qs = lambda v: {str(p): float(np.quantile(v, p)) for p in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1)}  # noqa: E731
    S["labels"] = dict(
        n_queries=L.nq, n_queries_with_tp=int(has.sum()), n_queries_without_tp=int((~has).sum()),
        total_tp_pairs=int(n_tp.sum()), n_tp_quantiles_queries_with_tp=qs(n_tp[has]),
        queries_n_tp_gt={str(x): int((n_tp > x).sum()) for x in (1000, 4000, 6000, 10000, 50000)},
        famsize_sum_total=int(L.famsize_sum.sum()), queries_famsize_differs_from_n_tp=int((L.famsize_sum != n_tp).sum()),
        queries_famsize_gt_4000=int((L.famsize_sum > 4000).sum()),
        n_families=len(L.fam_names), n_folds=len(L.fold_names), n_query_families=int(np.unique(L.q_fam_idx).size),
        query_families_absent_from_db=int((L.fam_size_db[np.unique(L.q_fam_idx)] == 0).sum()),
        n_prop_regex_variants_differ=int(arr["n_prop_regex_diff"]),
        queries_with_propeller=int(L.q_prop.sum()), queries_with_class_e=int(L.q_e.sum()),
        annotated_with_propeller=int(L.ann_prop.sum()), annotated_with_class_e=int(L.ann_e.sum()),
        query_source_in_db=int(selfm.sum()), query_source_in_db_and_tp=int(self_tp.sum()),
        query_source_not_in_db_but_in_annotation_file=int(sum(
            1 for i, q in enumerate(q_ids) if not selfm[i] and BC.query_original(q).encode() in annot)),
        n_tp_excluding_own_source=int(L.denominators("ntp", "own").sum()),
        queries_with_tp_excluding_own_source=int((L.denominators("ntp", "own") > 0).sum()),
        queries_with_tp_excluding_all_sources=int((L.denominators("ntp", "global") > 0).sum()),
        by_class={c: dict(n=int((L.q_class == c).sum()), with_tp=int(((L.q_class == c) & has).sum()))
                  for c in sorted(set(L.q_class.tolist()))},
        by_len_bin={g: dict(n=int((L.q_lenbin == b).sum()), with_tp=int(((L.q_lenbin == b) & has).sum()))
                    for b, g in enumerate(BC.LEN_LABELS)},
        largest_families=[(str(L.fam_names[i]), int(L.fam_size_db[i]))
                          for i in np.argsort(-L.fam_size_db)[:10]])
    log("labels", {k: v for k, v in S["labels"].items() if not isinstance(v, (dict, list))})

    # ------------------------------------------------------------------ human-readable tables
    import pyarrow as pa
    import pyarrow.parquet as pq
    fam_str = {}
    for tid, f in zip(ann_ok, ann_ok_fams):
        fam_str[tid] = " ".join(f)
    tab = pa.table({"target_id": pa.array(all_ids, pa.binary()).cast(pa.string()),
                    "kind": pa.array(all_kind, pa.int8()), "length": pa.array(np.asarray(all_len, np.int32)),
                    "families": pa.array([fam_str.get(x, "") for x in all_ids], pa.string())})
    pq.write_table(tab, os.path.join(args.out, "targets.parquet"), compression="zstd")
    del tab
    qcol = pa.DictionaryArray.from_arrays(
        pa.array(np.repeat(np.arange(L.nq, dtype=np.int32), np.diff(L.tp_ptr))), pa.array(L.q_ids.tolist()))
    tcol = pa.DictionaryArray.from_arrays(pa.array(L.tp_idx.astype(np.int32)),
                                          pa.array([x.decode("ascii") for x in L.ann_ids]))
    pq.write_table(pa.table({"query": qcol, "target": tcol}), os.path.join(args.out, "tp_pairs.parquet"),
                   compression="zstd")
    with open(os.path.join(args.out, "queries.tsv"), "w") as fh:
        fh.write("query\tlength\tlen_bin\tscop_class\tfamilies\tn_tp\tfamsize_sum\tsource_in_db\tsource_is_tp\n")
        for i in range(L.nq):
            fams = " ".join(L.fam_names[L.q_fam_idx[L.q_fam_ptr[i]:L.q_fam_ptr[i + 1]]])
            fh.write(f"{L.q_ids[i]}\t{L.q_len[i]}\t{BC.LEN_LABELS[L.q_lenbin[i]]}\t{L.q_class[i]}\t{fams}\t"
                     f"{L.n_tp[i]}\t{L.famsize_sum[i]}\t{int(selfm[i])}\t{int(self_tp[i])}\n")
    log("wrote targets.parquet, tp_pairs.parquet, queries.tsv")

    # ------------------------------------------------------------------ cross-check: the research 50k sample
    if not args.no_crosscheck:
        S["crosscheck_50k"] = crosscheck(L, annot)
    S["files"] = {f: {"sha256": BC.sha256_file(os.path.join(args.out, f)),
                      "bytes": os.path.getsize(os.path.join(args.out, f))}
                  for f in ("labels.npz", "targets.parquet", "tp_pairs.parquet", "queries.tsv")}
    S["provenance"] = AC.provenance(fm=False, kernel=False)
    BC.write_json(S, os.path.join(args.out, "labels_summary.json"))
    log("wrote labels_summary.json; done")


def crosscheck(L: "BC.Labels", annot: dict) -> dict:
    """The research 50k sample (P8 Bench, labels.npz by plan.py) against these rules and this label store."""
    B = AC.Bench()
    out = dict(bench_paths=B.paths, bench_counts=AC.BENCH_COUNTS)
    rev = B.rev
    # (a) these rules on the sample's own headers
    LS, tids, info = BC.labels_from_headers(B.q_headers, B.qlen.astype(int), B.t_headers, B.tlen.astype(int))
    kind, ann = LS.classify(tids)
    assert (kind != BC.KIND_UNKNOWN).all()
    st = LS.dense_status(np.arange(LS.nq), kind, ann)
    TP, FP = st == BC.ST_TP, st == BC.ST_FP
    FPA = FP & ~rev[None, :]
    out["a_own_headers"] = dict(
        info=info, tp=int(TP.sum()), fp_ann=int(FPA.sum()), ignored=int((~TP & ~FP).sum()),
        q_with_tp=int(TP.any(1).sum()), equal_TP=bool(np.array_equal(TP, B.TP)), equal_FP=bool(np.array_equal(FP, B.FP)),
        equal_FPA=bool(np.array_equal(FPA, B.REAL)), kinds={BC.KIND_NAMES[k]: int((kind == k).sum())
                                                           for k in np.unique(kind)},
        acc_common_label_matrices_equal=None)
    lm = AC.label_matrices(B.q_headers, B.t_headers)
    out["a_own_headers"]["acc_common_label_matrices_equal"] = bool(
        np.array_equal(lm["TP"], TP) and np.array_equal(lm["FP"], FP) and np.array_equal(lm["FPA"], FPA))
    del st, lm
    # (b) the full-DB label store restricted to the sample
    t_full_ids = []
    for h in B.t_headers:
        tok = h.split()
        t_full_ids.append(tok[1] if tok[0].startswith("rev_") else tok[0])      # 'rev_000123 tr|...' -> 'tr|...'
    fk, fa = L.classify(np.array(t_full_ids, dtype="S"))
    ann_cols = ~rev
    fam_equal = 0
    fam_diff = []
    for j in np.flatnonzero(ann_cols & (fk == BC.KIND_ANN)):
        sf = BC.families_evaluator(BC.comment_of(B.t_headers[j]))
        ff = L.fam_names[L.ann_fam_idx[L.ann_fam_ptr[fa[j]]:L.ann_fam_ptr[fa[j] + 1]]].tolist()
        if sf == ff:
            fam_equal += 1
        else:
            fam_diff.append(B.t_headers[j].split()[0])
    qmap = np.array([L.qpos.get(q, -1) for q in B.qids])
    mm = np.flatnonzero(qmap >= 0)
    qfam_equal = int(sum(BC.families_research(B.q_headers[i]) ==
                         L.fam_names[L.q_fam_idx[L.q_fam_ptr[qmap[i]]:L.q_fam_ptr[qmap[i] + 1]]].tolist()
                         for i in mm))
    rows_eq = dict(TP=0, FP=0, FPA=0)
    rows_eq_usable = dict(TP=0, FP=0, FPA=0)
    tp_cnt = 0
    # the sample's annotated targets that the full DB does not hold get no label from the store (reported; the
    # comparison is made on all columns and on the columns the full DB holds)
    usable = fk != BC.KIND_UNKNOWN
    for i in mm:
        s = np.full(B.nt, BC.ST_NONE, np.int8)
        s[usable] = L.pair_status(np.full(int(usable.sum()), qmap[i]), fk[usable], fa[usable])
        tp, fp = s == BC.ST_TP, s == BC.ST_FP
        tp_cnt += int(tp.sum())
        for name, mine, ref in (("TP", tp, B.TP[i]), ("FP", fp, B.FP[i]), ("FPA", fp & ~rev, B.REAL[i])):
            rows_eq[name] += bool(np.array_equal(mine, ref))
            rows_eq_usable[name] += bool(np.array_equal(mine[usable], ref[usable]))
    out["b_full_store_restricted"] = dict(
        sample_targets_kind_in_full_db={BC.KIND_NAMES[k]: int((fk == k).sum()) for k in np.unique(fk)},
        sample_annotated_not_in_full_db=[t_full_ids[j] for j in np.flatnonzero(ann_cols & (fk != BC.KIND_ANN))],
        sample_annotated_in_annotation_file=int(sum(t_full_ids[j].encode() in annot for j in np.flatnonzero(ann_cols))),
        sample_decoys_are_full_db_decoys=int((fk[rev] == BC.KIND_DECOY).sum()), n_sample_decoys=int(rev.sum()),
        annotated_family_equal=fam_equal, annotated_family_diff=fam_diff[:20],
        sample_queries_in_full_query_set=int(mm.size), sample_query_families_equal=qfam_equal,
        rows_equal=rows_eq, rows_equal_on_columns_in_full_db=rows_eq_usable, tp_pairs_in_those_rows=tp_cnt,
        bench_tp_pairs_in_those_rows=int(B.TP[mm].sum()),
        bench_q_with_tp_in_those_rows=int(B.TP[mm].any(1).sum()))
    a, b = out["a_own_headers"], out["b_full_store_restricted"]
    out["pass"] = bool(a["equal_TP"] and a["equal_FP"] and a["equal_FPA"] and a["tp"] == 31099
                       and a["q_with_tp"] == 959 and all(v == mm.size for v in b["rows_equal"].values()))
    log("crosscheck", {"a": {k: a[k] for k in ("tp", "fp_ann", "q_with_tp", "equal_TP", "equal_FP", "equal_FPA")},
                       "b": {k: b[k] for k in ("sample_queries_in_full_query_set", "rows_equal",
                                               "tp_pairs_in_those_rows", "bench_tp_pairs_in_those_rows")},
                       "pass": out["pass"]})
    return out


if __name__ == "__main__":
    main()
