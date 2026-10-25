#!/usr/bin/env python
"""Independent check of auc1.py on a real hit table: a literal, line-by-line Python port of EvaluateResults.cpp
(mmseqs-benchmark 3230f76b: readFamDefFromFasta, parseM8, evaluateResult, main's qFamSize / roc5val) - plain dicts,
strings and loops, no numpy and no code shared with auc1.py / bench_common.Labels.

  python evaluate_results_port.py --hits TSV --auc1-dir DIR/<method> [--target-column theader --target-first-token]

Target annotations: the 'annotated' rows of labels/targets.parquet (= targetannotation.fasta headers of the
targetdb-lt2000 entries; the regex families); the query families from query-lt2000.fasta (regex). Two runs:
  paper  the hit lines sorted like 'sort -k1,1 -k11,11g' (E-value; ties by target id, then line order), resSize 4000,
         rocx 1 and 5, qFamSize = min(sum of family sizes, 4000);
  spec   sorted by E-value, bit score descending, target id; whole list; denominator = distinct TP targets.
Each per-query value is compared with auc1.py's per_query.tsv of the same protocol (tie mode 'order').
"""
import argparse
import csv
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_common as BC  # noqa: E402  (paths, write_json and the FASTA reader only)

FAM = re.compile(r"[a-z]+\.[0-9]+\.[0-9]+\.[0-9]+")
IGN_SF = re.compile(r"^b\.(67|68|69|70).*")
IGN_CL = re.compile(r"^e\..*")


def fams_of_comment(comment):
    return sorted(set(FAM.findall(comment.split("|")[0])))


def fold(f):
    return f[:f.rfind(".")][:f[:f.rfind(".")].rfind(".")]


def evaluate_result(q_fams, lookup, results, rocx):
    fp_cnt = tp_cnt = ign = 0
    auc = 0
    for r_key, _ in results:
        tp = fp = ignore = False
        r_fams = lookup.get(r_key)
        if r_fams is None:
            fp = True
        else:
            for rf in r_fams:
                for qf in q_fams:
                    tp = rf == qf
                    if tp:
                        break
                    if (fold(rf) == fold(qf) or (IGN_SF.match(qf) and IGN_SF.match(rf)) or IGN_CL.match(qf)
                            or IGN_CL.match(rf)):
                        ignore = True
                    else:
                        fp = True
                if tp:
                    break
        if fp_cnt < rocx:
            if tp:
                tp_cnt += 1
            elif ignore:
                ign += 1
            elif fp:
                fp_cnt += 1
                auc += tp_cnt
    return tp_cnt, fp_cnt, auc


def log10e(s):
    m, _, x = s.upper().partition("E")
    m = float(m)
    return -math.inf if m == 0 else math.log10(m) + float(x or 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hits", required=True)
    ap.add_argument("--auc1-dir", required=True)
    ap.add_argument("--labels", default=BC.LABELS_DIR)
    ap.add_argument("--query-fasta", default=BC.QUERY_FASTA)
    ap.add_argument("--target-column", default="target")
    ap.add_argument("--target-first-token", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import pyarrow.parquet as pq
    BC.log("reading annotations")
    t = pq.read_table(os.path.join(args.labels, "targets.parquet"), columns=["target_id", "kind", "families"])
    tid, kind, fams = (t[c].to_pylist() for c in ("target_id", "kind", "families"))
    lookup, size = {}, {}
    for a, k, f in zip(tid, kind, fams):
        if k == 1:
            fl = f.split()
            lookup[a] = fl
            for x in fl:
                size[x] = size.get(x, 0) + 1
    del tid, kind, fams, t
    qf = {h.split()[0]: fams_of_comment(BC.comment_of(h)) for h, _ in BC.AC.read_fasta(args.query_fasta)}
    BC.log(f"{len(lookup)} annotated targets, {len(qf)} queries; reading {args.hits}")
    hits = {}
    with open(args.hits) as fh:
        head = fh.readline().rstrip("\n").split("\t")
        iq, it, ie, ib = (head.index(c) for c in ("query", args.target_column, "evalue", "bits"))
        for n, line in enumerate(fh):
            p = line.rstrip("\n").split("\t")
            tg = p[it].split()[0] if args.target_first_token else p[it]
            hits.setdefault(p[iq], []).append((log10e(p[ie]), float(p[ib]), tg, n))
    BC.log(f"{sum(len(v) for v in hits.values())} hit lines")
    out = {}
    for proto in ("paper", "spec"):
        mine = {r["query"]: r for r in csv.DictReader(open(os.path.join(args.auc1_dir, proto, "per_query.tsv")),
                                                      delimiter="\t")}
        worst, n_eq, n = 0.0, 0, 0
        mean = []
        for q, fl in qf.items():
            res = list(hits.get(q, []))
            if proto == "paper":
                res.sort(key=lambda h: (h[0], h[2].encode(), h[3]))
                res = res[:4000]
            else:
                res.sort(key=lambda h: (h[0], -h[1], h[2].encode(), h[3]))
            seen, single = set(), []
            for h in res:
                if h[2] not in seen:
                    single.append((h[2], h[0]))
                seen.add(h[2])
            if proto == "paper":
                den = 0
                for f in fl:
                    den = min(den + size.get(f, 0), 4000)
            else:
                den = int(mine[q]["n_tp_db"])          # distinct TP targets (label store; recounted below)
            vals = []
            for rocx in (1, 5):
                tp_cnt, fp_cnt, auc = evaluate_result(fl, lookup, single, rocx)
                all_auc = auc + (rocx - fp_cnt) * tp_cnt
                vals.append(all_auc / (rocx * den) if den > 0 else float("nan"))
            m1, m5 = float(mine[q]["auc1"]), float(mine[q]["auc5"])
            if den > 0:
                n += 1
                d = max(abs(vals[0] - m1), abs(vals[1] - m5))
                worst = max(worst, d)
                n_eq += d == 0.0
                mean.append(vals[0])
        out[proto] = dict(n_queries=n, n_exactly_equal=n_eq, max_abs_diff=worst, port_mean_auc1=sum(mean) / len(mean))
        BC.log(proto, out[proto])
    # the spec denominator: distinct TP targets recounted here from the annotation dicts (independent of tp_idx)
    inv = {}
    for a, fl in lookup.items():
        for f in fl:
            inv.setdefault(f, []).append(a)
    mine = {r["query"]: r for r in csv.DictReader(open(os.path.join(args.auc1_dir, "spec", "per_query.tsv")),
                                                  delimiter="\t")}
    bad = 0
    for q, fl in qf.items():
        s = set()
        for f in fl:
            s.update(inv.get(f, ()))
        bad += len(s) != int(mine[q]["n_tp_db"])
    out["spec_denominator_recount_mismatches"] = bad
    BC.log("spec denominators recounted, mismatches:", bad)
    BC.write_json(out, args.out or os.path.join(args.auc1_dir, "evaluate_results_port.json"))


if __name__ == "__main__":
    main()
