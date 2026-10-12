#!/usr/bin/env python
"""P0.7 long-sequence calibration slice, step 1 (NERSC, CPU; the login node is enough, ~1 min): candidates and sample.

SOURCE OF THE PROTEINS (stated, because UniRef50 has no SCOP labels): the MMseqs2-benchmark files that A5 itself was
drawn from (calib3di/sample.py), so the null is built with exactly the A5 rules (evalue/score/plan.py):
  * SCOP-annotated UniProt sequences  .../mmseqs-db/db/targetannotation.fasta
      header '>annotated_<acc> sccs1 [sccs2 ...] |evalues |scop domains |domain ranges |target ranges'
      -> A5 real null: annotated, no shared SCOP fold (a.b), pair not excepted (both b.67-70 propellers / either class e)
  * reversed UniProt background       .../mmseqs-db/db/targetdb-gt1000.fasta, entries without 'annotated_'
      (the long counterpart of A5's 25,000 reversed decoys, which came from targetdb-lt1000.fasta; the sequences are
      stored reversed, checked: they end in M) -> reversed null, homology-free by construction
  UniRef50 representatives (panel/pool) carry no SCOP labels, so they cannot give an A5-type real null; not used.

ROLES (seed 20260927, numpy default_rng; bins are [lo, hi))
  long_q  queries, annotated: 6 bins [300,600) [600,1000) [1000,1500) [1500,2000) [2000,2500) [2500,3000], NQ_BIN each.
          The two short bins are the POPULATION-MATCHED BASELINE (same source file and draw as the long ones; inside the
          table's interpolated range / the calibrated domain), so a long-bin excess is a length effect, not a
          population (full-length multi-domain UniProt vs benchmark query) effect.
  long_t  targets, annotated: 4 bins [1000,1500) ... [2500,3000], NT_BIN each (disjoint from long_q).
  long_r  targets, reversed decoys: the same 4 bins, NR_BIN each.
Eligible annotated protein: L in [300, 3000]; >= 1 sccs; no class-e sccs; non-standard letters <= 1% of L; every letter
in the ESM3 vocabulary; not an exact duplicate; NOT in the benchmark or the shipped panel: accession not a benchmark
query / annotated-target / decoy-source accession or a panel representative accession (UniRef50_<acc>); sequence not
identical to a benchmark query / target / decoy (either direction) or panel member; no benchmark query (>= 16 aa) as an
exact substring (checked for every protein the sampler considers). Diversity: at most CAP proteins per SCOP-fold-set key
(sorted set of a.b over the protein's sccs) per role and bin. Reversed decoys: L in [1000, 3000], vocabulary and
non-standard rules, not identical to any excluded sequence (either direction), no duplicates; uniform per bin.

OUTPUTS (this directory): long_candidates.tsv (every eligible annotated protein: id, L, n_sccs, folds, coverage =
union of the header's target ranges / L, nonstd), long_q.fasta, long_t.fasta, long_r.fasta (reversed ids renamed
'lrev_<n> <acc>'), select_long.json (counts at every filter, input md5s, per-role stats).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import Counter

import numpy as np

DB = "/pscratch/sd/j/jsegura/structure-embedding/mmseqs-db/db"
ANN = f"{DB}/targetannotation.fasta"
RDB = f"{DB}/targetdb-gt1000.fasta"
BQ = "/pscratch/sd/j/jsegura/fm-db/calib3di/data/queries.fasta"
BT = "/pscratch/sd/j/jsegura/fm-db/calib3di/data/targets.fasta"
PANEL = "/pscratch/sd/j/jsegura/embcomp/panel/fit2/results/shipped_panel.fasta"
OUT = os.path.dirname(os.path.abspath(__file__))
SEED = 20260927
Q_EDGES = [300, 600, 1000, 1500, 2000, 2500, 3001]
T_EDGES = [1000, 1500, 2000, 2500, 3001]
NQ_BIN = int(os.environ.get("NQ_BIN", "500"))
NT_BIN = int(os.environ.get("NT_BIN", "500"))
NR_BIN = int(os.environ.get("NR_BIN", "500"))
CAP = int(os.environ.get("CAP", "2"))
STD = set("ACDEFGHIKLMNPQRSTVWY")
VOCAB = set("LAGVSERTIDPKQNFYMHWCXBUZO")
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def records(path):
    header, chunks = None, []
    with open(path) as handle:
        for line in handle:
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header, chunks = line[1:].rstrip("\n"), []
            else:
                chunks.append(line.strip())
    if header is not None:
        yield header, "".join(chunks)


def sccs_of(h):
    out = []
    for t in h.split()[1:]:
        if t.startswith("|"):
            break
        out.append(t)
    return sorted(set(out))


def target_ranges(h):
    parts = h.split("|")
    if len(parts) < 5:
        return []
    rr = []
    for t in parts[-1].split():
        m = re.fullmatch(r"(\d+)-(\d+)", t)
        if m:
            rr.append((int(m.group(1)), int(m.group(2))))
    return rr


def coverage(rr, L):
    cov = np.zeros(L + 2, bool)
    for a, b in rr:
        cov[max(a, 1):min(b, L) + 1] = True
    return float(cov.sum() / L)


def seq_ok(s, c, prefix):
    letters = set(s)
    if letters - VOCAB:
        c[prefix + "x_outside_vocab"] += 1
        return False, 0
    nonstd = sum(s.count(ch) for ch in letters - STD)
    if nonstd > 0.01 * len(s):
        c[prefix + "x_nonstd_gt1pct"] += 1
        return False, nonstd
    return True, nonstd


def main():
    info = {"params": dict(seed=SEED, q_edges=Q_EDGES, t_edges=T_EDGES, nq_bin=NQ_BIN, nt_bin=NT_BIN, nr_bin=NR_BIN,
                           cap=CAP), "counts": {}}
    excl_acc, excl_seq = set(), set()
    bq, bt, pan = list(records(BQ)), list(records(BT)), list(records(PANEL))
    for h, s in bq:
        excl_acc.add(h.split()[0].rsplit("_", 1)[0])
        excl_seq.add(s)
    for h, s in bt:
        tok = h.split()
        excl_acc.add(tok[1].split("|")[1] if tok[0].startswith("rev_") and "|" in tok[1] else tok[0].replace("annotated_", ""))
        excl_seq.add(s)
        excl_seq.add(s[::-1])
    for h, s in pan:
        excl_acc.add(h.split()[0].replace("UniRef50_", ""))
        excl_seq.add(s)
    qseqs = [s for _, s in bq if len(s) >= 16]
    info["inputs"] = {"targetannotation": [ANN, os.path.getsize(ANN)], "targetdb_gt1000": [RDB, os.path.getsize(RDB)],
                      "benchmark_queries": [BQ, md5(BQ), len(bq)], "benchmark_targets": [BT, md5(BT), len(bt)],
                      "shipped_panel": [PANEL, md5(PANEL), len(pan)]}
    info["counts"].update(exclusion_accessions=len(excl_acc), exclusion_sequences=len(excl_seq))

    # ---- annotated scan
    c = Counter()
    elig, seen = [], set()
    for h, s in records(ANN):
        c["ann_records"] += 1
        L = len(s)
        if L < Q_EDGES[0] or L > 3000:
            continue
        c["ann_len_300_3000"] += 1
        sc = sccs_of(h)
        if not sc:
            c["ann_x_no_sccs"] += 1
            continue
        if any(x.startswith("e.") for x in sc):
            c["ann_x_class_e"] += 1
            continue
        ok, nonstd = seq_ok(s, c, "ann_")
        if not ok:
            continue
        if h.split()[0].replace("annotated_", "") in excl_acc:
            c["ann_x_accession_in_benchmark_or_panel"] += 1
            continue
        if s in excl_seq:
            c["ann_x_sequence_in_benchmark_or_panel"] += 1
            continue
        if s in seen:
            c["ann_x_duplicate_sequence"] += 1
            continue
        seen.add(s)
        elig.append(dict(id=h.split()[0], header=h, seq=s, L=L, sccs=sc,
                         folds=sorted({".".join(x.split(".")[:2]) for x in sc}),
                         cov=coverage(target_ranges(h), L), nonstd=nonstd))
    c["ann_eligible"] = len(elig)
    log("annotated scan", dict(c))
    L = np.array([e["L"] for e in elig])
    cov = np.array([e["cov"] for e in elig])
    keys = [";".join(e["folds"]) for e in elig]
    with open(os.path.join(OUT, "long_candidates.tsv"), "w") as fh:
        fh.write("id\tL\tn_sccs\tfolds\tcoverage\tnonstd\n")
        for e in elig:
            fh.write(f"{e['id']}\t{e['L']}\t{len(e['sccs'])}\t{','.join(e['folds'])}\t{e['cov']:.4f}\t{e['nonstd']}\n")
    info["counts"]["ann_eligible_by_q_bin"] = np.histogram(L, Q_EDGES)[0].tolist()

    rng = np.random.default_rng(SEED)
    taken = set()
    picked = {}
    for role, edges, want in (("long_q", Q_EDGES, NQ_BIN), ("long_t", T_EDGES, NT_BIN)):
        picked[role] = []
        for k in range(len(edges) - 1):
            idx = np.flatnonzero((L >= edges[k]) & (L < edges[k + 1]))
            perm = idx[rng.permutation(len(idx))]
            used, got = Counter(), []
            for i in perm:
                i = int(i)
                if i in taken or used[keys[i]] >= CAP:
                    continue
                if any(q in elig[i]["seq"] for q in qseqs):
                    c["ann_x_contains_benchmark_query_sampler"] += 1
                    continue
                used[keys[i]] += 1
                got.append(i)
                if len(got) == want:
                    break
            assert len(got) == want, (role, k, len(got))
            taken.update(got)
            picked[role].extend(got)
        ii = picked[role]
        fn = f"{role}.fasta"
        with open(os.path.join(OUT, fn), "w") as fh:
            for i in ii:
                fh.write(f">{elig[i]['header']}\n{elig[i]['seq']}\n")
        Ls, cv = L[ii], cov[ii]
        info[role] = dict(file=fn, md5=md5(os.path.join(OUT, fn)), n=len(ii), residues=int(Ls.sum()),
                          by_bin=np.histogram(Ls, edges)[0].tolist(), edges=edges,
                          len_pct={str(p): float(np.percentile(Ls, p)) for p in (0, 25, 50, 75, 100)},
                          coverage_pct={str(p): float(np.percentile(cv, p)) for p in (5, 25, 50, 75, 95)},
                          coverage_median_by_bin=[float(np.median(cv[(Ls >= edges[k]) & (Ls < edges[k + 1])]))
                                                  for k in range(len(edges) - 1)],
                          fold_set_keys=len({keys[i] for i in ii}),
                          distinct_folds=len({f for i in ii for f in elig[i]["folds"]}),
                          n_sccs_mean=float(np.mean([len(elig[i]["sccs"]) for i in ii])))
        log(role, json.dumps(info[role]))
    assert not set(picked["long_q"]) & set(picked["long_t"])

    # ---- reversed decoys
    rev, rseen = [], set()
    for h, s in records(RDB):
        if h.startswith("annotated_"):
            continue
        c["rev_records"] += 1
        if len(s) < 1000 or len(s) > 3000:
            continue
        c["rev_len_1000_3000"] += 1
        ok, _ = seq_ok(s, c, "rev_")
        if not ok:
            continue
        if s in excl_seq or s[::-1] in excl_seq:
            c["rev_x_sequence_in_benchmark_or_panel"] += 1
            continue
        if s in rseen:
            c["rev_x_duplicate_sequence"] += 1
            continue
        rseen.add(s)
        tok = h.split()[0]
        rev.append((tok.split("|")[1] if "|" in tok else tok, s))
    c["rev_eligible"] = len(rev)
    RL = np.array([len(s) for _, s in rev])
    got = []
    for k in range(len(T_EDGES) - 1):
        idx = np.flatnonzero((RL >= T_EDGES[k]) & (RL < T_EDGES[k + 1]))
        got.extend(int(i) for i in idx[rng.permutation(len(idx))[:NR_BIN]])
    with open(os.path.join(OUT, "long_r.fasta"), "w") as fh:
        for n, i in enumerate(got):
            fh.write(f">lrev_{n:06d} {rev[i][0]}\n{rev[i][1]}\n")
    Ls = RL[got]
    info["long_r"] = dict(file="long_r.fasta", md5=md5(os.path.join(OUT, "long_r.fasta")), n=len(got),
                          residues=int(Ls.sum()), by_bin=np.histogram(Ls, T_EDGES)[0].tolist(), edges=T_EDGES,
                          len_pct={str(p): float(np.percentile(Ls, p)) for p in (0, 25, 50, 75, 100)})
    log("long_r", json.dumps(info["long_r"]))
    info["counts"].update(dict(c))
    info["counts"]["rev_eligible_by_bin"] = np.histogram(RL, T_EDGES)[0].tolist()
    info["seconds"] = round(time.time() - T0, 1)
    json.dump(info, open(os.path.join(OUT, "select_long.json"), "w"), indent=1)
    log("done")


if __name__ == "__main__":
    main()
