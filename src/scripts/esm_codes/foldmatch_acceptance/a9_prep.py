#!/usr/bin/env python
"""P8 A9 (long sequences), step 1: the FoldMatch input FASTAs of the P0.7 slice and the per-sequence facts that the
calibrated-domain (S13) checks need. NERSC; light (a login node is enough).

Writes <P8_ROOT>/A9/inputs/:
  a9_nq.fasta       the 1,465 benchmark queries (calib3di/results/queries.fasta: same records, same order)
  a9_lq.fasta       the 3,000 P0.7 long queries (p0/long/long_q.fasta, 300-3,000 aa)
  a9_db_long.fasta  the P0.7 long targets: long_t (2,000 annotated, 1,000-3,000 aa), then long_r (2,000 long reversed
                    decoys, 1,000-3,000 aa). This is the column order of the research nq_long matrices
                    (p0/long/out/index.npz long_ids).
  a9_db_ben.fasta   the 30,000 benchmark targets of P0.7's lq_ben columns: the 25,000 annotated targets and 5,000 of
                    the 25,000 reversed decoys, in that column order. The decoys follow the score_long.py rule
                    np.sort(default_rng(20260927).choice(decoy rows, 5000, replace=False)); the rule is recomputed here
                    and checked against index.npz ben_cols.
  a9_inputs.npz     per set: ids, lengths and non-standard counts (FoldMatch's S13 rule: residues outside the 20
                    standard amino acids)
  a9_inputs.json    counts and md5s; FoldMatch's own strict FASTA pre-flight of every file (the one that
                    --residue-codes builds and esm-codes queries apply); the S13 query-side counts; provenance

Every record keeps its original header (id = first token, as FoldMatch and the research index read it) and its
sequence on one line. The sequences are checked to be FoldMatch-normalised already, so ESM3 sees the research string.
Usage:  python a9_prep.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import acc_common as C  # noqa: E402  (first: it sets the BLAS thread variables before numpy loads)
import numpy as np  # noqa: E402

SEED = 20260927          # score_long.py SEED
REV_SUB = 5000           # score_long.py REV_SUB
STD = set("ACDEFGHIKLMNPQRSTVWY")


def write_fasta(path, recs):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        for h, s in recs:
            fh.write(f">{h}\n{s}\n")
    os.replace(tmp, path)
    return C.md5_file(path)


def facts(recs):
    ids = np.array([h.split()[0] for h, _ in recs])
    L = np.array([len(s) for _, s in recs], np.int64)
    ns = np.array([sum(c not in STD for c in s) for _, s in recs], np.int64)
    return ids, L, ns


def main():
    from pathlib import Path
    from foldmatch.utils.sequence import normalize_sequence, preflight_fasta
    out = C.result_dir("A9", "inputs")
    ix = np.load(C.inp("p0/long/out/index.npz"))
    bix = np.load(C.inp("embcomp/evalue/score/out/index.npz"))
    rec = {"title": "P8 A9 inputs (P0.7 long-sequence slice through FoldMatch)", "sources": {}}
    for k in ("p0/long/long_q.fasta", "p0/long/long_t.fasta", "p0/long/long_r.fasta", "p0/long/out/index.npz",
              "calib3di/results/queries.fasta", "calib3di/results/targets.fasta", "embcomp/evalue/score/out/index.npz",
              "p0/long/score_long.py", "p0/long/analyze_long.py"):
        rec["sources"][k] = {"path": C.inp(k), "realpath": os.path.realpath(C.inp(k)), "md5": C.md5_file(C.inp(k))}

    lq = C.read_fasta(C.inp("p0/long/long_q.fasta"))
    lt = C.read_fasta(C.inp("p0/long/long_t.fasta"))
    lr = C.read_fasta(C.inp("p0/long/long_r.fasta"))
    nq = C.read_fasta(C.inp("calib3di/results/queries.fasta"))
    bt = C.read_fasta(C.inp("calib3di/results/targets.fasta"))

    # the research orders (index.npz) and the score_long.py decoy subset
    assert [h.split()[0] for h, _ in lq] == ix["lq_ids"].astype(str).tolist(), "long_q order"
    assert [h.split()[0] for h, _ in lt + lr] == ix["long_ids"].astype(str).tolist(), "long_t + long_r order"
    assert [h.split()[0] for h, _ in nq] == ix["nq_ids"].astype(str).tolist() == bix["query_ids"].astype(str).tolist()
    tids = np.array([h.split()[0] for h, _ in bt])
    assert tids.tolist() == bix["target_ids"].astype(str).tolist(), "benchmark target order"
    rev = np.array([t.startswith("rev_") for t in tids])
    ann_rows = np.flatnonzero(~rev)
    rev_rows = np.sort(np.random.default_rng(SEED).choice(np.flatnonzero(rev), REV_SUB, replace=False))
    ben_cols = np.concatenate([ann_rows, rev_rows])
    assert np.array_equal(ben_cols, ix["ben_cols"]), "ben_cols differ from the score_long.py rule"
    rec["ben_cols"] = dict(annotated=int(len(ann_rows)), reversed_subset=int(len(rev_rows)), seed=SEED,
                           rule="score_long.py load(): ann rows + np.sort(default_rng(20260927).choice(rev rows, "
                                "5000, replace=False))", equals_index_npz=True)
    ben = [bt[j] for j in ben_cols.tolist()]

    sets = {"nq": ("a9_nq.fasta", nq), "lq": ("a9_lq.fasta", lq), "db_long": ("a9_db_long.fasta", lt + lr),
            "db_ben": ("a9_db_ben.fasta", ben)}
    arrays = {"ben_cols": ben_cols}
    rec["sets"] = {}
    for key, (fn, recs) in sets.items():
        not_norm = [h.split()[0] for h, s in recs if normalize_sequence(s) != s]
        assert not not_norm, f"{key}: {len(not_norm)} sequences are not FoldMatch-normalised, e.g. {not_norm[:3]}"
        ids, L, ns = facts(recs)
        assert len(set(ids.tolist())) == len(ids), f"{key}: duplicate ids"
        path = os.path.join(out, fn)
        md5 = write_fasta(path, recs)
        pf = preflight_fasta(Path(path), 0)
        pf.check(strict=True)                      # raises on duplicate ids, '<' / '>' or non-ASCII sequences
        frac = ns / L
        rec["sets"][key] = dict(
            file=path, md5=md5, n=int(len(ids)), residues=int(L.sum()), min_len=int(L.min()), max_len=int(L.max()),
            foldmatch_preflight=dict(entries_kept=int(pf.entries_kept), problems=list(pf.problems())),
            n_len_gt_966=int((L > 966).sum()), n_len_lt_21=int((L < 21).sum()), n_len_gt_3000=int((L > 3000).sum()),
            n_nonstandard_gt_1pct=int((frac > 0.01).sum()), nonstandard_residues=int(ns.sum()),
            s13_query_side=dict(short=int((L < 21).sum()), long=int((L > 966).sum()),
                                nonstandard=int((frac > 0.01).sum()),
                                inside=int(((L >= 21) & (L <= 966) & (frac <= 0.01)).sum())))
        arrays[f"{key}_ids"], arrays[f"{key}_len"], arrays[f"{key}_nonstd"] = ids, L, ns
        C.log(key, rec["sets"][key])
    # the research column orders the production matrices are written in
    assert np.array_equal(arrays["db_long_len"], ix["long_len"]) and np.array_equal(arrays["lq_len"], ix["lq_len"])
    assert np.array_equal(arrays["db_ben_len"], ix["ben_len"]) and np.array_equal(arrays["nq_len"], ix["nq_len"])
    np.savez(os.path.join(out, "a9_inputs.npz"), **arrays)
    rec["a9_inputs_npz_md5"] = C.md5_file(os.path.join(out, "a9_inputs.npz"))
    rec["provenance"] = C.provenance(kernel=False)
    sha = C.write_json(rec, os.path.join(out, "a9_inputs.json"))
    C.log("wrote", os.path.join(out, "a9_inputs.json"), sha)


if __name__ == "__main__":
    main()
