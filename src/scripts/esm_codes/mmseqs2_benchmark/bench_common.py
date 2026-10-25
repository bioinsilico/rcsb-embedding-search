#!/usr/bin/env python
"""Shared module of the full MMseqs2 benchmark (Steinegger & Soding, Nat Biotechnol 2017, nbt.3988) run with FoldMatch.

Benchmark: queries query-lt2000.fasta (6,294) against targetdb-lt2000.fasta (30,308,746 = the SCOP-annotated
'annotated_<acc>' UniRef50 sequences + the reversed UniProt decoys), sequences longer than 2,000 aa left out.

Label rules (the research calibration rules = MMseqs2 EvaluateResults.cpp, mmseqs-benchmark commit 3230f76b,
EvaluateResults.cpp md5 e571ee5194082b35646a4a6734a540e7, EvaluateResults.h md5 a574f39d5a689ed60db6bb5a388bd0ad =
the research copy calib3di/auc1/reconcile/EvaluateResults.h):
  * the SCOP families of an entry are the regex [a-z]+\\.[0-9]+\\.[0-9]+\\.[0-9]+ matches in the FASTA comment before
    its first '|', de-duplicated (readFamDefFromFasta); the fold is the first two levels;
  * TP  = the query and the target share a SCOP family (some domain pair);
  * IGN = not TP, and some domain pair shares the fold, or both domains are b.67-b.70 beta-propellers, or either
          domain is in class e (EvaluateResults: ignore_superfam '^b\\.(67|68|69|70).*', ignoreClass '^e\\..*');
  * FP  = a target without SCOP annotation (the reversed decoys: every entry of the target DB that is not an
          'annotated_' entry), or an annotated target that is neither TP nor IGN;
  * an annotated entry whose comment carries no family is neither (EvaluateResults: an empty domain vector -> the hit
    is not counted at all) - 'unannotated' below; none is expected in targetdb-lt2000 (counted by build_labels.py).
Target classes (KIND): 0 decoy (reversed UniProt), 1 annotated, 2 annotated id without any family ('unannotated'),
3 'annotated_' id absent from targetannotation.fasta (EvaluateResults: not in the lookup -> FP).

Files (NERSC; override the root with BENCH_EVAL):
  /pscratch/sd/j/jsegura/embcomp/bench/eval/labels/   label store written by build_labels.py, read by auc1.py
  /pscratch/sd/j/jsegura/embcomp/bench/eval/<method>/ per-method AUC1 results
Mac copy: /Users/joan/devel/foldmatch-p2p3-sandbox/bench/.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "foldmatch_acceptance"))
import acc_common as AC  # noqa: E402  (P8 library: read_fasta, sccs_of, lvl, label_matrices, Bench, write_json)

T0 = time.time()
IS_NERSC = AC.IS_NERSC
DB_DIR = "/pscratch/sd/j/jsegura/structure-embedding/mmseqs-db/db"
QUERY_FASTA = f"{DB_DIR}/query-lt2000.fasta"
TARGET_FASTA = f"{DB_DIR}/targetdb-lt2000.fasta"
ANNOT_FASTA = f"{DB_DIR}/targetannotation.fasta"
EVAL_ROOT = os.environ.get("BENCH_EVAL") or ("/pscratch/sd/j/jsegura/embcomp/bench/eval" if IS_NERSC
                                             else "/Users/joan/devel/foldmatch-p2p3-sandbox/bench")
LABELS_DIR = os.path.join(EVAL_ROOT, "labels")
MM_RESULT_DB = "/global/homes/j/jsegura/tmp/mm-out/query-lt-2000-vs-targetdb-lt2000-sen-3"
MM_QUERY_DB = "/pscratch/sd/j/jsegura/structure-embedding/mmseqs-db/mm-db/query-lt2000"
MM_TARGET_DB = "/pscratch/sd/j/jsegura/structure-embedding/mmseqs-db/mm-db/targetdb-lt2000"
MMSEQS_BIN = "/global/homes/j/jsegura/tmp/mmseqs/bin/mmseqs"
EVALUATOR_SOURCE = {"repo": "https://bitbucket.org/martin_steinegger/mmseqs-benchmark",
                    "commit": "3230f76b200e0f4e0a9df44e918a02559d6be4b7",
                    "EvaluateResults.cpp_md5": "e571ee5194082b35646a4a6734a540e7",
                    "EvaluateResults.h_md5": "a574f39d5a689ed60db6bb5a388bd0ad",
                    "paper_call": "evaluate_results query.fas db.fas results.m8 out 4000 1 (resSize 4000, rocx 1), "
                                  "results sorted with 'sort -k1,1 -k11,11g' (scripts/run_mmseqs_blast.sh)"}

KIND_DECOY, KIND_ANN, KIND_EMPTY, KIND_MISSING, KIND_UNKNOWN = 0, 1, 2, 3, -1
KIND_NAMES = {0: "decoy", 1: "annotated", 2: "unannotated_no_family", 3: "annotated_id_missing_from_annotation",
              -1: "not_in_db"}
ST_TP, ST_FP, ST_IGN, ST_NONE = 0, 1, 2, 3
ST_NAMES = {0: "TP", 1: "FP", 2: "IGN", 3: "NONE"}

FAM_RX = re.compile(r"[a-z]+\.[0-9]+\.[0-9]+\.[0-9]+")            # readFamDefFromFasta scopDomainRegex
PROP_RX = re.compile(r"^b\.(67|68|69|70).*")                        # evaluateResult ignore_superfam
CLASS_E_RX = re.compile(r"^e\..*")                                  # evaluateResult ignoreClass
PROP_RX_RESEARCH = re.compile(r"^b\.(67|68|69|70)\.")               # plan.py / acc_common (checked equal on vocab)
ANN_PREFIX = b"annotated_"

# query length bins: research EDGES (rlib / auc_final) extended to the lt2000 range
LEN_EDGES = [0, 40, 60, 100, 200, 400, 1000, 2001]
LEN_LABELS = ["0-40", "40-60", "60-100", "100-200", "200-400", "400-1000", "1000-2000"]


def log(*a):
    print(f"[{time.time() - T0:8.1f}s] " + " ".join(str(x) for x in a), flush=True)


write_json = AC.write_json
read_json = AC.read_json
md5_file = AC.md5_file
sha256_file = AC.sha256_file


# =====================================================================================================================
# parsing
# =====================================================================================================================
def comment_of(header: str) -> str:
    """The FASTA comment (kseq: everything after the first whitespace of the header line)."""
    p = header.split(None, 1)
    return p[1] if len(p) > 1 else ""


def families_evaluator(comment: str) -> List[str]:
    """readFamDefFromFasta: family regex matches in the comment before its first '|', as a sorted set."""
    return sorted(set(FAM_RX.findall(comment.split("|", 1)[0])))


def families_research(header: str) -> List[str]:
    """auc_final.sccs_of (= acc_common.sccs_of): '>id s1 s2 |...' -> the tokens before the first '|' token
    (sorted set); a header without ' |' -> its second token; 'rev_' ids -> none."""
    return AC.sccs_of(header)


def fold_of(fam: str) -> str:
    return AC.lvl(fam, 2)


def scop_class_label(fams: Sequence[str]) -> str:
    """The query's SCOP class (a, b, ...), 'multi' when its domains span several classes, 'none' without any."""
    c = sorted({f.split(".")[0] for f in fams})
    return c[0] if len(c) == 1 else ("multi" if c else "none")


def len_bin(lengths) -> np.ndarray:
    lengths = np.asarray(lengths)
    b = np.digitize(lengths, LEN_EDGES[1:-1], right=False)
    assert ((lengths >= np.asarray(LEN_EDGES)[b]) & (lengths < np.asarray(LEN_EDGES)[b + 1])).all()
    return b


def id_hash(b: bytes) -> int:
    """64-bit key of a target id (blake2b-8, little endian) for the decoy membership table."""
    return int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "little")


def id_hashes(ids: Iterable[bytes]) -> np.ndarray:
    return np.fromiter((id_hash(x) for x in ids), dtype=np.uint64)


def query_original(qid: str) -> str:
    """The target id of a query's source sequence: query 'P30340_1' comes from 'annotated_P30340'
    (mmseqs-benchmark selectQueries.py names a query '<db id>_1'; addSuffixToFasta.py prefixes 'annotated_')."""
    return "annotated_" + qid.rsplit("_", 1)[0]


def csr(lists: Sequence[Sequence[int]], dtype=np.int32) -> Tuple[np.ndarray, np.ndarray]:
    ptr = np.zeros(len(lists) + 1, np.int64)
    ptr[1:] = np.cumsum([len(x) for x in lists])
    idx = np.fromiter((v for x in lists for v in x), dtype=dtype, count=int(ptr[-1]))
    return ptr, idx


def csr_rows(ptr: np.ndarray, idx: np.ndarray, rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Gather the CSR rows `rows`: returns (row_of_each_value (index into `rows`), values)."""
    rows = np.asarray(rows, np.int64)
    cnt = ptr[rows + 1] - ptr[rows]
    rep = np.repeat(np.arange(len(rows), dtype=np.int64), cnt)
    off = np.cumsum(cnt) - cnt
    pos = ptr[rows][rep] + (np.arange(int(cnt.sum()), dtype=np.int64) - off[rep])
    return rep, idx[pos]


# =====================================================================================================================
# the label store
# =====================================================================================================================
def build_label_arrays(q_ids: Sequence[str], q_len: Sequence[int], q_fams: Sequence[Sequence[str]],
                       ann_ids: Sequence[bytes], ann_len: Sequence[int], ann_fams: Sequence[Sequence[str]],
                       decoy_hash: np.ndarray, other_ids: Sequence[bytes] = (), other_kind: Sequence[int] = (),
                       fam_size_file: Optional[Dict[str, int]] = None, log_fn=log) -> Dict[str, np.ndarray]:
    """The label store as arrays (what build_labels.py saves and auc1.Labels loads).

    ann_ids must be sorted (bytes order) and unique; the TP set of a query = the annotated targets carrying any of its
    families (CSR tp_ptr / tp_idx, sorted annotated indices); fam_size_db = annotated targets of the DB carrying the
    family (once per target: the evaluator's scopSizeLoopup over the searched DB); famsize_sum = sum over the query's
    distinct families of fam_size_db (the evaluator's qFamSize before the resSize cap)."""
    ann_ids = np.asarray(ann_ids, dtype="S")
    assert (ann_ids[1:] > ann_ids[:-1]).all(), "annotated ids must be sorted and unique"
    vocab = sorted(set(f for l_ in q_fams for f in l_) | set(f for l_ in ann_fams for f in l_))
    fpos = {f: i for i, f in enumerate(vocab)}
    folds = sorted(set(fold_of(f) for f in vocab))
    dpos = {d: i for i, d in enumerate(folds)}
    fam_fold = np.array([dpos[fold_of(f)] for f in vocab], np.int32)
    fam_prop = np.array([bool(PROP_RX.match(f)) for f in vocab])
    fam_e = np.array([bool(CLASS_E_RX.match(f)) for f in vocab])
    prop_research = np.array([bool(PROP_RX_RESEARCH.match(f)) for f in vocab])
    n_prop_regex_diff = int((fam_prop != prop_research).sum())

    a_fptr, a_fidx = csr([[fpos[f] for f in l_] for l_ in ann_fams])
    q_fptr, q_fidx = csr([[fpos[f] for f in l_] for l_ in q_fams])

    def fold_csr(fptr, fidx):
        out = []
        for i in range(len(fptr) - 1):
            out.append(sorted(set(fam_fold[fidx[fptr[i]:fptr[i + 1]]].tolist())))
        return csr(out)
    a_dptr, a_didx = fold_csr(a_fptr, a_fidx)
    q_dptr, q_didx = fold_csr(q_fptr, q_fidx)

    def any_flag(fptr, fidx, flag):
        hit = flag[fidx].astype(np.int64)
        row = np.repeat(np.arange(len(fptr) - 1), np.diff(fptr))
        return np.bincount(row, weights=hit, minlength=len(fptr) - 1) > 0
    a_prop, a_e = any_flag(a_fptr, a_fidx, fam_prop), any_flag(a_fptr, a_fidx, fam_e)
    q_prop, q_e = any_flag(q_fptr, q_fidx, fam_prop), any_flag(q_fptr, q_fidx, fam_e)

    # family -> member annotated targets (CSR over vocab)
    n_ann = len(ann_ids)
    a_row = np.repeat(np.arange(n_ann, dtype=np.int64), np.diff(a_fptr))
    o = np.lexsort((a_row, a_fidx))
    mem_idx = a_row[o].astype(np.int32)
    fam_size_db = np.bincount(a_fidx, minlength=len(vocab)).astype(np.int64)
    mem_ptr = np.zeros(len(vocab) + 1, np.int64)
    mem_ptr[1:] = np.cumsum(fam_size_db)
    log_fn(f"label arrays: {len(q_ids)} queries, {n_ann} annotated targets, {len(vocab)} families, {len(folds)} "
           f"folds; building TP sets")
    tp_lists, n_tp = [], np.zeros(len(q_ids), np.int64)
    for i in range(len(q_ids)):
        fs = q_fidx[q_fptr[i]:q_fptr[i + 1]]
        parts = [mem_idx[mem_ptr[f]:mem_ptr[f + 1]] for f in fs]
        u = np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int32)
        tp_lists.append(u.astype(np.int32))
        n_tp[i] = u.size
    tp_ptr = np.zeros(len(q_ids) + 1, np.int64)
    tp_ptr[1:] = np.cumsum(n_tp)
    tp_idx = np.concatenate(tp_lists) if tp_lists else np.zeros(0, np.int32)
    famsize_sum = np.array([int(fam_size_db[q_fidx[q_fptr[i]:q_fptr[i + 1]]].sum()) for i in range(len(q_ids))],
                           np.int64)
    # each query's source sequence ('annotated_<acc>' for '<acc>_1') among the annotated targets
    q_self = np.full(len(q_ids), -1, np.int64)
    for i, q in enumerate(q_ids):
        key = query_original(q).encode()
        j = int(np.searchsorted(ann_ids, key))
        if j < n_ann and ann_ids[j] == key:
            q_self[i] = j
    out = dict(
        q_ids=np.array(q_ids, dtype=str), q_len=np.asarray(q_len, np.int64), q_fam_ptr=q_fptr, q_fam_idx=q_fidx,
        q_fold_ptr=q_dptr, q_fold_idx=q_didx, q_prop=q_prop, q_e=q_e,
        q_class=np.array([scop_class_label([vocab[f] for f in q_fidx[q_fptr[i]:q_fptr[i + 1]]])
                          for i in range(len(q_ids))], dtype=str),
        q_self=q_self, n_tp=n_tp, famsize_sum=famsize_sum, tp_ptr=tp_ptr, tp_idx=tp_idx,
        ann_ids=ann_ids, ann_len=np.asarray(ann_len, np.int64), ann_fam_ptr=a_fptr, ann_fam_idx=a_fidx,
        ann_fold_ptr=a_dptr, ann_fold_idx=a_didx, ann_prop=a_prop, ann_e=a_e,
        fam_names=np.array(vocab, dtype=str), fam_fold=fam_fold, fam_prop=fam_prop, fam_e=fam_e,
        fam_size_db=fam_size_db, fold_names=np.array(folds, dtype=str),
        decoy_hash=np.sort(np.asarray(decoy_hash, np.uint64)),
        other_ids=np.array(list(other_ids), dtype="S") if len(other_ids) else np.zeros(0, "S1"),
        other_kind=np.asarray(other_kind, np.int8),
        n_prop_regex_diff=np.array(n_prop_regex_diff))
    if fam_size_file is not None:
        out["fam_size_file"] = np.array([fam_size_file.get(f, 0) for f in vocab], np.int64)
    return out


class Labels:
    """The label store (build_label_arrays) with the classification and pair-status logic used by auc1.py."""

    def __init__(self, arrays: Dict[str, np.ndarray], source: str = ""):
        self.a = arrays
        self.source = source
        for k, v in arrays.items():
            setattr(self, k, v)
        self.q_ids = np.asarray(self.q_ids).astype(str)
        self.nq = len(self.q_ids)
        self.n_ann = len(self.ann_ids)
        self.n_fold = len(self.fold_names)
        self.qpos = {q: i for i, q in enumerate(self.q_ids)}
        assert len(self.qpos) == self.nq
        self.q_lenbin = len_bin(self.q_len)
        self._tp_keys = None
        self._qfold_keys = None
        self.other = {bytes(x): int(k) for x, k in zip(self.other_ids, self.other_kind)}

    @classmethod
    def load(cls, labels_dir: str = LABELS_DIR) -> "Labels":
        z = np.load(os.path.join(labels_dir, "labels.npz"), allow_pickle=False)
        return cls({k: z[k] for k in z.files}, source=labels_dir)

    # ------------------------------------------------------------------ targets
    def classify(self, ids_bytes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """KIND and annotated index (-1 if not annotated) of each target id (numpy 'S' array)."""
        ids_bytes = np.asarray(ids_bytes, dtype="S")
        n = len(ids_bytes)
        kind = np.full(n, KIND_UNKNOWN, np.int8)
        ann = np.full(n, -1, np.int64)
        if n == 0:
            return kind, ann
        j = np.searchsorted(self.ann_ids, ids_bytes)
        jc = np.minimum(j, self.n_ann - 1)
        is_ann = self.ann_ids[jc] == ids_bytes
        kind[is_ann] = KIND_ANN
        ann[is_ann] = jc[is_ann]
        rest = np.flatnonzero(~is_ann)
        for r in rest:
            k = self.other.get(bytes(ids_bytes[r]))
            if k is not None:
                kind[r] = k
        rest = rest[kind[rest] == KIND_UNKNOWN]
        if rest.size:
            h = id_hashes(bytes(x) for x in ids_bytes[rest])
            p = np.minimum(np.searchsorted(self.decoy_hash, h), len(self.decoy_hash) - 1)
            ok = self.decoy_hash[p] == h
            kind[rest[ok]] = KIND_DECOY
        return kind, ann

    # ------------------------------------------------------------------ pairs
    def tp_keys(self) -> np.ndarray:
        if self._tp_keys is None:
            q = np.repeat(np.arange(self.nq, dtype=np.int64), np.diff(self.tp_ptr))
            self._tp_keys = q * self.n_ann + self.tp_idx.astype(np.int64)   # sorted by q then ann
        return self._tp_keys

    def qfold_keys(self) -> np.ndarray:
        if self._qfold_keys is None:
            q = np.repeat(np.arange(self.nq, dtype=np.int64), np.diff(self.q_fold_ptr))
            self._qfold_keys = np.sort(q * self.n_fold + self.q_fold_idx.astype(np.int64))
        return self._qfold_keys

    def pair_status(self, qi: np.ndarray, kind: np.ndarray, ann: np.ndarray, *, fp_mode: str = "all",
                    exceptions: bool = True) -> np.ndarray:
        """EvaluateResults status of (query, target) rows: ST_TP / ST_FP / ST_IGN / ST_NONE.
        fp_mode: 'all' (decoys and annotated wrong folds are FPs), 'real' (decoys ignored: the research real null),
        'decoy' (annotated wrong folds ignored: EvaluateResults --ignore-fp). exceptions=False drops the propeller /
        class-e rule (the research P0 'spec' protocol)."""
        qi = np.asarray(qi, np.int64)
        kind = np.asarray(kind)
        ann = np.asarray(ann, np.int64)
        st = np.full(len(qi), ST_NONE, np.int8)
        if np.any(kind == KIND_UNKNOWN):
            raise ValueError(f"{int((kind == KIND_UNKNOWN).sum())} hit rows have a target that is not in the DB")
        dec = (kind == KIND_DECOY) | (kind == KIND_MISSING)
        st[dec] = ST_IGN if fp_mode == "real" else ST_FP
        a = np.flatnonzero(kind == KIND_ANN)
        if a.size:
            keys = qi[a] * self.n_ann + ann[a]
            tk = self.tp_keys()
            p = np.minimum(np.searchsorted(tk, keys), max(len(tk) - 1, 0))
            is_tp = (tk[p] == keys) if len(tk) else np.zeros(a.size, bool)
            st[a[is_tp]] = ST_TP
            nt = a[~is_tp]
            if nt.size:
                rep, folds = csr_rows(self.ann_fold_ptr, self.ann_fold_idx, ann[nt])
                fk = qi[nt][rep] * self.n_fold + folds.astype(np.int64)
                qk = self.qfold_keys()
                p = np.minimum(np.searchsorted(qk, fk), max(len(qk) - 1, 0))
                hit = (qk[p] == fk) if len(qk) else np.zeros(fk.size, bool)
                shared = np.bincount(rep, weights=hit.astype(np.float64), minlength=nt.size) > 0
                ign = shared.copy()
                if exceptions:
                    ign |= (self.q_prop[qi[nt]] & self.ann_prop[ann[nt]]) | self.q_e[qi[nt]] | self.ann_e[ann[nt]]
                st[nt[ign]] = ST_IGN
                st[nt[~ign]] = ST_IGN if fp_mode == "decoy" else ST_FP
        # KIND_EMPTY stays ST_NONE (EvaluateResults: annotated entry without domains -> not counted)
        return st

    def dense_status(self, rows: np.ndarray, kind: np.ndarray, ann: np.ndarray, **kw) -> np.ndarray:
        """[len(rows), n_targets] status matrix for the given target columns (validation on the 50k sample)."""
        nt = len(kind)
        out = np.empty((len(rows), nt), np.int8)
        for i, r in enumerate(rows):
            out[i] = self.pair_status(np.full(nt, r), kind, ann, **kw)
        return out

    # ------------------------------------------------------------------ denominators
    def excluded_ann(self, exclude_self: str) -> Optional[np.ndarray]:
        """Annotated targets removed from every list and every denominator: 'global' = all query originals
        (research P1 'db_noself'); 'none' / 'own' -> None ('own' is applied per query in auc1)."""
        if exclude_self != "global":
            return None
        m = np.zeros(self.n_ann, bool)
        m[self.q_self[self.q_self >= 0]] = True
        return m

    def denominators(self, kind: str = "ntp", exclude_self: str = "none", cap: Optional[int] = None) -> np.ndarray:
        """Per-query AUC denominator (float64).
        'ntp'     = the query's distinct TP targets in the DB;
        'famsize' = EvaluateResults qFamSize: sum over the query's distinct families of the family's member count in
                    the DB (a target carrying two of the query's families counts twice);
        exclude_self 'own' removes the query's own source sequence, 'global' every query's source sequence
        (from all TP sets and all family sizes); cap = min(., cap) (EvaluateResults resSize)."""
        if kind == "ntp":
            d = self.n_tp.astype(np.int64).copy()
            ex = self.excluded_ann(exclude_self)
            if ex is not None:
                q = np.repeat(np.arange(self.nq), np.diff(self.tp_ptr))
                d -= np.bincount(q, weights=ex[self.tp_idx].astype(np.float64), minlength=self.nq).astype(np.int64)
            elif exclude_self == "own":
                d -= self._self_is_tp().astype(np.int64)
        elif kind == "famsize":
            ex = self.excluded_ann(exclude_self)
            fs = self.fam_size_db.astype(np.int64).copy()
            if ex is not None:
                rep, fam = csr_rows(self.ann_fam_ptr, self.ann_fam_idx, np.flatnonzero(ex))
                fs -= np.bincount(fam, minlength=len(fs)).astype(np.int64)
            q = np.repeat(np.arange(self.nq), np.diff(self.q_fam_ptr))
            d = np.bincount(q, weights=fs[self.q_fam_idx].astype(np.float64), minlength=self.nq).astype(np.int64)
            if exclude_self == "own":
                # the own source sequence counts once per shared family in qFamSize
                for i in np.flatnonzero(self.q_self >= 0):
                    a = self.q_self[i]
                    shared = np.intersect1d(self.q_fam_idx[self.q_fam_ptr[i]:self.q_fam_ptr[i + 1]],
                                            self.ann_fam_idx[self.ann_fam_ptr[a]:self.ann_fam_ptr[a + 1]])
                    d[i] -= shared.size
        else:
            raise ValueError(kind)
        d = d.astype(np.float64)
        if cap is not None:
            d = np.minimum(d, float(cap))
        return d

    def _self_is_tp(self) -> np.ndarray:
        out = np.zeros(self.nq, bool)
        for i in np.flatnonzero(self.q_self >= 0):
            s = self.tp_idx[self.tp_ptr[i]:self.tp_ptr[i + 1]]
            j = np.searchsorted(s, self.q_self[i])
            out[i] = j < s.size and s[j] == self.q_self[i]
        return out


def labels_from_headers(q_headers: Sequence[str], q_len: Sequence[int], t_headers: Sequence[str],
                        t_len: Sequence[int], *, query_rule: str = "research",
                        target_rule: str = "evaluator") -> Tuple[Labels, np.ndarray, Dict]:
    """A Labels store for an explicit query x target set given as FASTA headers (the research 50k sample): targets
    whose id starts with 'rev_' or does not start with 'annotated_' are decoys. Returns (labels, target ids as 'S'
    in input order, info)."""
    q_ids = [h.split()[0] for h in q_headers]
    qf_r = [families_research(h) for h in q_headers]
    qf_e = [families_evaluator(comment_of(h)) for h in q_headers]
    q_fams = qf_r if query_rule == "research" else qf_e
    t_ids = [h.split()[0] for h in t_headers]
    ann_rows = [j for j, t in enumerate(t_ids) if t.startswith("annotated_")]
    tf_e = {j: families_evaluator(comment_of(t_headers[j])) for j in ann_rows}
    tf_r = {j: families_research(t_headers[j]) for j in ann_rows}
    t_fams = tf_e if target_rule == "evaluator" else tf_r
    order = sorted(ann_rows, key=lambda j: t_ids[j].encode())
    ann_set = set(ann_rows)
    dec = [t_ids[j].encode() for j in range(len(t_ids)) if j not in ann_set]
    empty = [t_ids[j].encode() for j in order if not t_fams[j]]
    order = [j for j in order if t_fams[j]]
    arr = build_label_arrays(q_ids, q_len, q_fams, [t_ids[j].encode() for j in order], [t_len[j] for j in order],
                             [t_fams[j] for j in order], id_hashes(dec), other_ids=empty,
                             other_kind=[KIND_EMPTY] * len(empty), log_fn=lambda *a: None)
    info = dict(query_rule_mismatch=int(sum(a != b for a, b in zip(qf_r, qf_e))),
                target_rule_mismatch=int(sum(tf_e[j] != tf_r[j] for j in ann_rows)),
                n_ann=len(order), n_decoy=len(dec), n_empty=len(empty))
    return Labels(arr, source="headers"), np.array(t_ids, dtype="S"), info
