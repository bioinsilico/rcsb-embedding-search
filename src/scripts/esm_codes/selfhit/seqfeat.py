"""Self-hit study: sequence-only features of a protein (no ESM3). Imported by analyze.py.

* ``lc_frac``: the panel strata's SEG-like low-complexity fraction (PREREGISTRATION.txt section 4): the fraction of
  residues covered by at least one 12-residue window whose Shannon entropy (20 standard letters + 1 pooled
  non-standard class) is < 2.2 bits; trigger only, no SEG extension. Sequences shorter than 12 use one window.
* ``topidp``: mean TOP-IDP disorder propensity (Campen et al. 2008) over standard residues (panel strata scale).
* ``comp_entropy``: Shannon entropy (bits) of the whole-sequence composition.
* ``kmer3_redundancy``: 1 - distinct 3-mers / all 3-mers (0 for a sequence without a repeated 3-mer).
* ``blosum_diag``: mean BLOSUM62 score of a residue with itself (the per-residue self score of the BLOSUM62 path).
* ``blosum_repeat``: best ungapped BLOSUM62 segment score between the sequence and itself off the main diagonal
  (offsets >= 4): an internal-repeat score, ~ 15-25 for a random 300-aa sequence.
* header flags: ``uncharacterized``, ``fragment``, cluster size ``n`` (UniRef50 headers ``n=``).
"""
from __future__ import annotations

import math
import re

import numpy as np

STD = "ACDEFGHIKLMNPQRSTVWY"
TOPIDP = dict(W=-.884, F=-.697, Y=-.510, I=-.486, M=-.397, L=-.326, V=-.121, N=.007, C=.020, T=.059, A=.060, G=.166,
              R=.180, D=.192, H=.303, Q=.318, S=.341, K=.586, E=.736, P=.987)
_IDX = np.full(256, 20, np.int64)
for _k, _c in enumerate(STD):
    _IDX[ord(_c)] = _k
_BLOSUM = None


def _blosum():
    global _BLOSUM
    if _BLOSUM is None:
        from biotite.sequence.align import SubstitutionMatrix
        m = SubstitutionMatrix.std_protein_matrix()
        alph = m.get_alphabet1()
        score = m.score_matrix()
        B = np.full((21, 21), -1, np.int64)
        for a, ca in enumerate(STD):
            for b, cb in enumerate(STD):
                B[a, b] = score[alph.encode(ca), alph.encode(cb)]
        _BLOSUM = B
    return _BLOSUM


def _entropy(counts):
    c = counts[counts > 0].astype(np.float64)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def lc_frac(seq: str, w: int = 12, k1: float = 2.2) -> float:
    x = _IDX[np.frombuffer(seq.encode("ascii"), np.uint8)]
    L = len(x)
    if L == 0:
        return 0.0
    if L < w:
        return 1.0 if _entropy(np.bincount(x, minlength=21)) < k1 else 0.0
    onehot = np.zeros((L + 1, 21), np.int64)
    onehot[np.arange(1, L + 1), x] = 1
    cum = onehot.cumsum(0)
    win = cum[w:] - cum[:-w]                      # [L - w + 1, 21]
    p = win / w
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(np.where(p > 0, p * np.log2(p), 0.0)).sum(1)
    trig = np.flatnonzero(h < k1)
    cover = np.zeros(L + 1, np.int64)
    np.add.at(cover, trig, 1)
    np.add.at(cover, trig + w, -1)
    return float((cover.cumsum()[:L] > 0).mean())


def features(seq: str, header: str = "") -> dict:
    x = _IDX[np.frombuffer(seq.encode("ascii"), np.uint8)]
    L = len(x)
    std = x[x < 20]
    topidp = float(np.mean([TOPIDP[STD[i]] for i in std])) if len(std) else math.nan
    kmers = [seq[i:i + 3] for i in range(L - 2)]
    red = 1.0 - len(set(kmers)) / len(kmers) if kmers else 0.0
    B = _blosum()
    diag = float(B[x, x].mean()) if L else math.nan
    best = 0
    if L > 4:
        M = B[x[:, None], x[None, :]]
        run = np.zeros(L, np.int64)               # Kadane on every diagonal d >= 4 at once, run[d]
        for k in range(L - 4):
            ds = np.arange(4, L - k)
            r = run[ds]
            r = np.where(r > 0, r, 0) + M[k, k + ds]
            run[ds] = r
            best = max(best, int(r.max()))
    h = header
    m = re.search(r"\bn=(\d+)", h)
    return dict(L=L, lc_frac=lc_frac(seq), topidp=topidp, comp_entropy=_entropy(np.bincount(x, minlength=21)),
                kmer3_redundancy=red, blosum_diag=diag, blosum_repeat=float(best),
                uncharacterized=bool(re.search(r"uncharacteri[sz]ed", h, re.I)),
                fragment=bool(re.search(r"fragment", h, re.I)), n=int(m.group(1)) if m else -1)
