"""Configurable pairwise alignment scoring for on-the-fly dataset generation.

The scorer wraps biotite's ``align_optimal`` and reports the **fraction of
aligned residues** rather than the sequence identity: the number of alignment
columns where *both* segments contribute a residue, divided by the segment
length.

Local (Smith-Waterman) alignment is the default because it is the only mode in
which the fraction spreads over [0, 1] — a global alignment of two equal-length
segments matches them end to end, so the fraction sits near 1.0 even for
unrelated pairs.
"""
from __future__ import annotations

import logging

import biotite.sequence.align as align
import numpy as np
from biotite.sequence import ProteinSequence

logger = logging.getLogger(__name__)

# Residues accepted by ``ProteinSequence`` but not by the token vocabulary and
# vice versa.  Selenocysteine and pyrrolysine are folded onto their standard
# counterparts so that the substitution matrix has an entry for them.
_ALIGN_TABLE = str.maketrans({'U': 'C', 'O': 'K'})


def to_alignment_alphabet(seq: str) -> str:
    """Map a normalized residue string onto biotite's protein alphabet."""
    return seq.translate(_ALIGN_TABLE)


class AlignmentFractionScorer:
    """Score a segment pair by the fraction of residues that align.

    Args:
        matrix: substitution matrix name resolvable by biotite
            (``BLOSUM62``, ``BLOSUM50``, ``PAM250``, ...).  ``None`` uses
            biotite's standard protein matrix (BLOSUM62).
        gap_open: gap opening penalty (negative).  Used as a linear penalty
            when ``gap_extend`` is ``None``.
        gap_extend: gap extension penalty (negative) for an affine gap model.
            ``None`` selects a linear gap penalty.
        local: ``True`` for Smith-Waterman, ``False`` for Needleman-Wunsch.
        terminal_penalty: whether terminal gaps are penalized.  Only relevant
            for global alignment.
    """

    def __init__(
        self,
        matrix: str | None = 'BLOSUM62',
        gap_open: int = -10,
        gap_extend: int | None = -1,
        local: bool = True,
        terminal_penalty: bool = False,
        identity_weighted: bool = False,
    ):
        self.matrix_name = matrix
        self.gap_open = gap_open
        self.gap_extend = gap_extend
        self.local = local
        self.terminal_penalty = terminal_penalty
        # Counting only identically-matched columns instead of all aligned ones.
        # (coverage x identity-within-alignment collapses to matches/denominator.)
        self.identity_weighted = identity_weighted
        self._matrix = None
        logger.info(
            f"Alignment scorer: matrix={matrix} gap_penalty={self.gap_penalty} "
            f"local={local} terminal_penalty={terminal_penalty} "
            f"identity_weighted={identity_weighted}"
        )

    @property
    def gap_penalty(self):
        """Linear (int) or affine (tuple) gap penalty as biotite expects it."""
        return self.gap_open if self.gap_extend is None else (self.gap_open, self.gap_extend)

    @property
    def matrix(self):
        """Substitution matrix, built lazily so it is created inside workers."""
        if self._matrix is None:
            if self.matrix_name is None:
                self._matrix = align.SubstitutionMatrix.std_protein_matrix()
            else:
                alphabet = ProteinSequence.alphabet
                self._matrix = align.SubstitutionMatrix(alphabet, alphabet, self.matrix_name)
        return self._matrix

    def __getstate__(self):
        # The matrix is rebuilt on first use; keep it out of the pickle so that
        # ``spawn``-started dataloader workers stay cheap to launch.
        state = self.__dict__.copy()
        state['_matrix'] = None
        return state

    def score(self, seq_i: str, seq_j: str, denominator: int) -> float:
        """Fraction of ``denominator`` residues that align between two segments.

        Args:
            seq_i: first segment, already normalized to the residue alphabet.
            seq_j: second segment, likewise.
            denominator: residue count the fraction is expressed against —
                the constant segment length for this dataset.

        Returns:
            ``aligned_columns / denominator`` in [0, 1], where an aligned column
            is one in which neither segment has a gap.
        """
        if denominator <= 0:
            return 0.0
        a_seq = ProteinSequence(to_alignment_alphabet(seq_i))
        b_seq = ProteinSequence(to_alignment_alphabet(seq_j))
        alignments = align.align_optimal(
            a_seq, b_seq,
            self.matrix,
            gap_penalty=self.gap_penalty,
            terminal_penalty=self.terminal_penalty,
            local=self.local,
            max_number=1,
        )
        if not alignments:
            return 0.0
        trace = alignments[0].trace
        if len(trace) == 0:
            return 0.0
        both = (trace[:, 0] != -1) & (trace[:, 1] != -1)
        if not self.identity_weighted:
            return int(np.count_nonzero(both)) / denominator
        matched = sum(1 for p, q in trace[both] if a_seq[p] == b_seq[q])
        return matched / denominator
