"""Minimizer index over sequence windows, used to *propose* candidate pairs.

Uniform random pairing cannot find related sequences: measured on CATH, fewer
than 0.02% of random window pairs align appreciably, because relatedness is
sparse in pair space no matter how redundant the corpus is.  The rejection
balancer can only up-weight what it draws, so it ends up filling its top bins
with chance alignments.

This index fixes the *proposal* distribution, never the label — a proposed pair
is still scored by the same alignment as any other, so no ground truth is
invented here.

Two choices make it sensitive to remote homology rather than just duplicates:

* **Reduced alphabet.**  Residues are folded into 11 physico-chemical groups
  (Murphy-style), so a conservative substitution does not destroy a seed match.
* **Minimizers.**  For every window of ``w`` consecutive k-mers the smallest by
  hash is kept.  This gives the guarantee an ad-hoc sketch cannot: two segments
  sharing any stretch of ``w + k - 1`` residues are *certain* to share a
  minimizer, while storage stays at roughly ``2/(w+1)`` entries per position.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Murphy-style reduction: conservative substitutions collapse to one symbol.
_GROUPS = ('LVIM', 'C', 'A', 'G', 'ST', 'P', 'FYW', 'EDNQ', 'KR', 'H')
_UNKNOWN = len(_GROUPS)                       # group 10: X and anything unmapped
_ALPHABET_SIZE = len(_GROUPS) + 1
# Ambiguity codes folded onto their closest standard residue.
_FOLD = {'U': 'C', 'O': 'K', 'B': 'N', 'Z': 'Q'}

# Odd 64-bit constant; multiplicative hashing to order k-mers pseudo-randomly.
_HASH_MULT = np.uint64(0x9E3779B97F4A7C15)


def _residue_table(reduced: bool) -> np.ndarray:
    """256-entry byte -> symbol table for the chosen alphabet."""
    table = np.full(256, _UNKNOWN if reduced else 25, dtype=np.uint8)
    for code in range(256):
        ch = chr(code).upper()
        ch = _FOLD.get(ch, ch)
        if reduced:
            for g, members in enumerate(_GROUPS):
                if ch in members:
                    table[code] = g
                    break
        elif 'A' <= ch <= 'Z':
            table[code] = ord(ch) - ord('A')
    return table


class MinimizerIndex:
    """Maps a minimizer to the windows containing it.

    Args:
        blob: the concatenated residue string the windows are cut from.
        window_start: blob offset of every window, one entry per window id.
        segment_length: window length in residues.
        k: k-mer size, in symbols of the (possibly reduced) alphabet.
        w: number of consecutive k-mers each minimizer is chosen from.
        reduced: fold residues into physico-chemical groups before indexing.
        chunk: windows processed per batch while building, to bound peak memory.
    """

    def __init__(
        self,
        blob: str,
        window_start: np.ndarray,
        segment_length: int,
        k: int = 6,
        w: int = 10,
        reduced: bool = True,
        chunk: int = 200_000,
    ):
        if segment_length < k + w:
            raise ValueError(
                f"segment_length {segment_length} is too short for k={k}, w={w}"
            )
        self.k = int(k)
        self.w = int(w)
        # Window ids are positions in the corpus this index was built from;
        # reusing it against a different corpus would silently return segments
        # from unrelated windows, so callers must be able to check.
        self.n_windows = int(len(window_start))
        self.reduced = bool(reduced)
        self.segment_length = int(segment_length)
        self._alphabet = _ALPHABET_SIZE if reduced else 26
        self._table = _residue_table(reduced)
        self._powers = (self._alphabet ** np.arange(self.k, dtype=np.int64))[::-1]

        codes = np.frombuffer(blob.encode('ascii', 'replace'), dtype=np.uint8)
        keys_parts: list[np.ndarray] = []
        wids_parts: list[np.ndarray] = []
        n_windows = len(window_start)
        for lo in range(0, n_windows, chunk):
            hi = min(lo + chunk, n_windows)
            keys, wids = self._minimizers_for(codes, window_start[lo:hi], lo)
            keys_parts.append(keys)
            wids_parts.append(wids)

        keys = np.concatenate(keys_parts)
        wids = np.concatenate(wids_parts)
        order = np.argsort(keys, kind='stable')
        self._keys = keys[order]
        self._wids = wids[order]
        self._unique, self._first = np.unique(self._keys, return_index=True)

        logger.info(
            f"Minimizer index: {len(self._keys):,} entries over {n_windows:,} windows "
            f"({len(self._unique):,} distinct, k={k}, w={w}, "
            f"{'reduced' if reduced else 'full'} alphabet, "
            f"{(self._keys.nbytes + self._wids.nbytes) / 1e6:.0f} MB); "
            f"any shared stretch of {k + w - 1} residues yields a shared minimizer"
        )

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def _kmers(self, symbols: np.ndarray) -> np.ndarray:
        """(n, L) symbols -> (n, L-k+1) integer k-mer codes."""
        n_pos = symbols.shape[1] - self.k + 1
        out = np.zeros((symbols.shape[0], n_pos), dtype=np.int64)
        for offset in range(self.k):
            out += symbols[:, offset:offset + n_pos].astype(np.int64) * self._powers[offset]
        return out

    def _minimizers_for(self, codes, starts, base_id):
        """Minimizer keys and window ids for one chunk of windows."""
        idx = starts[:, None] + np.arange(self.segment_length)[None, :]
        symbols = self._table[codes[idx]]
        kmers = self._kmers(symbols)

        hashed = (kmers.astype(np.uint64) * _HASH_MULT)
        n_windows = kmers.shape[1] - self.w + 1
        picks = np.empty((kmers.shape[0], n_windows), dtype=np.int64)
        for i in range(n_windows):
            local = np.argmin(hashed[:, i:i + self.w], axis=1)
            picks[:, i] = kmers[np.arange(kmers.shape[0]), i + local]

        # Consecutive minimizer windows usually select the same k-mer; keeping
        # only the changes collapses that run-length redundancy.
        keep = np.ones_like(picks, dtype=bool)
        keep[:, 1:] = picks[:, 1:] != picks[:, :-1]
        rows, _ = np.nonzero(keep)
        return picks[keep].astype(np.int64), (rows + base_id).astype(np.int64)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def minimizers_of(self, segment: str) -> np.ndarray:
        """Minimizer keys of a single segment string."""
        codes = np.frombuffer(segment.encode('ascii', 'replace'), dtype=np.uint8)
        symbols = self._table[codes][None, :]
        kmers = self._kmers(symbols)
        hashed = (kmers.astype(np.uint64) * _HASH_MULT)
        n_windows = kmers.shape[1] - self.w + 1
        picks = np.empty(n_windows, dtype=np.int64)
        for i in range(n_windows):
            picks[i] = kmers[0, i + int(np.argmin(hashed[0, i:i + self.w]))]
        return np.unique(picks)

    def partner(self, segment: str, rng: np.random.Generator, exclude: int = -1) -> int:
        """A random window sharing a minimizer with ``segment``; -1 if none.

        One minimizer is chosen at random rather than all of them being pooled:
        pooling would bias towards windows that happen to share many seeds,
        which correlates with low complexity rather than with homology.
        """
        keys = self.minimizers_of(segment)
        if len(keys) == 0:
            return -1
        for key in rng.permutation(keys):
            pos = int(np.searchsorted(self._unique, key))
            if pos >= len(self._unique) or self._unique[pos] != key:
                continue
            lo = int(self._first[pos])
            hi = int(self._first[pos + 1]) if pos + 1 < len(self._first) else len(self._keys)
            if hi - lo <= 0:
                continue
            candidate = int(self._wids[rng.integers(lo, hi)])
            if candidate != exclude:
                return candidate
        return -1
