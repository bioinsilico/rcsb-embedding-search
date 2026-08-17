"""On-the-fly segment-pair dataset for the sequence autoencoder.

Unlike :class:`~dataset.sequence_identity_dataset.SequenceIdentityDataset`,
which reads precomputed pairs and scores from a TSV, this dataset needs nothing
but a single FASTA file: it samples constant-length segments, aligns them, and
derives the target score from the alignment itself.
"""
from __future__ import annotations

import argparse
import logging
import os
import re

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from dataset.sequence_identity_dataset import _load_exclude, collate_sequence_pairs
from dataset.utils.alignment_scorer import AlignmentFractionScorer
from dataset.utils.tm_score_weight import fraction_score_of
from networks.sequence_autoencoder import tokenize_sequence

logger = logging.getLogger(__name__)

# Single-letter residues covered by the token vocabulary; anything else that
# shows up in a FASTA (gaps, stop codons, 'J', ...) is normalized to 'X'.
_RESIDUE_ALPHABET = 'ACDEFGHIKLMNPQRSTVWYXBZUO'
_NON_RESIDUE = re.compile(f'[^{_RESIDUE_ALPHABET}]')

# More local ranks per node than any real machine has, so that
# node_rank * span + local_rank stays collision-free across nodes.
_LOCAL_RANK_SPAN = 4096


def _rank_identity() -> int:
    """An integer unique to each participating process across all nodes.

    Used only to decorrelate random streams — every rank draws its own i.i.d.
    pairs, so nothing depends on this being the *true* global rank, only on it
    differing between processes.

    ``dist.get_rank()`` is unavailable exactly where it matters most: dataloader
    workers started with ``spawn`` (the default on macOS) do not inherit the
    process group, so ``dist.is_initialized()`` is ``False`` inside them even
    on rank 1.  The launcher's environment variables *are* inherited by both
    fork and spawn, so they are the reliable fallback.  ``LOCAL_RANK`` alone is
    not enough — it repeats on every node — hence the ``NODE_RANK`` term.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    for var in ('RANK', 'SLURM_PROCID'):          # global rank: torchrun, SLURM
        value = os.environ.get(var)
        if value is not None:
            return int(value)
    node = os.environ.get('NODE_RANK') or os.environ.get('GROUP_RANK') \
        or os.environ.get('SLURM_NODEID') or 0
    local = os.environ.get('LOCAL_RANK') or os.environ.get('SLURM_LOCALID') or 0
    return int(node) * _LOCAL_RANK_SPAN + int(local)


def _build_segment_index(
    fasta_file: str,
    segment_length: int,
    window_step: int,
    exclude: set[str],
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
    """Stream a FASTA file into a flat residue blob plus per-sequence offsets.

    Sequences shorter than ``segment_length`` are dropped, so every window that
    can be drawn from the index is exactly ``segment_length`` residues long.

    Windows sit on a fixed grid of stride ``window_step``.  When the last
    grid window stops short of the C-terminus, one extra window anchored at the
    end of the sequence is added, mirroring ``window_split`` in
    ``scripts/pairwise_sequence_identity.py`` so that no residue range is
    unreachable.

    Keeping the residues in a single string (instead of a dict of per-record
    strings) matters here: dataloader workers inherit the index by fork, and one
    large immutable object touches far fewer refcounted pages than millions of
    small ones.

    Returns:
        ``(blob, starts, n_grid, last_offset, window_counts)`` where
        ``blob[starts[i]:...]`` is the i-th retained sequence, ``n_grid[i]`` is
        how many of its windows lie on the stride grid, ``last_offset[i]`` is
        the in-sequence offset of its trailing window, and ``window_counts[i]``
        is its total window count.
    """
    chunks: list[str] = []
    lengths: list[int] = []
    n_records = n_short = n_excluded = n_normalized = 0

    def add(header: str | None, parts: list[str]) -> None:
        nonlocal n_records, n_short, n_excluded, n_normalized
        if header is None:
            return
        n_records += 1
        if header in exclude:
            n_excluded += 1
            return
        raw = ''.join(parts).upper()
        if len(raw) < segment_length:
            n_short += 1
            return
        seq = _NON_RESIDUE.sub('X', raw)
        if seq != raw:
            n_normalized += 1
        chunks.append(seq)
        lengths.append(len(seq))

    header: str | None = None
    parts: list[str] = []
    with open(fasta_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                add(header, parts)
                header = line[1:].split()[0]
                parts = []
            elif header is not None:
                parts.append(line)
        add(header, parts)

    if not lengths:
        raise ValueError(
            f"No sequence in {fasta_file} is at least {segment_length} residues long"
        )

    lengths_arr = np.asarray(lengths, dtype=np.int64)
    starts = np.zeros(len(lengths_arr), dtype=np.int64)
    np.cumsum(lengths_arr[:-1], out=starts[1:])

    n_grid = (lengths_arr - segment_length) // window_step + 1
    last_offset = lengths_arr - segment_length
    has_trailing = (n_grid - 1) * window_step < last_offset
    window_counts = n_grid + has_trailing.astype(np.int64)

    logger.info(
        f"Indexed {len(lengths_arr)} sequences from {fasta_file} "
        f"({int(window_counts.sum()):,} windows of length {segment_length} "
        f"at step {window_step}, {int(has_trailing.sum())} of them trailing); "
        f"read {n_records}, dropped {n_short} shorter than {segment_length}, "
        f"excluded {n_excluded}, normalized non-standard residues in {n_normalized}"
    )
    return ''.join(chunks), starts, n_grid, last_offset, window_counts


class SequenceAlignmentIterableDataset(IterableDataset):
    """Segment pairs scored by alignment coverage, generated on the fly.

    Each sample is built by drawing two windows of ``segment_length`` residues
    uniformly at random from all windows in the FASTA file, aligning them with
    ``aligner``, and using the resulting fraction of aligned residues as the
    target score.  Windows lie on a moving-window grid of stride
    ``window_step``, so a step below ``segment_length`` yields overlapping
    segments — the same windowing ``scripts/pairwise_sequence_identity.py``
    applies up front, done here on the fly instead.

    **Score balancing.**  Two random segments from an unrelated pair of proteins
    almost always align over a small fraction of their length, so the raw score
    distribution collapses onto the lowest bin.  Because an ``IterableDataset``
    cannot be driven by a ``WeightedRandomSampler``, balancing happens here: the
    score range [0, 1] is split into ``n_intervals`` bins and a candidate landing
    in bin *k* is accepted with probability
    ``((min_count + 1) / (count_k + 1)) ** balance_alpha``.  The least populated
    bin therefore always accepts, and over-represented bins are throttled.
    ``balance_alpha=0`` disables balancing; ``1.0`` flattens the distribution as
    hard as the corpus allows.

    Rejected candidates cost a full alignment, so ``max_attempts`` caps the work
    spent on one sample — after that many rejections the last candidate is
    emitted regardless.  This bounds the alignment cost per sample at
    ``max_attempts`` and keeps a corpus with no close homologs from starving the
    stream.

    Each ``__iter__`` yields ``(tokens_i, tokens_j, score)`` — the same tuple as
    ``SequenceIdentityDataset``, so :func:`collate_sequence_pairs` and the
    existing Lightning module apply unchanged.

    Args:
        fasta_file: FASTA file the segments are drawn from.
        aligner: object exposing ``score(seq_i, seq_j, denominator) -> float``.
            Defaults to :class:`AlignmentFractionScorer` (local, BLOSUM62).
        segment_length: constant residue length of every emitted segment.
        window_step: stride between consecutive window starts.  ``None`` uses
            ``segment_length``, i.e. non-overlapping windows; a smaller value
            makes them overlap.  A sequence whose length is not an exact fit
            gets one extra window anchored at its C-terminus.
        samples_per_epoch: samples produced per epoch **per rank**, split evenly
            across that rank's dataloader workers.
        score_method: optional post-processing of the raw fraction, e.g.
            ``fraction_score_of(10)`` to snap it to a 0.1 grid.  Binning uses
            the post-processed score, i.e. the value the model is trained on.
        n_intervals: number of balancing bins over [0, 1].
        balance_alpha: balancing strength, ``0`` disables it.
        max_attempts: alignment budget per emitted sample.
        exclude_ids_file: optional file of FASTA headers to skip, one per line.
        seed: base seed; combined with rank, worker id and epoch so that every
            worker on every rank of every node draws an independent stream.
        deterministic: replay the same stream on every epoch — use for the
            validation split, where a ``Subset`` of a map-style dataset is not
            available.
        log_every: emit accept-rate and bin-occupancy statistics every N
            samples per worker; ``0`` disables the logging.
    """

    def __init__(
        self,
        fasta_file: str,
        aligner=None,
        segment_length: int = 50,
        window_step: int | None = None,
        samples_per_epoch: int = 100_000,
        score_method=None,
        n_intervals: int = 5,
        balance_alpha: float = 1.0,
        max_attempts: int = 50,
        exclude_ids_file: str | None = None,
        seed: int = 42,
        deterministic: bool = False,
        log_every: int = 10_000,
    ):
        super().__init__()
        if segment_length < 1:
            raise ValueError(f"segment_length must be positive, got {segment_length}")
        if window_step is not None and window_step < 1:
            raise ValueError(f"window_step must be positive, got {window_step}")
        if n_intervals < 1:
            raise ValueError(f"n_intervals must be positive, got {n_intervals}")

        self.segment_length = int(segment_length)
        self.window_step = self.segment_length if window_step is None else int(window_step)
        self.aligner = AlignmentFractionScorer() if aligner is None else aligner
        self.samples_per_epoch = int(samples_per_epoch)
        self.score_method = score_method
        self.n_intervals = int(n_intervals)
        self.balance_alpha = float(balance_alpha)
        self.max_attempts = max(1, int(max_attempts))
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self.log_every = int(log_every)
        self._epoch = 0

        blob, starts, n_grid, last_offset, window_counts = _build_segment_index(
            fasta_file, self.segment_length, self.window_step, _load_exclude(exclude_ids_file)
        )
        self._blob = blob
        self._starts = starts
        self._n_grid = n_grid
        self._last_offset = last_offset
        self._window_cum = np.cumsum(window_counts)
        self._total_windows = int(self._window_cum[-1])

    # ------------------------------------------------------------------
    # Sharding and seeding
    # ------------------------------------------------------------------

    def _shard(self) -> tuple[int, np.random.SeedSequence]:
        """Return ``(n_samples, seed_sequence)`` for the calling worker.

        ``samples_per_epoch`` is split across the workers of this rank, with the
        remainder handed to the lowest worker ids.

        The stream is keyed on ``(seed, rank, worker_id)`` so that no two
        processes anywhere in the job coincide — Lightning seeds every rank
        identically, so without the rank term all GPUs would train on the same
        pairs.  ``SeedSequence`` is what turns those identifiers into
        independent streams; hand-mixing them risks collisions it is built to
        avoid.

        For training runs, ``torch.initial_seed()`` and the epoch counter add
        per-epoch entropy: the DataLoader redraws the former for every epoch,
        while the counter covers ``num_workers=0``, where it stays constant.
        """
        info = get_worker_info()
        worker_id, num_workers = (0, 1) if info is None else (info.id, info.num_workers)
        rank = _rank_identity()

        n_samples = self.samples_per_epoch // num_workers
        if worker_id < self.samples_per_epoch % num_workers:
            n_samples += 1

        entropy = [self.seed, rank, worker_id]
        if not self.deterministic:
            entropy += [torch.initial_seed(), self._epoch]
        return n_samples, np.random.SeedSequence(entropy)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _draw_segment(self, rng: np.random.Generator) -> str:
        """Draw one window uniformly at random from all windows in the corpus."""
        r = int(rng.integers(0, self._total_windows))
        idx = int(np.searchsorted(self._window_cum, r, side='right'))
        window = r - (int(self._window_cum[idx - 1]) if idx > 0 else 0)
        # Windows past the stride grid are the trailing, C-terminus-anchored one.
        offset = (
            window * self.window_step
            if window < self._n_grid[idx]
            else int(self._last_offset[idx])
        )
        start = int(self._starts[idx]) + offset
        return self._blob[start:start + self.segment_length]

    def _bin_of(self, score: float) -> int:
        return min(max(int(score * self.n_intervals), 0), self.n_intervals - 1)

    def _accept_prob(self, counts: np.ndarray, k: int) -> float:
        if self.balance_alpha <= 0:
            return 1.0
        return float(((counts.min() + 1) / (counts[k] + 1)) ** self.balance_alpha)

    def __len__(self) -> int:
        """Samples this rank yields per epoch (across all of its workers)."""
        return self.samples_per_epoch

    def __iter__(self):
        n_samples, seed_sequence = self._shard()
        self._epoch += 1
        rng = np.random.default_rng(seed_sequence)

        counts = np.zeros(self.n_intervals, dtype=np.int64)
        attempts_total = 0

        for emitted in range(1, n_samples + 1):
            for _ in range(self.max_attempts):
                seg_i = self._draw_segment(rng)
                seg_j = self._draw_segment(rng)
                raw = self.aligner.score(seg_i, seg_j, self.segment_length)
                score = raw if self.score_method is None else self.score_method(raw)
                k = self._bin_of(score)
                attempts_total += 1
                if rng.random() < self._accept_prob(counts, k):
                    break
            counts[k] += 1

            if self.log_every > 0 and emitted % self.log_every == 0:
                logger.info(
                    f"Emitted {emitted}/{n_samples} pairs "
                    f"({attempts_total / emitted:.1f} alignments per pair) | "
                    + ", ".join(f"bin{b}={counts[b]}" for b in range(self.n_intervals))
                )

            yield (
                torch.tensor(tokenize_sequence(seg_i), dtype=torch.long),
                torch.tensor(tokenize_sequence(seg_j), dtype=torch.long),
                torch.tensor(score, dtype=torch.float32),
            )


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--fasta_file', type=str, required=True)
    parser.add_argument('--segment_length', type=int, default=50)
    parser.add_argument('--window_step', type=int, default=None,
                        help='window stride; defaults to segment_length (non-overlapping)')
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--balance_alpha', type=float, default=1.0)
    parser.add_argument('--max_attempts', type=int, default=50)
    parser.add_argument('--matrix', type=str, default='BLOSUM62')
    parser.add_argument('--gap_open', type=int, default=-10)
    parser.add_argument('--gap_extend', type=int, default=-1)
    parser.add_argument('--global_alignment', action='store_true')
    args = parser.parse_args()

    dataset = SequenceAlignmentIterableDataset(
        fasta_file=args.fasta_file,
        aligner=AlignmentFractionScorer(
            matrix=args.matrix,
            gap_open=args.gap_open,
            gap_extend=args.gap_extend,
            local=not args.global_alignment,
        ),
        segment_length=args.segment_length,
        window_step=args.window_step,
        samples_per_epoch=args.samples,
        score_method=fraction_score_of(10),
        balance_alpha=args.balance_alpha,
        max_attempts=args.max_attempts,
        log_every=0,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        collate_fn=collate_sequence_pairs,
    )

    for seq_i, seq_j, score in dataloader:
        print(seq_i.shape, seq_j.shape, score)
