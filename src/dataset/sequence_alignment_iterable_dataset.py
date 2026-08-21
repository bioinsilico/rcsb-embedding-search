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

from dataset.sequence_identity_dataset import collate_sequence_pairs
from dataset.utils.alignment_scorer import AlignmentFractionScorer
from dataset.utils.cath_superfamily import load_superfamilies, parse_cath_domain_id
from dataset.utils.minimizer_index import MinimizerIndex
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


def _load_ids(paths) -> set[str]:
    """Read one id per line from a path, or from several paths.

    Ids may be given either as full FASTA headers or as bare CATH domain ids
    (``101mA00``); :func:`_build_segment_index` accepts both forms, so a holdout
    list stays readable and does not have to mirror the header format.
    """
    if paths is None:
        return set()
    if isinstance(paths, str):
        paths = [paths]
    ids: set[str] = set()
    for path in paths:
        if not path:
            continue
        with open(path) as handle:
            # Only the first whitespace-separated token, matching how FASTA
            # headers are parsed below -- otherwise a file listing full
            # description lines matches nothing and filters silently no-op.
            ids |= {line.split()[0] for line in handle if line.strip()}
    return ids


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


def _world_size() -> int:
    """Number of ranks in the job, resolved the same way as :func:`_rank_identity`.

    ``samples_per_epoch`` is a whole-job figure, matching how ``epoch_size``
    behaves for the map-style datasets (where ``DistributedSampler`` divides it),
    so it has to be divided here.  Getting this wrong is expensive and silent —
    falling back to 1 on a 32-rank job makes every epoch 32x longer and stretches
    an epoch-indexed LR warmup by the same factor — so the resolved value is
    logged once per rank on the first epoch.
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    for var in ('WORLD_SIZE', 'SLURM_NTASKS'):
        value = os.environ.get(var)
        if value is not None:
            try:
                return max(1, int(value))
            except ValueError:
                pass
    return 1


def _build_segment_index(
    fasta_file: str,
    segment_length: int,
    window_step: int,
    exclude: set[str],
    include: set[str] | None = None,
    group_of=None,
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
    groups: list[int] = []
    n_records = n_short = n_excluded = n_normalized = 0

    def add(header: str | None, parts: list[str]) -> None:
        nonlocal n_records, n_short, n_excluded, n_normalized
        if header is None:
            return
        n_records += 1
        if exclude or include:
            # Accept either the raw header or the bare CATH domain id.
            domain = parse_cath_domain_id(header)
            if header in exclude or domain in exclude:
                n_excluded += 1
                return
            if include and header not in include and domain not in include:
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
        # Resolved here, while the header is still in hand: keeping 600k header
        # strings alive just to group them later would cost far more memory.
        groups.append(-1 if group_of is None else group_of(header))

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

    # A filter that matches nothing is nearly always a format mismatch between
    # the id file and the FASTA headers, and it fails in the worst way: training
    # silently keeps the domains meant to be held out.  Catch it at startup.
    if exclude and n_excluded == 0:
        logger.warning(
            f"exclude list has {len(exclude):,} ids but matched no header in {fasta_file} "
            f"-- nothing was excluded; check the id format"
        )
    if not lengths:
        detail = (
            f" (include list of {len(include):,} ids matched no header -- check the id format)"
            if include else ""
        )
        raise ValueError(
            f"No sequence in {fasta_file} is at least {segment_length} residues long{detail}"
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
    return (''.join(chunks), starts, n_grid, last_offset, window_counts,
            np.asarray(groups, dtype=np.int64))


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
        samples_per_epoch: samples produced per epoch across the **whole job**,
            divided across ranks and then across each rank's dataloader workers.
            This matches how ``epoch_size`` behaves for the map-style datasets,
            so the same config value means the same thing in both.
        score_method: optional post-processing of the raw fraction, e.g.
            ``fraction_score_of(10)`` to snap it to a 0.1 grid.  Binning uses
            the post-processed score, i.e. the value the model is trained on.
        n_intervals: number of balancing bins over [0, 1].
        balance_alpha: balancing strength, ``0`` disables it.
        max_attempts: alignment budget per emitted sample.
        exclude_ids_file: file (or list of files) of ids to drop, one per line;
            full FASTA headers and bare CATH domain ids are both accepted.
        include_ids_file: if given, keep *only* these ids.  Pointing a
            validation dataset at the same file the training dataset excludes
            gives a clean holdout split.
        p_kmer: probability of proposing a pair via the minimizer index, which
            is what makes genuinely related pairs reachable — uniform draws hit
            an appreciable alignment in under 0.02% of cases on CATH.
        p_offset: probability of proposing two windows of the same sequence at a
            drawn offset, giving a guaranteed score gradient.  The remaining
            probability mass proposes a uniform random pair.
        p_superfamily: probability of proposing two domains of one CATH
            homologous superfamily.  This is the only strategy that supplies
            *real* remote homology (roughly the 15-60% identity band); the
            minimizer index mostly surfaces near-duplicates.  Requires
            ``superfamily_file``.
        superfamily_file: CATH ``cath-domain-list.txt``, mapping domain ids to
            their C.A.T.H classification.
        max_self_offset: largest offset used by ``p_offset``; defaults to
            ``segment_length``, which sweeps overlap from full to none.
        kmer_index: a prebuilt :class:`MinimizerIndex` to share between the
            training and validation datasets — they must use the same FASTA,
            ``segment_length`` and ``window_step`` for window ids to agree.
            Built internally when omitted and ``p_kmer > 0``.
        kmer_size, minimizer_window, reduced_alphabet: index parameters.
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
        exclude_ids_file=None,
        include_ids_file=None,
        p_kmer: float = 0.0,
        p_offset: float = 0.0,
        p_superfamily: float = 0.0,
        superfamily_file: str | None = None,
        max_self_offset: int | None = None,
        kmer_index: MinimizerIndex | None = None,
        kmer_size: int = 6,
        minimizer_window: int = 10,
        reduced_alphabet: bool = True,
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
        self.p_kmer = float(p_kmer)
        self.p_offset = float(p_offset)
        self.p_superfamily = float(p_superfamily)
        weights = (self.p_kmer, self.p_offset, self.p_superfamily)
        if any(w < 0 for w in weights) or sum(weights) > 1:
            raise ValueError(
                f"p_kmer + p_offset + p_superfamily must lie in [0, 1], got {weights}"
            )
        self.max_self_offset = (
            self.segment_length if max_self_offset is None else int(max_self_offset)
        )
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self.log_every = int(log_every)
        self._epoch = 0

        group_of = None
        if superfamily_file is not None and self.p_superfamily > 0:
            domains = load_superfamilies(superfamily_file)
            group_of = lambda h: domains.get(parse_cath_domain_id(h), -1)  # noqa: E731

        blob, starts, n_grid, last_offset, window_counts, seq_group = _build_segment_index(
            fasta_file, self.segment_length, self.window_step,
            _load_ids(exclude_ids_file), _load_ids(include_ids_file), group_of,
        )
        self._blob = blob
        self._starts = starts
        self._seq_length = np.diff(np.append(starts, len(blob)))

        # Flattening every window's blob offset once turns drawing into a single
        # array lookup, and gives the minimizer index a stable window id space.
        window_cum = np.cumsum(window_counts)
        self._total_windows = int(window_cum[-1])
        self._window_seq = np.repeat(np.arange(len(starts)), window_counts)
        first_of_seq = window_cum - window_counts
        rank_in_seq = np.arange(self._total_windows) - first_of_seq[self._window_seq]
        offsets = np.where(
            rank_in_seq < n_grid[self._window_seq],
            rank_in_seq * self.window_step,
            last_offset[self._window_seq],
        )
        self._window_start = starts[self._window_seq] + offsets
        self._first_window = first_of_seq
        self._window_counts = window_counts
        self._build_superfamily_groups(seq_group)

        self.kmer_index = kmer_index
        if self.kmer_index is not None and self.kmer_index.n_windows != self._total_windows:
            raise ValueError(
                f"kmer_index was built for {self.kmer_index.n_windows:,} windows but this "
                f"corpus has {self._total_windows:,}. Sharing an index across different "
                f"corpora (e.g. a train/validation holdout split) would return segments "
                f"from unrelated windows -- build a separate index instead."
            )
        if self.kmer_index is None and self.p_kmer > 0:
            self.kmer_index = MinimizerIndex(
                blob, self._window_start, self.segment_length,
                k=kmer_size, w=minimizer_window, reduced=reduced_alphabet,
            )
        if self.kmer_index is None:
            self.p_kmer = 0.0

    # ------------------------------------------------------------------
    # Sharding and seeding
    # ------------------------------------------------------------------

    def _shard(self) -> tuple[int, np.random.SeedSequence]:
        """Return ``(n_samples, seed_sequence)`` for the calling worker.

        ``samples_per_epoch`` counts the whole job, so it is divided first
        across ranks and then across that rank's dataloader workers, with the
        per-worker remainder handed to the lowest worker ids.  Any remainder
        from the rank division is dropped rather than assigned, because
        :func:`_rank_identity` is not guaranteed to be contiguous; at realistic
        sizes that is at most ``world_size - 1`` samples out of the epoch.

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

        world = _world_size()
        per_rank = max(1, self.samples_per_epoch // world)
        n_samples = per_rank // num_workers
        if worker_id < per_rank % num_workers:
            n_samples += 1

        if self._epoch == 0 and worker_id == 0:
            logger.info(
                f"Sharding {self.samples_per_epoch} samples/epoch over world_size={world} "
                f"-> {per_rank} per rank over {num_workers} worker(s) "
                f"-> {n_samples} for worker 0 (rank id {rank})"
            )

        entropy = [self.seed, rank, worker_id]
        if not self.deterministic:
            entropy += [torch.initial_seed(), self._epoch]
        return n_samples, np.random.SeedSequence(entropy)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _build_superfamily_groups(self, seq_group: np.ndarray) -> None:
        """Group sequences by superfamily, keeping only families with >=2 members.

        Stored CSR-style (members sorted by family, plus per-family offsets) so a
        proposal is two array lookups rather than a dict of lists.
        """
        self._sf_members = self._sf_first = self._sf_count = None
        if self.p_superfamily <= 0:
            return
        order = np.argsort(seq_group, kind='stable')
        _, first, counts = np.unique(seq_group[order], return_index=True, return_counts=True)
        keep = (counts >= 2) & (seq_group[order][first] >= 0)
        if not keep.any():
            logger.warning("No superfamily has two members; disabling p_superfamily")
            self.p_superfamily = 0.0
            return
        self._sf_members = order
        self._sf_first = first[keep]
        self._sf_count = counts[keep]
        logger.info(
            f"Superfamily proposals: {len(self._sf_first):,} families with >=2 members "
            f"covering {int(self._sf_count.sum()):,} sequences"
        )

    def _superfamily_pair(self, rng: np.random.Generator):
        """Two windows from different domains of one CATH superfamily.

        Families are drawn uniformly rather than in proportion to their size.
        CATH is extremely skewed — the largest superfamily holds 49,516 domains
        against a median of 7 — so size-weighted draws would spend almost all
        their budget on a handful of folds, and an embedding used for search has
        to generalise across folds, not memorise the popular ones.
        """
        if self._sf_first is None:
            return None
        family = int(rng.integers(0, len(self._sf_first)))
        lo = int(self._sf_first[family])
        count = int(self._sf_count[family])
        a, b = rng.choice(count, size=2, replace=False)
        seq_a = int(self._sf_members[lo + int(a)])
        seq_b = int(self._sf_members[lo + int(b)])
        return (
            self._segment_at(int(self._first_window[seq_a])
                             + int(rng.integers(0, self._window_counts[seq_a]))),
            self._segment_at(int(self._first_window[seq_b])
                             + int(rng.integers(0, self._window_counts[seq_b]))),
        )

    def _segment_at(self, window_id: int) -> str:
        """The residues of one window, by window id."""
        start = int(self._window_start[window_id])
        return self._blob[start:start + self.segment_length]

    def _draw_segment(self, rng: np.random.Generator) -> str:
        """Draw one window uniformly at random from all windows in the corpus."""
        return self._segment_at(int(rng.integers(0, self._total_windows)))

    def _offset_pair(self, rng: np.random.Generator):
        """Two windows of one sequence, separated by a uniformly drawn offset.

        Offset ``d`` maps directly onto overlap ``(L - d) / L``, so this sweeps
        the whole score range with guaranteed coverage at every level — the one
        source of mid-range examples that needs no search at all.  Starts are
        not constrained to the stride grid here; the point is score control.
        """
        seq = int(self._window_seq[int(rng.integers(0, self._total_windows))])
        span = int(self._seq_length[seq]) - self.segment_length
        if span < 1:
            return None
        d = int(rng.integers(0, min(self.max_self_offset, span) + 1))
        s1 = int(rng.integers(0, span - d + 1))
        base = int(self._starts[seq])
        L = self.segment_length
        return self._blob[base + s1:base + s1 + L], self._blob[base + s1 + d:base + s1 + d + L]

    def _kmer_pair(self, rng: np.random.Generator):
        """A window plus another sharing a minimizer with it."""
        wid = int(rng.integers(0, self._total_windows))
        segment = self._segment_at(wid)
        partner = self.kmer_index.partner(segment, rng, exclude=wid)
        if partner < 0:
            return None
        return segment, self._segment_at(partner)

    def _draw_pair(self, rng: np.random.Generator):
        """Propose one candidate pair from the configured mixture of strategies.

        Only the *proposal* distribution changes here — every pair is still
        labelled by the same alignment, so no ground truth is invented.  Any
        strategy that cannot produce a pair falls back to uniform rather than
        retrying, which keeps the mixture weights honest.
        """
        u = rng.random()
        if u < self.p_offset:
            pair = self._offset_pair(rng)
        elif u < self.p_offset + self.p_kmer:
            pair = self._kmer_pair(rng)
        elif u < self.p_offset + self.p_kmer + self.p_superfamily:
            pair = self._superfamily_pair(rng)
        else:
            pair = None
        if pair is not None:
            return pair
        return self._draw_segment(rng), self._draw_segment(rng)

    def _bin_of(self, score: float) -> int:
        return min(max(int(score * self.n_intervals), 0), self.n_intervals - 1)

    def _accept_prob(self, counts: np.ndarray, k: int) -> float:
        if self.balance_alpha <= 0:
            return 1.0
        return float(((counts.min() + 1) / (counts[k] + 1)) ** self.balance_alpha)

    def __len__(self) -> int:
        """Samples this rank yields per epoch (across all of its workers)."""
        return max(1, self.samples_per_epoch // _world_size())

    def __iter__(self):
        n_samples, seed_sequence = self._shard()
        self._epoch += 1
        rng = np.random.default_rng(seed_sequence)

        counts = np.zeros(self.n_intervals, dtype=np.int64)
        attempts_total = 0

        for emitted in range(1, n_samples + 1):
            for _ in range(self.max_attempts):
                seg_i, seg_j = self._draw_pair(rng)
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
