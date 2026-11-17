"""TM-score pair training over a packed embedding store, with a bin-balanced, length-bucketed sampler.

Replaces :class:`~dataset.tm_score_from_embeddings_dataset.TmScoreFromEmbeddingsDataset` plus
:class:`~dataset.utils.custom_weighted_random_sampler.CustomWeightedRandomSampler` for large runs:

* embeddings come from one :class:`~dataset.utils.packed_embeddings.PackedEmbeddingStore` (a memmap,
  optionally staged to node-local storage) instead of two ``torch.load`` calls per sample;
* pairs are numpy arrays (domain indices and TM) instead of a DataFrame searched with ``.loc``;
* :class:`BinBalancedPairBatchSampler` draws the production distribution - weight 1 / count of the
  pair's TM bin (:class:`~dataset.utils.tm_score_weight.TmScoreWeight` with 5 bins of width 0.2 and
  TM == 1 in its own bin at identity scale 0.25) - by drawing a bin by its mass and then a uniform
  pair inside it. Every rank draws its own stream from (seed, rank, epoch), and draws are sorted by
  length within pools so batches carry little padding.

The pairs file is an ``.npz`` with ``domains`` (names), ``i`` and ``j`` (int32 indices into
``domains``) and ``score`` (float32 TM), written by ``scripts/build_tm_pair_store.py``.
"""
from __future__ import annotations

import logging
import math

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler

from dataset.utils.packed_embeddings import PackedEmbeddingStore
from dataset.utils.tm_score_weight import FractionScore

logger = logging.getLogger(__name__)

N_BINS = 5
IDENTITY_SCALE = 0.25


def _distributed_context() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    return 1, 0


def cath_twins(excluded, domains):
    """CATH domains cut from the same PDB chain as an excluded SCOP domain (d<pdb><chain><n>)."""
    chains = {(d[1:5] + d[5]).lower() for d in excluded if d.startswith("d")}
    return {d for d in domains if not d.startswith("d") and d[:5].lower() in chains}


class BinaryScore:
    """1 when TM > thr else 0: the production validation label (binary_score(0.7)), picklable."""

    def __init__(self, thr=0.7):
        self.thr = thr

    def __call__(self, score):
        return 1.0 if score > self.thr else 0.0


def tm_bins(score):
    """Bin of every pair as TmScoreWeight.get_weight indexes it: floor(5 * TM), TM == 1 -> bin 5."""
    b = np.minimum(np.floor(score.astype(np.float64) * N_BINS).astype(np.int64), N_BINS)
    b[score == 1.0] = N_BINS
    return b


class PackedTmPairDataset(Dataset):
    """``(embedding_i, embedding_j, target)`` for every kept pair.

    Args:
        store_dir: packed store with one ``[L + 2, 1536]`` ESM3 tensor per domain.
        pairs_file: pairs ``.npz`` (see module docstring).
        exclude_domains: domains whose pairs are dropped (both sides), as ``exclude_domains_file``
            did in the original training.
        exclude_cath_twins: also drop CATH domains from the same PDB chain as an excluded SCOP
            domain (they are the same structure under another classification).
        fraction_score: targets are TM snapped to a 1 / fraction_score grid (production: 10).
        target: a callable label (e.g. :class:`BinaryScore` for validation); overrides fraction_score.
        seq_store_dir: optional packed store of sequence-only ESM3 residues of the same domains.
        seq_mix: probability that each side of a pair is independently replaced by its sequence-only
            view (when the domain is in ``seq_store_dir``); 0 = structure views only.
    """

    def __init__(self, store_dir, pairs_file, exclude_domains=None, exclude_cath_twins=True, fraction_score=10,
                 target=None, seq_store_dir=None, seq_mix=0.0):
        self.store = PackedEmbeddingStore(store_dir)
        self.seq_store = PackedEmbeddingStore(seq_store_dir) if seq_store_dir and seq_mix > 0 else None
        self.seq_mix = float(seq_mix)
        self.return_meta = False
        z = np.load(pairs_file)
        domains = [str(d) for d in z["domains"]]
        i, j, score = z["i"].astype(np.int32), z["j"].astype(np.int32), z["score"].astype(np.float32)
        present = np.array([d in self.store for d in domains])
        bad = ~present
        excluded = set(exclude_domains or [])
        if excluded:
            drop = set(excluded)
            if exclude_cath_twins:
                twins = cath_twins(excluded, domains)
                logger.info(f"excluding {len(excluded)} domains and {len(twins)} CATH twins")
                drop |= twins
            bad |= np.array([d in drop for d in domains])
        keep = ~(bad[i] | bad[j])
        logger.info(f"pairs {len(i):,}; kept {int(keep.sum()):,}; domains missing from the store {int((~present).sum())}")
        self.domains = domains
        self.i, self.j, self.score = i[keep], j[keep], score[keep]
        self.domain_length = np.array([self.store.length(d) if p else 0 for d, p in zip(domains, present)],
                                      dtype=np.int32)
        self.target = target if target is not None else FractionScore(fraction_score)
        if self.seq_store is not None:
            self.has_seq = np.array([d in self.seq_store for d in domains])
            logger.info(f"sequence-only views for {int(self.has_seq[present].sum()):,} of {int(present.sum()):,} domains, "
                        f"mixed with probability {self.seq_mix}")

    def __len__(self):
        return len(self.i)

    def pair_lengths(self):
        return np.maximum(self.domain_length[self.i], self.domain_length[self.j])

    def view(self, d):
        """Residue embeddings of domain index d and whether they are the sequence-only view: structure
        view, or with probability seq_mix the sequence-only view (torch RNG: per-worker streams seeded by
        seed_everything(workers=True))."""
        if self.seq_store is not None and self.has_seq[d] and torch.rand(()).item() < self.seq_mix:
            return self.seq_store.get(self.domains[d]).float(), True
        return self.store.get(self.domains[d]).float(), False

    def residue_lengths(self):
        """Residues per domain (store rows minus the ESM3 BOS / EOS rows); 0 for domains missing from the store."""
        return np.maximum(self.domain_length - 2, 0)

    def __getitem__(self, k):
        # k is a pair index, or (pair index, stratum) when the sampler draws containment strata
        k, stratum = k if isinstance(k, tuple) else (k, None)
        x, x_seq = self.view(self.i[k])
        y, y_seq = self.view(self.j[k])
        # np.float32 in, as production (pandas .iloc gives float32): 10 * TM is rounded in float32
        t = torch.tensor(self.target(self.score[k]), dtype=torch.float32)
        if self.return_meta:            # domain indices and views (and stratum), for teacher lookups (distillation)
            meta = [self.i[k], self.j[k], x_seq, y_seq] + ([] if stratum is None else [stratum])
            return x, y, t, torch.tensor(meta, dtype=torch.long)
        return x, y, t


def containment_strata(dataset: PackedTmPairDataset, pep_len=50, ratio=2.0, tm_pos=0.7, tm_neg=0.5):
    """Pairs whose shorter domain has fewer than ``pep_len`` residues and is at most 1 / ``ratio`` of the longer
    one (a peptide-like domain inside a larger one): positives with TM >= ``tm_pos`` and length-matched negatives
    with TM < ``tm_neg``, as indices into the dataset's pairs (train_heads.py ``contain``)."""
    res = dataset.residue_lengths()
    li, lj = res[dataset.i], res[dataset.j]
    lo, hi = np.minimum(li, lj), np.maximum(li, lj)
    base = (lo < pep_len) & (hi >= ratio * lo)
    return np.nonzero(base & (dataset.score >= tm_pos))[0], np.nonzero(base & (dataset.score < tm_neg))[0]


class BinBalancedPairBatchSampler(Sampler):
    """Batches of pair indices drawn from the production TM-bin-balanced distribution.

    Each epoch every rank draws ``samples_per_epoch // world // batch_size`` batches from its own
    generator (seed, rank, epoch), so ranks see different pairs and the same number of batches.
    Draws are cut into pools of ``pool_batches`` batches, sorted by pair length within a pool, cut
    into batches, and the batches put in an order that every rank shares (a permutation seeded by
    (seed, epoch) only): at each step all ranks hold batches from the same length quantile, so no
    GPU waits long for another. The pair distribution is unchanged; padding drops.

    Resuming: set ``resume = (epoch, batches_done)`` before ``trainer.fit`` (see
    ``training/embedding_tm_score_v2.py``). Lightning builds the first iterator before it calls
    ``set_epoch``, so the epoch is set here directly, and the batches already trained in that epoch
    are skipped; the skip lasts until ``set_epoch`` moves to another epoch.

    Strata: ``strata = [(pair indices, mass), ...]`` replaces a fixed share ``mass`` of the draws by uniform
    draws from each pool (:func:`containment_strata`); the rest follow the production distribution. Batches
    then hold ``(pair index, stratum)`` tuples (stratum 0 = production draw), which the dataset passes on in
    ``meta`` so distillation can leave strata rows out. Without strata the draws are unchanged.

    Pass ``use_distributed_sampler=False`` to the Trainer (this shards itself). The rank is read
    lazily, at iteration time, for the reason given in ``length_bucket_sampler``.
    """

    def __init__(self, dataset: PackedTmPairDataset, batch_size: int, samples_per_epoch: int,
                 pool_batches: int = 64, seed: int = 0, num_replicas: int | None = None, rank: int | None = None,
                 strata=None):
        super().__init__(None)
        # an empty pool keeps its place with weight 0 (the others renormalise), as train_heads.py's with_strata
        self.strata = [(np.asarray(p, dtype=np.int64), float(m)) for p, m in (strata or [])]
        assert sum(m for _, m in self.strata) < 1.0, "strata masses must leave room for the production draws"
        if not any(len(p) and m > 0 for p, m in self.strata):
            self.strata = []
        self.batch_size = batch_size
        self.samples_per_epoch = samples_per_epoch
        self.pool = max(1, pool_batches) * batch_size
        self.seed = seed
        self.epoch = 0
        self._num_replicas, self._rank = num_replicas, rank
        self.lengths = dataset.pair_lengths()
        self.resume = None
        bins = tm_bins(dataset.score)
        order = np.argsort(bins, kind="stable").astype(np.int32)      # 137M pairs: int32 halves memory
        counts = np.bincount(bins, minlength=N_BINS + 1)
        self.order = order
        self.starts = np.cumsum(counts) - counts
        self.counts = counts
        mass = (counts > 0).astype(np.float64)
        mass[N_BINS] = (1.0 / IDENTITY_SCALE) * (counts[N_BINS] > 0)
        self.mass = mass / mass.sum()
        logger.info(f"pair bins {counts.tolist()}; draw probability per bin {np.round(self.mass, 4).tolist()}")
        if self.strata:
            logger.info("strata (pairs, mass): " + ", ".join(f"({len(p):,}, {m})" for p, m in self.strata))
        self.sampler = self                  # Lightning calls set_epoch on batch_sampler.sampler

    @property
    def num_replicas(self) -> int:
        return self._num_replicas if self._num_replicas is not None else _distributed_context()[0]

    @property
    def rank(self) -> int:
        return self._rank if self._rank is not None else _distributed_context()[1]

    def set_epoch(self, epoch: int) -> None:
        if self.resume is not None and self.resume[0] != epoch:
            self.resume = None
        self.epoch = epoch

    def __len__(self) -> int:
        return self.samples_per_epoch // self.num_replicas // self.batch_size

    def draw(self, n, rng):
        b = rng.choice(len(self.mass), size=n, p=self.mass)
        off = np.floor(rng.random(n) * self.counts[b]).astype(np.int64)
        return self.order[self.starts[b] + off]

    def draw_strata(self, n, rng):
        """Pair indices and strata (0 = production draw) of n draws."""
        p = np.array([1.0 - sum(m for _, m in self.strata)] + [m if len(pool) else 0.0 for pool, m in self.strata])
        st = rng.choice(len(p), size=n, p=p / p.sum())
        idx = self.draw(n, rng).astype(np.int64)
        for k, (pool, _) in enumerate(self.strata, 1):
            sel = np.nonzero(st == k)[0]
            if len(sel) and len(pool):
                idx[sel] = pool[np.floor(rng.random(len(sel)) * len(pool)).astype(np.int64)]
        return idx, st

    def __iter__(self):
        rng = np.random.Generator(np.random.PCG64([self.seed, self.rank, self.epoch]))
        n_batches = len(self)
        if self.strata:
            idx, st = self.draw_strata(n_batches * self.batch_size, rng)
        else:
            idx, st = self.draw(n_batches * self.batch_size, rng), None
        batches = []
        for a in range(0, len(idx), self.pool):
            chunk = idx[a:a + self.pool]
            o = np.argsort(self.lengths[chunk], kind="stable")
            chunk = chunk[o]
            if st is not None:
                cst = st[a:a + self.pool][o]
                batches += [list(zip(chunk[b:b + self.batch_size].tolist(), cst[b:b + self.batch_size].tolist()))
                            for b in range(0, len(chunk), self.batch_size)]
                continue
            batches += [chunk[b:b + self.batch_size].tolist() for b in range(0, len(chunk), self.batch_size)]
        batches = [b for b in batches if len(b) == self.batch_size][:n_batches]
        shared = np.random.Generator(np.random.PCG64([self.seed, self.epoch, 2 ** 31]))
        batches = [batches[k] for k in shared.permutation(len(batches))]
        if self.resume is not None and self.resume[0] == self.epoch:
            batches = batches[self.resume[1]:]
        return iter(batches)


class ShardedSequentialBatchSampler(Sampler):
    """Fixed validation batches: all pairs sorted by length, batched, dealt to ranks round-robin.

    Every rank gets the same number of full batches (the tail is dropped), so distributed metric
    reductions and all_gather stay in step.
    """

    def __init__(self, dataset: PackedTmPairDataset, batch_size: int,
                 num_replicas: int | None = None, rank: int | None = None):
        super().__init__(None)
        order = np.argsort(dataset.pair_lengths(), kind="stable")
        # full batches only: every rank then holds the same number of predictions (all_gather)
        self.batches = [order[a:a + batch_size].tolist() for a in range(0, len(order) - batch_size + 1, batch_size)]
        self._num_replicas, self._rank = num_replicas, rank
        self.sampler = self

    @property
    def num_replicas(self) -> int:
        return self._num_replicas if self._num_replicas is not None else _distributed_context()[0]

    @property
    def rank(self) -> int:
        return self._rank if self._rank is not None else _distributed_context()[1]

    def set_epoch(self, epoch: int) -> None:
        pass

    def __len__(self) -> int:
        return len(self.batches) // self.num_replicas

    def __iter__(self):
        n = len(self)
        return iter(self.batches[self.rank::self.num_replicas][:n])


def expected_steps(draws: int, batch_size: int, world: int) -> int:
    return math.ceil(draws / (batch_size * world))
