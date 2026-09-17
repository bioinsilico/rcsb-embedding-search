"""Batch sampler that keeps similarly sized domains together, and shards across ranks.

Random batches of CATH/SCOP domains are about half padding, because a batch pads to
its longest member while lengths run from 16 to 1521 residues.  Sorting the whole
split by length would remove the padding but destroy shuffling, so this does the
usual compromise: shuffle, cut the order into pools of ``bucket_size``, sort each
pool by length, cut it into batches, then shuffle the batches.  Each batch is nearly
uniform in length while its composition still changes every epoch.

Two details exist because of how Lightning drives a batch sampler:

* **The rank is resolved lazily.**  Dataloaders are built before ``trainer.fit``
  starts the process group, so asking ``torch.distributed`` in ``__init__`` answers
  "world size 1" in every process, and every rank would then iterate the whole split
  - no error, no hang, just N times the work for one rank's progress.
* **``sampler`` points at this object.**  Lightning advances epochs by looking for
  ``dataloader.sampler`` or ``dataloader.batch_sampler.sampler`` and calling
  ``set_epoch`` on it.  A batch sampler is neither, so without this alias the epoch
  never advances and every epoch replays the same batches in the same order.

Pass ``use_distributed_sampler=False`` to the ``Trainer``: this shards itself, and
Lightning would otherwise wrap it and shard twice.
"""
from __future__ import annotations

import random

import torch.distributed as dist
from torch.utils.data import Sampler


def _distributed_context() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    return 1, 0


class LengthBucketBatchSampler(Sampler):
    """Yields lists of dataset indices, grouped by similar length.

    Args:
        lengths: residue count per dataset index.
        batch_size: samples per batch.
        bucket_size: how many samples are sorted together; larger means less padding
            and less randomness.
        shuffle: shuffle within an epoch (off for validation).
        seed: base seed; the epoch is added, so each epoch differs.
        num_replicas, rank: distributed sharding. Left as ``None`` they are read from
            the process group **at iteration time**, which is when it exists.
    """

    def __init__(self, lengths, batch_size: int, bucket_size: int = 2048,
                 shuffle: bool = True, seed: int = 0,
                 num_replicas: int | None = None, rank: int | None = None):
        super().__init__(None)
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.bucket_size = max(bucket_size, batch_size)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self._num_replicas = num_replicas
        self._rank = rank
        # Lightning looks for `dataloader.batch_sampler.sampler` to call set_epoch on.
        self.sampler = self

    @property
    def num_replicas(self) -> int:
        return self._num_replicas if self._num_replicas is not None else _distributed_context()[0]

    @property
    def rank(self) -> int:
        return self._rank if self._rank is not None else _distributed_context()[1]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _all_batches(self) -> list[list[int]]:
        order = list(range(len(self.lengths)))
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(order)
        batches = []
        for start in range(0, len(order), self.bucket_size):
            pool = sorted(order[start:start + self.bucket_size], key=lambda i: self.lengths[i])
            batches.extend(pool[i:i + self.batch_size] for i in range(0, len(pool), self.batch_size))
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        batches = self._all_batches()
        num_replicas, rank = self.num_replicas, self.rank
        # Every rank gets the same number of batches: a rank that runs out early leaves
        # the others waiting at a collective until the watchdog kills the job.
        per_rank = len(batches) // num_replicas
        yield from batches[rank * per_rank:(rank + 1) * per_rank]

    def __len__(self) -> int:
        # Bucket sizes depend only on the split size, so this needs no shuffling.
        total = 0
        for start in range(0, len(self.lengths), self.bucket_size):
            size = min(self.bucket_size, len(self.lengths) - start)
            total += -(-size // self.batch_size)
        return total // self.num_replicas
