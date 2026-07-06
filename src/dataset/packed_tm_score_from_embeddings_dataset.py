"""TM-score dataset backed by a packed, memory-mapped embedding store.

Drop-in replacement for :class:`~dataset.tm_score_from_embeddings_dataset.TmScoreFromEmbeddingsDataset`
that serves embeddings from a single memory-mapped blob (see
:mod:`dataset.utils.packed_embeddings`) instead of one ``torch.load`` per
sample. The item contract is identical -- ``(emb_i, emb_j, score)`` -- so the
existing ``collate_fn`` and training step work unchanged.

Build a store first with ``scripts/build_packed_embeddings.py``.
"""

import argparse
import logging

import numpy as np
import torch

from torch.utils.data import Dataset

from dataset.utils.packed_embeddings import PackedEmbeddingStore, stage_store_to_local
from dataset.utils.tm_score_weight import (
    binary_score,
    binary_weights,
    fraction_score,
    tm_score_weights,
)
from dataset.utils.tools import load_class_pairs

d_type = np.float32

logger = logging.getLogger(__name__)


class PackedTmScoreFromEmbeddingsDataset(Dataset):
    BINARY_THR = 0.7

    def __init__(
            self,
            tm_score_file,
            store_dir,
            score_method=None,
            weighting_method=None,
            exclude_domains_file=None,
            local_scratch=None,
            drop_missing=True
    ):
        self.score_method = binary_score(self.BINARY_THR) if not score_method else score_method
        self.weighting_method = binary_weights(self.BINARY_THR) if not weighting_method else weighting_method

        if local_scratch:
            store_dir = stage_store_to_local(store_dir, local_scratch)
        logger.info(f"Opening packed embedding store at {store_dir}")
        self.store = PackedEmbeddingStore(store_dir)
        logger.info(f"Store holds {len(self.store)} domains (dim={self.store.dim}, dtype={self.store.dtype})")

        self.class_pairs = self._load_pairs(tm_score_file, exclude_domains_file, drop_missing)

    def _load_pairs(self, tm_score_file, exclude_domains_file, drop_missing):
        logger.info(f"Loading pairs from file {tm_score_file}")
        pairs = load_class_pairs(tm_score_file, _exclude_domains(exclude_domains_file))
        logger.info(f"Total pairs: {len(pairs)}")
        if drop_missing:
            present = set(self.store.domains.tolist())
            keep = pairs["domain_i"].isin(present) & pairs["domain_j"].isin(present)
            dropped = int((~keep).sum())
            if dropped:
                logger.warning(
                    f"Dropping {dropped}/{len(pairs)} pairs referencing domains absent from the packed store"
                )
            pairs = pairs[keep].reset_index(drop=True)
        return pairs

    def weights(self):
        return self.weighting_method(
            np.expand_dims(self.class_pairs['score'].to_numpy(), axis=1)
        )

    def __len__(self):
        return len(self.class_pairs)

    def __getitem__(self, idx):
        row = self.class_pairs.iloc[idx]
        return (
            self.store.get(row['domain_i']),
            self.store.get(row['domain_j']),
            torch.from_numpy(
                np.array(self.score_method(row['score']), dtype=d_type)
            )
        )


def _exclude_domains(exclude_domains_file=None):
    if exclude_domains_file is None:
        return None
    with open(exclude_domains_file, 'r') as f:
        return [line.strip() for line in f]


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    from dataset.utils.custom_weighted_random_sampler import CustomWeightedRandomSampler
    from dataset.utils.tools import collate_fn

    parser = argparse.ArgumentParser()
    parser.add_argument('--tm_score_file', type=str, required=True)
    parser.add_argument('--store_dir', type=str, required=True)
    parser.add_argument('--local_scratch', type=str, default=None)
    args = parser.parse_args()

    dataset = PackedTmScoreFromEmbeddingsDataset(
        args.tm_score_file,
        args.store_dir,
        score_method=fraction_score,
        weighting_method=tm_score_weights(5, 0.25),
        local_scratch=args.local_scratch,
    )
    weights = dataset.weights()
    sampler = CustomWeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=16, collate_fn=collate_fn)
    for (x, x_mask), (y, y_mask), z in dataloader:
        print(z)
