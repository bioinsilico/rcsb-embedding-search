import argparse
import logging

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os

from dataset.tm_score_from_embeddings_dataset import TmScoreFromEmbeddingsDataset
from dataset.utils.custom_weighted_random_sampler import CustomWeightedRandomSampler
from dataset.utils.tm_score_weight import tm_score_weights, fraction_score, binary_score, binary_weights, \
    DualTmScoreWeight
from dataset.utils.tools import collate_fn, load_dual_pairs

d_type = np.float32

logger = logging.getLogger(__name__)


class DualTmScoreFromEmbeddingsDataset(TmScoreFromEmbeddingsDataset):
    BINARY_THR = 0.7

    def __init__(self, tm_score_file, embedding_path, score_method=None, weighting_method=None):
        super().__init__(
            tm_score_file,
            embedding_path,
            score_method,
            weighting_method
        )

    def load_class_pairs(self, tm_score_file):
        logger.info(f"Loading pairs from file {tm_score_file}")
        self.class_pairs = load_dual_pairs(tm_score_file)
        logger.info(f"Total pairs: {len(self.class_pairs)}")

    def weights(self):
        np_scores = np.column_stack(
            (self.class_pairs['score'].to_numpy(), self.class_pairs['score_max'].to_numpy())
        )
        return self.weighting_method(np_scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tm_score_file', type=str, required=True)
    parser.add_argument('--embedding_path', type=str, required=True)
    args = parser.parse_args()

    dataset = DualTmScoreFromEmbeddingsDataset(
        args.tm_score_file,
        args.embedding_path,
        score_method=fraction_score,
        weighting_method=tm_score_weights(5, 0.25, score_weight=DualTmScoreWeight)
    )
    weights = dataset.weights()
    sampler = CustomWeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1,
        collate_fn=collate_fn
    )
    for (x, x_mask), (y, y_mask), z in dataloader:
        print(z)
