import argparse
import logging

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

from dataset.utils.custom_weighted_random_sampler import CustomWeightedRandomSampler
from dataset.utils.embedding_provider import SqliteEmbeddingProvider
from dataset.utils.tm_score_weight import tm_score_weights, fraction_score, binary_score, binary_weights
from dataset.utils.tools import collate_fn, load_class_pairs

d_type = np.float32
logger = logging.getLogger(__name__)


class TmScoreFromEmbeddingsProviderDataset(Dataset):
    BINARY_THR = 0.7

    def __init__(
            self,
            tm_score_file,
            embedding_provider,
            score_method=None,
            weighting_method=None
    ):
        self.embedding = pd.DataFrame()
        self.class_pairs = pd.DataFrame()
        self.embedding_provider = embedding_provider
        self.score_method = binary_score(self.BINARY_THR) if not score_method else score_method
        self.weighting_method = binary_weights(self.BINARY_THR) if not weighting_method else weighting_method
        self.__exec(tm_score_file)

    def __exec(self, tm_score_file):
        self.load_class_pairs(tm_score_file)

    def load_class_pairs(self, tm_score_file):
        logger.info(f"Loading pairs from file {tm_score_file}")
        self.class_pairs = load_class_pairs(tm_score_file)
        logger.info(f"Total pairs: {len(self.class_pairs)}")

    def weights(self):
        return self.weighting_method(self.class_pairs['score'].to_numpy())

    def __len__(self):
        return len(self.class_pairs)

    def __getitem__(self, idx):
        domain_i = self.class_pairs.loc[idx, 'domain_i']
        embedding_i = torch.from_numpy(np.array(self.embedding_provider.get(domain_i)))

        domain_j = self.class_pairs.loc[idx, 'domain_j']
        embedding_j = torch.from_numpy(np.array(self.embedding_provider.get(domain_j)))

        return (
            embedding_i,
            embedding_j,
            torch.from_numpy(
                np.array(self.score_method(self.class_pairs.loc[idx, 'score']), dtype=d_type)
            )
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tm_score_file', type=str, required=True)
    parser.add_argument('--embedding_db', type=str, required=True)
    args = parser.parse_args()

    dataset = TmScoreFromEmbeddingsProviderDataset(
        args.tm_score_file,
        SqliteEmbeddingProvider(),
        score_method=fraction_score,
        weighting_method=tm_score_weights(5, 0.25)
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
        batch_size=16,
        collate_fn=collate_fn
    )
    for (x, x_mask), (y, y_mask), z in dataloader:
        print(z)
