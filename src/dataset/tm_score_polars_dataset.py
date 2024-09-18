import argparse

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import polars as pl
import os

from dataset.utils.custom_weighted_random_sampler import CustomWeightedRandomSampler
from dataset.utils.tm_score_weight import tm_score_weights, fraction_score, binary_score, binary_weights
from dataset.utils.tools import collate_fn, load_class_pairs

d_type = np.float32


class TmScorePolarsDataset(Dataset):
    BINARY_THR = 0.7

    def __init__(
            self,
            tm_score_file,
            embedding_path,
            score_method=None,
            weighting_method=None
    ):
        self.embedding = pl.DataFrame()
        self.class_pairs = pl.DataFrame()
        self.score_method = binary_score(self.BINARY_THR) if not score_method else score_method
        self.weighting_method = binary_weights(self.BINARY_THR) if not weighting_method else weighting_method
        self.__exec(tm_score_file, embedding_path)

    def __exec(self, tm_score_file, embedding_path):
        self.load_class_pairs(tm_score_file)
        self.load_embedding(embedding_path)

    def load_class_pairs(self, tm_score_file):
        print(f"Loading pairs from file {tm_score_file}")
        self.class_pairs = load_class_pairs(tm_score_file)
        print(f"Total pairs: {len(self.class_pairs)}")

    def load_embedding(self, embedding_path):
        print(f"Loading embeddings from path {embedding_path}")
        self.embedding = pl.DataFrame(
            data=[
                (dom_id, os.path.join(embedding_path, f"{dom_id}.pt")) for dom_id in pl.concat([
                    self.class_pairs["domain_i"], self.class_pairs["domain_j"]
                ]).unique()
            ],
            orient="row",
            schema=['domain', 'embedding'],
        )
        print(f"Total embeddings: {len(self.embedding)}")

    def weights(self):
        return self.weighting_method(self.class_pairs.select('score'))

    def __len__(self):
        return len(self.class_pairs)

    def __getitem__(self, idx):
        return (
            torch.load(self.embedding.row(
                by_predicate=(pl.col("domain") == self.class_pairs.row(idx, named=True)['domain_i']),
                named=True
            )['embedding']),
            torch.load(self.embedding.row(
                by_predicate=(pl.col("domain") == self.class_pairs.row(idx, named=True)['domain_j']),
                named=True
            )['embedding']),
            torch.from_numpy(
                np.array(self.score_method(self.class_pairs.row(idx, named=True)['score']), dtype=d_type)
            )
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tm_score_file', type=str, required=True)
    parser.add_argument('--embedding_path', type=str, required=True)
    args = parser.parse_args()

    dataset = TmScorePolarsDataset(
        args.tm_score_file,
        args.embedding_path,
        score_method=fraction_score,
        weighting_method=tm_score_weights(5)
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
