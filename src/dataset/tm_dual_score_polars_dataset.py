import argparse

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os

from dataset.utils.custom_weighted_random_sampler import CustomWeightedRandomSampler
from dataset.utils.tm_score_weight import tm_score_weights, fraction_score, binary_score, binary_weights
from dataset.utils.tools import collate_dual_fn

d_type = np.float32


class TmDualScorePolarsDataset(Dataset):
    BINARY_THR = 0.7

    def __init__(self, tm_score_file, embedding_path, score_method=None, weighting_method=None):
        self.embedding = pd.DataFrame()
        self.class_pairs = pd.DataFrame()
        self.score_method = binary_score(self.BINARY_THR) if not score_method else score_method
        self.weighting_method = binary_weights(self.BINARY_THR) if not weighting_method else weighting_method
        self.__exec(tm_score_file, embedding_path)

    def __exec(self, tm_score_file, embedding_path):
        self.load_class_pairs(tm_score_file)
        self.load_embedding(embedding_path)

    def load_class_pairs(self, tm_score_file):
        print(f"Loading pairs from file {tm_score_file}")
        dtype = {
            'domain_i': 'str',
            'domain_j': 'str',
            'score-max': 'float32',
            'score-min': 'float32'
        }
        self.class_pairs = pd.read_csv(
            tm_score_file,
            header=None,
            index_col=None,
            names=['domain_i', 'domain_j', 'score-max', 'score-min'],
            dtype=dtype
        )
        print(f"Total pairs: {len(self.class_pairs)}")

    def load_embedding(self, embedding_path):
        self.embedding = pd.DataFrame(
            data=[
                (dom_id, os.path.join(embedding_path, f"{dom_id}.pt")) for dom_id in pd.concat([
                    self.class_pairs["domain_i"], self.class_pairs["domain_j"]
                ]).unique()
            ],
            columns=['domain', 'embedding']
        )
        print(f"Total embedding: {len(self.embedding)}")

    def weights(self):
        return self.weighting_method([dp['score-max'] for dp in self.class_pairs.rows(named=True)])

    def __len__(self):
        return len(self.class_pairs)

    def __getitem__(self, idx):
        domain_i = self.class_pairs.loc[idx, 'domain_i']
        row_i = self.embedding.loc[self.embedding['domain'] == domain_i]
        embedding_i = row_i['embedding'].values[0]

        domain_j = self.class_pairs.loc[idx, 'domain_j']
        row_j = self.embedding.loc[self.embedding['domain'] == domain_j]
        embedding_j = row_j['embedding'].values[0]

        return (
            torch.load(embedding_i),
            torch.load(embedding_j),
            (
                torch.from_numpy(
                    np.array(self.score_method(self.class_pairs.loc[idx, 'score-max']), dtype=d_type)
                ),
                torch.from_numpy(
                    np.array(self.score_method(self.class_pairs.loc[idx, 'score-min']), dtype=d_type)
                )
            )
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tm_score_file', type=str, required=True)
    parser.add_argument('--embedding_path', type=str, required=True)
    args = parser.parse_args()

    dataset = TmDualScorePolarsDataset(
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
        collate_fn=collate_dual_fn
    )
    for (x, x_mask), (y, y_mask), (z_max, z_min) in dataloader:
        print("max", z_max)
        print("min", z_min)
        print("================")