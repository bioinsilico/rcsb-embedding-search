import torch
from polars import Schema, String, Float32
from torch.utils.data import Dataset, DataLoader
import numpy as np
import polars as pl
import os

from dataset.utils.custom_weighted_random_sampler import CustomWeightedRandomSampler
from dataset.utils.tm_score_weight import tm_score_weights, fraction_score, binary_score, binary_weights
from dataset.utils.tools import collate_dual_fn

d_type = np.float32


class TmDualScorePolarsDataset(Dataset):
    BINARY_THR = 0.7

    def __init__(self, tm_score_file, embedding_path, score_method=None, weighting_method=None):
        self.embedding = pl.DataFrame()
        self.class_pairs = pl.DataFrame()
        self.score_method = binary_score(self.BINARY_THR) if not score_method else score_method
        self.weighting_method = binary_weights(self.BINARY_THR) if not weighting_method else weighting_method
        self.__exec(tm_score_file, embedding_path)

    def __exec(self, tm_score_file, embedding_path):
        self.load_class_pairs(tm_score_file)
        self.load_embedding(tm_score_file, embedding_path)

    def load_class_pairs(self, tm_score_file):
        self.class_pairs = pl.DataFrame(
            data=[(lambda row: row.strip().split(","))(row) for row in open(tm_score_file)],
            orient="row",
            schema=Schema({'domain_i': String, 'domain_j': String, 'score-max': Float32, 'score-min': Float32})
        )
        print(f"Total pairs: {len(self.class_pairs)}")

    def load_embedding(self, tm_score_file, embedding_path):

        self.embedding = pl.DataFrame(
            data=[
                (dom_id, os.path.join(embedding_path, f"{dom_id}.pt")) for dom_id in list(set(
                    [(lambda row: row.strip().split(",")[0])(row) for row in open(tm_score_file)] +
                    [(lambda row: row.strip().split(",")[1])(row) for row in open(tm_score_file)]
                ))
            ],
            orient="row",
            schema=['domain', 'embedding'],
        )
        print(f"Total embedding: {len(self.embedding)}")

    def weights(self):
        return self.weighting_method([dp['score-max'] for dp in self.class_pairs.rows(named=True)])

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
            (
                torch.from_numpy(
                    np.array(self.score_method(self.class_pairs.row(idx, named=True)['score-max']), dtype=d_type)
                ),
                torch.from_numpy(
                    np.array(self.score_method(self.class_pairs.row(idx, named=True)['score-min']), dtype=d_type)
                )
            )
        )


if __name__ == '__main__':
    dataset = TmDualScorePolarsDataset(
        '/Users/joan/data/scop-zenodo/TMfast.dual.csv',
        '/Users/joan/data/structure-embedding/pst_t30_so/combined-cath_S40-scop-zenodo/embedding',
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