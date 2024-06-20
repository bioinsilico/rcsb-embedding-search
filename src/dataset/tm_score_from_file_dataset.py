import math

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

from src.dataset.utils import collate_fn, CustomWeightedRandomSampler

d_type = np.float32


class TmScoreDatasetFromFile(Dataset):

    BINARY_THR = 0.7

    def __init__(self, tm_score_file, embedding_path, score_method=None, weighting_method=None):
        self.domains = set()
        self.embedding = {}
        self.class_pairs = list()
        self.max_length = 0
        self.score_method = binary_score(self.BINARY_THR) if not score_method else score_method
        self.weighting_method = binary_weights(self.BINARY_THR) if not weighting_method else weighting_method
        self.__exec(tm_score_file, embedding_path)

    def __exec(self, tm_score_file, embedding_path):
        self.load_class_pairs(tm_score_file)
        self.load_embedding(embedding_path)

    def load_class_pairs(self, tm_score_file):
        for row in open(tm_score_file):
            r = row.strip().split(",")
            dom_i = r[0]
            dom_j = r[1]
            tm_score = float(r[2])
            self.domains.add(dom_i)
            self.domains.add(dom_j)
            self.class_pairs.append([
                tm_score,
                dom_i,
                dom_j
            ])
        print(f"Total pairs: {len(self.class_pairs)}")

    def load_embedding(self, embedding_path):
        for dom_id in self.domains:
            self.embedding[dom_id] = os.path.join(embedding_path, f"{dom_id}.pt")

    def weights(self):
        return self.weighting_method([dp[0] for dp in self.class_pairs])

    def __len__(self):
        return len(self.class_pairs)

    def __getitem__(self, idx):
        embedding_i = torch.load(self.embedding[self.class_pairs[idx][1]])
        embedding_j = torch.load(self.embedding[self.class_pairs[idx][2]])
        score = self.score_method(self.class_pairs[idx][0])
        label = torch.from_numpy(np.array(score, dtype=d_type))

        return embedding_i, embedding_j, label


def binary_score(thr):
    def __binary_score(score):
        return 1. if score > thr else 0.
    return __binary_score


def binary_weights(thr):
    def __binary_weights(scores):
        p = 1 / sum([1 for s in scores if s >= thr])
        n = 1 / sum([1 for s in scores if s < thr])
        return torch.tensor([p if s >= thr else n for s in scores])
    return __binary_weights


def fraction_score(score):
    return round(10 * score) / 10


def tm_score_weights(n_intervals):
    def __tm_score_weights(scores):
        class_weights = TmScoreWeight(scores, n_intervals)
        return torch.tensor([class_weights.get_weight(s) for s in scores])
    return __tm_score_weights


class TmScoreWeight:

    def __init__(self, scores, n_intervals=5):
        self.weights = []
        self.n_intervals = n_intervals
        self.__compute(scores)

    def __compute(self, scores):
        h = 1 / self.n_intervals
        for idx in range(self.n_intervals):
            f = sum([1 for s in scores if idx * h <= s < (idx + 1) * h])
            self.weights.append(
                1 / f
            )

    def get_weight(self, score):
        idx = math.floor(score * self.n_intervals)
        if idx == self.n_intervals:
            idx = self.n_intervals - 1
        return self.weights[idx]


if __name__ == '__main__':
    dataset = TmScoreDatasetFromFile(
        '/Users/joan/data/cath_23M/cath_23M_ch_ids.csv',
        '/Users/joan/cs-data/structure-embedding/pst_t30_so/cath_S40/embedding',
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
