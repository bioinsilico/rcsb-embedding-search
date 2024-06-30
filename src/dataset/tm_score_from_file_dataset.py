
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

from utils.custom_weighted_random_sampler import CustomWeightedRandomSampler
from utils.tm_score_weight import tm_score_weights, fraction_score, binary_score, binary_weights
from utils.tools import collate_fn

d_type = np.float32


class TmScoreFileDataset(Dataset):

    BINARY_THR = 0.7

    def __init__(self, tm_score_file, embedding_path, score_method=None, weighting_method=None):
        self.domains = set()
        self.embedding = {}
        self.class_pairs = list()
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


if __name__ == '__main__':
    dataset = TmScoreFileDataset(
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
