import torch
from torch.utils.data import Dataset
import numpy as np
import os

d_type = np.float32


class TmScoreDataset(Dataset):
    def __init__(self, tm_score_file, embedding_path):
        self.domains = set()
        self.embedding = {}
        self.class_pairs = list()
        self.max_length = 0
        self.__exec(tm_score_file, embedding_path)

    def __exec(self, tm_score_file, embedding_path):
        self.load_class_pairs(tm_score_file)
        self.load_embedding(embedding_path)

    def load_class_pairs(self, tm_score_file):
        for row in open(tm_score_file):
            r = row.strip().split(",")
            dom_i = r[0]
            dom_j = r[1]
            tm_score = r[2]
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
            self.embedding[dom_id] = torch.load(os.path.join(embedding_path, f"{dom_id}.pt"))

    def weights(self):
        p = 0.5 / sum([dp[0] for dp in self.class_pairs])
        n = 0.5 / sum([1 - dp[0] for dp in self.class_pairs])
        return torch.tensor([p if dp[0] == 1. else n for dp in self.class_pairs])

    def __len__(self):
        return len(self.class_pairs)

    def __getitem__(self, idx):
        embedding_i = self.embedding[self.class_pairs[idx][1]]
        embedding_j = self.embedding[self.class_pairs[idx][2]]
        label = torch.from_numpy(np.array(self.class_pairs[idx][0], dtype=d_type))

        return embedding_i, embedding_j, label


if __name__ == '__main__':
    TmScoreDataset('/Users/joan/data/cath_23M/cath_23M.csv',
                   '/Users/joan/data/structure-embedding/pst_t30_so/cath_S40/embedding')
