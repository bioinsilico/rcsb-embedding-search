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

        self.load_embedding(embedding_path)
        self.load_class_pairs(tm_score_file)

    def load_class_pairs(self, tm_score_file):
        for row in open(tm_score_file):
            r = row.strip().split(",")
            dom_i = r[0]
            dom_j = r[1]
            tm_score = r[2]
            if dom_i in self.domains and dom_j in self.domains:
                self.class_pairs.append([
                    tm_score,
                    dom_i,
                    dom_j
                ])
        print(f"Total pairs: {len(self.class_pairs)}")

    def load_embedding(self, embedding_path):
        for file in os.listdir(embedding_path):
            dom_id = ".".join(file.split(".")[0:-2])
            self.domains.add(dom_id)
            self.embedding[dom_id] = torch.load(f"{embedding_path}/{file}")
            if self.embedding[dom_id].shape[0] > self.max_length:
                self.max_length = self.embedding[dom_id].shape[0]

    def __len__(self):
        return len(self.class_pairs)

    def __getitem__(self, idx):
        embedding_i = self.embedding[self.class_pairs[idx][1]]
        embedding_j = self.embedding[self.class_pairs[idx][2]]
        label = torch.from_numpy(np.array(self.class_pairs[idx][0], dtype=d_type))

        return embedding_i, embedding_j, label


if __name__ == '__main__':
    TmScoreDataset('/Users/joan/data/cath_23M/cath_23M.csv',
                   '/Users/joan/data/structure-embedding/pst_t30_so/cath_23M/embedding')
