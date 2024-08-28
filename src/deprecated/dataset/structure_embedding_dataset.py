import torch
from torch.utils.data import Dataset
import numpy as np

d_type = np.float32


def class_match_score(class_i, class_j):
    r_i = class_i.split(".")
    r_j = class_j.split(".")
    if r_i[0] != r_j[0]:
        return 0.
    if r_i[1] != r_j[1]:
        return 0.25
    if r_i[2] != r_j[2]:
        return 0.5
    if r_i[3] != r_j[3]:
        return 0.75
    return 1


def class_strict_match(class_i, class_j):
    return 1. if class_i == class_j else 0.


class StructureEmbeddingDataset(Dataset):
    def __init__(self, class_file, embedding_path, match_score=class_strict_match):
        self.classes = {}
        self.embedding = {}
        self.class_pairs = list()
        self.match_score = match_score
        self.__exec(class_file, embedding_path)

    def __exec(self, class_file, embedding_path):
        self.load_classes(class_file)
        self.load_class_pairs()
        self.load_embedding(embedding_path)

    def load_classes(self, class_file):
        for pdb_class in open(class_file, "r"):
            r = pdb_class.strip().split('\t')
            self.classes[r[0]] = r[1]

    def load_class_pairs(self):
        domains = list(self.classes.keys())
        for i in range(len(domains)):
            for j in range(i + 1, len(domains)):
                self.class_pairs.append([
                    self.match_score(self.classes[domains[i]], self.classes[domains[j]]),
                    domains[i],
                    domains[j]
                ])

    def load_embedding(self, embedding_path):
        for dom_id in self.classes.keys():
            self.embedding[dom_id] = torch.load(f"{embedding_path}/{dom_id}.pt")

    def weights(self):
        p = 0.5 / sum([dp[0] for dp in self.class_pairs])
        n = 0.5 / sum([1 - dp[0] for dp in self.class_pairs])
        return torch.tensor([p if dp[0] == 1. else n for dp in self.class_pairs])

    def get_classes(self, idx):
        return "%s / **%s** - %s / **%s**" % (
            self.class_pairs[idx][1],
            self.classes[self.class_pairs[idx][1]],
            self.class_pairs[idx][2],
            self.classes[self.class_pairs[idx][2]]
        )

    def __len__(self):
        return len(self.class_pairs)

    def __getitem__(self, idx):
        embedding_i = self.embedding[self.class_pairs[idx][1]]
        embedding_j = self.embedding[self.class_pairs[idx][2]]
        label = torch.from_numpy(np.array(self.class_pairs[idx][0], dtype=d_type))

        return embedding_i, embedding_j, label
