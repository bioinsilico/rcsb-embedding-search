import torch
from torch.utils.data import Dataset
import numpy as np

d_type = np.float32


class StructureEmbeddingDataset(Dataset):
    def __init__(self, class_file, embedding_path):
        self.classes = {}
        self.embedding = {}
        self.class_pairs = list()
        self.max_length = 0

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
                    1. if self.classes[domains[i]] == self.classes[domains[j]] else 0.,
                    domains[i],
                    domains[j]
                ])

    def load_embedding(self, embedding_path):
        for dom_id in self.classes.keys():
            self.embedding[dom_id] = torch.load(f"{embedding_path}/{dom_id}.pt")
            if self.embedding[dom_id].shape[0] > self.max_length:
                self.max_length = self.embedding[dom_id].shape[0]

    def pad_embedding(self):
        for dom_id in self.classes.keys():
            target = torch.zeros(self.max_length, self.embedding[dom_id].shape[1])
            target[:self.embedding[dom_id].shape[0], :] = self.embedding[dom_id]
            self.embedding[dom_id] = target

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
