import torch
from torch.utils.data import Dataset
import numpy as np

d_type = np.float32


class TripletEmbeddingDataset(Dataset):
    def __init__(self, class_file, embedding_path):
        self.domain_to_class = {}
        self.class_to_domains = {}
        self.embedding = {}
        self.class_triplets = list()
        self.__exec(class_file, embedding_path)

    def __exec(self, class_file, embedding_path):
        self.load_classes(class_file)
        self.load_class_triplets()
        self.load_embedding(embedding_path)

    def load_classes(self, class_file):
        for pdb_class in open(class_file, "r"):
            r = pdb_class.strip().split('\t')
            self.domain_to_class[r[0]] = r[1]
            if r[1] not in self.class_to_domains:
                self.class_to_domains[r[1]] = []
            self.class_to_domains[r[1]].append(r[0])

    def load_class_triplets(self):
        for dom_class, dom_list in self.class_to_domains.items():
            for d_a in dom_list:
                for d_p in dom_list:
                    for d_n, d_n_class in self.domain_to_class.items():
                        if dom_class != d_n_class:
                            self.class_triplets.append([d_a, d_p, d_n])

    def load_embedding(self, embedding_path):
        for dom_id in self.domain_to_class.keys():
            self.embedding[dom_id] = torch.load(f"{embedding_path}/{dom_id}.pt")

    def __len__(self):
        return len(self.class_triplets)

    def __getitem__(self, idx):
        embedding_a = self.embedding[self.class_triplets[idx][0]]
        embedding_p = self.embedding[self.class_triplets[idx][1]]
        embedding_n = self.embedding[self.class_triplets[idx][2]]

        return embedding_a, embedding_p, embedding_n
