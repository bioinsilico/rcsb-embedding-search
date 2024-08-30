import argparse
import os

import torch
from torch.utils.data import Dataset, DataLoader
import polars as pl

from dataset.utils.tools import pair_at_index


class ResidueEmbeddingPairsDataset(Dataset):
    def __init__(
            self,
            embedding_list,
            embedding_path
    ):
        self.n_pairs = None
        self.__pair_at_index = None
        self.embedding = pl.DataFrame()
        self.__exec(embedding_list, embedding_path)

    def __exec(self, embedding_list, embedding_path):
        self.load_embedding(embedding_list, embedding_path)

    def load_embedding(self, embedding_list, embedding_path):
        self.embedding = pl.DataFrame(
            data=[
                (
                    row[0],
                    row[1],
                    (lambda v: v/torch.norm(v, p=2))(torch.load(os.path.join(embedding_path, f"{row[0]}.pt")))
                )
                for row in [row.strip().split("\t") for row in open(embedding_list)]
            ],
            orient="row",
            schema=['dom_id', 'class_id', 'embedding'],
        )
        n = len(self.embedding)
        self.n_pairs = n * (n-1) // 2
        self.__pair_at_index = pair_at_index(n)
        print(f"Total embedding: {n}")

    def __len__(self):
        return self.n_pairs

    def __getitem__(self, idx):
        idx_i, idx_j = self.__pair_at_index(idx)
        emb_i = self.embedding.row(idx_i, named=True)
        emb_j = self.embedding.row(idx_j, named=True)
        return (
            (emb_i['embedding'], emb_i['dom_id'], emb_i['class_id']),
            (emb_j['embedding'], emb_j['dom_id'], emb_j['class_id'])
        )


def collate(emb):
    return (
        (
            torch.stack([emb_i for (emb_i, d_i, c_i), _ in emb], dim=0),
            tuple([(d_i, c_i) for (emb_i, d_i, c_i), _ in emb])
        ),
        (
            torch.stack([emb_j for _, (emb_j, d_j, c_j) in emb], dim=0),
            tuple([(d_j, c_j) for _, (emb_j, d_j, c_j) in emb])
        )
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_list', type=str, required=True)
    parser.add_argument('--embedding_path', type=str, required=True)
    args = parser.parse_args()

    dataset = ResidueEmbeddingPairsDataset(
        args.embedding_list,
        args.embedding_path
    )

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        collate_fn=collate
    )

    for (x, x_id), (y, y_id) in dataloader:
        print(x, x_id, y_id)
