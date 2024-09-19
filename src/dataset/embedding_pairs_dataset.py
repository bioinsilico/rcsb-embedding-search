import argparse
import os

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class EmbeddingPairsDataset(Dataset):
    def __init__(
            self,
            embedding_list,
            embedding_path
    ):
        self.embedding = None
        self.pairs = None
        self.__exec(embedding_list, embedding_path)

    def __exec(self, embedding_list, embedding_path):
        self.load_embedding(embedding_list, embedding_path)
        self.load_pairs()

    def load_embedding(self, embedding_list, embedding_path):
        self.embedding = pd.DataFrame(
            data=[
                (
                    row[0],
                    row[1],
                    (lambda v: v/torch.norm(v, p=2))(torch.load(os.path.join(embedding_path, f"{row[0]}.pt")))
                )
                for row in [row.strip().split("\t") for row in open(embedding_list)]
            ],
            columns=['dom_id', 'class_id', 'embedding'],
        )
        print(f"Total embeddings: {len(self.embedding)}")

    def load_pairs(self):
        n_embedding = len(self.embedding)
        self.pairs = pd.DataFrame(
            data=[(i, j) for i in range(0, n_embedding) for j in range(i+1, n_embedding)],
            columns=['i', 'j']
        )
        print(f"Total pairs: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        idx_i = self.pairs.loc[idx, 'i']
        idx_j = self.pairs.loc[idx, 'j']
        emb_i = self.embedding.loc[idx_i]
        emb_j = self.embedding.loc[idx_j]
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


def display_progress(current, total):
    percent = (current / total) * 100
    print(f"\rProgress: {percent:.2f}% ({current}/{total} files)", end='')
    if current == total:
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_list', type=str, required=True)
    parser.add_argument('--embedding_path', type=str, required=True)
    args = parser.parse_args()

    dataset = EmbeddingPairsDataset(
        args.embedding_list,
        args.embedding_path
    )

    dataloader = DataLoader(
        dataset,
        batch_size=256,
        collate_fn=collate
    )

    n = len(dataset)
    for idx, ((x, x_id), (y, y_id)) in enumerate(dataloader):
        if idx % 10000 == 0:
            print(display_progress(idx, n))

