import argparse
import logging
import os

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from dataset.utils.tools import collate_seq_embeddings

logger = logging.getLogger(__name__)


class EmbeddingsDataset(Dataset):
    def __init__(
            self,
            embedding_list,
            embedding_path
    ):
        self.embedding = pd.DataFrame()
        self.__exec(embedding_list, embedding_path)

    def __exec(self, embedding_list, embedding_path):
        self.load_embedding(embedding_list, embedding_path)

    def load_embedding(self, embedding_list, embedding_path):
        logger.info(f"Loading embeddings from path {embedding_path}")
        if type(embedding_list) is not list:
            logger.info(f"Embedding list file {embedding_list}")
        self.embedding = pd.DataFrame(
            data=[
                (dom_id, os.path.join(embedding_path, f"{dom_id}.pt"))
                for dom_id in (
                    embedding_list if type(embedding_list) is list
                    else list(set([row.strip() for row in open(embedding_list)]))
                )
            ],
            columns=['dom_id', 'embedding'],
        )
        logger.info(f"Total embeddings: {len(self.embedding)}")

    def __len__(self):
        return len(self.embedding)

    def __getitem__(self, idx):
        try:
            return torch.load(
                self.embedding.loc[idx, 'embedding'],
                map_location=torch.device('cpu')
            ), self.embedding.loc[idx, 'dom_id']
        except Exception as e:
            logger.error(f"Error loading embedding from {self.embedding.loc[idx, 'embedding']}, {self.embedding.loc[idx, 'dom_id']}: {e}")
            raise e


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_list', type=str, required=True)
    parser.add_argument('--embedding_path', type=str, required=True)
    args = parser.parse_args()

    dataset = EmbeddingsDataset(
        args.embedding_list,
        args.embedding_path
    )

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        collate_fn=lambda emb: (
            collate_seq_embeddings([x for x, z in emb]),
            tuple([z for x, z in emb])
        )
    )

    for (x, x_mask), z in dataloader:
        print(x.shape, x_mask.shape, len(z))
