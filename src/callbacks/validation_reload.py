import os.path
from collections import deque

import lightning as L
import torch
from torch.utils.data import DataLoader

from dataset.embeddings_dataset import EmbeddingsDataset
from dataset.utils.tools import collate_seq_embeddings


class ReloadValidationDataLoaderCallback(L.Callback):

    def __init__(self):
        self.validation_loader = None

    def on_validation_start(self, trainer, pl_module):
        print("on_validation_start")

    def on_validation_epoch_start(self, trainer, pl_module):
        cfg = pl_module.cfg
        dataset = EmbeddingsDataset(
            list(set(
                get_column_from_file(cfg.validation_set.tm_score_file, 0) +
                get_column_from_file(cfg.validation_set.tm_score_file, 1)
            )),
            pl_module.cfg.validation_set.data_path
        )
        dataloader = DataLoader(
            dataset,
            batch_size=pl_module.cfg.validation_set.batch_size,
            collate_fn=lambda emb: (
                collate_seq_embeddings([x for x, z in emb]),
                tuple([z for x, z in emb])
            )
        )

        compute_embeddings(
            pl_module.model.embedding_pooling,
            pl_module.device,
            pl_module.cfg.validation_set.embedding_tmp_path,
            dataloader
        )


def compute_embeddings(embedding_pooling, device, out_path, dataloader):
    for (x, x_mask), z in dataloader:
        x_pred = embedding_pooling(
            x.to(device),
            x_mask.to(device)
        )
        indexes, embeddings = zip(*enumerate(x_pred))
        deque(map(
            save_embedding,
            [e / torch.norm(e, p=2) for e in embeddings],
            [os.path.join(out_path, f"{z[idx]}.pt") for idx in indexes]
        ))


def save_embedding(embedding, path):
    torch.save(
        embedding.to('cpu'),
        path
    )


def get_column_from_file(file_name, column_n):
    return [row.strip().split(",")[column_n] for row in open(file_name)]
