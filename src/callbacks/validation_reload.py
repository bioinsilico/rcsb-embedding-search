import logging

import pandas as pd
import lightning as L
from torch.utils.data import DataLoader

from dataset.embeddings_dataset import EmbeddingsDataset
from dataset.utils.tools import collate_seq_embeddings, load_class_pairs
from scripts.populate_embeddings import populate_zero_embeddings

logger = logging.getLogger(__name__)


class ReloadValidationDataLoaderCallback(L.Callback):

    def __init__(
            self,
            embedding_provider
    ):
        self.embedding_provider = embedding_provider

    def on_fit_start(self, trainer, pl_module):
        self.embedding_provider.build(
            pl_module.cfg.validation_set.embedding_tmp_path,
            str(pl_module.local_rank)
        )
        populate_zero_embeddings(
            tm_score_file=pl_module.cfg.validation_set.tm_score_file,
            dim=pl_module.cfg.network_parameters.hidden_layer,
            embedding_provider=self.embedding_provider
        )

    def on_validation_epoch_start(self, trainer, pl_module):
        logger.info(f"Generating validation embeddings from {pl_module.cfg.validation_set.tm_score_file}, device: {pl_module.device}, rank: {pl_module.local_rank}")
        class_pairs = load_class_pairs(pl_module.cfg.validation_set.tm_score_file)
        dataset = EmbeddingsDataset(
            list(pd.concat([
                class_pairs["domain_i"], class_pairs["domain_j"]
            ]).unique()),
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
        for dom_id, embedding in compute_embeddings(
                pl_module.model.embedding_pooling,
                pl_module.device,
                dataloader
        ):
            self.embedding_provider.update(dom_id, embedding.tolist())
        logger.info(f"New validation embeddings available")


def compute_embeddings(embedding_pooling, device, dataloader):
    for (x, x_mask), z in dataloader:
        x_pred = embedding_pooling(
            x.to(device),
            x_mask.to(device)
        )
        for idx, embedding in enumerate(x_pred):
            yield z[idx], embedding

