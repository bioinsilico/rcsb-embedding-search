import logging

import lightning as L
import torch
from torch.utils.data import DataLoader

from dataset.embeddings_dataset import EmbeddingsDataset
from dataset.utils.tools import collate_seq_embeddings, get_unique_pairs
from scripts.populate_embeddings import populate_zero_embeddings

logger = logging.getLogger(__name__)


class ReloadValidationDataLoaderCallback(L.Callback):

    def __init__(
            self,
            embedding_provider
    ):
        self.device = None
        self.path = None
        self.source = None
        self.dim = None
        self.rank = None
        self.embedding_provider = embedding_provider

    def on_fit_start(self, trainer, pl_module):
        self.rank = pl_module.global_rank
        self.device = pl_module.device
        self.path = pl_module.cfg.validation_set.embedding_tmp_path
        self.dim = pl_module.cfg.network_parameters.hidden_layer
        self.source = pl_module.cfg.validation_set.tm_score_file

        logger.info(f"Initializing zero embeddings from {self.source}, device: {self.device}, rank: {self.rank}")
        self.embedding_provider.build(
            self.path,
            str(self.rank)
        )
        populate_zero_embeddings(
            tm_score_file=self.source,
            dim=self.dim,
            embedding_provider=self.embedding_provider
        )

    def on_validation_epoch_start(self, trainer, pl_module):
        logger.info(f"#new line#")
        dataset = EmbeddingsDataset(
            get_unique_pairs(self.source),
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
        with torch.no_grad():
            logger.info(f"Generating validation embeddings from {self.source}, device: {self.device}, rank: {self.rank}")
            pl_module.model.eval()
            z, x_pred = compute_embeddings(
                    pl_module.model.embedding_pooling,
                    self.device,
                    dataloader
            )
            logger.info(f"Updating validation embeddings from {self.source}, device: {self.device}, rank: {self.rank}")
            self.embedding_provider.set_many([(z[idx], embedding.tolist()) for idx, embedding in enumerate(x_pred)])
        pl_module.model.train()
        logger.info(f"New validation embeddings available from {self.source}, device: {self.device}, rank: {self.rank}")


def compute_embeddings(embedding_pooling, device, dataloader):
    predictions = [(z, embedding_pooling(x.to(device), x_mask.to(device))) for (x, x_mask), z in dataloader]
    return (
        [s for strings, _ in predictions for s in strings],
        torch.cat([embedding for _, embedding in predictions], dim=0)
    )

