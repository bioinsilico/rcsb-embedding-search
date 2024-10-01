import logging

import lightning as L
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
        logger.info(f"Generating validation embeddings from {self.source}, device: {self.device}, rank: {self.rank}")
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
        for dom_id, embedding in compute_embeddings(
                pl_module.model.embedding_pooling,
                self.device,
                dataloader
        ):
            self.embedding_provider.update(dom_id, embedding.tolist())
        logger.info(f"New validation embeddings available from {self.source}, device: {self.device}, rank: {self.rank}")


def compute_embeddings(embedding_pooling, device, dataloader):
    for (x, x_mask), z in dataloader:
        x_pred = embedding_pooling(
            x.to(device),
            x_mask.to(device)
        )
        for idx, embedding in enumerate(x_pred):
            yield z[idx], embedding

