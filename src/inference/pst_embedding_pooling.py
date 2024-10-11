import logging

import hydra
from hydra.core.config_store import ConfigStore
import lightning as L
from hydra.utils import instantiate

from torch.utils.data import DataLoader

from config.utils import get_config_path
from dataset.embeddings_dataset import EmbeddingsDataset
from dataset.utils.tools import collate_seq_embeddings
from lightning_module.inference.lightning_pst_embedding_pooling import LitStructurePstEmbeddingPooling
from config.schema_config import InferenceConfig

cs = ConfigStore.instance()
cs.store(name="inference_default", node=InferenceConfig)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../config", config_name="inference_config")
def main(cfg: InferenceConfig):

    logger.info(f"Using config file: {get_config_path()}")

    inference_set = EmbeddingsDataset(
        embedding_list=cfg.inference_set.embedding_source,
        embedding_path=cfg.inference_set.embedding_path
    )

    inference_dataloader = DataLoader(
        dataset=inference_set,
        batch_size=cfg.inference_set.batch_size,
        num_workers=cfg.inference_set.workers,
        collate_fn=lambda emb: (
            collate_seq_embeddings([x for x, z in emb]),
            tuple([z for x, z in emb])
        )
    )

    nn_model = instantiate(
        cfg.embedding_network
    )

    model = LitStructurePstEmbeddingPooling.load_from_checkpoint(
        cfg.checkpoint,
        nn_model=nn_model,
        params=cfg
    )
    inference_writer = instantiate(
        cfg.inference_writer
    )
    trainer = L.Trainer(callbacks=[inference_writer])
    trainer.predict(
        model,
        inference_dataloader
    )


if __name__ == '__main__':
    main()
