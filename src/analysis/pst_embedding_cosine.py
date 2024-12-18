import logging

import hydra
from hydra._internal.instantiate._instantiate2 import instantiate
from hydra.core.config_store import ConfigStore
import lightning as L

from torch.utils.data import DataLoader

from config.utils import get_config_path
from dataset.embedding_pairs_dataset import EmbeddingPairsDataset, collate
from lightning_module.analysis.lightning_pst_embedding_cosine import LitStructurePstEmbeddingCosine
from config.schema_config import InferenceConfig

cs = ConfigStore.instance()
cs.store(name="analysis_default", node=InferenceConfig)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../config", config_name="inference_config")
def main(cfg: InferenceConfig):

    logger.info(f"Using config file: {get_config_path()}")

    inference_set = EmbeddingPairsDataset(
        embedding_list=cfg.inference_set.embedding_source,
        embedding_path=cfg.inference_set.embedding_path
    )

    inference_dataloader = DataLoader(
        dataset=inference_set,
        batch_size=cfg.inference_set.batch_size,
        num_workers=cfg.inference_set.workers,
        collate_fn=collate
    )

    nn_model = instantiate(
        cfg.embedding_network
    )

    model = LitStructurePstEmbeddingCosine.load_from_checkpoint(
        cfg.checkpoint,
        nn_model=nn_model,
        params=cfg
    )

    trainer = L.Trainer(
        callbacks=[],
        devices=cfg.computing_resources.devices
    )
    trainer.predict(
        model,
        inference_dataloader
    )


if __name__ == '__main__':
    main()
