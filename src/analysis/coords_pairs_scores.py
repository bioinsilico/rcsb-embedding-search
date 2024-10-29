import logging

import hydra
from hydra.core.config_store import ConfigStore
import lightning as L
from hydra.utils import instantiate
from torch_geometric.loader import DataLoader

from config.utils import get_config_path
from dataset.tm_score_from_coord_dataset import TmScoreFromCoordDataset
from config.schema_config import TrainingConfig
from dataset.utils.embedding_builder import graph_ch_coords_builder
from lightning_module.analysis.lightning_pst_embedding_cosine import LitStructurePstEmbeddingScore

cs = ConfigStore.instance()
cs.store(name="analysis_default", node=TrainingConfig)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../config", config_name="identity_analysis_config")
def main(cfg: TrainingConfig):

    logger.info(f"Using config file: {get_config_path()}")

    inference_set = TmScoreFromCoordDataset(
        tm_score_file=cfg.training_set.tm_score_file,
        coords_path=cfg.training_set.data_path,
        num_workers=cfg.training_set.workers,
        ext="cif",
        graph_builder=graph_ch_coords_builder
    )

    inference_dataloader = DataLoader(
        dataset=inference_set,
        batch_size=cfg.training_set.batch_size,
        num_workers=cfg.training_set.workers
    )

    nn_model = instantiate(
        cfg.embedding_network
    )

    model = LitStructurePstEmbeddingScore.load_from_checkpoint(
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
