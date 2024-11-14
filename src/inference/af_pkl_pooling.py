import logging

import hydra
from hydra.core.config_store import ConfigStore
import lightning as L
from hydra.utils import instantiate
from torch_geometric.loader import DataLoader

from config.utils import get_config_path
from dataset.graph_from_pickle_dataset import GraphFromPklDataset
from lightning_module.inference.lightning_pst_embedding_pooling import LitStructurePstEmbeddingPoolingFromGraph
from config.schema_config import InferenceConfig

cs = ConfigStore.instance()
cs.store(name="inference_default", node=InferenceConfig)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../config", config_name="inference_config")
def main(cfg: InferenceConfig):

    logger.info(f"Using config file: {get_config_path()}")

    inference_set = GraphFromPklDataset(
        pkl_file=f"{cfg.metadata.af_root}/{cfg.metadata.af_folders[cfg.metadata.af_index]}.pkl.gz",
    )

    inference_dataloader = DataLoader(
        dataset=inference_set,
        batch_size=cfg.inference_set.batch_size,
        num_workers=cfg.inference_set.workers
    )

    nn_model = instantiate(
        cfg.embedding_network
    )

    model = LitStructurePstEmbeddingPoolingFromGraph.load_from_checkpoint(
        cfg.checkpoint,
        nn_model=nn_model,
        params=cfg
    )
    inference_writer = instantiate(
        cfg.inference_writer,
        df_id=cfg.metadata.af_folders[cfg.metadata.af_index]
    )
    trainer = L.Trainer(callbacks=[inference_writer])
    trainer.predict(
        model,
        inference_dataloader
    )


if __name__ == '__main__':
    main()
