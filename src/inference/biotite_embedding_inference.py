import logging

import hydra
from hydra.core.config_store import ConfigStore
import lightning as L
from hydra.utils import instantiate

from torch.utils.data import DataLoader

from config.utils import get_config_path
from dataset.biotite_structure_from_list import BiotiteStructureFromList
from lightning_module.inference.biotite_embeding_pooling import LitStructureBiotiteEmbeddingPooling
from config.schema_config import InferenceConfig

cs = ConfigStore.instance()
cs.store(name="inference_default", node=InferenceConfig)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../config", config_name="inference_config")
def main(cfg: InferenceConfig):

    logger.info(f"Using config file: {get_config_path()}")

    def __build_url(af_id):
        return f"https://alphafold.ebi.ac.uk/files/{af_id}-model_v4.cif"

    inference_set = BiotiteStructureFromList(
        rcsb_file_list=cfg.inference_set.embedding_source,
        build_url=__build_url
    )

    inference_dataloader = DataLoader(
        dataset=inference_set,
        batch_size=cfg.inference_set.batch_size,
        num_workers=cfg.inference_set.workers,
        collate_fn=lambda emb: emb
    )

    nn_model = instantiate(
        cfg.embedding_network
    )

    model = LitStructureBiotiteEmbeddingPooling.load_from_checkpoint(
        cfg.checkpoint,
        nn_model=nn_model,
        params=cfg
    )
    inference_writer = instantiate(
        cfg.inference_writer,
        df_id=cfg.metadata.task_id
    )
    trainer = L.Trainer(
        callbacks=[inference_writer],
        num_nodes=cfg.computing_resources.nodes,
        devices=cfg.computing_resources.devices,
        strategy=cfg.computing_resources.strategy
    )
    trainer.predict(
        model,
        inference_dataloader
    )


if __name__ == '__main__':
    main()
