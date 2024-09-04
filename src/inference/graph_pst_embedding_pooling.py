import hydra
from hydra.core.config_store import ConfigStore
import lightning as L
from hydra.utils import instantiate
from torch_geometric.loader import DataLoader

from dataset.graph_from_list_dataset import GraphFromListDataset
from lightning_module.inference.lightning_pst_embedding_pooling import LitStructurePstEmbeddingPoolingFromGraph
from config_schema.config import InferenceConfig

cs = ConfigStore.instance()
cs.store(name="inference_default", node=InferenceConfig)


@hydra.main(version_base=None, config_path="../../config", config_name="inference_config")
def main(cfg: InferenceConfig):

    inference_set = GraphFromListDataset(
        graph_list=cfg.inference_set.embedding_source,
        graph_path=cfg.inference_set.embedding_path
    )

    inference_dataloader = DataLoader(
        dataset=inference_set,
        batch_size=cfg.inference_set.batch_size,
        num_workers=cfg.computing_resources.workers
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
        cfg.inference_writer
    )
    trainer = L.Trainer(callbacks=[inference_writer])
    trainer.predict(
        model,
        inference_dataloader
    )


if __name__ == '__main__':
    main()
