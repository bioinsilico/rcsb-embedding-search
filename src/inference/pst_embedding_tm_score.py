import hydra
from hydra.core.config_store import ConfigStore
import lightning as L

from torch.utils.data import DataLoader

from dataset.residue_embeddings_dataset import ResidueEmbeddingsDataset
from dataset.utils.tools import embedding_collate_fn
from lightning_module.lightning_pst_graph import LitStructurePstGraph
from networks.transformer_pst import TransformerPstEmbeddingCosine
from schema.config import InferenceConfig


cs = ConfigStore.instance()
cs.store(name="base_config", node=InferenceConfig)


@hydra.main(version_base=None, config_path="../../config", config_name="inference_default")
def main(cfg: InferenceConfig):

    inference_set = ResidueEmbeddingsDataset(
        embedding_list=cfg.inference_set.embedding_source,
        embedding_path=cfg.inference_set.embedding_path
    )

    inference_dataloader = DataLoader(
        dataset=inference_set,
        batch_size=16,
        collate_fn=embedding_collate_fn
    )

    nn_model = TransformerPstEmbeddingCosine(
        pst_model_path=cfg.embedding_network.pst_model_path,
        input_features=cfg.embedding_network.input_layer,
        nhead=cfg.embedding_network.nhead,
        num_layers=cfg.embedding_network.num_layers,
        dim_feedforward=cfg.embedding_network.dim_feedforward,
        hidden_layer=cfg.embedding_network.hidden_layer,
        res_block_layers=cfg.embedding_network.res_block_layers
    )

    model = LitStructurePstGraph(
        nn_model=nn_model,
        params=cfg
    ).load_from_checkpoint(cfg.checkpoint)
    trainer = L.Trainer()
    predictions = trainer.predict(
        model,
        inference_dataloader
    )
    for x, z in predictions:
        print(x.shape, len(z))


if __name__ == '__main__':
    main()
