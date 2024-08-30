import hydra
from hydra.core.config_store import ConfigStore
import lightning as L
from hydra.utils import instantiate

from torch.utils.data import DataLoader

from dataset.residue_embeddings_dataset import ResidueEmbeddingsDataset
from dataset.utils.tools import collate_seq_embeddings
from lightning_module.inference.lightning_pst_embedding_pooling import LitStructurePstEmbeddingPooling
from networks.transformer_pst import TransformerPstEmbeddingCosine
from config_schema.config import InferenceConfig

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
        batch_size=cfg.inference_set.batch_size,
        num_workers=cfg.computing_resources.workers,
        collate_fn=lambda emb: (
            collate_seq_embeddings([x for x, z in emb]),
            tuple([z for x, z in emb])
        )
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
