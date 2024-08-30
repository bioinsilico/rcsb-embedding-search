import hydra
from hydra.core.config_store import ConfigStore
import lightning as L

from torch.utils.data import DataLoader

from dataset.residue_embedding_pairs_dataset import ResidueEmbeddingPairsDataset, collate
from lightning_module.analysis.lightning_pst_embedding_cosine import LitStructurePstEmbeddingCosine
from schema.config import InferenceConfig

cs = ConfigStore.instance()
cs.store(name="base_config", node=InferenceConfig)


@hydra.main(version_base=None, config_path="../../config", config_name="inference_default")
def main(cfg: InferenceConfig):

    inference_set = ResidueEmbeddingPairsDataset(
        embedding_list=cfg.inference_set.embedding_source,
        embedding_path=cfg.inference_set.embedding_path
    )

    inference_dataloader = DataLoader(
        dataset=inference_set,
        batch_size=cfg.inference_set.batch_size,
        num_workers=cfg.computing_resources.workers,
        collate_fn=collate
    )

    model = LitStructurePstEmbeddingCosine()
    trainer = L.Trainer(
        callbacks=[],
        accelerator='cpu',
        devices=cfg.computing_resources.devices
    )
    trainer.predict(
        model,
        inference_dataloader
    )


if __name__ == '__main__':
    main()
