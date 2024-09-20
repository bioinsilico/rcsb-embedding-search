import hydra
from hydra.core.config_store import ConfigStore
import lightning as L
from hydra.utils import instantiate
from torch_geometric.data import DataLoader

from dataset.identity_coords_dataset import IdentityCoordsDataset
from dataset.utils.coords_augmenter import SelfAugmenterRandomFraction
from config_schema.config import TrainingConfig
from lightning_module.analysis.lightning_pst_embedding_cosine import LitStructurePstEmbeddingScore

cs = ConfigStore.instance()
cs.store(name="training_default", node=TrainingConfig)


@hydra.main(version_base=None, config_path="../../config", config_name="identity_analysis_config")
def main(cfg: TrainingConfig):

    inference_set = IdentityCoordsDataset(
        coords_list=cfg.training_set.tm_score_file,
        coords_path=cfg.training_set.data_path,
        num_workers=cfg.computing_resources.workers,
        ext="pdb",
        coords_augmenter=SelfAugmenterRandomFraction(0.1, 10)
    )

    inference_dataloader = DataLoader(
        dataset=inference_set,
        batch_size=cfg.training_set.batch_size,
        num_workers=cfg.computing_resources.workers
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
