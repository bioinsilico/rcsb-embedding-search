import signal

import hydra
import lightning as L
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from lightning import seed_everything
from lightning.pytorch.plugins.environments import SLURMEnvironment
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from torch_geometric.loader import DataLoader as GraphDataLoader

from dataset.pst.pst import load_pst_model
from dataset.tm_score_from_coord_dataset import TmScoreFromCoordDataset
from dataset.tm_score_polars_dataset import TmScorePolarsDataset
from dataset.utils.custom_weighted_random_sampler import CustomWeightedRandomSampler
from dataset.utils.tm_score_weight import fraction_score, tm_score_weights
from dataset.utils.tools import collate_fn
from lightning_module.lightning_pst_graph import LitStructurePstGraph
from networks.transformer_pst import TransformerPstEmbeddingCosine
from schema.config import Config
from src.params.structure_embedding_params import StructureEmbeddingParams


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: Config):
    seed_everything(cfg.global_seed, workers=True)

    training_set = instantiate(
        cfg.training_set,
        tm_score_file=cfg.training_sources.train_class_file,
        coords_path=cfg.training_sources.train_embedding_path,
        score_method=fraction_score,
        weighting_method=tm_score_weights(5),
        num_workers=cfg.computing_resources.workers
    )

    exit(0)
    weights = training_set.weights()
    sampler = CustomWeightedRandomSampler(
        weights=weights,
        num_samples=cfg.training_parameters.epoch_size if cfg.training_parameters.epoch_size > 0 else len(weights),
        replacement=True
    )
    train_dataloader = instantiate(
        cfg.training_loader,
        dataset=training_set,
        sampler=sampler,
        batch_size=cfg.training_parameters.batch_size,
        num_workers=cfg.computing_resources.workers,
        persistent_workers=True if cfg.computing_resources.workers > 0 else False
    )
    validation_set = instantiate(
        cfg.validation_set,

    )
    validation_dataloader = instantiate(
        cfg.validation_loader,
        dataset=validation_set,
        batch_size=cfg.training_parameters.testing_batch_size,
        num_workers=cfg.computing_resources.workers,
        persistent_workers=True if cfg.computing_resources.workers > 0 else False,
        collate_fn=collate_fn
    )
    exit(0)
    nn_model = instantiate(
        cfg.nn_model
    )
    model = LitStructurePstGraph(
        nn_model=nn_model,
        learning_rate=cfg.training_parameters.learning_rate,
        params=cfg,
    )

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        monitor=LitStructurePstGraph.PR_AUC_METRIC_NAME,
        mode='max',
        filename='{epoch}-{' + LitStructurePstGraph.PR_AUC_METRIC_NAME + ':.2f}'
    )

    lr_monitor = L.pytorch.callbacks.LearningRateMonitor(
        logging_interval='step'
    )

    trainer = L.Trainer(
        max_epochs=cfg.training_parameters.epochs,
        check_val_every_n_epoch=cfg.training_parameters.check_val_every_n_epoch,
        devices=cfg.computing_resources.devices,
        strategy=cfg.computing_resources.strategy,
        callbacks=[checkpoint_callback, lr_monitor],
        plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)],
        default_root_dir=cfg.training_sources.default_root_dir
    )
    trainer.fit(
        model,
        train_dataloader,
        validation_dataloader,
        ckpt_path=params.checkpoint
    )


if __name__ == '__main__':
    main()
