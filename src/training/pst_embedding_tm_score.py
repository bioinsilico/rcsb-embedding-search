import logging
import signal

import hydra
import lightning as L
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from lightning import seed_everything
from lightning.pytorch.plugins.environments import SLURMEnvironment
from torch.utils.data import DataLoader
from torch import stack

from torch_geometric.loader import DataLoader as GraphDataLoader

from callbacks.validation_reload import ReloadValidationDataLoaderCallback
from config.utils import get_config_path
from dataset.tm_score_from_embeddings_provider_dataset import TmScoreFromEmbeddingsProviderDataset
from dataset.tm_score_from_pickle_dataset import TmScoreFromCoordDataset
from dataset.utils.custom_weighted_random_sampler import CustomWeightedRandomSampler
from dataset.utils.embedding_provider import SqliteEmbeddingProvider
from dataset.utils.tm_score_weight import fraction_score, tm_score_weights

from lightning_module.training.lightning_pst_graph import LitStructurePstGraph
from config.schema_config import TrainingConfig


cs = ConfigStore.instance()
cs.store(name="training_default", node=TrainingConfig)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../config", config_name="training_config")
def main(cfg: TrainingConfig):
    logger.info(f"Using config file: {get_config_path()}")
    seed_everything(cfg.global_seed, workers=True)
    training_set = TmScoreFromCoordDataset(
        tm_score_file=cfg.training_set.tm_score_file,
        coords_path=cfg.training_set.data_path,
        ext=cfg.training_set.data_ext,
        score_method=fraction_score,
        weighting_method=tm_score_weights(5, 0.25),
        num_workers=cfg.training_set.workers,
        include_self_comparison=cfg.training_set.include_self_comparison,
        coords_augmenter=instantiate(
            cfg.training_set.data_augmenter
        ) if cfg.training_set.data_augmenter is not None else None
    )

    weights = training_set.weights()
    sampler = CustomWeightedRandomSampler(
        weights=weights,
        num_samples=cfg.training_parameters.epoch_size if cfg.training_parameters.epoch_size > 0 else len(weights),
        replacement=True
    )
    train_dataloader = GraphDataLoader(
        dataset=training_set,
        sampler=sampler,
        batch_size=cfg.training_set.batch_size,
        num_workers=cfg.training_set.workers,
        persistent_workers=True if cfg.training_set.workers > 0 else False
    )

    embedding_provider = SqliteEmbeddingProvider()

    validation_set = TmScoreFromEmbeddingsProviderDataset(
        tm_score_file=cfg.validation_set.tm_score_file,
        embedding_provider=embedding_provider
    )
    validation_dataloader = DataLoader(
        dataset=validation_set,
        batch_size=cfg.validation_set.batch_size,
        num_workers=cfg.validation_set.workers,
        persistent_workers=True if cfg.validation_set.workers > 0 else False,
        pin_memory=True,
        collate_fn=lambda emb: (
            stack([x for x, y, z in emb], dim=0),
            stack([y for x, y, z in emb], dim=0),
            stack([z for x, y, z in emb], dim=0)
        )
    )

    nn_model = instantiate(
        cfg.embedding_network
    )

    model = LitStructurePstGraph(
        nn_model=nn_model,
        learning_rate=cfg.training_parameters.learning_rate,
        params=cfg
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
        num_nodes=cfg.computing_resources.nodes,
        devices=cfg.computing_resources.devices,
        strategy=cfg.computing_resources.strategy,
        callbacks=[checkpoint_callback, lr_monitor, ReloadValidationDataLoaderCallback(embedding_provider)],
        plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)],
        default_root_dir=cfg.default_root_dir
    )
    trainer.fit(
        model,
        train_dataloader,
        validation_dataloader,
        ckpt_path=cfg.checkpoint
    )


if __name__ == '__main__':
    main()
