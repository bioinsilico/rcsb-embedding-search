import logging
import signal

import hydra
import lightning as L
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from lightning import seed_everything
from lightning.pytorch.plugins.environments import SLURMEnvironment
from torch.utils.data import DataLoader

from config.utils import get_config_path
from dataset.tm_score_from_embeddings_dataset import TmScoreFromEmbeddingsDataset
from dataset.utils.custom_weighted_random_sampler import CustomWeightedRandomSampler
from dataset.utils.tm_score_weight import fraction_score_of, tm_score_weights
from dataset.utils.tools import collate_fn
from lightning_module.training.embedding_training import LitEmbeddingTraining

from config.schema_config import TrainingConfig
from lightning.pytorch.loggers import TensorBoardLogger


cs = ConfigStore.instance()
cs.store(name="training_default", node=TrainingConfig)
logger = logging.getLogger(__name__)



@hydra.main(version_base=None, config_path="../../config", config_name="training_config")
def main(cfg: TrainingConfig):
    logger.info(f"Using config file: {get_config_path()}")
    seed_everything(cfg.global_seed, workers=True)
    training_set = TmScoreFromEmbeddingsDataset(
        tm_score_file=cfg.training_set.tm_score_file,
        embedding_path=cfg.training_set.data_path,
        score_method=fraction_score_of(
            f=cfg.metadata.fraction_score if cfg.metadata is not None and 'fraction_score' in cfg.metadata else 10
        ),
        weighting_method=tm_score_weights(5, 0.25),
        exclude_domains_file=cfg.metadata.exclude_domains_file if cfg.metadata is not None and 'exclude_domains_file' in cfg.metadata else None
    )

    weights = training_set.weights()
    sampler = CustomWeightedRandomSampler(
        weights=weights,
        num_samples=cfg.training_parameters.epoch_size if cfg.training_parameters.epoch_size > 0 else len(weights),
        replacement=True
    )
    train_dataloader = DataLoader(
        dataset=training_set,
        sampler=sampler,
        batch_size=cfg.training_set.batch_size,
        num_workers=cfg.training_set.workers,
        persistent_workers=True if cfg.training_set.workers > 0 else False,
        pin_memory=True,
        collate_fn=collate_fn
    )

    validation_set = TmScoreFromEmbeddingsDataset(
        tm_score_file=cfg.validation_set.tm_score_file,
        embedding_path=cfg.validation_set.data_path
    )
    validation_dataloader = DataLoader(
        dataset=validation_set,
        batch_size=cfg.validation_set.batch_size,
        num_workers=cfg.validation_set.workers,
        persistent_workers=True if cfg.validation_set.workers > 0 else False,
        pin_memory=True,
        collate_fn=collate_fn
    )

    nn_model = instantiate(
        cfg.embedding_network
    )

    model = LitEmbeddingTraining(
        nn_model=nn_model,
        learning_rate=cfg.training_parameters.learning_rate,
        params=cfg
    )

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        monitor=LitEmbeddingTraining.PR_AUC_METRIC_NAME,
        mode='max',
        filename='{epoch}-{' + LitEmbeddingTraining.PR_AUC_METRIC_NAME + ':.2f}'
    )

    lr_monitor = L.pytorch.callbacks.LearningRateMonitor(
        logging_interval='step'
    )

    logger_tb = TensorBoardLogger(
    save_dir=cfg.logger.save_dir,
    name=cfg.logger.name
    )   

    trainer = L.Trainer(
        max_epochs=cfg.training_parameters.epochs,
        check_val_every_n_epoch=cfg.training_parameters.check_val_every_n_epoch,
        num_nodes=cfg.computing_resources.nodes,
        devices=cfg.computing_resources.devices,
        strategy=cfg.computing_resources.strategy,
        callbacks=[checkpoint_callback, lr_monitor],
        plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)],
        default_root_dir=cfg.default_root_dir,
        logger= logger_tb
    )
    trainer.fit(
        model,
        train_dataloader,
        validation_dataloader,
        ckpt_path=cfg.checkpoint
    )


if __name__ == '__main__':
    main()
