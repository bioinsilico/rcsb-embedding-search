"""Train the sequence autoencoder on segment pairs aligned on the fly.

Same model and losses as ``sequence_autoencoder.py``, but the pairs and their
scores are generated from a single FASTA file at training time instead of being
read from a precomputed TSV.  Because the dataset is an ``IterableDataset``,
score balancing lives inside the dataset rather than in a weighted sampler, and
validation uses a deterministic stream instead of a ``Subset``.
"""
import logging
import signal

import hydra
import lightning as L
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from lightning import seed_everything
from lightning.pytorch.plugins.environments import SLURMEnvironment
from torch.utils.data import DataLoader

from config.schema_config import TrainingConfig
from config.utils import get_config_path
from dataset.sequence_alignment_iterable_dataset import SequenceAlignmentIterableDataset
from dataset.sequence_identity_dataset import collate_sequence_pairs
from dataset.utils.tm_score_weight import fraction_score_of
from lightning_module.training.sequence_autoencoder_training import LitSequenceAutoencoderTraining

from lightning.pytorch.loggers import TensorBoardLogger


cs = ConfigStore.instance()
cs.store(name="training_default", node=TrainingConfig)
logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../../config", config_name="training_config")
def main(cfg: TrainingConfig):
    logger.info(f"Using config file: {get_config_path()}")
    seed_everything(cfg.global_seed, workers=True)

    meta = cfg.metadata or {}
    aligner = instantiate(meta.aligner) if 'aligner' in meta else None
    segment_length = meta.get('segment_length', 50)
    window_step = meta.get('window_step', None)

    training_set = SequenceAlignmentIterableDataset(
        fasta_file=cfg.training_set.data_path,
        aligner=aligner,
        segment_length=segment_length,
        window_step=window_step,
        samples_per_epoch=cfg.training_parameters.epoch_size,
        score_method=fraction_score_of(f=meta.get('fraction_score', 10)),
        n_intervals=cfg.training_set.tm_score_intervals,
        balance_alpha=meta.get('balance_alpha', 1.0),
        max_attempts=meta.get('max_attempts', 50),
        exclude_ids_file=meta.get('exclude_domains_file', None),
        seed=cfg.global_seed,
        deterministic=False,
    )

    # A fixed seed and deterministic=True replay the same pairs every epoch, so
    # validation numbers stay comparable across epochs.
    validation_set = SequenceAlignmentIterableDataset(
        fasta_file=cfg.training_set.data_path,
        aligner=aligner,
        segment_length=segment_length,
        window_step=window_step,
        samples_per_epoch=meta.get('validation_size', 1000),
        score_method=fraction_score_of(f=meta.get('fraction_score', 10)),
        n_intervals=cfg.training_set.tm_score_intervals,
        balance_alpha=meta.get('balance_alpha', 1.0),
        max_attempts=meta.get('max_attempts', 50),
        exclude_ids_file=meta.get('exclude_domains_file', None),
        seed=cfg.global_seed + 1,
        deterministic=True,
    )

    train_dataloader = DataLoader(
        dataset=training_set,
        batch_size=cfg.training_set.batch_size,
        num_workers=cfg.training_set.workers,
        persistent_workers=True if cfg.training_set.workers > 0 else False,
        pin_memory=True,
        collate_fn=collate_sequence_pairs,
    )

    validation_dataloader = DataLoader(
        dataset=validation_set,
        batch_size=cfg.validation_set.batch_size,
        num_workers=cfg.validation_set.workers,
        persistent_workers=True if cfg.validation_set.workers > 0 else False,
        pin_memory=True,
        collate_fn=collate_sequence_pairs,
    )

    nn_model = instantiate(cfg.embedding_network)

    model = LitSequenceAutoencoderTraining(
        nn_model=nn_model,
        learning_rate=cfg.training_parameters.learning_rate,
        reconstruction_weight=cfg.metadata.reconstruction_weight if cfg.metadata is not None and 'reconstruction_weight' in cfg.metadata else 1.0,
        similarity_weight=cfg.metadata.similarity_weight if cfg.metadata is not None and 'similarity_weight' in cfg.metadata else 1.0,
        cfg=cfg,
    )

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        monitor=LitSequenceAutoencoderTraining.PR_AUC_METRIC_NAME,
        mode='max',
        filename='{epoch}-{' + LitSequenceAutoencoderTraining.PR_AUC_METRIC_NAME + ':.2f}',
    )

    lr_monitor = L.pytorch.callbacks.LearningRateMonitor(
        logging_interval='step',
    )

    logger_tb = TensorBoardLogger(
        save_dir=cfg.logger.save_dir,
        name=cfg.logger.name,
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
        logger=logger_tb,
        # Each rank seeds its own independent stream; no sampler to distribute.
        use_distributed_sampler=False,
    )
    trainer.fit(
        model,
        train_dataloader,
        validation_dataloader,
        ckpt_path=cfg.checkpoint,
    )


if __name__ == '__main__':
    main()
