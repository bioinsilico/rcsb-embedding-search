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
    # In full-sequence mode `segment_length` keeps two jobs: it is the window
    # size the minimizer index is built on (proposals stay local) and the
    # minimum length a sequence must have to be kept at all.
    full_sequence = meta.get('full_sequence', False)
    sequence_kwargs = dict(
        full_sequence=full_sequence,
        max_sequence_length=meta.get('max_sequence_length', None),
        length_bucket=meta.get('length_bucket', 0),
    )

    # Holdout split: whole CATH superfamilies kept out of training and used as
    # the entire validation corpus, so the validation score measures transfer to
    # folds the model has never seen rather than to fresh pairs of seen domains.
    holdout = meta.get('holdout_domains_file', None)
    excluded = meta.get('exclude_domains_file', None)
    train_exclude = [f for f in (excluded, holdout) if f] or None
    logger.info(
        f"Validating on held-out superfamilies from {holdout}" if holdout
        else "No holdout file: validation draws from the same domains as training"
    )
    if full_sequence:
        logger.info(
            f"Full-sequence mode: whole sequences emitted, coverage taken over "
            f"min(len_i, len_j); index windows of {segment_length} residues, "
            f"max length {sequence_kwargs['max_sequence_length']}, "
            f"length_bucket {sequence_kwargs['length_bucket']}"
        )

    # Uniform pairing cannot reach related sequences: on CATH fewer than 0.02%
    # of random window pairs align appreciably, so the balancer ends up filling
    # its top bins with chance alignments.  p_kmer proposes pairs through a
    # minimizer index and p_offset takes two windows of one sequence at a drawn
    # offset; both change only which pairs are looked at, never the label.
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
        exclude_ids_file=train_exclude,
        p_kmer=meta.get('p_kmer', 0.0),
        p_offset=meta.get('p_offset', 0.0),
        max_self_offset=meta.get('max_self_offset', None),
        p_superfamily=meta.get('p_superfamily', 0.0),
        superfamily_file=meta.get('superfamily_file', None),
        **sequence_kwargs,
        kmer_size=meta.get('kmer_size', 6),
        minimizer_window=meta.get('minimizer_window', 10),
        reduced_alphabet=meta.get('reduced_alphabet', True),
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
        exclude_ids_file=excluded,
        include_ids_file=holdout,
        p_kmer=meta.get('p_kmer', 0.0),
        p_offset=meta.get('p_offset', 0.0),
        max_self_offset=meta.get('max_self_offset', None),
        p_superfamily=meta.get('p_superfamily', 0.0),
        superfamily_file=meta.get('superfamily_file', None),
        **sequence_kwargs,
        kmer_size=meta.get('kmer_size', 6),
        minimizer_window=meta.get('minimizer_window', 10),
        reduced_alphabet=meta.get('reduced_alphabet', True),
        # Window ids index the corpus the index was built from, so it can only be
        # shared when both splits see the same domains.  With a holdout the
        # validation set builds its own -- far smaller, and quick.
        kmer_index=None if holdout else training_set.kmer_index,
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

    # Monitor the thresholded metric: pr_auc against continuous targets can sit
    # at a constant value for a whole run, which leaves ModelCheckpoint holding
    # an early epoch forever.  This callback tracks the best model only; keeping
    # the newest weights is the rolling callback's job below.
    monitor = LitSequenceAutoencoderTraining.PR_AUC_BIN_METRIC_NAME
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        monitor=monitor,
        mode='max',
        filename='best-{epoch}-{' + monitor + ':.2f}',
    )

    # A monitored checkpoint only writes when the metric improves, and save_last
    # is refreshed only alongside such a write — so neither preserves the newest
    # weights once the monitor plateaus.  This second, unmonitored callback saves
    # unconditionally on a step cadence, which is also the only granularity that
    # protects a wall-clock-limited job when one epoch spans hours.  Resume from
    # 'rolling.ckpt'.
    rolling_callback = L.pytorch.callbacks.ModelCheckpoint(
        monitor=None,
        save_top_k=1,
        every_n_train_steps=meta.get('checkpoint_every_n_steps', 1000),
        filename='rolling',
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
        callbacks=[checkpoint_callback, rolling_callback, lr_monitor],
        plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)],
        default_root_dir=cfg.default_root_dir,
        logger=logger_tb,
        # Each rank seeds its own independent stream; no sampler to distribute.
        use_distributed_sampler=False,
        # Bounds the loss spikes seen mid-run; 0 disables clipping.
        gradient_clip_val=meta.get('gradient_clip_val', 1.0),
    )
    trainer.fit(
        model,
        train_dataloader,
        validation_dataloader,
        ckpt_path=cfg.checkpoint,
    )


if __name__ == '__main__':
    main()
