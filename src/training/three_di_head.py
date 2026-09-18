"""Train a sequence -> 3Di head on cached ESM3 embeddings.

    python src/training/three_di_head.py \\
        --config-path=/path/to/config/training --config-name=three_di_head_config

Swapping architectures is a config change - ``head._target_`` selects the module and
its keyword arguments come from the same block, so a new architecture only has to be
an ``nn.Module`` with the ``ThreeDiHead`` signature:

    head:
      _target_: networks.three_di_head.CnnThreeDiHead   # or LinearThreeDiHead, Transformer...,
      hidden: 512                                       # or anything you write yourself
"""
import logging
import os

import hydra
import lightning as L
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from lightning import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from torch.utils.data import DataLoader

from config.schema_config import ThreeDiTrainingConfig
from config.utils import get_config_path
from dataset.three_di_from_embeddings_dataset import ThreeDiFromEmbeddingsDataset, collate_three_di
from dataset.utils.length_bucket_sampler import LengthBucketBatchSampler
from dataset.utils.packed_embeddings import stage_store_to_local
from lightning_module.training.three_di_training import LitThreeDiTraining

cs = ConfigStore.instance()
cs.store(name="three_di_training_default", node=ThreeDiTrainingConfig)
logger = logging.getLogger(__name__)


def build_dataset(cfg_set, shuffle: bool, seed: int = 0):
    store_path = str(cfg_set.store_path)
    embeddings_dir = os.path.join(store_path, "embeddings")
    ss8_dir = os.path.join(store_path, "ss8_logits")
    if cfg_set.local_scratch:
        # Node-local staging: the store is read randomly all epoch and the shared
        # filesystem is the slow part of this pipeline (see packed_embeddings). Each
        # packed store is staged separately, and the *returned* paths are what the
        # dataset must read - staging and then reading the shared copy would pay the
        # copy for nothing.
        embeddings_dir = stage_store_to_local(embeddings_dir, cfg_set.local_scratch)
        if cfg_set.with_ss8:
            ss8_dir = stage_store_to_local(ss8_dir, cfg_set.local_scratch)
        logger.info(f"embeddings served from {embeddings_dir}")
    dataset = ThreeDiFromEmbeddingsDataset(
        store_path=store_path,
        labels_path=str(cfg_set.labels_path),
        split=cfg_set.split,
        use_reasons=cfg_set.use_reasons,
        with_ss8=cfg_set.with_ss8,
        min_length=cfg_set.min_length,
        max_length=cfg_set.max_length,
        embeddings_dir=embeddings_dir,
        ss8_dir=ss8_dir,
    )
    if cfg_set.length_bucket:
        sampler = LengthBucketBatchSampler(
            lengths=dataset.lengths(),
            batch_size=cfg_set.batch_size,
            bucket_size=cfg_set.length_bucket,
            shuffle=shuffle,
            seed=seed,
        )
        loader = DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=cfg_set.workers,
            persistent_workers=cfg_set.workers > 0,
            pin_memory=True,
            collate_fn=collate_three_di,
        )
    else:
        loader = DataLoader(
            dataset=dataset,
            batch_size=cfg_set.batch_size,
            shuffle=shuffle,
            num_workers=cfg_set.workers,
            persistent_workers=cfg_set.workers > 0,
            pin_memory=True,
            collate_fn=collate_three_di,
        )
    return dataset, loader


@hydra.main(version_base=None, config_path="../../config", config_name="three_di_head_config")
def main(cfg: ThreeDiTrainingConfig):
    logger.info(f"Using config file: {get_config_path()}")
    seed_everything(cfg.global_seed, workers=True)

    if (cfg.training_set.length_bucket is None) != (cfg.validation_set.length_bucket is None):
        # use_distributed_sampler is one Trainer-wide flag: with bucketing on one split
        # only, either that sampler is wrapped and shards twice, or the other split is
        # never sharded at all.
        raise ValueError("length_bucket must be set (or unset) on both training_set and validation_set")

    training_set, train_dataloader = build_dataset(cfg.training_set, shuffle=True, seed=cfg.global_seed)
    _, val_dataloader = build_dataset(cfg.validation_set, shuffle=False, seed=cfg.global_seed)

    head = instantiate(cfg.head)
    loss_fn = instantiate(cfg.loss)
    loss_fn.set_class_counts(training_set.label_counts())
    logger.info(f"head {type(head).__name__} with "
                f"{sum(p.numel() for p in head.parameters()):,} parameters; "
                f"loss {type(loss_fn).__name__}")

    model = LitThreeDiTraining(
        nn_model=head,
        loss_fn=loss_fn,
        learning_rate=cfg.training_parameters.learning_rate,
        cfg=cfg,
    )

    trainer = L.Trainer(
        max_epochs=cfg.training_parameters.epochs,
        check_val_every_n_epoch=cfg.training_parameters.check_val_every_n_epoch,
        devices=cfg.computing_resources.devices,
        num_nodes=cfg.computing_resources.nodes,
        strategy=cfg.computing_resources.strategy,
        # The batch sampler shards itself and gives every rank the same batch count.
        use_distributed_sampler=cfg.training_set.length_bucket is None,
        default_root_dir=cfg.default_root_dir,
        logger=TensorBoardLogger(save_dir=cfg.logger.save_dir, name=cfg.logger.name),
        # Select on validation NLL, not expected_score: expected_score rewards confidence,
        # and overfitting raises confidence, so it kept the most overfit epochs. "nll" is
        # plain cross-entropy whatever the training loss is, so it ranks every loss variant
        # on the same scale (validation_loss would not, under soft targets or class weights).
        callbacks=[ModelCheckpoint(monitor="nll", mode="min", save_last=True, save_top_k=3,
                                   filename="{epoch}-{nll:.4f}")],
        plugins=[SLURMEnvironment(auto_requeue=False)] if SLURMEnvironment().detect() else None,
    )
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=cfg.checkpoint)


if __name__ == "__main__":
    main()
