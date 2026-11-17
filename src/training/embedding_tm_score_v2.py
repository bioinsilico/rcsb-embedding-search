"""Stage-1 retraining of the structure-embedding model on a packed store.

    srun python src/training/embedding_tm_score_v2.py --config-path=<dir> --config-name=<name> [overrides]

Differences from ``embedding_tm_score.py``:

* data: :class:`PackedTmPairDataset` + :class:`BinBalancedPairBatchSampler` (the production pair
  distribution, no per-sample file reads, optional node-local staging of the store);
* ``data.exclude_domains_file`` drops a superfamily-held-out domain list (and its CATH twins) for
  held-out replicates;
* ``init.checkpoint`` (backbone; any tensor whose shape matches) and ``init.head`` (a Linear state dict,
  e.g. a frozen-network head) initialise the model;
* :class:`LitEmbeddingTrainingV2`: warm-up + cosine to 0 over the actual number of steps, separate
  learning rates for the final Linear and the rest, gradient clipping, optional bf16;
* weights-only checkpoints every ``checkpoint_every_epochs`` epochs (all kept, for offline selection
  by retrieval) plus a full ``last.ckpt`` for requeue;
* ``data.seq_store_path`` / ``data.seq_mix``: sequence-only views mixed into training pairs;
* ``kd.teacher`` (.npz of unit teacher vectors per domain, ``z`` and optionally ``z_seq``) with
  ``kd.weight`` / ``kd.tau`` / ``kd.gather``: relational distillation (:class:`LitEmbeddingTrainingKD`);
  ``kd.cosent``: weight of an added CoSENT ranking term over the gathered pairs; ``kd.min_len``: domains with
  fewer residues are left out of the distillation batch;
* ``data.contain`` (``{mass, ratio, tm_pos, tm_neg, pep_len}``): containment strata - a fixed share ``mass`` of
  the draws each from the short-inside-long positive and negative pairs (:func:`containment_strata`), kept out
  of the distillation batch.

One "epoch" is ``data.samples_per_epoch`` pair draws over all ranks; the run lasts ``data.draws`` draws.
"""
import glob
import logging
import math
import os
import signal
import time

import hydra
import lightning as L
import numpy as np
import torch
from hydra.utils import instantiate
from lightning import seed_everything
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader

from dataset.packed_tm_pairs_dataset import (BinaryScore, BinBalancedPairBatchSampler, PackedTmPairDataset,
                                             ShardedSequentialBatchSampler, containment_strata)
from dataset.utils.packed_embeddings import stage_store_to_local
from dataset.utils.tools import collate_fn, collate_meta_fn
from lightning_module.training.embedding_training import LitEmbeddingTrainingKD, LitEmbeddingTrainingV2

logger = logging.getLogger(__name__)


def load_state(path):
    """State dict of a FoldMatch ``.pt``, a bare state dict, or a Lightning checkpoint (``model.`` keys)."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(obj, "state_dict") and not isinstance(obj, dict):
        obj = obj.state_dict()
    if "state_dict" in obj:
        obj = {k[len("model."):]: v for k, v in obj["state_dict"].items() if k.startswith("model.")}
    return obj


def initialise(model, checkpoint=None, head=None):
    if checkpoint:
        state = load_state(checkpoint)
        own = model.state_dict()
        loaded = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
        model.load_state_dict(loaded, strict=False)
        logger.info(f"init from {checkpoint}: {len(loaded)} tensors loaded; not loaded "
                    f"{sorted(set(state) - set(loaded))}; left at init {sorted(set(own) - set(loaded))}")
    if head:
        h = torch.load(head, map_location="cpu", weights_only=False)
        weight = h["net.weight"] if "net.weight" in h else h["weight"]
        bias = h["net.bias"] if "net.bias" in h else h["bias"]
        with torch.no_grad():
            model.embedding.linear.weight.copy_(weight)
            model.embedding.linear.bias.copy_(bias)
        logger.info(f"head init from {head}: {tuple(weight.shape)}")


class Throughput(Callback):
    """Logs pairs per second per GPU every ``every`` steps (after a short warm-up)."""

    def __init__(self, batch_size, every=50, warmup=20):
        self.batch_size, self.every, self.warmup = batch_size, every, warmup
        self.t0, self.s0, self.start = None, None, 0

    def on_train_start(self, trainer, pl_module):
        self.start = trainer.global_step              # non-zero after a resume

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if step == self.start + self.warmup:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            self.t0, self.s0 = time.time(), step
        elif self.t0 is not None and step > self.s0 and step % self.every == 0:
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            dt = (time.time() - self.t0) / (step - self.s0)
            pl_module.log("sec_per_step", dt, rank_zero_only=True)
            pl_module.log("pairs_per_sec_per_gpu", self.batch_size / dt, rank_zero_only=True)
            if trainer.is_global_zero:
                logger.info(f"step {step}: {dt:.3f} s/step, {self.batch_size / dt:.1f} pairs/s/GPU")


def resume_checkpoint(root, resume):
    """``auto``: the newest of the SIGUSR1 checkpoint Lightning writes before a requeue
    (``<root>/hpc_ckpt_*.ckpt``) and ``checkpoints/last.ckpt``; otherwise the given path or None."""
    if resume != "auto":
        return resume
    cands = glob.glob(os.path.join(root, "hpc_ckpt_*.ckpt")) + glob.glob(os.path.join(root, "checkpoints", "last.ckpt"))
    return max(cands, key=os.path.getmtime) if cands else None


def load_teacher(path, domains):
    """Teacher table (.npz: ``domains``, unit vectors ``z`` and optionally ``z_seq``) in the order of
    ``domains`` (the pairs file); a domain without a teacher vector gets zeros (and a warning)."""
    z = np.load(path)
    pos = {str(d): n for n, d in enumerate(z["domains"])}
    idx = np.array([pos.get(d, -1) for d in domains])
    missing = int((idx < 0).sum())
    if missing:
        logger.warning(f"teacher {path}: {missing} of {len(domains)} domains have no vector (zeros)")

    def table(a):
        out = np.zeros((len(domains), a.shape[1]), np.float32)
        out[idx >= 0] = a[idx[idx >= 0]].astype(np.float32)
        return out
    logger.info(f"teacher {path}: {z['z'].shape[1]}-d, {len(domains) - missing} domains, "
                f"sequence-only views {'z_seq' in z.files}")
    return table(z["z"]), (table(z["z_seq"]) if "z_seq" in z.files else None)


def exclusion(path):
    if not path:
        return None
    with open(path) as fh:
        return {line.strip() for line in fh if line.strip()}


@hydra.main(version_base=None, config_path=None, config_name=None)
def main(cfg):
    torch.set_float32_matmul_precision("high")
    seed_everything(cfg.global_seed, workers=True)
    d = cfg.data
    store = str(d.store_path)
    if d.get("local_scratch"):
        store = stage_store_to_local(store, d.local_scratch)
    excluded = exclusion(d.get("exclude_domains_file"))
    twins = d.get("exclude_cath_twins", True)
    seq_store = d.get("seq_store_path")
    if seq_store and d.get("seq_mix", 0.0) > 0 and d.get("local_scratch"):
        seq_store = stage_store_to_local(str(seq_store), d.local_scratch)
    train_set = PackedTmPairDataset(store, d.train_pairs, excluded, twins, d.get("fraction_score", 10),
                                    seq_store_dir=seq_store, seq_mix=d.get("seq_mix", 0.0))
    kd = cfg.get("kd") or {}
    train_set.return_meta = bool(kd.get("teacher"))         # domain ids and views for the teacher lookups
    # validation as in production: the scop_low_sen pairs, binary label TM > 0.7, nothing excluded
    val_set = PackedTmPairDataset(store, d.val_pairs, target=BinaryScore(0.7))

    res = cfg.computing_resources
    world = res.nodes * res.devices
    steps_per_epoch = d.samples_per_epoch // (d.batch_size * world)
    epochs = math.ceil(d.draws / d.samples_per_epoch)
    total_steps = epochs * steps_per_epoch
    logger.info(f"{world} GPUs x {d.batch_size} pairs = {world * d.batch_size} pairs/step; "
                f"{d.draws:,} draws = {total_steps} steps, {epochs} epochs of {d.samples_per_epoch:,}")
    workers = d.get("workers", 4)
    strata = None
    if d.get("contain"):
        c = d.contain
        pos, neg = containment_strata(train_set, c.get("pep_len", 50), c.get("ratio", 2.0), c.get("tm_pos", 0.7),
                                      c.get("tm_neg", 0.5))
        logger.info(f"containment strata: {len(pos):,} positive and {len(neg):,} negative pairs, mass {c.mass} each")
        strata = [(pos, c.mass), (neg, c.mass)]
    sampler = BinBalancedPairBatchSampler(train_set, d.batch_size, d.samples_per_epoch,
                                          d.get("pool_batches", 64), cfg.global_seed, strata=strata)
    train_loader = DataLoader(
        train_set,
        batch_sampler=sampler,
        num_workers=workers, persistent_workers=workers > 0, pin_memory=True,
        collate_fn=collate_meta_fn if train_set.return_meta else collate_fn,
        prefetch_factor=4 if workers > 0 else None)
    val_loader = DataLoader(
        val_set, batch_sampler=ShardedSequentialBatchSampler(val_set, d.get("val_batch_size", d.batch_size)),
        num_workers=workers, persistent_workers=workers > 0, pin_memory=True, collate_fn=collate_fn)

    nn_model = instantiate(cfg.embedding_network)
    initialise(nn_model, cfg.init.get("checkpoint"), cfg.init.get("head"))
    common = dict(lr_backbone=cfg.optim.lr_backbone, lr_head=cfg.optim.lr_head, total_steps=total_steps,
                  warmup_frac=cfg.optim.get("warmup_frac", 0.05), weight_decay=cfg.optim.get("weight_decay", 0.0),
                  params=cfg)
    if kd.get("teacher"):
        teacher, teacher_seq = load_teacher(kd.teacher, train_set.domains)
        model = LitEmbeddingTrainingKD(nn_model, teacher=teacher, teacher_seq=teacher_seq,
                                       kd_weight=kd.get("weight", 1.0), kd_tau=kd.get("tau", 0.05),
                                       gather=kd.get("gather", True), cosent_weight=kd.get("cosent", 0.0),
                                       kd_min_len=kd.get("min_len", 0), domain_length=train_set.residue_lengths(),
                                       **common)
        logger.info(f"distillation: weight {model.kd_weight}, tau {model.kd_tau}, cosent {model.cosent_weight}, "
                    f"min_len {model.kd_min_len}")
    else:
        model = LitEmbeddingTrainingV2(nn_model, **common)

    root = cfg.default_root_dir
    callbacks = [
        ModelCheckpoint(dirpath=os.path.join(root, "checkpoints"), filename="epoch{epoch:03d}",
                        auto_insert_metric_name=False, every_n_epochs=cfg.get("checkpoint_every_epochs", 1),
                        save_top_k=-1, save_weights_only=True),
        # full state for requeue; a step interval keeps its state_key distinct from the callback above
        ModelCheckpoint(dirpath=os.path.join(root, "checkpoints"), save_last=True, save_top_k=0,
                        every_n_train_steps=cfg.get("last_every_n_steps", 500)),
        LearningRateMonitor(logging_interval="step"),
        Throughput(d.batch_size),
    ]
    trainer = L.Trainer(
        max_epochs=epochs,
        max_steps=total_steps,
        accelerator=res.get("accelerator", "auto"),
        num_nodes=res.nodes,
        devices=res.devices,
        # the buffers (teacher tables, domain lengths, input statistics) are constant and equal on every rank:
        # DDP's default re-broadcast of all buffers before each forward would move ~0.5 GB per step
        strategy=DDPStrategy(broadcast_buffers=False) if res.get("strategy", "ddp") == "ddp" else res.get("strategy"),
        precision=cfg.get("precision", "32-true"),
        gradient_clip_val=cfg.optim.get("grad_clip"),
        use_distributed_sampler=False,
        check_val_every_n_epoch=cfg.get("check_val_every_n_epoch", 1),
        limit_val_batches=cfg.get("limit_val_batches", 1.0),
        log_every_n_steps=cfg.get("log_every_n_steps", 50),
        callbacks=callbacks,
        plugins=[SLURMEnvironment(requeue_signal=signal.SIGUSR1)],
        default_root_dir=root,
        logger=TensorBoardLogger(save_dir=cfg.logger.save_dir, name=cfg.logger.name, version=0),
    )
    resume = resume_checkpoint(root, cfg.get("resume"))
    if resume:
        # Lightning builds the first data iterator before set_epoch and does not skip consumed
        # batches itself: put the sampler at the checkpoint's epoch and position (one optimizer step
        # per batch, steps_per_epoch batches per epoch on every rank).
        step = torch.load(resume, map_location="cpu", weights_only=False, mmap=True)["global_step"]
        epoch, done = divmod(step, steps_per_epoch)
        sampler.set_epoch(epoch)
        sampler.resume = (epoch, done)
        logger.info(f"resuming from {resume}: global step {step} = epoch {epoch} + {done} batches")
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume)


if __name__ == "__main__":
    main()
