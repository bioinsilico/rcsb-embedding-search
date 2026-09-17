"""Lightning module for the sequence -> 3Di head.

The head is any ``nn.Module`` with the ``ThreeDiHead`` signature and the loss any
module with ``ThreeDiLoss``'s, so both are swapped from the config alone.

What it logs, and why these and not just accuracy: the benchmark says a 65%-accurate
head keeps 96% of the alignment F1 if its mistakes land on geometrically similar
states, so per-residue accuracy alone cannot rank two heads.

* ``accuracy`` / ``top3_accuracy`` - the familiar numbers, reported but not optimized.
* ``expected_score`` - mean ``sum_k p(k) * M[k, true]``, the quantity the aligner
  computes from a soft profile. ``expected_score_ceiling`` is what a perfect, confident
  prediction would score on the same labels (below the matrix diagonal's mean, because
  the states are not equally frequent).
* ``nll`` and ``ece`` - calibration. No published 3Di predictor reports it, and the
  soft profile is only as good as its calibration.
* ``confidence`` - mean max probability, which is what a deployment would threshold
  on to mask uncertain query positions.
"""
from __future__ import annotations

import pathlib

import lightning as L
import torch
import yaml
from omegaconf import OmegaConf
from torch import nn, optim
from torch.nn import functional as F

from config.schema_config import LrInterval, Strategy
from lightning_module.utils import get_cosine_schedule_with_warmup


N_BINS = 15


def calibration_bins(probability: torch.Tensor, correct: torch.Tensor,
                     n_bins: int = N_BINS) -> torch.Tensor:
    """Per-bin (count, summed confidence, summed correctness), ready to sum across ranks."""
    index = torch.clamp((probability * n_bins).long(), max=n_bins - 1)
    stats = torch.zeros((n_bins, 3), dtype=torch.float64, device=probability.device)
    stats[:, 0] = torch.bincount(index, minlength=n_bins)
    stats[:, 1] = torch.bincount(index, weights=probability.double(), minlength=n_bins)
    stats[:, 2] = torch.bincount(index, weights=correct.double(), minlength=n_bins)
    return stats


def expected_calibration_error(stats: torch.Tensor) -> torch.Tensor:
    """|confidence - accuracy| over bins, weighted by bin population."""
    count = stats[:, 0]
    total = count.sum()
    if total == 0:
        return torch.zeros((), dtype=stats.dtype, device=stats.device)
    filled = count > 0
    gap = (stats[filled, 1] / count[filled] - stats[filled, 2] / count[filled]).abs()
    return (count[filled] * gap).sum() / total


class LitThreeDiTraining(L.LightningModule):
    """Train a 3Di head on cached ESM3 embeddings."""

    def __init__(
            self,
            nn_model: nn.Module,
            loss_fn: nn.Module,
            learning_rate: float = 1e-3,
            cfg=None,
    ):
        super().__init__()
        self.model = nn_model
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.cfg = cfg
        self._validation: list[tuple[torch.Tensor, torch.Tensor]] = []

    def on_fit_start(self):
        if self.cfg is not None and hasattr(self.logger.experiment, "add_text"):
            yaml.add_representer(pathlib.PurePosixPath, lambda d, v: d.represent_str(str(v)))
            yaml.add_representer(pathlib.PosixPath, lambda d, v: d.represent_str(str(v)))
            yaml.add_representer(LrInterval, lambda d, v: d.represent_str(str(v)))
            yaml.add_representer(Strategy, lambda d, v: d.represent_str(str(v)))
            self.logger.experiment.add_text("Config", yaml.dump(OmegaConf.to_container(self.cfg, resolve=True)))

    def forward(self, batch) -> torch.Tensor:
        return self.model(batch["embedding"], padding_mask=batch["padding_mask"], ss8=batch["ss8"])

    def training_step(self, batch, batch_idx):
        logits = self(batch)
        loss = self.loss_fn(logits, batch["label"], batch["loss_mask"])
        # Weighted by residues, not domains: length bucketing makes batches differ by an
        # order of magnitude in residue count, and the loss is a per-residue mean.
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=int(batch["loss_mask"].sum()), sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self(batch)
        loss = self.loss_fn(logits, batch["label"], batch["loss_mask"])
        self.log("validation_loss", loss, on_epoch=True, prog_bar=True,
                 batch_size=int(batch["loss_mask"].sum()), sync_dist=True)
        mask = batch["loss_mask"]
        self._validation.append((logits[mask].detach().float().cpu(), batch["label"][mask].detach().cpu()))
        return loss

    def on_validation_epoch_end(self):
        if not self._validation:
            return
        logits = torch.cat([item[0] for item in self._validation])
        label = torch.cat([item[1] for item in self._validation])
        self._validation.clear()

        probability = F.softmax(logits, dim=-1)
        top = probability.max(dim=-1)
        correct = (top.indices == label).double()
        top3 = logits.topk(3, dim=-1).indices.eq(label.unsqueeze(-1)).any(dim=-1).double()
        matrix = self.loss_fn.matrix.cpu()

        # Sums and counts rather than means: ranks hold different numbers of residues, so
        # averaging per-rank means (what sync_dist does) would weight a rank with few
        # residues as heavily as one with many, and the checkpoint monitor reads this.
        totals = torch.stack([
            torch.tensor(float(label.numel()), dtype=torch.float64),
            correct.sum(),
            top3.sum(),
            (probability * matrix[label]).sum(dim=-1).double().sum(),
            matrix[label, top.indices].double().sum(),
            F.cross_entropy(logits, label, reduction="sum").double(),
            top.values.double().sum(),
            # The ceiling a perfect prediction could reach on *these* labels, which is
            # below the matrix diagonal's mean because states are not equally frequent.
            matrix.diagonal()[label].double().sum(),
        ])
        bins = calibration_bins(top.values, correct)
        # Sum in float64 on CPU (where the logits already are), then cross ranks in
        # float32: MPS has no float64, and the gathered vectors are tiny either way.
        totals = totals.to(device=self.device, dtype=torch.float32)
        bins = bins.to(device=self.device, dtype=torch.float32)

        totals = self.all_gather(totals).reshape(-1, totals.numel()).sum(dim=0)
        bins = self.all_gather(bins).reshape(-1, *bins.shape).sum(dim=0)
        count = totals[0].clamp_min(1.0)

        self.log_dict({
            "accuracy": totals[1] / count,
            "top3_accuracy": totals[2] / count,
            "expected_score": totals[3] / count,
            "argmax_score": totals[4] / count,
            "nll": totals[5] / count,
            "confidence": totals[6] / count,
            "expected_score_ceiling": totals[7] / count,
            "ece": expected_calibration_error(bins),
        }, sync_dist=False)

    def configure_optimizers(self):
        parameters = self.cfg.training_parameters if self.cfg is not None else None
        optimizer = optim.AdamW(
            params=self.parameters(),
            lr=self.learning_rate,
            weight_decay=getattr(parameters, "weight_decay", 0.0) or 0.0,
        )
        if parameters is None:
            return optimizer
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            warmup_epochs=parameters.warmup_epochs,
            max_epochs=parameters.epochs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": parameters.lr_interval,
                "frequency": parameters.lr_frequency,
            },
        }
