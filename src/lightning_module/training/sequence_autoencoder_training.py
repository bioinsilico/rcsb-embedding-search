import pathlib
import lightning as L
import torch
import yaml
from omegaconf import OmegaConf
from torch import nn, optim, cat
from torcheval.metrics.functional import binary_auprc, binary_auroc

from config.schema_config import TrainingConfig, Strategy, LrInterval
from lightning_module.utils import get_cosine_schedule_with_warmup
from networks.sequence_autoencoder import AA_PAD_IDX


def _token_accuracy(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Fraction of non-padding positions the decoder reconstructs exactly."""
    mask = tokens != AA_PAD_IDX
    total = mask.sum()
    if total == 0:
        return torch.zeros((), device=tokens.device)
    correct = (logits.argmax(dim=-1) == tokens) & mask
    return correct.sum().float() / total


def _mean_offdiag_cosine(latent: torch.Tensor) -> torch.Tensor:
    """Mean cosine between distinct members of a batch of unit-norm latents."""
    n = latent.size(0)
    if n < 2:
        return torch.zeros((), device=latent.device)
    sim = latent @ latent.t()
    return (sim.sum() - sim.diagonal().sum()) / (n * (n - 1))


def _average_ranks(x: torch.Tensor) -> torch.Tensor:
    """1-based ranks with ties averaged.

    Fraction scores snap onto a coarse grid, so ties are the rule rather than
    the exception; breaking them arbitrarily would understate the correlation.
    """
    _, inverse = torch.unique(x, return_inverse=True)
    counts = torch.bincount(inverse).float()
    starts = torch.cumsum(counts, 0) - counts
    return (starts + (counts + 1) / 2)[inverse]


def _spearman(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Spearman rank correlation — how well predictions order the targets."""
    if pred.numel() < 2:
        return torch.zeros((), device=pred.device)
    a = _average_ranks(pred.detach().float())
    b = _average_ranks(target.detach().float())
    a = a - a.mean()
    b = b - b.mean()
    denominator = a.norm() * b.norm()
    if denominator == 0:
        return torch.zeros((), device=pred.device)
    return (a @ b) / denominator


class LitSequenceAutoencoderTraining(L.LightningModule):
    """Lightning module for the protein sequence autoencoder.

    Training uses paired protein sequences with known sequence identity scores.
    Two losses are combined:

    1. **Reconstruction loss** (cross-entropy): ensures the autoencoder can
       faithfully reconstruct each input sequence from its latent vector.
    2. **Similarity alignment loss** (MSE): forces the cosine similarity
       between the two latent vectors in a pair to match the ground-truth
       sequence identity score.

    The total loss is::

        loss = reconstruction_weight * CE + similarity_weight * MSE

    Expected batch format from the dataloader::

        (tokens_i, tokens_j, seq_identity_score)

    where ``tokens_i`` and ``tokens_j`` are (B, L) long tensors of padded
    amino acid token indices and ``seq_identity_score`` is a (B,) float
    tensor in [0, 1].
    """

    PR_AUC_METRIC_NAME = 'pr_auc'
    ROC_AUC_METRIC_NAME = 'roc_auc'
    TRAIN_LOSS_METRIC_NAME = 'train_loss'
    VALIDATION_LOSS_METRIC_NAME = 'validation_loss'

    # ``pr_auc``/``roc_auc`` above are computed against the raw scores, which are
    # continuous when a fraction score is in use.  They still rank models, but
    # their scale is not interpretable (a perfect predictor does not reach 1.0).
    # The ``_BIN`` variants threshold the target first, which is what the
    # torcheval binary metrics expect, and are the better checkpoint monitor.
    PR_AUC_BIN_METRIC_NAME = 'pr_auc_bin'
    ROC_AUC_BIN_METRIC_NAME = 'roc_auc_bin'
    BINARY_THR = 0.7

    def __init__(
        self,
        nn_model: nn.Module,
        learning_rate: float = 1e-4,
        reconstruction_weight: float = 1.0,
        similarity_weight: float = 1.0,
        cfg: TrainingConfig = None,
    ):
        super().__init__()
        self.model = nn_model
        self.learning_rate = learning_rate
        self.reconstruction_weight = reconstruction_weight
        self.similarity_weight = similarity_weight
        self.cfg = cfg

        self.z = None
        self.z_pred = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_fit_start(self):
        self.z = torch.empty(0).to(self.device)
        self.z_pred = torch.empty(0).to(self.device)
        if self.cfg is not None and hasattr(self.logger.experiment, 'add_text'):
            yaml.add_representer(pathlib.PurePosixPath, lambda d, v: d.represent_str(str(v)))
            yaml.add_representer(pathlib.PosixPath, lambda d, v: d.represent_str(str(v)))
            yaml.add_representer(pathlib.WindowsPath, lambda d, v: d.represent_str(str(v)))
            yaml.add_representer(Strategy, lambda d, v: d.represent_str(str(v)))
            yaml.add_representer(LrInterval, lambda d, v: d.represent_str(str(v)))

            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            config = OmegaConf.to_container(self.cfg, resolve=True)
            config['model_parameters'] = f"{trainable_params}/{total_params}"
            config_text = yaml.dump(config)
            self.logger.experiment.add_text(
                "Config",
                config_text
            )
        if hasattr(self.logger.experiment, 'add_graph'):
            try:
                dummy = torch.ones(1, 16, dtype=torch.long, device=self.device)
                with torch.no_grad():
                    self.logger.experiment.add_graph(self.model, dummy)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Shared step logic
    # ------------------------------------------------------------------

    def _step(self, batch):
        tokens_i, tokens_j, score = batch

        logits_i, latent_i = self.model(tokens_i)
        logits_j, latent_j = self.model(tokens_j)

        # Reconstruction loss (cross-entropy, ignoring padding)
        recon_loss_i = nn.functional.cross_entropy(
            logits_i.transpose(1, 2), tokens_i, ignore_index=AA_PAD_IDX,
        )
        recon_loss_j = nn.functional.cross_entropy(
            logits_j.transpose(1, 2), tokens_j, ignore_index=AA_PAD_IDX,
        )
        recon_loss = (recon_loss_i + recon_loss_j) / 2.0

        # Similarity alignment loss
        cos_sim = nn.functional.cosine_similarity(latent_i, latent_j)
        sim_loss = nn.functional.mse_loss(cos_sim, score)

        loss = self.reconstruction_weight * recon_loss + self.similarity_weight * sim_loss

        with torch.no_grad():
            stats = {
                # Cross-entropy in nats says little about whether the decoder
                # actually recovers residues; this does.
                'recon_acc': (
                    _token_accuracy(logits_i, tokens_i)
                    + _token_accuracy(logits_j, tokens_j)
                ) / 2.0,
                # Mean similarity between unrelated members of the batch.  Rises
                # towards 1 if the encoder collapses every sequence onto one
                # point — a failure the paired cosine alone can hide.
                'batch_cos': _mean_offdiag_cosine(latent_i),
            }
        return loss, cos_sim, score, recon_loss, sim_loss, stats

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        loss, cos_sim, score, recon_loss, sim_loss, stats = self._step(batch)
        self.z = cat((self.z, score), dim=0)
        self.z_pred = cat((self.z_pred, cos_sim), dim=0)

        self.log('recon_loss', recon_loss, prog_bar=True)
        self.log('sim_loss', sim_loss, prog_bar=True)
        # Relative pull of the two objectives; with equal weights the
        # reconstruction term typically supplies most of the gradient.
        self.log('loss_ratio', recon_loss / sim_loss.clamp(min=1e-8))
        for name, value in stats.items():
            self.log(name, value)
        return loss

    def on_train_epoch_end(self):
        self._log_metrics(self.TRAIN_LOSS_METRIC_NAME)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        loss, cos_sim, score, recon_loss, sim_loss, stats = self._step(batch)
        self.z = cat((self.z, score), dim=0)
        self.z_pred = cat((self.z_pred, cos_sim), dim=0)

        self.log('val_recon_loss', recon_loss, prog_bar=True)
        self.log('val_sim_loss', sim_loss, prog_bar=True)
        for name, value in stats.items():
            self.log(f'val_{name}', value)

    def on_validation_epoch_start(self):
        self._log_metrics(self.TRAIN_LOSS_METRIC_NAME)

    def on_validation_epoch_end(self):
        z = self.z
        z_pred = self.z_pred
        if len(z) > 0:
            pr_auc = binary_auprc(z_pred, z)
            self.log(self.PR_AUC_METRIC_NAME, pr_auc, sync_dist=True)
            if self.device.type == 'mps':
                roc_auc = binary_auroc(z_pred.to('cpu'), z.to('cpu'))
            else:
                roc_auc = binary_auroc(z_pred, z)
            self.log(self.ROC_AUC_METRIC_NAME, roc_auc, sync_dist=True)

            # Same metrics on a thresholded target — interpretable, and the
            # checkpoint monitor.  A degenerate epoch (every target on one side
            # of the threshold) has no meaningful value, but it must still be
            # logged: ModelCheckpoint errors out on a missing monitor, and under
            # DDP a metric logged by only some ranks desynchronises the
            # sync_dist reduction.  Zero is the right sentinel for mode='max' —
            # an epoch we cannot score is never selected as the best one.
            z_bin = (z >= self.BINARY_THR).float()
            positives = int(z_bin.sum())
            degenerate = not (0 < positives < len(z_bin))
            zero = torch.zeros((), device=self.device)
            if degenerate:
                pr_bin, roc_bin = zero, zero
            else:
                pr_bin = binary_auprc(z_pred, z_bin)
                if self.device.type == 'mps':
                    # torcheval returns float64 here, a dtype MPS cannot hold —
                    # leave it on the CPU as float32 and let Lightning place it.
                    roc_bin = binary_auroc(z_pred.to('cpu'), z_bin.to('cpu')).float()
                else:
                    roc_bin = binary_auroc(z_pred, z_bin)
            self.log(self.PR_AUC_BIN_METRIC_NAME, pr_bin, sync_dist=True)
            self.log(self.ROC_AUC_BIN_METRIC_NAME, roc_bin, sync_dist=True)
            self.log('val_frac_positive', z_bin.mean(), sync_dist=True)
        self._log_metrics(self.VALIDATION_LOSS_METRIC_NAME)

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            params=self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.cfg.training_parameters.weight_decay if self.cfg and self.cfg.training_parameters.weight_decay is not None else 0,
        )
        if self.cfg:
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                warmup_epochs=self.cfg.training_parameters.warmup_epochs,
                max_epochs=self.cfg.training_parameters.epochs,
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': lr_scheduler,
                    'interval': self.cfg.training_parameters.lr_interval,
                    'frequency': self.cfg.training_parameters.lr_frequency,
                },
            }
        return optimizer

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_metrics(self, step_name: str):
        if self.z is None or len(self.z) == 0:
            return
        loss = nn.functional.mse_loss(self.z_pred, self.z)
        self.log(step_name, loss, sync_dist=True)
        prefix = 'train' if step_name == self.TRAIN_LOSS_METRIC_NAME else 'val'
        self._log_stream_metrics(prefix, self.z_pred, self.z)
        self.z = torch.empty(0).to(self.device)
        self.z_pred = torch.empty(0).to(self.device)

    def _log_stream_metrics(self, prefix: str, pred: torch.Tensor, target: torch.Tensor):
        """Log distribution and rank-agreement metrics over a whole epoch.

        MSE alone cannot separate a model that has learned the relationship from
        one that has collapsed onto the mean of the targets — predicting a
        constant can beat a genuinely weak model on MSE.  The spread of the
        predictions and their rank agreement with the target distinguish them.
        """
        self.log(f'{prefix}_cos_mean', pred.mean(), sync_dist=True)
        self.log(f'{prefix}_cos_std', pred.std(), sync_dist=True)
        # What the model is actually being shown, after the dataset's balancing.
        self.log(f'{prefix}_target_mean', target.mean(), sync_dist=True)
        self.log(f'{prefix}_target_std', target.std(), sync_dist=True)
        self.log(f'{prefix}_target_frac_high',
                 (target >= self.BINARY_THR).float().mean(), sync_dist=True)
        self.log(f'{prefix}_spearman', _spearman(pred, target), sync_dist=True)
