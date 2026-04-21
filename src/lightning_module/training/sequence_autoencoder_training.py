import pathlib
import lightning as L
import torch
import yaml
from omegaconf import OmegaConf
from torch import nn, optim, cat
from torcheval.metrics.functional import binary_auprc, binary_auroc

from config.schema_config import TrainingConfig, Strategy, LrInterval
from lightning_module.utils import get_cosine_schedule_with_warmup
from networks.sequence_autoencoder import AA_PAD_IDX, AA_TOKENS


class LitSequenceAutoencoderTraining(L.LightningModule):
    """Lightning module for the protein sequence autoencoder.

    Training uses paired protein sequences with known sequence identity scores.
    Two losses are combined:

    1. **Reconstruction loss** (cross-entropy): ensures the autoencoder can
       faithfully reconstruct each input sequence from its latent vector.
       Sequences include a trailing ``<eos>`` token so the model also learns
       where each sequence ends.
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
        self.recon_correct = 0
        self.recon_total = 0
        if self.cfg is not None and hasattr(self.logger.experiment, 'add_text'):
            yaml.add_representer(pathlib.PurePosixPath, lambda d, v: d.represent_str(str(v)))
            yaml.add_representer(pathlib.PosixPath, lambda d, v: d.represent_str(str(v)))
            yaml.add_representer(pathlib.WindowsPath, lambda d, v: d.represent_str(str(v)))
            yaml.add_representer(Strategy, lambda d, v: d.represent_str(str(v)))
            yaml.add_representer(LrInterval, lambda d, v: d.represent_str(str(v)))

            self.logger.experiment.add_text(
                "Config",
                yaml.dump(OmegaConf.to_container(self.cfg, resolve=True))
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
        return loss, cos_sim, score, recon_loss, sim_loss

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        loss, cos_sim, score, recon_loss, sim_loss = self._step(batch)
        self.z = cat((self.z, score), dim=0)
        self.z_pred = cat((self.z_pred, cos_sim), dim=0)

        self.log('recon_loss', recon_loss, prog_bar=True)
        self.log('sim_loss', sim_loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self._log_metrics(self.TRAIN_LOSS_METRIC_NAME)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        loss, cos_sim, score, recon_loss, sim_loss = self._step(batch)
        self.z = cat((self.z, score), dim=0)
        self.z_pred = cat((self.z_pred, cos_sim), dim=0)

        self.log('val_recon_loss', recon_loss, prog_bar=True)
        self.log('val_sim_loss', sim_loss, prog_bar=True)

        # Autoregressive reconstruction accuracy
        tokens_i, tokens_j, _ = batch
        latent_i = self.model.encode(tokens_i)
        latent_j = self.model.encode(tokens_j)
        for latent, tokens in [(latent_i, tokens_i), (latent_j, tokens_j)]:
            predicted = self.model.decode_sequence(latent)
            for pred, gt_row in zip(predicted, tokens):
                # Ground-truth amino acids: strip <eos> and <pad>
                gt = gt_row[(gt_row != AA_PAD_IDX) & (gt_row != AA_TOKENS['<eos>'])]
                length = min(len(pred), len(gt))
                self.recon_correct += (pred[:length] == gt[:length]).sum().item()
                self.recon_total += len(gt)

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
        if self.recon_total > 0:
            self.log('val_recon_accuracy', self.recon_correct / self.recon_total, sync_dist=True)
            self.recon_correct = 0
            self.recon_total = 0
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
        self.z = torch.empty(0).to(self.device)
        self.z_pred = torch.empty(0).to(self.device)
