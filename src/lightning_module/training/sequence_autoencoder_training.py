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


class LitSequenceAutoencoderTraining(L.LightningModule):
    """Lightning module for the protein sequence autoencoder.

    Training uses paired protein sequences with known sequence identity scores.
    Three losses are combined:

    1. **Reconstruction loss** (cross-entropy): ensures the autoencoder can
       faithfully reconstruct each input sequence from its latent vector.
    2. **Similarity alignment loss** (MSE): forces the cosine similarity
       between the two latent vectors in a pair to match the ground-truth
       sequence identity score.
    3. **Length prediction loss** (MSE on log-length): trains the length head
       to predict sequence length from the latent, enabling length-free decoding
       at inference time.

    The total loss is::

        loss = reconstruction_weight * CE + similarity_weight * MSE_sim + length_weight * MSE_len

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
        length_weight: float = 1.0,
        cfg: TrainingConfig = None,
    ):
        super().__init__()
        self.model = nn_model
        self.learning_rate = learning_rate
        self.reconstruction_weight = reconstruction_weight
        self.similarity_weight = similarity_weight
        self.length_weight = length_weight
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

        # Length prediction loss — MSE in log-space against true non-padding lengths
        true_len_i = (tokens_i != AA_PAD_IDX).sum(dim=1).float()
        true_len_j = (tokens_j != AA_PAD_IDX).sum(dim=1).float()
        len_loss_i = nn.functional.mse_loss(
            self.model.predict_length(latent_i), torch.log(true_len_i)
        )
        len_loss_j = nn.functional.mse_loss(
            self.model.predict_length(latent_j), torch.log(true_len_j)
        )
        len_loss = (len_loss_i + len_loss_j) / 2.0

        loss = (
            self.reconstruction_weight * recon_loss
            + self.similarity_weight * sim_loss
            + self.length_weight * len_loss
        )
        return loss, cos_sim, score, recon_loss, sim_loss, len_loss

    def _val_step(self, batch):
        """Validation step using predicted length for decoding.

        Unlike :meth:`_step`, the decoder is evaluated with the length
        produced by :meth:`predict_length` rather than the true input length,
        mirroring inference. Because ``decode`` takes a single ``seq_len``
        per call, reconstruction is computed per-sample in a Python loop.
        """
        tokens_i, tokens_j, score = batch

        latent_i = self.model.encode(tokens_i)
        latent_j = self.model.encode(tokens_j)

        cos_sim = nn.functional.cosine_similarity(latent_i, latent_j)
        sim_loss = nn.functional.mse_loss(cos_sim, score)

        true_len_i = (tokens_i != AA_PAD_IDX).sum(dim=1).float()
        true_len_j = (tokens_j != AA_PAD_IDX).sum(dim=1).float()

        pred_log_len_i = self.model.predict_length(latent_i)
        pred_log_len_j = self.model.predict_length(latent_j)
        len_loss_i = nn.functional.mse_loss(pred_log_len_i, torch.log(true_len_i))
        len_loss_j = nn.functional.mse_loss(pred_log_len_j, torch.log(true_len_j))
        len_loss = (len_loss_i + len_loss_j) / 2.0

        recon_loss_i = self._per_sample_recon_loss(latent_i, tokens_i, true_len_i, pred_log_len_i)
        recon_loss_j = self._per_sample_recon_loss(latent_j, tokens_j, true_len_j, pred_log_len_j)
        recon_loss = (recon_loss_i + recon_loss_j) / 2.0

        loss = (
            self.reconstruction_weight * recon_loss
            + self.similarity_weight * sim_loss
            + self.length_weight * len_loss
        )
        return loss, cos_sim, score, recon_loss, sim_loss, len_loss

    def _per_sample_recon_loss(self, latent, tokens, true_len, pred_log_len):
        """CE reconstruction loss with per-sample predicted decoding length.

        Decodes each sample at its predicted length, then compares the first
        ``min(pred_len, true_len)`` positions against the true tokens. Samples
        where that overlap is zero contribute nothing.
        """
        max_positions = self.model.decoder_queries.num_embeddings
        pred_len = pred_log_len.exp().round().long().clamp(min=1, max=max_positions)
        true_len_long = true_len.long()

        losses = []
        for b in range(latent.size(0)):
            n = int(min(pred_len[b].item(), true_len_long[b].item()))
            if n == 0:
                continue
            pl = int(pred_len[b].item())
            logits_b = self.model.decode(latent[b : b + 1], pl)  # (1, pl, V)
            losses.append(
                nn.functional.cross_entropy(logits_b[0, :n], tokens[b, :n])
            )
        if not losses:
            return torch.zeros((), device=latent.device)
        return torch.stack(losses).mean()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        loss, cos_sim, score, recon_loss, sim_loss, len_loss = self._step(batch)
        self.z = cat((self.z, score), dim=0)
        self.z_pred = cat((self.z_pred, cos_sim), dim=0)

        self.log('recon_loss', recon_loss, prog_bar=True)
        self.log('sim_loss', sim_loss, prog_bar=True)
        self.log('len_loss', len_loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self._log_metrics(self.TRAIN_LOSS_METRIC_NAME)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch, batch_idx):
        # loss, cos_sim, score, recon_loss, sim_loss, len_loss = self._step(batch)
        loss, cos_sim, score, recon_loss, sim_loss, len_loss = self._val_step(batch)
        self.z = cat((self.z, score), dim=0)
        self.z_pred = cat((self.z_pred, cos_sim), dim=0)

        self.log('val_recon_loss', recon_loss, prog_bar=True)
        self.log('val_sim_loss', sim_loss, prog_bar=True)
        self.log('val_len_loss', len_loss, prog_bar=True)

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
