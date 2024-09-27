import lightning as L
import torch
from omegaconf import OmegaConf
from torch import nn, optim, cat
from torcheval.metrics.functional import binary_auprc, binary_auroc
from config.schema_config import TrainingConfig
from src.lightning_module.utils import get_cosine_schedule_with_warmup


class LitStructureCore(L.LightningModule):
    PR_AUC_METRIC_NAME = 'pr_auc'

    def __init__(
            self,
            nn_model: nn.Module,
            learning_rate: float = 1e-6,
            cfg: TrainingConfig = None
    ):
        super().__init__()
        self.model = nn_model
        self.learning_rate = learning_rate
        self.cfg = cfg
        self.z = torch.empty(1).to(self.device)
        self.z_pred = torch.empty(1).to(self.device)

    def on_fit_start(self):
        if self.cfg is not None and hasattr(self.logger.experiment, 'add_text'):
            self.logger.experiment.add_text(
                "Config",
                OmegaConf.to_yaml(self.cfg)
            )

    def on_train_epoch_start(self):
        self.reset_z()

    def on_train_epoch_end(self):
        self.train_loss()

    def on_validation_epoch_start(self):
        self.train_loss()
        self.reset_z()

    def on_validation_epoch_end(self):
        z = self.z
        z_pred = self.z_pred
        pr_auc = binary_auprc(z_pred, z)
        self.log(self.PR_AUC_METRIC_NAME, pr_auc, sync_dist=True)
        if self.device.type == 'mps':
            roc_auc = binary_auroc(z_pred.to('cpu'), z.to('cpu'))
        else:
            roc_auc = binary_auroc(z_pred, z)
        self.log("roc_auc", roc_auc, sync_dist=True)
        loss = nn.functional.mse_loss(z_pred, z)
        self.log("validation_loss", loss, sync_dist=True)
        self.reset_z()

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            params=self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.cfg.training_parameters.weight_decay if self.cfg.training_parameters.weight_decay is not None else 0
        )
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            warmup_epochs=self.cfg.training_parameters.warmup_epochs,
            max_epochs=self.cfg.training_parameters.epochs
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': self.cfg.training_parameters.lr_interval,
                'frequency': self.cfg.training_parameters.lr_frequency
            }
        }

    def train_loss(self):
        if len(self.z) == 0:
            return
        z = self.z
        z_pred = self.z_pred
        loss = nn.functional.mse_loss(z_pred, z)
        self.log("train_loss", loss, sync_dist=True)

    def reset_z(self):
        self.z = torch.empty(1).to(self.device)
        self.z_pred = torch.empty(1).to(self.device)
