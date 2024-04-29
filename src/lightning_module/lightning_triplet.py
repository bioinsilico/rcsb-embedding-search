import lightning as L
import torch
from torch import nn, optim, cat
from torcheval.metrics.functional import binary_auprc, binary_auroc

from src.lightning_module.utils import get_cosine_schedule_with_warmup


class LitTripletEmbedding(L.LightningModule):
    PR_AUC_METRIC_NAME = 'pr_auc'
    TRIPLET_LOSS_MARGIN = 0.5

    def __init__(
            self,
            net,
            learning_rate=1e-6,
            params=None
    ):
        super().__init__()
        self.model = net
        self.learning_rate = learning_rate
        self.params = params
        self.margin = self.TRIPLET_LOSS_MARGIN
        self.z = []
        self.z_pred = []

    def on_fit_start(self):
        if self.params and hasattr(self.logger.experiment, 'add_text'):
            summary = L.pytorch.utilities.model_summary.LayerSummary(self)
            self.logger.experiment.add_text(
                "Param Description",
                self.params.text_params({
                    "Number-parameters": f"{summary.num_parameters // 1e6}M"
                })
            )

    def training_step(self, batch, batch_idx):
        (a, a_mask), (p, p_mask), (n, n_mask) = batch
        p_pred = self.model(a, a_mask, p, p_mask)
        self.z.append(1.)
        self.z_pred.append(p_pred)
        n_pred = self.model(a, a_mask, n, n_mask)
        self.z.append(0.)
        self.z_pred.append(n_pred)
        losses = torch.max(p_pred - n_pred + self.margin, torch.zeros_like(p_pred))
        return losses.mean()

    def on_train_epoch_end(self):
        z = cat(self.z, dim=0)
        z_pred = cat(self.z_pred, dim=0)
        loss = nn.functional.mse_loss(z_pred, z)
        self.log("mse_loss", loss, sync_dist=True)
        self.reset_z()

    def validation_step(self, batch, batch_idx):
        (x, x_mask), (y, y_mask), z = batch
        self.z.append(z)
        self.z_pred.append(self.model(x, x_mask, y, y_mask))

    def on_validation_epoch_end(self):
        z = cat(self.z, dim=0)
        z_pred = cat(self.z_pred, dim=0)
        pr_auc = binary_auprc(z_pred, z)
        self.log(self.PR_AUC_METRIC_NAME, pr_auc, sync_dist=True)
        roc_auc = binary_auroc(z_pred, z)
        self.log("roc_auc", roc_auc, sync_dist=True)
        self.reset_z()

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            params=self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.params.weight_decay if self.params.weight_decay else 0
        )
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            warmup_epochs=self.params.warmup_epochs,
            max_epochs=self.params.epochs
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': "step",
                'frequency': self.params.test_every_n_steps
            }
        }

    def reset_z(self):
        self.z = []
        self.z_pred = []