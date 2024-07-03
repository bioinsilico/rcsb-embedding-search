import lightning as L
from torch import nn, optim, cat
from torcheval.metrics.functional import binary_auprc, binary_auroc

from lightning_module.lightning_base import LitStructureBase
from src.lightning_module.utils import get_cosine_schedule_with_warmup


class LitStructureGeoGraph(LitStructureBase):

    def __init__(
            self,
            nn_model,
            learning_rate=1e-6,
            params=None
    ):
        super().__init__(nn_model, learning_rate, params)

    def training_step(self, batch, batch_idx):
        g_i, g_j, z = batch
        z_pred = self.model(g_i, g_j)
        self.z.append(z)
        self.z_pred.append(z_pred)
        return nn.functional.mse_loss(z_pred, z)

    def validation_step(self, batch, batch_idx):
        g_i, g_j, z = batch
        self.z.append(z)
        self.z_pred.append(self.model(g_i,g_j))

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
                'interval': self.params.lr_interval,
                'frequency': self.params.lr_frequency
            }
        }
