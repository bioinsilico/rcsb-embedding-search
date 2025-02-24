from torch import nn, cat
from lightning_module.lightning_core import LitStructureCore


class LitEmbeddingTraining(LitStructureCore):

    def __init__(
            self,
            nn_model,
            learning_rate=1e-6,
            params=None
    ):
        super().__init__(nn_model, learning_rate, params)

    def training_step(self, batch, batch_idx):
        (x, x_mask), (y, y_mask), z = batch
        z_pred = self.model(x, x_mask, y, y_mask)
        self.z = cat((self.z, z), dim=0)
        self.z_pred = cat((self.z_pred, z_pred), dim=0)
        return nn.functional.mse_loss(z_pred, z)

    def validation_step(self, batch, batch_idx):
        (x, x_mask), (y, y_mask), z = batch
        z_pred = self.model(x, x_mask, y, y_mask)
        self.z = cat((self.z, z), dim=0)
        self.z_pred = cat((self.z_pred, z_pred), dim=0)
