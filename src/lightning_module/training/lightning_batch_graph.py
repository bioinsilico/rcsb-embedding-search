from torch import nn, cat
from lightning_module.lightning_core import LitStructureCore


class LitStructureBatchGraph(LitStructureCore):

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
        self.z = cat((self.z, z), dim=0)
        self.z_pred = cat((self.z_pred, z_pred), dim=0)
        return nn.functional.mse_loss(z_pred, z)

    def validation_step(self, batch, batch_idx):
        g_i, g_j, z = batch
        self.z = cat((self.z, z), dim=0)
        self.z_pred = cat((self.z_pred, self.model(g_i, g_j)), dim=0)
