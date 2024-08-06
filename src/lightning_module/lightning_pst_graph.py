from torch import nn, mean, stack

from lightning_module.lightning_base import LitStructureBase


class LitStructurePstGraph(LitStructureBase):

    def __init__(
            self,
            nn_model,
            learning_rate=1e-6,
            unit_norm_scale=1e-2,
            params=None
    ):
        super().__init__(nn_model, learning_rate, params)
        self.unit_norm_scale = unit_norm_scale

    def training_step(self, batch, batch_idx):
        g_i, g_j, z = batch
        z_pred, x_norm, y_norm = self.model(g_i, g_j)
        mean_norm = mean((stack((x_norm, y_norm), dim=0) - 1)**2)
        self.z.append(z)
        self.z_pred.append(z_pred)
        return nn.functional.mse_loss(z_pred, z) + self.unit_norm_scale * mean_norm

    def validation_step(self, batch, batch_idx):
        (x, x_mask), (y, y_mask), z = batch
        self.z.append(z)
        self.z_pred.append(
            self.model.validation_forward(x, x_mask, y, y_mask)
        )
