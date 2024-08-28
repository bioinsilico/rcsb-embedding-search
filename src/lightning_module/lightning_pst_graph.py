from torch import nn
from lightning_module.lightning_core import LitStructureCore


class LitStructurePstGraph(LitStructureCore):

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
        (x, x_mask), (y, y_mask), z = batch
        self.z.append(z)
        self.z_pred.append(
            self.model.validation_forward(x, x_mask, y, y_mask)
        )

    def predict_step(self, batch, batch_idx):
        (x, x_mask), z = batch
        return self.model.embedding_pooling(x, x_mask), z
