from torch import nn, cat, sum
from lightning_module.training.lightning_batch_embedding import LitStructureBatchEmbedding


class LitStructureBatchEmbeddingStaticValidation(LitStructureBatchEmbedding):

    def __init__(
            self,
            nn_model,
            learning_rate=1e-6,
            params=None
    ):
        super().__init__(nn_model, learning_rate, params)

    def validation_step(self, batch, batch_idx):
        (x, x_mask), (y, y_mask), z = batch
        z_pred = self.model(x, x_mask, y, y_mask)
        self.z = cat((self.z, z), dim=0)
        self.z_pred = cat((self.z_pred, z_pred), dim=0)
