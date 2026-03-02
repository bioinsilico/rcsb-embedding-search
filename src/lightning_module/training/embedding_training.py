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

        # Compute main loss
        loss = nn.functional.mse_loss(z_pred, z)

        # Add load balancing loss if MoE model
        if hasattr(self.model, 'get_load_balance_loss'):
            load_balance_loss = self.model.get_load_balance_loss()
            loss = loss + load_balance_loss
            self.log('train_load_balance_loss', load_balance_loss, on_step=True, on_epoch=True, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        (x, x_mask), (y, y_mask), z = batch
        z_pred = self.model(x, x_mask, y, y_mask)
        self.z = cat((self.z, z), dim=0)
        self.z_pred = cat((self.z_pred, z_pred), dim=0)
