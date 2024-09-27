from torch import sum, cat

from lightning_module.training.lightning_batch_graph import LitStructureBatchGraph


class LitStructurePstGraph(LitStructureBatchGraph):

    def __init__(
            self,
            nn_model,
            learning_rate=1e-6,
            params=None
    ):
        super().__init__(nn_model, learning_rate, params)

    def validation_step(self, batch, batch_idx):
        x, y, z = batch
        self.z = cat((self.z, z), dim=0)
        self.z_pred = cat((self.z_pred, sum(x*y, dim=1)), dim=0)
