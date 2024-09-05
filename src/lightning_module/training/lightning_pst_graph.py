import torch.nn as nn

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
        self.z.append(z)
        self.z_pred.append(
            nn.functional.cosine_similarity(x, y)
        )
