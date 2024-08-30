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
        (x, x_mask), (y, y_mask), z = batch
        self.z.append(z)
        self.z_pred.append(
            self.model.validation_forward(x, x_mask, y, y_mask)
        )
