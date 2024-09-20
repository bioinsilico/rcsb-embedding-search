import lightning as L
from torch import sum


class LitStructurePstEmbeddingCosine(L.LightningModule):

    def __init__(self):
        super().__init__()

    def predict_step(self, batch, batch_idx):
        (x, dom_x), (y, dom_y) = batch
        return sum(x * y, dim=1)


class LitStructurePstEmbeddingScore(L.LightningModule):

    def __init__(
            self,
            nn_model
    ):
        super().__init__()
        self.model = nn_model

    def predict_step(self, batch, batch_idx):
        g_i, g_j, z = batch
        print(self.model(g_i, g_j))
        return self.model(g_i, g_j)
