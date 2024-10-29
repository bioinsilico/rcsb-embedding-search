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
        score = self.model.forward_graph_ch_list(g_i, g_j)
        print("\n", score)
        return score
