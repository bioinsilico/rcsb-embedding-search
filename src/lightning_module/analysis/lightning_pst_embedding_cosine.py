
import lightning as L
from torch import sum

class LitStructurePstEmbeddingCosine(L.LightningModule):

    def __init__(self):
        super().__init__()

    def predict_step(self, batch, batch_idx):
        (x, dom_x), (y, dom_y) = batch
        return sum(x * y, dim=1)
