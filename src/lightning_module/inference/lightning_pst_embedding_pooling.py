import lightning as L


class LitStructurePstEmbeddingPooling(L.LightningModule):

    def __init__(
            self,
            nn_model
    ):
        super().__init__()
        self.model = nn_model

    def predict_step(self, batch, batch_idx):
        (x, x_mask), dom_id = batch
        return self.model.embedding_pooling(x, x_mask), dom_id


class LitStructurePstOrNullEmbeddingPooling(L.LightningModule):

    def __init__(
            self,
            nn_model
    ):
        super().__init__()
        self.model = nn_model

    def predict_step(self, batch, batch_idx):
        if batch[0] is None:
            return None
        (x, x_mask), dom_id = batch
        return self.model.embedding_pooling(x, x_mask), dom_id


class LitStructurePstEmbeddingPoolingFromGraph(L.LightningModule):

    def __init__(
            self,
            nn_model
    ):
        super().__init__()
        self.model = nn_model

    def predict_step(self, batch, batch_idx):
        graph, graph_id = batch
        if len(graph) == 0:
            return None
        return self.model.graph_pooling(graph), graph_id
