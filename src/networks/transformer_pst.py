
import torch.nn as nn
from collections import OrderedDict

from torch_geometric.utils import unbatch

from dataset.utils.tools import collate_seq_embeddings


class TransformerPstEmbeddingCosine(nn.Module):
    dropout = 0.1

    def __init__(self, pst_model, input_features=640, dim_feedforward=1280, hidden_layer=640, nhead=10, num_layers=6):
        super().__init__()
        self.pst_model = pst_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_features,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.embedding = nn.Sequential(OrderedDict([
            ('norm', nn.LayerNorm(input_features)),
            ('dropout', nn.Dropout(p=self.dropout)),
            ('linear', nn.Linear(input_features, hidden_layer)),
            ('activation', nn.ReLU())
        ]))

    def graph_transformer_forward(self, graph):
        self.pst_model.eval()
        embedding = self.pst_model(
            graph,
            return_repr=True,
            aggr=None
        )
        return collate_seq_embeddings(unbatch(
            embedding[graph.idx_mask],
            graph.batch[graph.idx_mask]
        ))

    def validation_forward(self, x, x_mask, y, y_mask):
        return nn.functional.cosine_similarity(
            self.embedding(self.transformer(x, src_key_padding_mask=x_mask).sum(dim=1)),
            self.embedding(self.transformer(y, src_key_padding_mask=y_mask).sum(dim=1))
        )

    def forward(self, g_i, g_j):
        x_batch, x_mask = self.graph_transformer_forward(g_i)
        y_batch, y_mask = self.graph_transformer_forward(g_j)
        if x_batch.shape[0] != y_batch.shape[0]:
            print("ERROR!!!")
        return nn.functional.cosine_similarity(
            self.embedding(self.transformer(x_batch, src_key_padding_mask=x_mask).sum(dim=1)),
            self.embedding(self.transformer(y_batch, src_key_padding_mask=y_mask).sum(dim=1))
        )

    def get_weights(self):
        return [(name, param) for name, param in self.embedding.named_parameters()]


class TransformerPstCosine(nn.Module):
    dropout = 0.1

    def __init__(self, pst_model, input_features=640, dim_feedforward=1280, hidden_layer=640, nhead=10, num_layers=6):
        super().__init__()
        self.pst_model = pst_model

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_features,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.embedding = nn.Sequential(OrderedDict([
            ('norm', nn.LayerNorm(input_features)),
            ('dropout', nn.Dropout(p=self.dropout)),
            ('linear', nn.Linear(input_features, hidden_layer)),
            ('activation', nn.ReLU())
        ]))

    def graph_transformer_forward(self, graph):
        self.pst_model.eval()
        embedding = self.pst_model(
            graph,
            return_repr=True,
            aggr=None
        )
        return collate_seq_embeddings(unbatch(
            embedding[graph.idx_mask],
            graph.batch[graph.idx_mask]
        ))

    def validation_forward(self, x, x_mask, y, y_mask):
        return nn.functional.cosine_similarity(
            self.embedding(x.sum(dim=1)),
            self.embedding(y.sum(dim=1))
        )

    def forward(self, g_i, g_j):
        x_batch, x_mask = self.graph_transformer_forward(g_i)
        y_batch, y_mask = self.graph_transformer_forward(g_j)
        if x_batch.shape[0] != y_batch.shape[0]:
            print("ERROR!!!")
        return nn.functional.cosine_similarity(
            self.embedding(x_batch.sum(dim=1)),
            self.embedding(y_batch.sum(dim=1))
        )

    def get_weights(self):
        return [(name, param) for name, param in self.embedding.named_parameters()]