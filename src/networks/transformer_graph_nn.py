from collections import OrderedDict

import torch.nn as nn
from torch_geometric.nn import global_add_pool

from dataset.utils.tools import collate_seq_embeddings
from networks.layers import UniMP
from torch_geometric.utils import unbatch


class TransformerGraphEmbeddingCosine(nn.Module):

    dropout = 0

    def __init__(self, node_dim=4, edge_dim=8, out_dim=640, num_layers=6):
        super().__init__()

        self.graph_transformer = UniMP(
            in_channels=node_dim,
            out_channels=out_dim,
            edge_channels=edge_dim,
            num_layers=num_layers
        )

        self.embedding = nn.Sequential(OrderedDict([
            ('norm', nn.LayerNorm(out_dim)),
            ('dropout', nn.Dropout(p=self.dropout)),
            ('linear', nn.Linear(out_dim, out_dim)),
            ('activation', nn.ReLU())
        ]))

    def graph_transformer_forward(self, g):
        x = self.graph_transformer(g.x, g.edge_index, g.edge_attr)
        return global_add_pool(x, g.batch)

    def forward(self, g_i, g_j):
        return nn.functional.cosine_similarity(
            self.embedding(self.graph_transformer_forward(g_i)),
            self.embedding(self.graph_transformer_forward(g_j))
        )


class BiTransformerGraphEmbeddingCosine(nn.Module):

    dropout = 0

    def __init__(self, node_dim=4, edge_dim=8, out_dim=640, num_layers=6, nhead=10, dim_feedforward=1280):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=out_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.graph_transformer = UniMP(
            in_channels=node_dim,
            out_channels=out_dim,
            edge_channels=edge_dim,
            num_layers=num_layers
        )
        self.embedding = nn.Sequential(OrderedDict([
            ('norm', nn.LayerNorm(out_dim)),
            ('dropout', nn.Dropout(p=self.dropout)),
            ('linear', nn.Linear(out_dim, out_dim)),
            ('activation', nn.ReLU())
        ]))

    def graph_transformer_forward(self, g):
        x = self.graph_transformer(g.x, g.edge_index, g.edge_attr)
        x_batch, x_mask = collate_seq_embeddings(unbatch(x, g.batch))
        return self.transformer(x_batch, src_key_padding_mask=x_mask).sum(dim=1)

    def forward(self, g_i, g_j):
        return nn.functional.cosine_similarity(
            self.embedding(self.graph_transformer_forward(g_i)),
            self.embedding(self.graph_transformer_forward(g_j))
        )
