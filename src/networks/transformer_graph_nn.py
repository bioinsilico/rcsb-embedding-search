from collections import OrderedDict

import torch.nn as nn
from torch_geometric.nn import global_add_pool
from torch_geometric.nn.conv import TransformerConv


class TransformerGraphEmbeddingCosine(nn.Module):

    dropout = 0

    def __init__(self, node_dim=8, edge_dim=1, out_dim=640, num_layers=6):
        super().__init__()

        self.graph_transformer = TransformerConv(node_dim, out_dim, edge_dim=edge_dim)
        self.graph_transformer_layers = []

        for i in range(1, num_layers):
            self.graph_transformer_layers.append(
                TransformerConv(out_dim, out_dim, edge_dim=edge_dim)
            )

        self.embedding = nn.Sequential(OrderedDict([
            ('norm', nn.LayerNorm(out_dim)),
            ('dropout', nn.Dropout(p=self.dropout)),
            ('linear', nn.Linear(out_dim, out_dim)),
            ('activation', nn.ReLU())
        ]))

    def graph_transformer_forward(self, g):
        x = self.graph_transformer(g.x, g.edge_index, g.edge_attr)
        for tr in self.graph_transformer_layers:
            x = tr(x, g.edge_index, g.edge_attr)
        return global_add_pool(x, g.batch)

    def forward(self, g_i, g_j):
        return nn.functional.cosine_similarity(
            self.embedding(self.graph_transformer_forward(g_i)),
            self.embedding(self.graph_transformer_forward(g_j))
        )
