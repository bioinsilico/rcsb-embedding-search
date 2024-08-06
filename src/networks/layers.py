import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv


class GatedResidualConnection(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GatedResidualConnection, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.gate = nn.Linear(in_channels, 1)

    def forward(self, x, out):
        residual = self.linear(x)
        gate = torch.sigmoid(self.gate(x))
        return gate * out + (1 - gate) * residual


class FeedForwardLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super(FeedForwardLayer, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.gated_residual = GatedResidualConnection(in_channels, out_channels)

    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)
        if self.dropout:
            out = self.dropout(out)
        out = self.gated_residual(x, out)
        return out


class GraphMultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_channels, out_channels, edge_channels=0, heads=1):
        super(GraphMultiHeadAttentionLayer, self).__init__()
        self.conv = TransformerConv(in_channels, out_channels, edge_dim=edge_channels, heads=heads)
        self.norm = nn.LayerNorm(out_channels)
        self.relu = nn.ReLU()
        self.gated_residual = GatedResidualConnection(in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr=None):
        out = self.conv(x, edge_index, edge_attr)
        out = self.norm(out)
        out = self.relu(out)
        out = self.gated_residual(x, out)
        return out


class GraphTransformerLayer(nn.Module):
    def __init__(self, in_channels, out_channels, edge_channels=1, hidden_channels=1280, heads=1):
        super(GraphTransformerLayer, self).__init__()
        self.head = GraphMultiHeadAttentionLayer(in_channels, hidden_channels, edge_channels, heads)
        self.feed_forward = FeedForwardLayer(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        x = self.head(x, edge_index, edge_attr)
        x = self.feed_forward(x)
        return x


class UniMP(nn.Module):
    def __init__(self, in_channels=4, out_channels=640, edge_channels=8, hidden_channels=1280, num_layers=6, heads=1):
        super(UniMP, self).__init__()
        self.layers = nn.ModuleList()
        in_heads = 1
        for _ in range(num_layers):
            self.layers.append(
                GraphTransformerLayer(in_channels, out_channels, edge_channels, hidden_channels, in_heads)
            )
            in_channels = out_channels
            in_heads = heads

    def forward(self, x, edge_index, edge_attr=None):
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.residual = nn.Identity()
        self.block = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(p=dropout),
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(in_dim),
            nn.Dropout(p=dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.activate = nn.ReLU()

    def forward(self, x):
        residual = self.residual(x)
        x = self.block(x)
        x = self.activate(x + residual)
        return x
