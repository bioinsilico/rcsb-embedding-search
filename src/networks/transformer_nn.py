import torch.nn as nn
from collections import OrderedDict

from networks.layers import ResBlock, MoETransformerEncoderLayer


class LinearEmbeddingCosine(nn.Module):

    def __init__(self, input_features=1280, hidden_layer=1280):
        super().__init__()
        self.embedding = nn.Sequential(OrderedDict([
            ('norm', nn.LayerNorm(input_features)),
            ('dropout', nn.Dropout(p=0.0)),
            ('linear', nn.Linear(input_features, hidden_layer)),
            ('activation', nn.ReLU())
        ]))

    def forward(self, x, x_mask, y, y_mask):
        return nn.functional.cosine_similarity(
            self.embedding(x.sum(dim=1)),
            self.embedding(y.sum(dim=1))
        )

    def get_weights(self):
        return [(name, param) for name, param in self.embedding.named_parameters()]


class TransformerEmbeddingCosine(nn.Module):
    dropout = 0.1

    def __init__(
        self,
        input_features=640,
        dim_feedforward=1280,
        hidden_layer=640,
        nhead=10,
        num_layers=6,
        res_block_layers=0
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_features,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        if res_block_layers == 0:
            self.embedding = nn.Sequential(OrderedDict([
                ('norm', nn.LayerNorm(input_features)),
                ('dropout', nn.Dropout(p=self.dropout)),
                ('linear', nn.Linear(input_features, hidden_layer)),
                ('activation', nn.ReLU())
            ]))
        else:
            res_block = OrderedDict([(
                f'block{i}',
                ResBlock(input_features, hidden_layer, self.dropout)
            ) for i in range(res_block_layers)])
            res_block.update([
                ('dropout', nn.Dropout(p=self.dropout)),
                ('linear', nn.Linear(input_features, hidden_layer)),
                ('activation', nn.ReLU())
            ])
            self.embedding = nn.Sequential(res_block)

    def embedding_pooling(self, x, x_mask):
        return self.embedding(self.transformer(x, src_key_padding_mask=x_mask).sum(dim=1))

    def forward(self, x, x_mask, y, y_mask):
        return nn.functional.cosine_similarity(
            self.embedding_pooling(x, x_mask),
            self.embedding_pooling(y, y_mask)
        )

    def get_weights(self):
        return [(name, param) for name, param in self.embedding.named_parameters()]


class MoETransformerEmbeddingCosine(nn.Module):
    """TransformerEmbeddingCosine with Mixture of Experts in the transformer layers."""
    dropout = 0.1

    def __init__(
        self,
        input_features=640,
        dim_feedforward=1280,
        hidden_layer=640,
        nhead=10,
        num_layers=6,
        num_experts=8,
        top_k=2,
        load_balance_weight=0.01,
        res_block_layers=0
    ):
        super().__init__()

        # Build transformer encoder with MoE layers
        encoder_layers = []
        for _ in range(num_layers):
            encoder_layers.append(
                MoETransformerEncoderLayer(
                    d_model=input_features,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    num_experts=num_experts,
                    top_k=top_k,
                    dropout=self.dropout,
                    load_balance_weight=load_balance_weight,
                    batch_first=True
                )
            )
        self.transformer = nn.ModuleList(encoder_layers)

        # Build embedding layer
        if res_block_layers == 0:
            self.embedding = nn.Sequential(OrderedDict([
                ('norm', nn.LayerNorm(input_features)),
                ('dropout', nn.Dropout(p=self.dropout)),
                ('linear', nn.Linear(input_features, hidden_layer)),
                ('activation', nn.ReLU())
            ]))
        else:
            res_block = OrderedDict([(
                f'block{i}',
                ResBlock(input_features, hidden_layer, self.dropout)
            ) for i in range(res_block_layers)])
            res_block.update([
                ('dropout', nn.Dropout(p=self.dropout)),
                ('linear', nn.Linear(input_features, hidden_layer)),
                ('activation', nn.ReLU())
            ])
            self.embedding = nn.Sequential(res_block)

    def embedding_pooling(self, x, x_mask):
        # Pass through MoE transformer layers
        for layer in self.transformer:
            x = layer(x, src_key_padding_mask=x_mask)
        # Pool and embed
        return self.embedding(x.sum(dim=1))

    def forward(self, x, x_mask, y, y_mask):
        return nn.functional.cosine_similarity(
            self.embedding_pooling(x, x_mask),
            self.embedding_pooling(y, y_mask)
        )

    def get_load_balance_loss(self):
        """Aggregate load balancing loss from all MoE layers."""
        total_loss = 0.0
        for layer in self.transformer:
            total_loss += layer.moe.get_load_balance_loss()
        return total_loss

    def get_weights(self):
        return [(name, param) for name, param in self.embedding.named_parameters()]
