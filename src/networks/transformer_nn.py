import torch.nn as nn
from torch import cat

from collections import OrderedDict

from networks.layers import ResBlock

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


class TransformerResBlockEmbeddingCosine(nn.Module):
    dropout = 0.1

    def __init__(self, input_features=1280, dim_feedforward=2048, hidden_layer=1280, nhead=8, num_layers=6):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_features,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.embedding = nn.Sequential(OrderedDict([
            ('block1', ResBlock(input_features, hidden_layer, self.dropout)),
            ('block2', ResBlock(input_features, hidden_layer, self.dropout)),
            ('block3', ResBlock(input_features, hidden_layer, self.dropout)),
            ('dropout', nn.Dropout(p=self.dropout)),
            ('linear', nn.Linear(input_features, hidden_layer)),
            ('activation', nn.ReLU())
        ]))

    def embedding_pooling(self, x, x_mask):
        return self.embedding(self.transformer(x, src_key_padding_mask=x_mask).sum(dim=1))

    def forward(self, x, x_mask, y, y_mask):
        return nn.functional.cosine_similarity(
            self.embedding_pooling(x, x_mask),
            self.embedding_pooling(y, y_mask)
        )

    def get_weights(self):
        return [(name, param) for name, param in self.embedding.named_parameters()]


class TransformerEmbeddingMultiBranchEmbeddingCosine(nn.Module):
    dropout = 0.1

    def __init__(
            self,
            input_features=640,
            dim_feedforward=1280,
            hidden_layer=640,
            nhead=10,
            num_layers=6,
            res_block_layers=0,
            num_path=2,
            path_block_layers=6
    ):
        super().__init__()

        self.transformer_paths = nn.ModuleList([TransformerEmbeddingCosine(
            input_features,
            dim_feedforward,
            hidden_layer,
            nhead,
            num_layers,
            res_block_layers
        ) for _ in range(0, num_path)])

        res_in_dim = num_path * input_features
        res_out_dim = num_path * hidden_layer
        if path_block_layers == 0:
            self.embedding = nn.Sequential(OrderedDict([
                ('norm', nn.LayerNorm(res_in_dim)),
                ('dropout', nn.Dropout(p=self.dropout)),
                ('linear', nn.Linear(res_in_dim, res_out_dim)),
                ('activation', nn.ReLU())
            ]))
        else:
            res_block = OrderedDict([(
                f'block{i}',
                ResBlock(res_in_dim, res_out_dim, self.dropout)
            ) for i in range(path_block_layers)])
            res_block.update([
                ('dropout', nn.Dropout(p=self.dropout)),
                ('linear', nn.Linear(res_in_dim, res_out_dim)),
                ('activation', nn.ReLU())
            ])
            self.embedding = nn.Sequential(res_block)

    def embedding_pooling(self, x_batch, x_mask):
        return self.embedding(cat(
            [transformer.embedding_pooling(x_batch, x_mask) for transformer in self.transformer_paths],
            dim=1
        ))

    def forward(self,  x, x_mask, y, y_mask):
        return nn.functional.cosine_similarity(
            self.embedding_pooling(x, x_mask),
            self.embedding_pooling(y, y_mask)
        )


class TransformerEmbedding(nn.Module):
    dropout = 0.1

    def __init__(self, input_features=1280, dim_feedforward=2048, hidden_layer=1280, nhead=8, num_layers=6):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_features,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.embedding = nn.Sequential(OrderedDict([
            ('dropout', nn.Dropout(p=self.dropout)),
            ('linear', nn.Linear(input_features, hidden_layer)),
            ('activation', nn.ReLU())
        ]))

    def forward(self, x, x_mask):
        return self.embedding(self.transformer(x, src_key_padding_mask=x_mask).sum(dim=1))

    def get_weights(self):
        return [(name, param) for name, param in self.embedding.named_parameters()]

