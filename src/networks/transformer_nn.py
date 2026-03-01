import torch
import torch.nn as nn
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


class TransformerDecoderEmbeddingCosine(nn.Module):
    dropout = 0.1

    def __init__(
        self,
        input_features=640,
        dim_feedforward=1280,
        hidden_layer=640,
        nhead=10,
        num_encoder_layers=6,
        num_decoder_layers=2,
        res_block_layers=0
    ):
        super().__init__()
        # Encoder to process input sequences
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_features,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Decoder for aggregation
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_features,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Learnable query token for aggregation
        self.query_token = nn.Parameter(torch.randn(1, 1, input_features))

        # Embedding layers
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
        # Encode the input sequence
        encoded = self.transformer_encoder(x, src_key_padding_mask=x_mask)

        # Expand query token to batch size
        batch_size = x.size(0)
        query = self.query_token.expand(batch_size, -1, -1)

        # Use decoder to aggregate encoded sequence with query token
        aggregated = self.transformer_decoder(
            query,
            encoded,
            memory_key_padding_mask=x_mask
        )

        # Take the single output token and pass through embedding
        return self.embedding(aggregated.squeeze(1))

    def forward(self, x, x_mask, y, y_mask):
        return nn.functional.cosine_similarity(
            self.embedding_pooling(x, x_mask),
            self.embedding_pooling(y, y_mask)
        )

    def get_weights(self):
        return [(name, param) for name, param in self.embedding.named_parameters()]
