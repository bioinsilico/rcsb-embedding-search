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


class MaskedTransformerEmbeddingCosine(nn.Module):
    """Transformer twin network with per-residue spatial attention masks.

    Same architecture as :class:`TransformerEmbeddingCosine` but the forward
    pass accepts an additional ``(B, L, L)`` boolean attention mask per protein
    that restricts each residue to attend only to its spatial neighbors (as
    produced by :func:`collate_masked_fn`).

    The mask is expanded to ``(B * nhead, L, L)`` before being passed to
    ``nn.TransformerEncoder`` as the ``mask`` argument.
    """
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
        self.nhead = nhead
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

    def embedding_pooling(self, x, x_pad_mask, x_attn_mask):
        # Expand (B, L, L) → (B * nhead, L, L) for multi-head attention
        attn_mask = x_attn_mask.repeat_interleave(self.nhead, dim=0)
        return self.embedding(
            self.transformer(x, mask=attn_mask, src_key_padding_mask=x_pad_mask).sum(dim=1)
        )

    def forward(self, x, x_pad_mask, x_attn_mask, y, y_pad_mask, y_attn_mask):
        return nn.functional.cosine_similarity(
            self.embedding_pooling(x, x_pad_mask, x_attn_mask),
            self.embedding_pooling(y, y_pad_mask, y_attn_mask)
        )

    def get_weights(self):
        return [(name, param) for name, param in self.embedding.named_parameters()]
