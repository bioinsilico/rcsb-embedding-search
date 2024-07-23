import torch.nn as nn
from collections import OrderedDict


class TransformerDualEmbeddingCosine(nn.Module):
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
        self.embedding_max = nn.Sequential(OrderedDict([
            ('norm', nn.LayerNorm(input_features)),
            ('dropout', nn.Dropout(p=self.dropout)),
            ('linear', nn.Linear(input_features, hidden_layer)),
            ('activation', nn.ReLU())
        ]))
        self.embedding_min = nn.Sequential(OrderedDict([
            ('norm', nn.LayerNorm(input_features)),
            ('dropout', nn.Dropout(p=self.dropout)),
            ('linear', nn.Linear(input_features, hidden_layer)),
            ('activation', nn.ReLU())
        ]))

    def embedding(self, x, x_mask):
        return self.transformer(x, src_key_padding_mask=x_mask).sum(dim=1)

    def forward(self, x, x_mask, y, y_mask):
        x_emb = self.embedding(x, x_mask)
        y_emb = self.embedding(y, y_mask)
        return nn.functional.cosine_similarity(
            self.embedding_max(x_emb),
            self.embedding_max(y_emb)
        ), nn.functional.cosine_similarity(
            self.embedding_min(x_emb),
            self.embedding_min(y_emb)
        )

    def get_weights(self):
        return [(name, param) for name, param in self.embedding.named_parameters()]
