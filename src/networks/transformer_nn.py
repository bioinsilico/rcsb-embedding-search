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


class TransformerEmbeddingCosineConfigurable(nn.Module):
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
            res_block = OrderedDict()
            in_dim = input_features
            block_id = 0  # needed for unique block names

            for out_dim, count in res_block_layers:
                for i in range(count):
                    res_block[f'block{block_id}'] = ResBlock(in_dim, out_dim, self.dropout)
                    in_dim = out_dim  # update input for next block
                    block_id += 1

            res_block.update([
                ('dropout', nn.Dropout(p=self.dropout)),
                ('linear', nn.Linear(in_dim, in_dim)),
                ('activation', nn.ReLU())
            ])

            self.embedding = nn.Sequential(res_block)
            self.output_features = in_dim  

        self.print_architecture()

    def embedding_pooling(self, x, x_mask):
        return self.embedding(self.transformer(x, src_key_padding_mask=x_mask).sum(dim=1))

    def forward(self, x, x_mask, y, y_mask):
        return nn.functional.cosine_similarity(
            self.embedding_pooling(x, x_mask),
            self.embedding_pooling(y, y_mask)
        )

    def get_weights(self):
        return [(name, param) for name, param in self.embedding.named_parameters()]

    def print_architecture(self):
        print("\nEmbedding Architecture:")
        prev_dim = None

        for name, module in self.embedding.named_modules():
            if isinstance(module, ResBlock):
                try:
                    in_dim = module.block[1].in_features
                    out_dim = module.block[-1].out_features
                    res_type = "Identity" if isinstance(module.residual, nn.Identity) else "Linear"
                    print(f"  {name}: ResBlock({in_dim} → {out_dim}), Residual: {res_type}")
                    prev_dim = out_dim
                except Exception as e:
                    print(f"  {name}: ResBlock (error reading dims: {e})")
            elif isinstance(module, nn.Linear) and name == "linear":
                print(f"  {name}: Final Linear({prev_dim} → {module.out_features})")
                prev_dim = module.out_features
            elif isinstance(module, nn.Dropout):
                print(f"  {name}: Dropout(p={module.p})")
            elif isinstance(module, nn.ReLU):
                print(f"  {name}: ReLU")
            elif isinstance(module, nn.LayerNorm):
                print(f"  {name}: LayerNorm({module.normalized_shape})")

        print()

