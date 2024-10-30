import torch.nn as nn
from torch import cat
from collections import OrderedDict

from torch_geometric.utils import unbatch

from dataset.pst.pst import load_pst_model
from dataset.utils.tools import collate_seq_embeddings
from networks.layers import ResBlock
from networks.transformer_nn import TransformerEmbeddingCosine


class TransformerPstEmbeddingCosine(nn.Module):
    dropout = 0.1

    def __init__(
            self,
            pst_model_path,
            input_features=640,
            dim_feedforward=1280,
            hidden_layer=640,
            nhead=10,
            num_layers=6,
            res_block_layers=0
    ):
        super().__init__()

        self.pst_model = load_pst_model({
            'model_path': pst_model_path
        })

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
                ('linear', nn.Linear(hidden_layer, hidden_layer)),
                ('activation', nn.ReLU())
            ])
            self.embedding = nn.Sequential(res_block)

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

    def embedding_pooling(self, x, x_mask):
        return self.embedding(self.transformer(x, src_key_padding_mask=x_mask).sum(dim=1))

    def graph_pooling(self, g_i):
        x_batch = []
        x_mask = []
        for g in g_i:
            batch, mask = self.graph_transformer_forward(g)
            x_batch.append(batch)
            x_mask.append(mask)
        return self.embedding_pooling(cat(x_batch, dim=1), cat(x_mask, dim=1))

    def validation_forward(self, x, x_mask, y, y_mask):
        return nn.functional.cosine_similarity(
            self.embedding_pooling(x, x_mask),
            self.embedding_pooling(y, y_mask)
        )

    def forward(self, g_i, g_j):
        x_batch, x_mask = self.graph_transformer_forward(g_i)
        y_batch, y_mask = self.graph_transformer_forward(g_j)
        if x_batch.shape[0] != y_batch.shape[0]:
            raise Exception("Batch shape mismatch")
        return nn.functional.cosine_similarity(
            self.embedding_pooling(x_batch, x_mask),
            self.embedding_pooling(y_batch, y_mask)
        )

    def get_weights(self):
        return [(name, param) for name, param in self.embedding.named_parameters()]


class TransformerPstMultiBranchEmbeddingCosine(nn.Module):
    dropout = 0.1

    def __init__(
            self,
            pst_model_path,
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

        self.pst_model = load_pst_model({
            'model_path': pst_model_path
        })
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

    def embedding_pooling(self, x_batch, x_mask):
        return self.embedding(cat(
            [transformer.embedding_pooling(x_batch, x_mask) for transformer in self.transformer_paths],
            dim=1
        ))

    def validation_forward(self, x, x_mask, y, y_mask):
        return nn.functional.cosine_similarity(
            self.embedding_pooling(x, x_mask),
            self.embedding_pooling(y, y_mask)
        )

    def forward_graph_ch_list(self, g_i, g_j):
        x_batch = []
        x_mask = []
        for g in g_i:
            batch, mask = self.graph_transformer_forward(g)
            x_batch.append(batch)
            x_mask.append(mask)
        y_batch = []
        y_mask = []
        for g in g_j:
            batch, mask = self.graph_transformer_forward(g)
            y_batch.append(batch)
            y_mask.append(mask)

        return nn.functional.cosine_similarity(
            self.embedding_pooling(cat(x_batch, dim=1), cat(x_mask, dim=1)),
            self.embedding_pooling(cat(y_batch, dim=1), cat(y_mask, dim=1))
        )

    def graph_pooling(self, g_i):
        x_batch = []
        x_mask = []
        for g in g_i:
            batch, mask = self.graph_transformer_forward(g)
            x_batch.append(batch)
            x_mask.append(mask)
        return self.embedding_pooling(cat(x_batch, dim=1), cat(x_mask, dim=1))

    def forward(self, g_i, g_j):
        x_batch, x_mask = self.graph_transformer_forward(g_i)
        y_batch, y_mask = self.graph_transformer_forward(g_j)

        return nn.functional.cosine_similarity(
            self.embedding_pooling(x_batch, x_mask),
            self.embedding_pooling(y_batch, y_mask)
        )


class TransformerPstMetaEmbeddingCosine(nn.Module):
    dropout = 0.1

    def __init__(
            self,
            pst_model_path,
            input_features=640,
            dim_feedforward=1280,
            hidden_layer=640,
            nhead=10,
            num_layers=6,
            num_path=2,
            path_block_layers=6
    ):
        super().__init__()

        self.pst_model = load_pst_model({
            'model_path': pst_model_path
        })
        self.transformer_paths = nn.ModuleList([nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=input_features,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        ), num_layers=num_layers) for _ in range(0, num_path)])

        res_in_dim = num_path * input_features
        res_out_dim = num_path * hidden_layer

        self.transformer_meta_embedding = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=res_in_dim,
            nhead=nhead,
            dim_feedforward=res_out_dim,
            dropout=self.dropout,
            batch_first=True
        ), num_layers=num_layers)

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

    def embedding_pooling(self, x_batch, x_mask):
        return self.embedding(self.transformer_meta_embedding(cat(
            [transformer(x_batch, src_key_padding_mask=x_mask) for transformer in self.transformer_paths],
            dim=2
        ), src_key_padding_mask=x_mask).sum(dim=1))

    def validation_forward(self, x, x_mask, y, y_mask):
        return nn.functional.cosine_similarity(
            self.embedding_pooling(x, x_mask),
            self.embedding_pooling(y, y_mask)
        )

    def graph_pooling(self, g_i):
        x_batch = []
        x_mask = []
        for g in g_i:
            batch, mask = self.graph_transformer_forward(g)
            x_batch.append(batch)
            x_mask.append(mask)
        return self.embedding_pooling(cat(x_batch, dim=1), cat(x_mask, dim=1))

    def forward(self, g_i, g_j):
        x_batch, x_mask = self.graph_transformer_forward(g_i)
        y_batch, y_mask = self.graph_transformer_forward(g_j)

        return nn.functional.cosine_similarity(
            self.embedding_pooling(x_batch, x_mask),
            self.embedding_pooling(y_batch, y_mask)
        )


class TransformerPstCosine(nn.Module):
    dropout = 0.1

    def __init__(self, pst_model, input_features=640, dim_feedforward=1280, hidden_layer=640, nhead=10, num_layers=6):
        super().__init__()
        self.pst_model = pst_model
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
