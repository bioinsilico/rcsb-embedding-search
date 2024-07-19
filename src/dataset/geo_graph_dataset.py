import numpy as np
import torch
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
import polars as pl

from dataset.utils.tm_score_weight import binary_score, binary_weights, fraction_score, tm_score_weights
from dataset.utils.tools import load_class_pairs, load_tensors
from networks.transformer_graph_nn import TransformerGraphEmbeddingCosine, BiTransformerGraphEmbeddingCosine

d_type = np.float32


class GeoGraphDataset(Dataset):

    BINARY_THR = 0.7

    def __init__(self, tm_score_file, geo_graph_path, score_method=None, weighting_method=None):
        super().__init__()
        self.graphs = pl.DataFrame()
        self.class_pairs = pl.DataFrame()
        self.score_method = binary_score(self.BINARY_THR) if not score_method else score_method
        self.weighting_method = binary_weights(self.BINARY_THR) if not weighting_method else weighting_method
        self.__exec(tm_score_file, geo_graph_path)

    def __exec(self, tm_score_file, geo_graph_path):
        self.load_class_pairs(tm_score_file)
        self.load_graphs(geo_graph_path, tm_score_file)

    def load_class_pairs(self, tm_score_file):
        self.class_pairs = load_class_pairs(tm_score_file)
        print(f"Total pairs: {len(self.class_pairs)}")

    def load_graphs(self, geo_graph_path, tm_score_file):
        self.graphs = load_tensors(geo_graph_path, tm_score_file)
        print(f"Total graphs: {len(self.graphs)}")

    def weights(self):
        return self.weighting_method([dp['score'] for dp in self.class_pairs.iter_rows(named=True)])

    def len(self):
        return len(self.class_pairs)

    def get(self, idx):
        return (
            torch.load(self.graphs.row(
                by_predicate=(pl.col("domain") == self.class_pairs.row(idx, named=True)['domain_i']),
                named=True
            )['embedding']),
            torch.load(self.graphs.row(
                by_predicate=(pl.col("domain") == self.class_pairs.row(idx, named=True)['domain_j']),
                named=True
            )['embedding']),
            torch.from_numpy(
                np.array(self.score_method(self.class_pairs.row(idx, named=True)['score']), dtype=d_type)
            )
        )


if __name__ == '__main__':
    dataset = GeoGraphDataset(
        "/Users/joan/data/cath_23M/cath_23M.csv",
        "/Users/joan/cs-data/structure-embedding/cath_S40/graph-geo",
        score_method=fraction_score,
        weighting_method=tm_score_weights(5)
    )
    dataset.weights()
    dataloader = DataLoader(
        dataset,
        batch_size=8
    )
    model = BiTransformerGraphEmbeddingCosine()
    for g_i, g_j, z in dataloader:
        p = model(g_i, g_j)
        print(p.shape, z.shape, z)