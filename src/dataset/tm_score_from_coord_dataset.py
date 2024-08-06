import argparse
import os

import numpy as np
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
import polars as pl
import torch

from dataset.pst.pst import load_pst_model
from dataset.utils.biopython_getter import get_coords_from_pdb_file
from dataset.utils.coords_augmenter import NullAugmenter, AbstractAugmenter, SelfAugmenterRandomFraction
from dataset.utils.custom_weighted_random_sampler import CustomWeightedRandomSampler
from dataset.utils.embedding_builder import graph_builder
from dataset.utils.tm_score_weight import binary_score, binary_weights, fraction_score, tm_score_weights
from dataset.utils.tools import load_class_pairs_with_self_comparison
from networks.transformer_pst import TransformerPstEmbeddingCosine

d_type = np.float32


class TmScoreFromCoordDataset(Dataset):
    BINARY_THR = 0.7

    def __init__(
            self,
            tm_score_file,
            coords_path,
            ext="ent",
            score_method=None,
            weighting_method=None,
            coords_augmenter: AbstractAugmenter = NullAugmenter(),
            num_workers=1
    ):
        super().__init__()
        self.coords = pl.DataFrame()
        self.class_pairs = pl.DataFrame()
        self.score_method = binary_score(self.BINARY_THR) if not score_method else score_method
        self.weighting_method = binary_weights(self.BINARY_THR) if not weighting_method else weighting_method
        self.coords_augmenter = coords_augmenter
        self.nun_workers = num_workers if num_workers > 1 else 1
        self.__exec(tm_score_file, coords_path, f".{ext}" if len(ext) > 0 else "")

    def __exec(self, tm_score_file, embedding_path, ext):
        self.load_class_pairs(tm_score_file)
        self.load_coords(tm_score_file, embedding_path, ext)

    def load_class_pairs(self, tm_score_file):
        self.class_pairs = load_class_pairs_with_self_comparison(tm_score_file)
        print(f"Total pairs: {len(self.class_pairs)}")

    def load_coords(self, tm_score_file, coords_path, ext):
        self.coords = pl.DataFrame(
            data=[
                (lambda coords: (
                    dom_id,
                    coords['cas'],
                    coords['seq']
                ))(
                    get_coords_from_pdb_file(os.path.join(coords_path, f"{dom_id}{ext}"))
                )
                for dom_id in list(set(
                    [(lambda row: row.strip().split(",")[0])(row) for row in open(tm_score_file)] +
                    [(lambda row: row.strip().split(",")[1])(row) for row in open(tm_score_file)]
                ))
            ],
            orient="row",
            schema=['domain', 'cas', 'seq']
        )
        print(f"Total structures: {len(self.coords)}")

    def weights(self):
        return self.weighting_method([dp['score'] for dp in self.class_pairs.rows(named=True)])

    def __build_graph(self, dom_id, add_change=lambda *abc: abc):
        coords_i = self.coords.row(
            by_predicate=(pl.col("domain") == dom_id),
            named=True
        )
        out = graph_builder(
            [{'cas': ch[0], 'seq': ch[1]} for ch in zip(coords_i['cas'], coords_i['seq'])],
            add_change=add_change,
            num_workers=self.nun_workers
        )
        return out

    def len(self):
        return len(self.class_pairs)

    def get(self, idx):
        dom_i = self.class_pairs.row(idx, named=True)['domain_i']
        dom_j = self.class_pairs.row(idx, named=True)['domain_j']
        return (
            self.__build_graph(
                dom_i,
                add_change=self.coords_augmenter.add_change(dom_i, dom_j, "dom_i")
            ),
            self.__build_graph(
                self.class_pairs.row(idx, named=True)['domain_j'],
                add_change=self.coords_augmenter.add_change(dom_i, dom_j, "dom_j")
            ),
            torch.from_numpy(
                np.array(self.score_method(self.class_pairs.row(idx, named=True)['score']), dtype=d_type)
            )
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tm_score_file', type=str, required=True)
    parser.add_argument('--pdb_path', type=str, required=True)
    parser.add_argument('--pst_model_path', type=str, required=True)
    args = parser.parse_args()

    dataset = TmScoreFromCoordDataset(
        args.tm_score_file,
        args.pdb_path,
        ext="",
        weighting_method=tm_score_weights(5),
        score_method=fraction_score,
        coords_augmenter=SelfAugmenterRandomFraction(
            fraction=0.1,
            max_remove=10
        )
    )
    weights = dataset.weights()
    sampler = CustomWeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        sampler=sampler
    )
    for (g_i), (g_j), z in dataloader:
        print(z)

    pst_model = load_pst_model({
        'model_path': args.pst_model_path,
        'device': 'cpu'
    })
    model = TransformerPstEmbeddingCosine(pst_model)
    for (g_i), (g_j), z in dataloader:
        z_pred = model(g_i, g_j)
        print(z_pred.shape, z.shape)
