from __future__ import annotations

import argparse
import logging
import os

import numpy as np
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
import pandas as pd
import torch

from dataset.pst.pst import load_pst_model
from dataset.utils.biopython_getter import get_coords_from_pdb_file, get_coords_from_cif_file
from dataset.utils.coords_augmenter import NullAugmenter, AbstractAugmenter, SelfAugmenterRandomFraction
from dataset.utils.custom_weighted_random_sampler import CustomWeightedRandomSampler
from dataset.utils.embedding_builder import graph_builder
from dataset.utils.tm_score_weight import binary_score, binary_weights, fraction_score, tm_score_weights
from dataset.utils.tools import load_class_pairs_with_self_comparison, load_class_pairs
from networks.transformer_pst import TransformerPstEmbeddingCosine

d_type = np.float32
logger = logging.getLogger(__name__)


class TmScoreFromCoordDataset(Dataset):
    BINARY_THR = 0.7

    def __init__(
            self,
            tm_score_file,
            coords_path,
            ext="pdb",
            score_method=None,
            weighting_method=None,
            coords_augmenter: AbstractAugmenter | None = NullAugmenter(),
            num_workers=1,
            include_self_comparison=False
    ):
        super().__init__()
        self.coords = pd.DataFrame()
        self.class_pairs = pd.DataFrame()
        self.score_method = binary_score(self.BINARY_THR) if not score_method else score_method
        self.weighting_method = binary_weights(self.BINARY_THR) if not weighting_method else weighting_method
        self.coords_augmenter = NullAugmenter() if coords_augmenter is None else coords_augmenter
        self.nun_workers = num_workers if num_workers > 1 else 1
        self.include_self_comparison = include_self_comparison
        self.__exec(tm_score_file, coords_path, f".{ext}" if len(ext) > 0 else "")

    def __exec(self, tm_score_file, embedding_path, ext):
        self.load_class_pairs(tm_score_file)
        self.load_coords(embedding_path, ext)

    def load_class_pairs(self, tm_score_file):
        logger.info(f"Loading pairs from file {tm_score_file}")
        if self.include_self_comparison:
            self.class_pairs = load_class_pairs_with_self_comparison(tm_score_file)
        else:
            self.class_pairs = load_class_pairs(tm_score_file)
        logger.info(f"Total pairs: {len(self.class_pairs)}")

    def load_coords(self, coords_path, ext):
        logger.info(f"Loading coords from path {coords_path}")
        self.coords = pd.DataFrame(
            data=[
                (lambda coords: (
                    dom_id,
                    coords['cas'],
                    coords['seq']
                ))(
                    get_coords_from_pdb_file(os.path.join(coords_path, f"{dom_id}{ext}")) if ext == ".pdb" or ext == ".ent" else
                    get_coords_from_cif_file(os.path.join(coords_path, f"{dom_id}{ext}"))
                )
                for dom_id in pd.concat([
                    self.class_pairs["domain_i"], self.class_pairs["domain_j"]
                ]).unique()
            ],
            columns=['domain', 'cas', 'seq']
        )
        logger.info(f"Total structures: {len(self.coords)}")

    def weights(self):
        return self.weighting_method(self.class_pairs['score'].to_numpy())

    def __build_graph(self, dom_id, add_change=lambda *abc: abc):
        coords_i = self.coords.loc[self.coords['domain'] == dom_id]
        if len(coords_i) != 1:
            raise Exception(f'Data error: found {len(coords_i)} rows for {dom_id}')
        out = graph_builder(
            [{'cas': ch[0], 'seq': ch[1]} for ch in zip(coords_i.values[0][1], coords_i.values[0][2])],
            add_change=add_change,
            num_workers=self.nun_workers
        )
        return out

    def len(self):
        return len(self.class_pairs)

    def get(self, idx):
        dom_i = self.class_pairs.loc[idx, 'domain_i']
        dom_j = self.class_pairs.loc[idx, 'domain_j']
        return (
            self.__build_graph(
                dom_i,
                add_change=self.coords_augmenter.add_change(dom_i, dom_j, "dom_i")
            ),
            self.__build_graph(
                dom_j,
                add_change=self.coords_augmenter.add_change(dom_i, dom_j, "dom_j")
            ),
            torch.from_numpy(
                np.array(self.score_method(self.class_pairs.loc[idx, 'score']), dtype=d_type)
            )
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tm_score_file', type=str, required=True)
    parser.add_argument('--pdb_path', type=str, required=True)
    # parser.add_argument('--pst_model_path', type=str, required=True)
    args = parser.parse_args()

    dataset = TmScoreFromCoordDataset(
        args.tm_score_file,
        args.pdb_path,
        ext="pdb",
        weighting_method=tm_score_weights(5, 0.25),
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
        print(z.shape)

    pst_model = load_pst_model({
        'model_path': args.pst_model_path,
        'device': 'cpu'
    })
    model = TransformerPstEmbeddingCosine(pst_model)
    for (g_i), (g_j), z in dataloader:
        z_pred = model(g_i, g_j)
        print(z_pred.shape, z.shape)
