from __future__ import annotations

import argparse
import os

import numpy as np

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


from dataset.utils.biopython_getter import get_coords_from_pdb_file
from dataset.utils.coords_augmenter import NullAugmenter, AbstractAugmenter
from dataset.utils.geometric_tools import get_res_attr
from dataset.utils.tm_score_weight import binary_score, binary_weights, fraction_score, tm_score_weights
from dataset.utils.tools import load_class_pairs_with_self_comparison, load_class_pairs, collate_fn

d_type = np.float32


class TmScoreGeoEmbeddingFromCoordDataset(Dataset):
    BINARY_THR = 0.7

    def __init__(
            self,
            tm_score_file,
            coords_path,
            ext="pdb",
            score_method=None,
            weighting_method=None,
            coords_augmenter: AbstractAugmenter | None = NullAugmenter()
    ):
        super().__init__()
        self.coords = pd.DataFrame()
        self.class_pairs = pd.DataFrame()
        self.score_method = binary_score(self.BINARY_THR) if not score_method else score_method
        self.weighting_method = binary_weights(self.BINARY_THR) if not weighting_method else weighting_method
        self.coords_augmenter = NullAugmenter() if coords_augmenter is None else coords_augmenter
        self.__exec(tm_score_file, coords_path, f".{ext}" if len(ext) > 0 else "")

    def __exec(self, tm_score_file, embedding_path, ext):
        self.load_class_pairs(tm_score_file)
        self.load_coords(embedding_path, ext)

    def load_class_pairs(self, tm_score_file):
        print(f"Loading pairs from file {tm_score_file}")
        self.class_pairs = load_class_pairs(tm_score_file) if isinstance(self.coords_augmenter, NullAugmenter) else \
                           load_class_pairs_with_self_comparison(tm_score_file)
        print(f"Total pairs: {len(self.class_pairs)}")

    def load_coords(self, coords_path, ext):
        print(f"Loading coords from path {coords_path}")
        self.coords = pd.DataFrame(
            data=[
                (lambda coords: (
                    dom_id,
                    coords['cas'],
                    coords['labels'],
                    coords['seq']
                ))(
                    get_coords_from_pdb_file(os.path.join(coords_path, f"{dom_id}{ext}"))
                )
                for dom_id in pd.concat([
                    self.class_pairs["domain_i"], self.class_pairs["domain_j"]
                ]).unique()
            ],
            columns=['domain', 'cas', 'labels', 'seq']
        )
        print(f"Total structures: {len(self.coords)}")

    def weights(self):
        return self.weighting_method(self.class_pairs['score'].to_numpy())

    def __build_graph(self, dom_id, add_change=lambda *abc: abc):
        coords_i = self.coords.loc[self.coords['domain'] == dom_id]
        if len(coords_i) != 1:
            raise Exception(f'Data error: found {len(coords_i)} rows for {dom_id}')
        graph_features = [get_res_attr(list(zip(ch[0], ch[1]))) for ch in zip(coords_i.values[0][1], coords_i.values[0][2])]
        return torch.stack([res for ch in graph_features for res in ch[0]], dim=0)

    def __len__(self):
        return len(self.class_pairs)

    def __getitem__(self, idx):
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
    args = parser.parse_args()

    dataset = TmScoreGeoEmbeddingFromCoordDataset(
        args.tm_score_file,
        args.pdb_path,
        ext="pdb",
        weighting_method=tm_score_weights(5, 0.25),
        score_method=fraction_score
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collate_fn
    )

    for (x, x_mask), (y, y_mask), z in dataloader:
        print(x.shape, y.shape, z.shape)
