import os

import numpy as np
from torch.utils.data import Dataset, DataLoader
import polars as pl
import torch

from dataset.pst.pst import load_pst_model
from dataset.utils.biopython_getter import get_coords_from_pdb_file
from dataset.utils.custom_weighted_random_sampler import CustomWeightedRandomSampler
from dataset.utils.embedding_builder import embedding_builder
from dataset.utils.tm_score_weight import binary_score, binary_weights, fraction_score, tm_score_weights
from dataset.utils.tools import load_class_pairs

d_type = np.float32


class TmScoreFromCoordDataset(Dataset):
    BINARY_THR = 0.7

    def __init__(
            self,
            tm_score_file,
            coords_path,
            pst_model,
            ext="ent",
            score_method=None,
            weighting_method=None,
            add_change=lambda *abc: abc
    ):
        self.coords = pl.DataFrame()
        self.class_pairs = pl.DataFrame()
        self.score_method = binary_score(self.BINARY_THR) if not score_method else score_method
        self.weighting_method = binary_weights(self.BINARY_THR) if not weighting_method else weighting_method
        self.pst_model = pst_model
        self.add_change = add_change
        self.__exec(tm_score_file, coords_path, f".{ext}" if len(ext) > 0 else "")

    def __exec(self, tm_score_file, embedding_path, ext):
        self.load_class_pairs(tm_score_file)
        self.load_coords(tm_score_file, embedding_path, ext)

    def load_class_pairs(self, tm_score_file):
        self.class_pairs = load_class_pairs(tm_score_file)
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

    def __build_embedding(self, dom_id):
        coords_i = self.coords.row(
            by_predicate=(pl.col("domain") == dom_id),
            named=True
        )
        out = embedding_builder(
            [{'cas': ch[0], 'seq': ch[1]} for ch in zip(coords_i['cas'], coords_i['seq'])],
            self.pst_model,
            add_change=self.add_change
        )
        return out

    def __len__(self):
        return len(self.class_pairs)

    def __getitem__(self, idx):
        return (
            self.__build_embedding(self.class_pairs.row(idx, named=True)['domain_i']),
            self.__build_embedding(self.class_pairs.row(idx, named=True)['domain_j']),
            torch.from_numpy(
                np.array(self.score_method(self.class_pairs.row(idx, named=True)['score']), dtype=d_type)
            )
        )


if __name__ == '__main__':
    model = load_pst_model({
        'model_path': '/Users/joan/devel/rcsb-embedding-search/src/dataset/pst/.cache/pst/pst_t30_so.pt',
        'device': 'cpu'
    })
    dataset = TmScoreFromCoordDataset(
        '/Users/joan/data/cath_23M/cath_23M.csv',
        '/Users/joan/cs-data/structure-embedding/cath_S40/pdb',
        pst_model=model,
        ext="",
        weighting_method=tm_score_weights(5),
        score_method=fraction_score
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
    for x, y, z in dataloader:
        print(x.shape, y.shape, z.shape)
