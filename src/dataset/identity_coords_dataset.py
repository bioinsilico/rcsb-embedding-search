from __future__ import annotations

import logging
import os

from torch_geometric.data import Dataset
import pandas as pd

from dataset.utils.biopython_getter import get_coords_from_pdb_file, get_coords_from_cif_file
from dataset.utils.coords_augmenter import AbstractAugmenter, NullAugmenter
from dataset.utils.embedding_builder import graph_builder


logger = logging.getLogger(__name__)


class IdentityCoordsDataset(Dataset):

    def __init__(
            self,
            coords_list,
            coords_path,
            ext="pdb",
            coords_augmenter: AbstractAugmenter | None = NullAugmenter(),
            num_workers=1
    ):
        super().__init__()
        self.coords = None
        self.coords_augmenter = coords_augmenter
        self.__exec(coords_list, coords_path, ext)
        self.nun_workers = num_workers if num_workers > 1 else 1

    def __exec(self, embedding_list, embedding_path, ext):
        self.load_coords(embedding_list, embedding_path, ext)

    def load_coords(self, coords_list, coords_path, ext):
        logger.info(f"Loading structures from file {coords_list} in path {coords_path}")
        self.coords = pd.DataFrame(
            data=[
                (lambda coords: (
                    row[0],
                    coords['cas'],
                    coords['seq']
                ))(
                    get_coords_from_pdb_file(os.path.join(coords_path, f"{row[0]}.{ext}")) if ext == "pdb" or ext == "ent" else
                    get_coords_from_cif_file(os.path.join(coords_path, f"{row[0]}.{ext}"))
                )
                for row in [row.strip().split(",") for row in open(coords_list)]
            ],
            columns=['domain', 'cas', 'seq']
        )
        logger.info(f"Total structures: {len(self.coords)}")

    def __build_graph(self, idx, add_change=lambda *abc: abc):
        coords_i = self.coords.loc[idx]
        out = graph_builder(
            [{'cas': ch[0], 'seq': ch[1]} for ch in zip(coords_i['cas'], coords_i['seq'])],
            add_change=add_change,
            num_workers=self.nun_workers
        )
        return out

    def len(self):
        return len(self.coords)

    def get(self, idx):
        coords_i = self.coords.loc[idx]
        return (
            self.__build_graph(
                idx,
                add_change=self.coords_augmenter.add_change(coords_i['domain'], coords_i['domain'], "dom_i")
            ),
            self.__build_graph(
                idx,
                add_change=self.coords_augmenter.add_change(coords_i['domain'], coords_i['domain'], "dom_j")
            ),
            1.
        )



