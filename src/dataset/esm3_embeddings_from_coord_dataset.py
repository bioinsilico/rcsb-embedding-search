from __future__ import annotations


import logging
import os

import numpy as np
import pandas as pd


from torch.utils.data import Dataset

d_type = np.float32
logger = logging.getLogger(__name__)


class EsmEmbeddingFromPdbDataset(Dataset):
    BINARY_THR = 0.7

    def __init__(
            self,
            coords_path,
            file_list_file=None
    ):
        super().__init__()
        self.coords = pd.DataFrame()
        self.load_coords(coords_path, file_list_file)

    def load_coords(self, coords_path, file_list_file=None):
        logger.info(f"Loading coords from path {coords_path}")
        dom_id_list = os.listdir(coords_path) if file_list_file is None else [r.strip() for r in open(file_list_file)]
        self.coords = pd.DataFrame(
            data=[(
                    dom_id[0:-4] if ".pdb" in dom_id or ".ent" in dom_id or ".cif" in dom_id else dom_id,
                    os.path.join(coords_path, f"{dom_id}")
            ) for dom_id in dom_id_list],
            columns=['domain', 'file']
        )
        logger.info(f"Total structures: {len(self.coords)}")

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        return self.coords.loc[idx, 'domain'], self.coords.loc[idx, 'file']
