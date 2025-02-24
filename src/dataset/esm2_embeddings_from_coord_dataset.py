from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch
from torch import squeeze

from torch.utils.data import Dataset, DataLoader

from dataset.utils.biopython_getter import get_coords_from_pdb_file

import esm


d_type = np.float32
logger = logging.getLogger(__name__)


class EsmEmbeddingFromCoordDataset(Dataset):
    BINARY_THR = 0.7

    def __init__(
            self,
            coords_path
    ):
        super().__init__()
        self.coords = pd.DataFrame()
        self.load_coords(coords_path)

    def load_coords(self, coords_path):
        logger.info(f"Loading coords from path {coords_path}")
        self.coords = pd.DataFrame(
            data=[
                (lambda coords: (
                    dom_id,
                    coords['cas'],
                    coords['seq']
                ))(
                    get_coords_from_pdb_file(os.path.join(coords_path, f"{dom_id}"))
                )
                for dom_id in os.listdir(coords_path)
            ],
            columns=['domain', 'cas', 'seq']
        )
        logger.info(f"Total structures: {len(self.coords)}")

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        return self.coords.loc[idx, 'domain'], "".join(["".join(aa) for aa in self.coords.loc[idx, 'seq']])


if __name__ == '__main__':
    import config.logging_config

    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    args = parser.parse_args()

    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model = model.eval()

    dataset = EsmEmbeddingFromCoordDataset(
        args.pdb_path
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1
    )

    for s in dataloader:
        batch_labels, batch_strs, batch_tokens = batch_converter([(s[0][0], s[1][0])])
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[30], return_contacts=False)
            torch.save(
                squeeze(results["representations"][30], dim =0),
                f"{args.out_path}/{s[0][0]}.pt"
            )


