from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader

from esm.utils.structure.protein_chain import ProteinChain
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL

d_type = np.float32
logger = logging.getLogger(__name__)


class EsmEmbeddingFromPdbDataset(Dataset):
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
            data=[(
                    dom_id[0:-4] if ".pdb" in dom_id or ".ent" in dom_id else dom_id,
                    os.path.join(coords_path, f"{dom_id}")
            )for dom_id in os.listdir(coords_path)],
            columns=['domain', 'file']
        )
        logger.info(f"Total structures: {len(self.coords)}")

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        return self.coords.loc[idx, 'domain'], self.coords.loc[idx, 'file']


if __name__ == '__main__':
    import config.logging_config

    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    args = parser.parse_args()

    dataset = EsmEmbeddingFromPdbDataset(
        args.pdb_path
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1
    )

    model: ESM3InferenceClient = ESM3.from_pretrained(ESM3_OPEN_SMALL, torch.device('cpu'))

    for s in dataloader:
        pdb_file = s[1][0]
        protein_chain = ProteinChain.from_pdb(pdb_file)
        protein = ESMProtein.from_protein_chain(protein_chain)
        protein_tensor = model.encode(protein)
        output = model.forward_and_sample(
            protein_tensor, SamplingConfig(return_per_residue_embeddings=True)
        )
        logger.info(f"{s[0][0]} {output.per_residue_embedding.shape}")
        torch.save(
            output.per_residue_embedding,
            f"{args.out_path}/{s[0][0]}.pt"
        )
