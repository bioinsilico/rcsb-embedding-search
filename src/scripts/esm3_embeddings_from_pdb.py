from __future__ import annotations

import argparse
import logging


import numpy as np
import torch

from torch.utils.data import DataLoader

from esm.utils.structure.protein_chain import ProteinChain
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL

from dataset.esm3_embeddings_from_coord_dataset import EsmEmbeddingFromPdbDataset

d_type = np.float32
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    import config.logging_config

    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    dataset = EsmEmbeddingFromPdbDataset(
        args.pdb_path
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1
    )

    model: ESM3InferenceClient = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=torch.device(args.device) if args.device is not None else None)

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
