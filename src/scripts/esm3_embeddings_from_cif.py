from __future__ import annotations

import argparse
import logging
import os

import numpy as np

import torch
from biotite.structure import filter_amino_acids, array
from biotite.structure.io.pdbx import CIFFile, get_structure

from torch.utils.data import  DataLoader

from esm.utils.structure.protein_chain import ProteinChain
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL

from dataset.esm3_embeddings_from_coord_dataset import EsmEmbeddingFromPdbDataset

d_type = np.float32
logger = logging.getLogger(__name__)

def rename_atom_attr(atom_ch):
    return array([__rename_atom(a) for a in atom_ch])

def __rename_atom(atom):
    atom.chain_id = "A"
    atom.hetero = False
    if len(atom.res_name) > 3:
        atom.res_name = 'UNK'
    return atom

if __name__ == '__main__':
    import config.logging_config

    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--only_sequence', action='store_true')
    args = parser.parse_args()

    dataset = EsmEmbeddingFromPdbDataset(
        args.pdb_path
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1
    )

    model: ESM3InferenceClient = ESM3.from_pretrained(ESM3_OPEN_SMALL, torch.device('cpu'))


    for s in os.listdir(args.pdb_path):
        src_structure = f"{args.pdb_path}/{s}"
        if os.path.isfile(f"{args.out_path}/{s.replace('.cif', '.pt')}"):
            logger.info(f"{s} is ready")
            continue
        logger.info(f"Processing {s}")
        cif_file = CIFFile.read(src_structure)
        protein_chain = get_structure(
            cif_file,
            model=1,
            use_author_fields=False
        )
        protein_chain = protein_chain[filter_amino_acids(protein_chain)]
        protein_chain = rename_atom_attr(protein_chain)
        protein_chain = ProteinChain.from_atomarray(protein_chain)
        if args.only_sequence:
            protein = ESMProtein(sequence=protein_chain.sequence)
        else:
            protein = ESMProtein.from_protein_chain(protein_chain)
        protein_tensor = model.encode(protein)
        output = model.forward_and_sample(
            protein_tensor, SamplingConfig(return_per_residue_embeddings=True)
        )
        logger.info(f"{s.replace('.cif', '')} {output.per_residue_embedding.shape}")
        torch.save(
            output.per_residue_embedding,
            f"{args.out_path}/{s.replace('.cif', '')}.pt"
        )
