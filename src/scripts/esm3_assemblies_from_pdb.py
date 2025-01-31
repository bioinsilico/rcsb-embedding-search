import argparse
import logging
import os.path

import torch
from biotite.structure import get_chains, chain_iter, filter_amino_acids, get_residues
from biotite.structure.io.pdb import PDBFile
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL
from esm.utils.structure.protein_chain import ProteinChain
from torch.utils.data import DataLoader

from dataset.esm3_embeddings_from_coord_dataset import EsmEmbeddingFromPdbDataset


logger = logging.getLogger(__name__)

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
        pdb_id = s[0][0]
        if os.path.isfile(f"{args.out_path}/{pdb_id}.pt"):
            print(f"{pdb_file} is ready")
            continue
        print(f"Processing {pdb_file}")
        pdb = PDBFile.read(pdb_file)
        assembly_ch = []
        for atom_ch in chain_iter(pdb.get_structure()):
            ch = get_chains(atom_ch)[0]
            atom_res = atom_ch[0][filter_amino_acids(atom_ch)]
            if len(atom_res) == 0 or len(get_residues(atom_res)[0]) < 10:
                logger.info(f"Ignoring chain {pdb_id}.{ch} with {0 if len(atom_res) == 0 else len(get_residues(atom_res)[0])} res")
                continue
            if len(get_chains(atom_ch)) > 1:
                raise ValueError("Inconsistent chain ids")
            protein_chain = ProteinChain.from_atomarray(atom_res)
            protein = ESMProtein.from_protein_chain(protein_chain)
            protein_tensor = model.encode(protein)
            output = model.forward_and_sample(
                protein_tensor, SamplingConfig(return_per_residue_embeddings=True)
            )
            logger.info(f"{pdb_id}.{ch} {output.per_residue_embedding.shape}")
            assembly_ch.append(output.per_residue_embedding)
        assembly_embedding = torch.cat(assembly_ch, dim=0)
        logger.info(f"{pdb_id} {assembly_embedding.shape}")
        torch.save(
            torch.cat(assembly_ch, dim=0),
            f"{args.out_path}/{pdb_id}.pt"
        )
