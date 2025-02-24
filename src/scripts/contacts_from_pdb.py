import argparse
import logging
import os.path

import numpy as np
import torch
from biotite.structure import get_chains, chain_iter, filter_amino_acids, get_residues
from biotite.structure.io.pdb import PDBFile

from torch.utils.data import DataLoader
import torch_geometric.nn as gnn

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
            coo = torch.from_numpy(np.array(atom_res[atom_res.atom_name == 'CA'].coord))
            edge_index = gnn.radius_graph(
                coo, r=8, loop=False
            )
            print(edge_index)
            exit(0)
