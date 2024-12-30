from __future__ import annotations

import shutil
import argparse
import logging
import torch

from biotite.database import rcsb
from biotite.structure import filter_amino_acids, get_chains, chain_iter, get_residues
from biotite.structure.io.pdbx import BinaryCIFFile, get_assembly, list_assemblies

from scripts.esm3_embeddings_from_rcsb import split_list_get_index

logger = logging.getLogger(__name__)


def concatenate_tensors(file_list, dim=0):
    """
    Concatenates a list of tensors stored in individual files along a specified dimension.

    Args:
        file_list (list of str): List of file paths to tensor files.
        dim (int): The dimension along which to concatenate the tensors. Default is 0.

    Returns:
        torch.Tensor: The concatenated tensor.
    """
    tensors = []
    for file in file_list:
        try:
            tensor = torch.load(file)
            tensors.append(tensor)
        except Exception as e:
            print(f"Error loading tensor from {file}: {e}")
            continue

    if tensors:
        return torch.cat(tensors, dim=dim)
    else:
        raise ValueError("No valid tensors were loaded to concatenate.")


def compute_esm3_assembly(pdb_id, esm3_embeddings, out_path):
    rcsb_fetch = rcsb.fetch(pdb_id, "bcif")
    bcif = BinaryCIFFile.read(rcsb_fetch)
    for assembly_id in list_assemblies(bcif):
        atom_array = get_assembly(
            bcif,
            model=1,
            assembly_id=assembly_id,
            use_author_fields=False
        )
        assembly_ch = []
        for atom_ch in chain_iter(atom_array):
            ch = get_chains(atom_ch)[0]
            atom_res = atom_ch[filter_amino_acids(atom_ch)]
            if len(atom_res) == 0 or len(get_residues(atom_res)[0]) < 10:
                logger.info(f"Ignoring chain {pdb_id}.{ch} with {0 if len(atom_res) == 0 else len(get_residues(atom_res)[0])} res")
                continue
            if len(get_chains(atom_ch)) > 1:
                raise ValueError("Inconsistent chain ids")
            assembly_ch.append(get_chains(atom_ch)[0])
        if len(assembly_ch) == 0:
            logger.info(f"Ignoring assembly {pdb_id}-{assembly_id} with no protein chains")
            continue

        logger.info(f"Building assembly {pdb_id}-{assembly_id} from chains {assembly_ch}")
        if len(assembly_ch) == 1:
            shutil.copy(f"{esm3_embeddings}/{pdb_id}.{assembly_ch[0]}.pt", f"{out_path}/{pdb_id}-{assembly_id}.pt")
        else:
            torch.save(
                concatenate_tensors([f"{esm3_embeddings}/{pdb_id}.{ch}.pt" for ch in assembly_ch]),
                f"{out_path}/{pdb_id}-{assembly_id}.pt"
            )


if __name__ == "__main__":
    import config.logging_config

    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_list_file', type=str, required=True)
    parser.add_argument('--esm3_embeddings', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--n_split', type=int, required=True)
    parser.add_argument('--n_idx', type=int, required=True)
    args = parser.parse_args()

    pdb_list_file = args.pdb_list_file
    esm3_embeddings = args.esm3_embeddings
    out_path = args.out_path
    n_split = args.n_split
    n_idx = args.n_idx

    failed_file = f"{out_path}/failed_{n_idx}.txt"

    full_pdb_list = [row.strip() for row in open(pdb_list_file)]
    pdb_list = split_list_get_index(full_pdb_list, n_split, n_idx)

    logger.info(f"Full list length {len(full_pdb_list)} sub-list length {len(pdb_list)} index {n_idx}")

    for pdb_id in pdb_list:
        compute_esm3_assembly(pdb_id, esm3_embeddings, out_path)
