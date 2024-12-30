from __future__ import annotations

import argparse

import torch
import logging

from biotite.database import rcsb
from biotite.structure import AtomArray, filter_amino_acids, chain_iter, get_chains, get_residues
from biotite.structure.io.pdbx import BinaryCIFFile, get_structure

from esm.utils.structure.protein_chain import ProteinChain
from esm.models.esm3 import ESM3
from esm.sdk.api import ESM3InferenceClient, ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL

logger = logging.getLogger(__name__)


def rename_atom_ch(atom_ch, ch = "A"):
    renamed_atom_ch = AtomArray(len(atom_ch))
    n = 0
    for atom in atom_ch:
        atom.chain_id = ch
        renamed_atom_ch[n] = atom
        n += 1
    return renamed_atom_ch


def compute_esm3_embeddings(model, pdb_id, out_path, failed_file):
    try:
        rcsb_fetch = rcsb.fetch(pdb_id, "bcif")
        bcif = BinaryCIFFile.read(rcsb_fetch)
        atom_array = get_structure(
            bcif,
            model=1,
            use_author_fields=False,
            extra_fields=["b_factor"]
        )
    except Exception as e:
        logger.info(f"PDB entry failed {pdb_id}")
        logger.exception(str(e))
        with open(failed_file, "a") as f:
            f.write(f"{pdb_id}#entry\n")
        return

    for atom_ch in chain_iter(atom_array):
        if len(get_chains(atom_ch)) == 0:
            logger.info(f"Ignoring null chain from {pdb_id}")
            continue
        ch = get_chains(atom_ch)[0]
        atom_res = atom_ch[filter_amino_acids(atom_ch)]
        if len(atom_res) == 0 or len(get_residues(atom_res)[0]) < 10:
            logger.info(f"Ignoring chain {pdb_id}.{ch} with {0 if len(atom_res) == 0 else len(get_residues(atom_res)[0])} res")
            continue
        if len(get_chains(atom_ch)) > 1:
            raise ValueError("Inconsistent chain ids")
        try:
            atom_ch = rename_atom_ch(atom_ch)
            protein_chain = ProteinChain.from_atomarray(atom_ch)
            protein = ESMProtein.from_protein_chain(protein_chain)
            protein_tensor = model.encode(protein)
            output = model.forward_and_sample(
                protein_tensor, SamplingConfig(return_per_residue_embeddings=True)
            )
            logger.info(f"{pdb_id}.{ch} {output.per_residue_embedding.shape}")
            torch.save(
                output.per_residue_embedding,
                f"{out_path}/{pdb_id}.{ch}.pt"
            )
        except Exception as e:
            logger.info(f"Chain failed {pdb_id}.{ch}")
            logger.exception(str(e))
            with open(failed_file, "a") as f:
                f.write(f"{pdb_id}.{ch}\n")


def split_list_get_index(lst, n, index):
    """
    Splits a list into N sublists and returns the sublist at the specified index.

    Parameters:
    - lst (list): The list to split.
    - n (int): The number of sublists to create.
    - index (int): The index of the sublist to return (0-based).

    Returns:
    - list: The sublist at the specified index.
    """
    if n <= 0:
        raise ValueError("Number of sublists (n) must be greater than 0.")
    if index < 0 or index >= n:
        raise ValueError("Index is out of range for the number of sublists.")

    # Determine the size of each chunk
    avg_size = len(lst) // n
    remainder = len(lst) % n

    # Generate sublists
    sublists = []
    start = 0
    for i in range(n):
        # Calculate end index based on remainder (distribute the extra items)
        extra = 1 if i < remainder else 0
        end = start + avg_size + extra
        sublists.append(lst[start:end])
        start = end

    return sublists[index]


if __name__ == "__main__":
    import config.logging_config

    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_list_file', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--n_split', type=int, required=True)
    parser.add_argument('--n_idx', type=int, required=True)
    args = parser.parse_args()

    pdb_list_file = args.pdb_list_file
    out_path = args.out_path
    n_split = args.n_split
    n_idx = args.n_idx

    failed_file = f"{out_path}/failed_{n_idx}.txt"

    model: ESM3InferenceClient = ESM3.from_pretrained(ESM3_OPEN_SMALL)

    full_pdb_list = [row.strip() for row in open(pdb_list_file)]
    pdb_list = split_list_get_index(full_pdb_list, n_split, n_idx)

    logger.info(f"Full list length {len(full_pdb_list)} sub-list length {len(pdb_list)} index {n_idx}")

    for pdb_id in pdb_list:
        compute_esm3_embeddings(model, pdb_id, out_path, failed_file)
