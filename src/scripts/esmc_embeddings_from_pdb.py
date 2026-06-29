from __future__ import annotations

import argparse
import functools
import logging
import os

import torch
import biotite.structure.io.pdb as pdb
from biotite.structure import (
    filter_polymer,
    filter_amino_acids,
    get_chain_starts,
    get_residues,
)
from biotite.structure.info import get_ccd

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def res_to_letter_map():
    """3-letter residue code -> 1-letter code, using each residue's own
    one_letter_code or its mon_nstd_parent_comp_id's. Missing entries map to 'X'."""
    chem_comp = get_ccd()['chem_comp']
    ids = chem_comp['id'].as_array()
    letters = chem_comp['one_letter_code'].as_array()
    parents = chem_comp['mon_nstd_parent_comp_id'].as_array()

    direct = {code: letter for code, letter in zip(ids, letters)
              if len(letter) == 1 and letter != '?'}
    mapping = dict(direct)
    for code, parent in zip(ids, parents):
        if code in mapping:
            continue
        if parent in direct:
            mapping[code] = direct[parent]
    return mapping


def iter_chains(pdb_file):
    pdb_struct = pdb.PDBFile.read(pdb_file)
    atom_array = pdb_struct.get_structure(model=1)
    atom_array = atom_array[filter_polymer(atom_array)]
    atom_array = atom_array[filter_amino_acids(atom_array)]
    if len(atom_array) == 0:
        return

    mapping = res_to_letter_map()
    chain_starts = get_chain_starts(atom_array, add_exclusive_stop=True)
    for i in range(len(chain_starts) - 1):
        chain_array = atom_array[chain_starts[i]:chain_starts[i + 1]]
        _, res_names = get_residues(chain_array)
        seq = ''.join(mapping.get(name, 'X') for name in res_names)
        yield chain_array.chain_id[0], seq


if __name__ == '__main__':
    import config.logging_config  # noqa: F401

    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--ignore_chain_id', action='store_true',
                        help='Omit the chain id from output filenames; fail if a '
                             'structure has more than one chain.')
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)

    client = ESMC.from_pretrained("esmc_600m").to(args.device).eval()

    for pdb_name in sorted(os.listdir(args.pdb_path)):
        if not (pdb_name.endswith('.pdb') or pdb_name.endswith('.ent')):
            continue
        pdb_file = os.path.join(args.pdb_path, pdb_name)
        stem = pdb_name.rsplit('.', 1)[0]

        chains = list(iter_chains(pdb_file))
        if args.ignore_chain_id and len(chains) > 1:
            raise ValueError(
                f"{pdb_name}: --ignore_chain_id set but structure has "
                f"{len(chains)} chains ({', '.join(c for c, _ in chains)})"
            )

        for chain_id, seq in chains:
            with torch.inference_mode():
                protein_tensor = client.encode(ESMProtein(sequence=seq))
                output = client.logits(
                    protein_tensor,
                    LogitsConfig(sequence=True, return_embeddings=True),
                )
            # drop BOS/EOS → (L, hidden)
            embedding = output.embeddings.squeeze(0)[1:-1].cpu()
            if args.ignore_chain_id or not chain_id.strip():
                name = stem
            else:
                name = f"{stem}.{chain_id}"
            out_file = os.path.join(args.out_path, f"{name}.pt")
            torch.save(embedding, out_file)
            logger.info(f"{name} {tuple(embedding.shape)} -> {out_file}")
