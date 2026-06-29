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
    # min_size=1: with the default min_size=2, filter_polymer silently drops chain
    # residues that gappy/non-monotonic numbering isolates into 1-residue segments
    # (~1.4k files in the CATH set, ~600 of them mid-chain). min_size=1 keeps them
    # while still filtering out non-peptide entities.
    atom_array = atom_array[filter_polymer(atom_array, min_size=1)]
    atom_array = atom_array[filter_amino_acids(atom_array)]
    if len(atom_array) == 0:
        return

    mapping = res_to_letter_map()
    # biotite starts a new "chain" whenever the residue id decreases
    # (get_chain_starts), so a single chain id can resolve to several disjoint
    # segments. In this data those extra segments are distinct entities sharing the
    # chain id (bound peptides flagged by an insertion code, His-tags, symmetry
    # mates, or two mislabelled chains) rather than a renumbered continuation of one
    # chain. Concatenating them would feed ESMC a chimeric, non-physical sequence, so
    # a chain id that splits into more than one segment is skipped, not guessed.
    segments = {}
    chain_starts = get_chain_starts(atom_array, add_exclusive_stop=True)
    for i in range(len(chain_starts) - 1):
        chain_array = atom_array[chain_starts[i]:chain_starts[i + 1]]
        _, res_names = get_residues(chain_array)
        seq = ''.join(mapping.get(name, 'X') for name in res_names)
        segments.setdefault(str(chain_array.chain_id[0]), []).append(seq)
    for chain_id, segs in segments.items():
        if len(segs) > 1:
            logger.warning(
                f"{os.path.basename(pdb_file)}: chain {chain_id!r} resolves to "
                f"{len(segs)} disjoint segments (sizes {[len(s) for s in segs]}) "
                f"- skipping (likely distinct entities sharing a chain id)"
            )
            continue
        yield chain_id, segs[0]


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

        try:
            chains = list(iter_chains(pdb_file))
        except Exception as e:
            # e.g. empty/corrupt files ("0 models" ValueError) — skip, don't abort.
            logger.warning(f"Skipping {pdb_name}: {type(e).__name__}: {e}")
            continue
        if not chains:
            logger.warning(f"Skipping {pdb_name}: no amino-acid chains")
            continue
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
