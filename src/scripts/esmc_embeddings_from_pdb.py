from __future__ import annotations

import argparse
import logging
import os

import torch
import biotite.structure.io.pdb as pdb
from biotite.structure import to_sequence, filter_polymer, filter_amino_acids, BadStructureError

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

logger = logging.getLogger(__name__)


def iter_chains(pdb_file):
    pdb_struct = pdb.PDBFile.read(pdb_file)
    atom_array = pdb_struct.get_structure(model=1)
    atom_array = atom_array[filter_polymer(atom_array)]
    atom_array = atom_array[filter_amino_acids(atom_array)]
    sequences, chain_starts = to_sequence(atom_array)
    for seq, start_idx in zip(sequences, chain_starts):
        yield atom_array.chain_id[start_idx], str(seq)


if __name__ == '__main__':
    import config.logging_config  # noqa: F401

    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cpu')
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
        except BadStructureError as e:
            logger.warning(f"Skipping {pdb_name}: {e}")
            continue

        for chain_id, seq in chains:
            with torch.inference_mode():
                protein_tensor = client.encode(ESMProtein(sequence=seq))
                output = client.logits(
                    protein_tensor,
                    LogitsConfig(sequence=True, return_embeddings=True),
                )
            # drop BOS/EOS → (L, hidden)
            embedding = output.embeddings.squeeze(0)[1:-1].cpu()
            name = f"{stem}.{chain_id}" if chain_id.strip() else stem
            out_file = os.path.join(args.out_path, f"{name}.pt")
            torch.save(embedding, out_file)
            logger.info(f"{name} {tuple(embedding.shape)} -> {out_file}")
