import argparse
import logging
import os.path
from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor
import multiprocessing as mp

import torch
from biotite.structure import get_chains, chain_iter, filter_amino_acids, get_residues
from biotite.structure.io.pdb import PDBFile, get_structure as get_pdb_structure
from biotite.structure.io.pdbx import CIFFile, get_structure as get_cif_structure
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL
from esm.utils.structure.protein_chain import ProteinChain
from torch.utils.data import DataLoader

from dataset.esm3_embeddings_from_coord_dataset import EsmEmbeddingFromPdbDataset
from scripts.esm3_embeddings_from_cif import rename_atom_attr

logger = logging.getLogger(__name__)


def compute_embedding(pdb_file, model, out_path):
    pdb_id, ext = os.path.splitext(pdb_file)
    pdb_id = os.path.basename(pdb_id)
    if os.path.isfile(f"{out_path}/{pdb_id}.pt"):
        print(f"{pdb_file} is ready")
        return
    print(f"Processing {pdb_file}")
    pdb = PDBFile.read(pdb_file) if ext == ".pdb" else CIFFile.read(pdb_file)
    assembly_structure = get_pdb_structure(pdb) if ext == ".pdb" else get_cif_structure(
        pdb,
        model=1,
        use_author_fields=False
    )

    assembly_ch = []
    for atom_ch in chain_iter(assembly_structure):
        ch = get_chains(atom_ch)[0]
        if ext == ".pdb":
            atom_ch = atom_ch[0]
        atom_res = atom_ch[filter_amino_acids(atom_ch)]
        if len(atom_res) == 0 or len(get_residues(atom_res)[0]) < 10:
            logger.info(f"Ignoring chain {pdb_id}.{ch} with {0 if len(atom_res) == 0 else len(get_residues(atom_res)[0])} res")
            continue
        if len(get_chains(atom_ch)) > 1:
            raise ValueError("Inconsistent chain ids")
        atom_res = rename_atom_attr(atom_res)
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
        f"{out_path}/{pdb_id}.pt"
    )

if __name__ == '__main__':
    import config.logging_config

    # Use the 'spawn' start method to avoid CUDA re-initialization errors when
    # using multiprocessing with GPUs.
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--n-device', type=int, default=None)
    args = parser.parse_args()

    dataset = EsmEmbeddingFromPdbDataset(
        args.pdb_path
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1
    )

    models = []
    if args.device == 'cuda' and args.n_device is not None and args.n_device > 1:
        for n in range(0,args.n_device):
            models.append( ESM3.from_pretrained(ESM3_OPEN_SMALL, device=torch.device(f"cuda:{n}")) )
    else:
        models.append( ESM3.from_pretrained(ESM3_OPEN_SMALL, device=torch.device(args.device) ))

    ctx = mp.get_context("spawn")
    executor = ProcessPoolExecutor(
        max_workers=args.n_device if args.n_device is not None else 1,
        mp_context=ctx,
    )
    future_to_command = {}
    for idx, s in enumerate(dataloader):
        structure_file = s[1][0]
        key = executor.submit(compute_embedding, s[1][0], models[idx % len(models)], args.out_path)
        future_to_command[key] = structure_file

    for future in as_completed(future_to_command):
        structure_file = future_to_command.get(future)
        try:
            future.result()
            logger.info(f"Finished {structure_file}")
        except Exception as e:
            logger.error(f"Failed {structure_file}: {e}")
