import argparse
import logging
import os
import os.path
import multiprocessing as mp
from typing import List

import torch
from torch.multiprocessing import get_context
from biotite.structure import get_chains, chain_iter, filter_amino_acids, get_residues
from biotite.structure.io.pdb import PDBFile, get_structure as get_pdb_structure
from biotite.structure.io.pdbx import CIFFile, get_structure as get_cif_structure
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL
from esm.utils.structure.protein_chain import ProteinChain

from scripts.esm3_embeddings_from_cif import rename_atom_attr

logger = logging.getLogger(__name__)

def compute_embedding_for_file(pdb_file: str, model: ESM3, out_path: str):
    pdb_id, ext = os.path.splitext(pdb_file)
    pdb_id = os.path.basename(pdb_id)
    out_file = os.path.join(out_path, f"{pdb_id}.pt")
    if os.path.isfile(out_file):
        logger.info(f"{pdb_file} is ready")
        return

    logger.info(f"Processing {pdb_file}")
    pdb = PDBFile.read(pdb_file) if ext == ".pdb" else CIFFile.read(pdb_file)
    assembly_structure = get_pdb_structure(pdb) if ext == ".pdb" else get_cif_structure(
        pdb, model=1, use_author_fields=False
    )

    assembly_ch = []
    with torch.inference_mode():
        for atom_ch in chain_iter(assembly_structure):
            ch = get_chains(atom_ch)[0]
            atom_sel = atom_ch[0] if ext == ".pdb" else atom_ch
            atom_res = atom_sel[filter_amino_acids(atom_sel)]
            if len(atom_res) == 0 or len(get_residues(atom_res)[0]) < 10:
                nres = 0 if len(atom_res) == 0 else len(get_residues(atom_res)[0])
                logger.info(f"Ignoring chain {pdb_id}.{ch} with {nres} res")
                continue
            if len(get_chains(atom_sel)) > 1:
                raise ValueError("Inconsistent chain ids")
            atom_res = rename_atom_attr(atom_res)
            protein_chain = ProteinChain.from_atomarray(atom_res)
            protein = ESMProtein.from_protein_chain(protein_chain)

            protein_tensor = model.encode(protein)  # stays on model device
            output = model.forward_and_sample(
                protein_tensor, SamplingConfig(return_per_residue_embeddings=True)
            )
            logger.info(f"{pdb_id}.{ch} {output.per_residue_embedding.shape}")
            assembly_ch.append(output.per_residue_embedding)

        if len(assembly_ch) == 0:
            logger.info(f"No valid chains found for {pdb_id}, skipping.")
            return

        assembly_embedding = torch.cat(assembly_ch, dim=0)
        logger.info(f"{pdb_id} {assembly_embedding.shape}")
        torch.save(assembly_embedding, out_file)

def _per_gpu_worker(rank: int, files: List[str], out_path: str, model_name: str):
    # Constrain CPU threading to avoid thrash when multiple GPU processes run.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    # Bind this process to a single GPU and instantiate the model *here*.
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    model = ESM3.from_pretrained(model_name, device=device)

    for pdb_file in files:
        try:
            compute_embedding_for_file(pdb_file, model, out_path)
        except Exception as e:
            logger.error(f"Failed {pdb_file} on gpu:{rank}: {e}")

def shard(lst, n, i):
    # Static round-robin sharding: process i handles items i, i+n, i+2n, ...
    return lst[i::n]

if __name__ == "__main__":
    import config.logging_config  # your logger setup

    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--pdb_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--n-device", type=int, default=1)
    args = parser.parse_args()

    coords_path = args.pdb_path
    files = [(
        dom_id[0:-4] if ".pdb" in dom_id or ".ent" in dom_id or ".cif" in dom_id else dom_id,
        os.path.join(coords_path, f"{dom_id}")
    ) for dom_id in os.listdir(coords_path)],

    ngpu = args.n_device
    assert ngpu >= 1, "n-device must be >= 1"

    ctx = get_context("spawn")
    procs = []
    for rank in range(ngpu):
        rank_files = shard(files, ngpu, rank)
        if not rank_files:
            continue
        p = ctx.Process(
            target=_per_gpu_worker,
            args=(rank, rank_files, args.out_path, ESM3_OPEN_SMALL),
            daemon=False,
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
