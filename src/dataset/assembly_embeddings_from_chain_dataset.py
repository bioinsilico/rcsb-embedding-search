import argparse
import logging
import torch

from biotite.database import rcsb
from biotite.structure import chain_iter, get_chains, filter_amino_acids, get_residues
from biotite.structure.io.pdbx import BinaryCIFFile, list_assemblies, get_assembly
from torch.utils.data import Dataset, DataLoader

from dataset.utils.tools import collate_seq_embeddings

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
    total_memory = 0
    for file in file_list:
        try:
            tensor = torch.load(
                file,
                map_location=torch.device('cpu')
            )
            total_memory += tensor.numel() * tensor.element_size()
            tensors.append(tensor)
        except Exception as e:
            logger.error(f"Error loading tensor from {file}: {e}")
            continue
        if total_memory > (200 * 1e9):
            logger.warning("Limiting number of assembly chains")
            break
    if tensors and len(tensors) > 0:
        tensor_cat = torch.cat(tensors, dim=dim)
        logger.info(f"Tensor shape {tensor_cat.shape} size {tensor_cat.numel() * tensor_cat.element_size()}")
        return tensor_cat
    else:
        raise ValueError("No valid tensors were loaded to concatenate.")


def compute_esm3_assembly(pdb_id, assembly_id, esm3_embeddings):
    rcsb_fetch = rcsb.fetch(pdb_id, "bcif")
    bcif = BinaryCIFFile.read(rcsb_fetch)
    for _assembly_id in list_assemblies(bcif):
        if assembly_id != _assembly_id:
            continue
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

        return concatenate_tensors([f"{esm3_embeddings}/{pdb_id}.{ch}.pt" for ch in assembly_ch])


class AssemblyEmbeddingsDataset(Dataset):

    def __init__(
            self,
            assembly_list,
            chain_embedding_path
    ):
        self.assembly_list = [assembly_id.strip() for assembly_id in open(assembly_list)]
        self.chain_embedding_path = chain_embedding_path

    def __len__(self):
        return len(self.assembly_list)

    def __getitem__(self, idx):
        [pdb_id, assembly_id] = self.assembly_list[idx].split("-")
        try:
            return compute_esm3_assembly(pdb_id, assembly_id, self.chain_embedding_path), self.assembly_list[idx]
        except Exception:
            logger.info(f"Assembly {pdb_id}-{assembly_id} failed")
            return None, self.assembly_list[idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--assembly_list', type=str, required=True)
    parser.add_argument('--chain_embedding_path', type=str, required=True)
    args = parser.parse_args()

    dataset = AssemblyEmbeddingsDataset(
        args.assembly_list,
        args.chain_embedding_path
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=lambda emb: (
            collate_seq_embeddings([x for x, z in emb]),
            tuple([z for x, z in emb])
        )
    )

    for (x, x_mask), z in dataloader:
        print(x.shape, x_mask.shape, z)

