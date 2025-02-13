import argparse
from io import StringIO
import requests

import pandas as pd
from biotite.structure import chain_iter, filter_amino_acids, get_residues
from biotite.structure.io.pdbx import get_structure, CIFFile
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, SamplingConfig
from esm.utils.constants.models import ESM3_OPEN_SMALL
from esm.utils.structure.protein_chain import ProteinChain
from torch import device
from torch.utils.data import Dataset, DataLoader


def stringio_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        return StringIO(response.text)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL: {e}")
        return None


class BiotiteStructureFromList(Dataset):

    def __init__(
            self,
            rcsb_file_list,
            build_url
    ):
        self.rcsb_pd = pd.read_csv(rcsb_file_list, names=["rcsb_id"], header=0)
        self.build_url = build_url

    def __len__(self):
        return len(self.rcsb_pd)

    def __getitem__(self, idx):
        acc = self.rcsb_pd.loc[idx, 'rcsb_id']
        file_stream = stringio_from_url(self.build_url(acc))
        cif_file = CIFFile.read(file_stream)
        structure = get_structure(
            cif_file,
            model=1
        )
        structure = structure[structure.chain_id == "A"]
        for atom_ch in chain_iter(structure):
            atom_res = atom_ch[filter_amino_acids(atom_ch)]
            if len(atom_res) == 0 or len(get_residues(atom_res)[0]) < 10:
                continue
            protein_chain = ProteinChain.from_atomarray(atom_ch)
            return ESMProtein.from_protein_chain(protein_chain), acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rcsb_file_list', type=str, required=True)
    args = parser.parse_args()

    # def __build_url(af_id):
    #   (_, acc, tag) = af_id.split("-")
    #    return f"http://csm200-west.rcsb.org/AF_AF{acc}{tag}.bcif.gz"

    def __build_url(af_id):
        return f"https://alphafold.ebi.ac.uk/files/{af_id}-model_v4.cif"

    dataset = BiotiteStructureFromList(
        args.rcsb_file_list,
        __build_url
    )

    dataloader = DataLoader(
        dataset,
        batch_size=2,
        collate_fn=lambda emb: emb
    )

    esm_model = ESM3.from_pretrained(ESM3_OPEN_SMALL, device=device('cpu'))
    for esm_prot_list in dataloader:
        for esm_prot, acc in esm_prot_list:
            protein_tensor = esm_model.encode(esm_prot)
            output = esm_model.forward_and_sample(
                protein_tensor, SamplingConfig(return_per_residue_embeddings=True)
            )
            print(acc, output.per_residue_embedding.shape)

