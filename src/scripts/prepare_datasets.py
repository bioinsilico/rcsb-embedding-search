import argparse
import os

import pandas as pd

from dataset.utils.biopython_getter import get_coords_from_pdb_file, get_coords_from_cif_file
from dataset.utils.tools import load_class_pairs_with_self_comparison


def prepare_dataset_from_coords(tm_score_file, coords_path, ext, out_file):
    class_pairs = load_class_pairs_with_self_comparison(tm_score_file)
    pd_coords = pd.DataFrame(
        data=[
            (lambda coords: (
                dom_id,
                coords['cas'],
                coords['seq']
            ))(
                get_coords_from_pdb_file(os.path.join(coords_path, f"{dom_id}.{ext}")) if ext == "pdb" or ext == "ent" else
                get_coords_from_cif_file(os.path.join(coords_path, f"{dom_id}.{ext}"))
            )
            for dom_id in pd.concat([
                class_pairs["domain_i"], class_pairs["domain_j"]
            ]).unique()
        ],
        columns=['domain', 'cas', 'seq']
    )
    pd.DataFrame.to_pickle(pd_coords, out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tm_score_file', type=str, required=True)
    parser.add_argument('--pdb_path', type=str, required=True)
    parser.add_argument('--ext', type=str, required=True)
    parser.add_argument('--out_file', type=str, required=True)
    args = parser.parse_args()

    prepare_dataset_from_coords(
        args.tm_score_file,
        args.pdb_path,
        args.ext,
        args.out_file
    )
