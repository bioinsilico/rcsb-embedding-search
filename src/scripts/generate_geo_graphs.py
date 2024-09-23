import argparse
import os
import itertools
from pathlib import Path

import torch
from torch_geometric.data import Data

from dataset.utils.biopython_getter import get_coords_from_pdb_file
from dataset.utils.geometric_tools import get_res_attr


def main(pdb_path, out_path):
    for pdb_file in os.listdir(pdb_path):
        coords = get_coords_from_pdb_file(os.path.join(pdb_path, pdb_file))
        cas = list(itertools.chain.from_iterable(coords['cas']))
        labels = list(itertools.chain.from_iterable(coords['labels']))
        graph_nodes, graph_edges, edge_attr = get_res_attr(list(zip(cas,labels)))
        graph = Data(
            graph_nodes,
            edge_index=graph_edges.t().contiguous(),
            edge_attr=edge_attr
        )
        torch.save(graph, f"{out_path}/{Path(pdb_file).stem}.pt")
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    args = parser.parse_args()
    main(args.pdb_path, args.out_path)
