import random

import numpy as np
import torch
import esm
import torch_geometric.nn as gnn
from torch_geometric.data import Data, Batch
from torch_geometric.utils import unbatch

ESM_ALPHABET = esm.data.Alphabet.from_architecture("ESM-1b")


def graph_builder(coords, eps=8, num_workers=1, add_change=lambda *x: x):
    random.shuffle(coords)
    structure, sequence = add_change(
        [xyz for ch in coords for xyz in ch['cas']],
        [aa for ch in coords for aa in ch['seq']]
    )
    return __graph_builder(
        torch.from_numpy(np.array(structure)),
        "".join(sequence),
        eps,
        num_workers
    )


def embedding_builder(coords, pst_model, eps=8, num_workers=1, add_change=lambda *x: x):
    random.shuffle(coords)
    structure, sequence = add_change(
        [xyz for ch in coords for xyz in ch['cas']],
        [aa for ch in coords for aa in ch['seq']]
    )
    graph = Batch.from_data_list([__graph_builder(
        torch.from_numpy(np.array(structure)),
        "".join(sequence),
        eps,
        num_workers
    )])
    embedding = pst_model(
        graph,
        return_repr=True,
        aggr=None
    )
    return unbatch(
        embedding[graph.idx_mask],
        graph.batch[graph.idx_mask]
    )[0]


def __graph_builder(structure, sequence, eps=8, num_workers=1):
    edge_index = gnn.radius_graph(
        structure, r=eps, loop=False, num_workers=num_workers
    )
    edge_index += 1  # shift for cls_idx
    x = torch.cat(
        [
            torch.LongTensor([ESM_ALPHABET.cls_idx]),
            torch.LongTensor([
                ESM_ALPHABET.get_idx(res) for res in
                ESM_ALPHABET.tokenize(sequence)
            ]),
            torch.LongTensor([ESM_ALPHABET.eos_idx]),
        ]
    )
    idx_mask = torch.zeros_like(x, dtype=torch.bool)
    idx_mask[1:-1] = True
    return Data(x=x, edge_index=edge_index, idx_mask=idx_mask)
