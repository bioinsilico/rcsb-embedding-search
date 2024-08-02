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


def no_change(*lists):
    return lists


def remove_random_residues(n):
    def __remove_n_random_elements_from_multiple_lists(*lists):
        if not lists:
            raise ValueError("At least one list must be provided")
        list_lengths = [len(lst) for lst in lists]
        if any(length < n for length in list_lengths):
            raise ValueError("n cannot be greater than the length of any of the lists")
        indices_to_remove = random.sample(range(list_lengths[0]), n)
        print(indices_to_remove)
        for lst in lists:
            for index in sorted(indices_to_remove, reverse=True):
                del lst[index]
        return lists

    return __remove_n_random_elements_from_multiple_lists


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
