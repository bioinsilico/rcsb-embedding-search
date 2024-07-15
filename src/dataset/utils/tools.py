import os
import torch
import polars as pl
from polars import Schema, String, Float32


def load_class_pairs(tm_score_file):
    return pl.DataFrame(
        data=[(lambda row: row.strip().split(","))(row) for row in open(tm_score_file)],
        orient="row",
        schema=Schema({'domain_i': String, 'domain_j': String, 'score': Float32})
    )


def load_tensors(tensor_path, tm_score_file):
    return pl.DataFrame(
        data=[
            (dom_id, os.path.join(tensor_path, f"{dom_id}.pt")) for dom_id in list(set(
                [(lambda row: row.strip().split(",")[0])(row) for row in open(tm_score_file)] +
                [(lambda row: row.strip().split(",")[1])(row) for row in open(tm_score_file)]
            ))
        ],
        orient="row",
        schema=['domain', 'embedding'],
    )


def collate_seq_embeddings(batch_list):
    """
    Pads the tensors in a batch to the same size.

    Args:
        batch_list: A list of samples.

    Returns:
        A batch tensor.
    """
    max_len = max([len(sample) for sample in batch_list])
    dim = batch_list[0].shape[1]
    padded_batch = []
    mask_batch = []
    for sample in batch_list:
        padded_sample = torch.zeros([max_len, dim])
        padded_sample[:len(sample)] = sample
        padded_batch.append(padded_sample)

        mask_sample = torch.ones([max_len])
        mask_sample[:len(sample)] = torch.zeros([len(sample)])
        mask_batch.append(mask_sample.bool())

    return torch.stack(padded_batch, dim=0), torch.stack(mask_batch, dim=0)


def collate_label(label_list):
    return torch.stack(label_list, dim=0)


def collate_fn(tuple_list):
    return (
        collate_seq_embeddings([x for x, y, z in tuple_list]),
        collate_seq_embeddings([y for x, y, z in tuple_list]),
        collate_label([z for x, y, z in tuple_list]),
    )


def triplet_collate_fn(tuple_list):
    return (
        collate_seq_embeddings([x for x, y, z in tuple_list]),
        collate_seq_embeddings([y for x, y, z in tuple_list]),
        collate_seq_embeddings([z for x, y, z in tuple_list])
    )
