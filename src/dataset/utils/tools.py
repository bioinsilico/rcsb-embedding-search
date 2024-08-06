import os
import torch
import polars as pl
from polars import Schema, String, Float32


def load_class_pairs(tm_score_file):
    with open(tm_score_file) as file:
        return pl.DataFrame(
            data=[(lambda row: row.strip().split(","))(row) for row in file],
            orient="row",
            schema=Schema({'domain_i': String, 'domain_j': String, 'score': Float32})
        )


def load_class_pairs_with_self_comparison(tm_score_file):
    with open(tm_score_file) as file:
        return pl.DataFrame(
            data=[(lambda row: row.strip().split(","))(row) for row in file] + [[r, r, 1.] for r in list(set(
                [(lambda row: row.strip().split(",")[0])(row) for row in open(tm_score_file)] +
                [(lambda row: row.strip().split(",")[1])(row) for row in open(tm_score_file)]
            ))],
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
        batch_list (list of torch.Tensor): A list of samples, where each sample is a tensor of shape (sequence_length, embedding_dim).

    Returns:
        tuple: A tuple containing:
            - padded_batch (torch.Tensor): A tensor of shape (batch_size, max_seq_length, embedding_dim), where each sample is padded to the max sequence length.
            - mask_batch (torch.Tensor): A tensor of shape (batch_size, max_seq_length) where padded positions are marked as False.
    """
    device = batch_list[0].device  # Get the device of the input tensors
    max_len = max(sample.size(0) for sample in batch_list)  # Determine the maximum sequence length
    dim = batch_list[0].size(1)  # Determine the embedding dimension
    batch_size = len(batch_list)  # Determine the batch size

    # Initialize tensors for the padded batch and masks on the same device as the input tensors
    padded_batch = torch.zeros((batch_size, max_len, dim), dtype=batch_list[0].dtype, device=device)
    mask_batch = torch.ones((batch_size, max_len), dtype=torch.bool, device=device)

    for i, sample in enumerate(batch_list):
        seq_len = sample.size(0)  # Get the length of the current sequence
        padded_batch[i, :seq_len] = sample  # Pad the sequence with zeros
        mask_batch[i, :seq_len] = False  # Set mask positions for the actual data to False

    return padded_batch, mask_batch


def collate_label(label_list):
    return torch.stack(label_list, dim=0)


def collate_dual_label(label_list):
    return torch.stack([x for x, y in label_list], dim=0), torch.stack([y for x, y in label_list], dim=0),


def collate_fn(tuple_list):
    return (
        collate_seq_embeddings([x for x, y, z in tuple_list]),
        collate_seq_embeddings([y for x, y, z in tuple_list]),
        collate_label([z for x, y, z in tuple_list]),
    )


def collate_dual_fn(tuple_list):
    return (
        collate_seq_embeddings([x for x, y, z in tuple_list]),
        collate_seq_embeddings([y for x, y, z in tuple_list]),
        collate_dual_label([z for x, y, z in tuple_list]),
    )


def triplet_collate_fn(tuple_list):
    return (
        collate_seq_embeddings([x for x, y, z in tuple_list]),
        collate_seq_embeddings([y for x, y, z in tuple_list]),
        collate_seq_embeddings([z for x, y, z in tuple_list])
    )
