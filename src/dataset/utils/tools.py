import os
import torch
import pandas as pd


def load_classes(class_file):
    dtype = {
        'domain': 'str',
        'class': 'str'
    }
    return pd.read_csv(
        class_file,
        delimiter="\t",
        header=None,
        index_col=None,
        names=['domain', 'class'],
        dtype=dtype
    )


def load_class_pairs(tm_score_file):
    dtype = {
        'domain_i': 'str',
        'domain_j': 'str',
        'score': 'float32'
    }
    return pd.read_csv(
        tm_score_file,
        header=None,
        index_col=None,
        names=['domain_i', 'domain_j', 'score'],
        dtype=dtype
    )


def get_unique_pairs(tm_score_file):
    class_pairs = load_class_pairs(tm_score_file)
    return list(pd.concat([
        class_pairs["domain_i"], class_pairs["domain_j"]
    ]).unique())


def load_class_pairs_with_self_comparison(tm_score_file):
    dtype = {
        'domain_i': 'str',
        'domain_j': 'str',
        'score': 'float32'
    }
    df = pd.read_csv(
        tm_score_file,
        header=None,
        index_col=None,
        names=['domain_i', 'domain_j', 'score'],
        dtype=dtype
    )
    unique_domains = pd.concat([df["domain_i"], df["domain_j"]]).unique()
    id_df = pd.DataFrame({
        'domain_i': unique_domains,
        'domain_j': unique_domains,
        'score': 1.0
    }, columns=['domain_i', 'domain_j', 'score'])
    return pd.concat([df, id_df], ignore_index=True)


def load_tensors(tensor_path, tm_score_file):
    dtype = {
        'domain_i': 'str',
        'domain_j': 'str',
        'score': 'float32'
    }
    df = pd.read_csv(
        tm_score_file,
        header=False,
        index_col=None,
        dtype=dtype
    )
    return pd.DataFrame(
        data=[
            (dom_id, os.path.join(tensor_path, f"{dom_id}.pt")) for dom_id in pd.concat([df["domain_i"], df["domain_j"]]).unique()
        ],
        names=['domain', 'embedding'],
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
    if batch_list[0] is None:
        return None
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


def pair_at_index(n):
    def __pair_at_index(index):
        total = 0
        for x in range(n):
            count = n - x - 1
            if index < total + count:
                y = x + 1 + (index - total)
                return (x, y)
            total += count

        raise IndexError("Index out of range for the given n")
    return __pair_at_index


if __name__ == "__main__":
    n = 15229
    index = 15228
    f = pair_at_index(n)
    result = f(index)
    print(result)