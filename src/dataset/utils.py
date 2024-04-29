import torch


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
