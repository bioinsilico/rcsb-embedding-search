import os
import torch


def load_class_pairs(tm_score_file):
    domains = set({})
    class_pairs = []
    for row in open(tm_score_file):
        r = row.strip().split(",")
        dom_i = r[0]
        dom_j = r[1]
        tm_score = float(r[2])
        domains.add(dom_i)
        domains.add(dom_j)
        class_pairs.append([
            tm_score,
            dom_i,
            dom_j
        ])
    print(f"Total pairs: {len(class_pairs)}")
    return domains, class_pairs


def load_tensors(tensor_path, file_name_list):
    data_map = {}
    for dom_id in file_name_list:
        data_map[dom_id] = torch.load(os.path.join(tensor_path, f"{dom_id}.pt"))
    return data_map


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



