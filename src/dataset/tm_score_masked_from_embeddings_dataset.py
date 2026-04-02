import argparse
import logging
import os

import torch
from torch.utils.data import DataLoader

from dataset.tm_score_from_embeddings_dataset import TmScoreFromEmbeddingsDataset
from dataset.utils.custom_weighted_random_sampler import CustomWeightedRandomSampler
from dataset.utils.neighbor_mask import neighbors_tsv_to_src_mask
from dataset.utils.tm_score_weight import tm_score_weights, fraction_score
from dataset.utils.tools import collate_seq_embeddings, collate_label

logger = logging.getLogger(__name__)


def collate_attn_masks(mask_list):
    """Pad square boolean attention masks to the max sequence length in the batch.

    Input masks have variable shapes ``(s_i, s_i)`` where ``s_i`` is the
    sequence length of sample *i*.  The output is a single tensor of shape
    ``(batch, max_len, max_len)`` where extra positions are set to ``True``
    (masked out), which is consistent with the PyTorch ``attn_mask`` convention.
    """
    max_len = max(m.size(0) for m in mask_list)
    batch_size = len(mask_list)
    padded = torch.ones(batch_size, max_len, max_len, dtype=torch.bool)
    for i, m in enumerate(mask_list):
        s = m.size(0)
        padded[i, :s, :s] = m
    return padded


def collate_masked_fn(tuple_list):
    """Collate function for :class:`TmScoreMaskedFromEmbeddingsDataset`.

    Each sample is a 5-tuple ``(emb_i, attn_mask_i, emb_j, attn_mask_j, score)``.

    Returns
    -------
    tuple
        ``((padded_x, pad_mask_x, attn_mask_x),``
        ``(padded_y, pad_mask_y, attn_mask_y),``
        ``labels)``

        where ``padded_*`` has shape ``(B, L, D)``, ``pad_mask_*`` has shape
        ``(B, L)`` and ``attn_mask_*`` has shape ``(B, L, L)``.
    """
    return (
        (
            *collate_seq_embeddings([x for x, xm, y, ym, z in tuple_list]),
            collate_attn_masks([xm for x, xm, y, ym, z in tuple_list]),
        ),
        (
            *collate_seq_embeddings([y for x, xm, y, ym, z in tuple_list]),
            collate_attn_masks([ym for x, xm, y, ym, z in tuple_list]),
        ),
        collate_label([z for x, xm, y, ym, z in tuple_list]),
    )


class TmScoreMaskedFromEmbeddingsDataset(TmScoreFromEmbeddingsDataset):
    """Dataset that extends :class:`TmScoreFromEmbeddingsDataset` with
    per-residue spatial attention masks.

    For every protein pair the dataset returns the ESM3 embeddings, the
    Ca-neighbor attention masks (pre-computed with ``ca_neighbors --esm3-offsets``),
    and the TM-score label.

    The attention masks are square boolean tensors of shape ``(seq_len, seq_len)``
    where ``True`` means the position is masked out.  They are intended to be
    passed as the ``mask`` (``src_mask``) argument of
    ``torch.nn.TransformerEncoder``.

    Neighbor TSV files are expected at ``{neighbor_path}/{domain_id}.tsv``
    following the same naming convention as the embedding files.

    Parameters
    ----------
    tm_score_file : str
        Path to the TSV file with domain pairs and TM-scores.
    embedding_path : str
        Directory containing pre-computed ``.pt`` embedding files.
    neighbor_path : str
        Directory containing neighbor ``.tsv`` files written with
        ``ca_neighbors --esm3-offsets`` (no ``--labels``).
    score_method : callable, optional
        Function to transform raw TM-scores (default: binary_score(0.7)).
    weighting_method : callable, optional
        Function to compute sample weights (default: binary_weights(0.7)).
    exclude_domains_file : str, optional
        Path to a file listing domain IDs to exclude.
    """

    def __init__(
        self,
        tm_score_file,
        embedding_path,
        neighbor_path,
        score_method=None,
        weighting_method=None,
        exclude_domains_file=None,
    ):
        self.neighbor_path = neighbor_path
        super().__init__(
            tm_score_file=tm_score_file,
            embedding_path=embedding_path,
            score_method=score_method,
            weighting_method=weighting_method,
            exclude_domains_file=exclude_domains_file,
        )

    def __getitem__(self, idx):
        embedding_i, embedding_j, score = super().__getitem__(idx)

        domain_i = self.class_pairs.iloc[idx]['domain_i']
        domain_j = self.class_pairs.iloc[idx]['domain_j']

        attn_mask_i = neighbors_tsv_to_src_mask(
            os.path.join(self.neighbor_path, f"{domain_i}.tsv"),
            seq_len=embedding_i.size(0),
        )
        attn_mask_j = neighbors_tsv_to_src_mask(
            os.path.join(self.neighbor_path, f"{domain_j}.tsv"),
            seq_len=embedding_j.size(0),
        )

        return embedding_i, attn_mask_i, embedding_j, attn_mask_j, score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tm_score_file', type=str, required=True)
    parser.add_argument('--embedding_path', type=str, required=True)
    parser.add_argument('--neighbor_path', type=str, required=True)
    args = parser.parse_args()

    dataset = TmScoreMaskedFromEmbeddingsDataset(
        args.tm_score_file,
        args.embedding_path,
        args.neighbor_path,
        score_method=fraction_score,
        weighting_method=tm_score_weights(5, 0.25)
    )
    weights = dataset.weights()
    sampler = CustomWeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True,
    )
    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=16,
        collate_fn=collate_masked_fn
    )
    for (x, x_pad_mask, x_attn_mask), (y, y_pad_mask, y_attn_mask), z in dataloader:
        print(f"x: {x.shape}, x_pad_mask: {x_pad_mask.shape}, x_attn_mask: {x_attn_mask.shape}")
        print(f"y: {y.shape}, y_pad_mask: {y_pad_mask.shape}, y_attn_mask: {y_attn_mask.shape}")
        print(f"z: {z.shape}")
        break
