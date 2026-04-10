import argparse
import logging
import math
import random

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from dataset.utils.tm_score_weight import binary_score, binary_weights, fraction_score, tm_score_weights, \
    fraction_score_of
from networks.sequence_autoencoder import tokenize_sequence, AA_PAD_IDX

d_type = np.float32

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 100_000  # rows per pandas read_csv chunk


def parse_fasta(fasta_file: str) -> dict[str, str]:
    """Read a FASTA file and return a dict mapping headers to sequences."""
    sequences = {}
    header = None
    parts = []
    with open(fasta_file) as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if header is not None:
                    sequences[header] = ''.join(parts)
                header = line[1:].split()[0]
                parts = []
            elif header is not None:
                parts.append(line)
        if header is not None:
            sequences[header] = ''.join(parts)
    logger.info(f"Loaded {len(sequences)} sequences from {fasta_file}")
    return sequences


def collate_sequence_pairs(tuple_list):
    """Collate function for SequenceIdentityDataset batches.

    Pads token sequences to the longest in the batch and returns
    ``(tokens_i, tokens_j, scores)`` where token tensors have shape
    ``(B, L)`` with ``AA_PAD_IDX`` in padding positions.
    """
    tokens_i_list, tokens_j_list, score_list = zip(*tuple_list)
    tokens_i = _pad_tokens(tokens_i_list)
    tokens_j = _pad_tokens(tokens_j_list)
    scores = torch.stack(score_list, dim=0)
    return tokens_i, tokens_j, scores


def _pad_tokens(token_list: list[torch.Tensor]) -> torch.Tensor:
    """Pad a list of 1-D long tensors to equal length."""
    max_len = max(t.size(0) for t in token_list)
    batch_size = len(token_list)
    padded = torch.full((batch_size, max_len), AA_PAD_IDX, dtype=torch.long)
    for i, t in enumerate(token_list):
        padded[i, :t.size(0)] = t
    return padded


def _load_exclude(exclude_ids_file: str | None) -> set[str]:
    if exclude_ids_file is None:
        return set()
    with open(exclude_ids_file) as f:
        return {line.strip() for line in f}


def _strip_header_row(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the first row if its score column is non-numeric (i.e. a header)."""
    if len(df) == 0:
        return df
    try:
        float(df.iloc[0]['score'])
    except (ValueError, TypeError):
        df = df.iloc[1:].reset_index(drop=True)
        df['score'] = df['score'].astype('float32')
    return df


class SequenceIdentityDataset(Dataset):
    """Paired protein-sequence dataset with sequence identity labels.

    Reads sequences from a FASTA file and pairs + scores from a
    3-column TSV (``header_i \\t header_j \\t identity``).

    For very large TSV files (billions of pairs) use ``max_pairs`` to cap
    memory usage.  Pairs are selected with **stratified reservoir sampling**:
    the score range [0, 1] is divided into ``n_intervals`` equal bins and up
    to ``max_pairs // n_intervals`` pairs are kept per bin, giving a balanced
    sample without loading the whole file into memory.

    Each ``__getitem__`` call returns::

        (tokens_i, tokens_j, score)

    where ``tokens_*`` are 1-D long tensors of amino acid token indices
    and ``score`` is a scalar float tensor.

    Use :func:`collate_sequence_pairs` as the DataLoader collate function
    to pad variable-length sequences within a batch.
    """

    BINARY_THR = 0.7

    def __init__(
        self,
        fasta_file: str,
        identity_file: str,
        score_method=None,
        weighting_method=None,
        exclude_ids_file: str | None = None,
        max_pairs: int | None = None,
        n_intervals: int = 5,
    ):
        self.score_method = binary_score(self.BINARY_THR) if score_method is None else score_method
        self.weighting_method = binary_weights(self.BINARY_THR) if weighting_method is None else weighting_method

        self.sequences = parse_fasta(fasta_file)
        exclude = _load_exclude(exclude_ids_file)

        if max_pairs is None:
            self.pairs = self._load_pairs_full(identity_file, exclude)
        else:
            self.pairs = self._load_pairs_sampled(identity_file, exclude, max_pairs, n_intervals)

        known = set(self.sequences)
        missing_mask = ~(self.pairs['domain_i'].isin(known) & self.pairs['domain_j'].isin(known))
        if missing_mask.any():
            logger.warning(
                f"{missing_mask.sum()} pairs reference IDs absent from the FASTA — dropping affected rows"
            )
            self.pairs = self.pairs[~missing_mask].reset_index(drop=True)

        logger.info(f"Total pairs: {len(self.pairs)} | Unique sequences: {len(self.sequences)}")

        # Pre-tokenize only sequences that appear in the retained pairs
        referenced = set(self.pairs['domain_i']) | set(self.pairs['domain_j'])
        self._token_cache: dict[str, torch.Tensor] = {
            h: torch.tensor(tokenize_sequence(self.sequences[h]), dtype=torch.long)
            for h in referenced
        }

    # ------------------------------------------------------------------
    # Pair loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_pairs_full(identity_file: str, exclude: set[str]) -> pd.DataFrame:
        """Load the entire TSV into memory (suitable for small-to-medium files)."""
        logger.info(f"Loading {identity_file} identity values.")
        dtype = {'domain_i': 'str', 'domain_j': 'str', 'score': 'float32'}
        df = pd.read_csv(
            identity_file,
            delimiter='\t',
            header=None,
            index_col=None,
            names=['domain_i', 'domain_j', 'score'],
            dtype=dtype,
            comment='#',
        )
        df = _strip_header_row(df)
        if exclude:
            df = df[~df['domain_i'].isin(exclude) & ~df['domain_j'].isin(exclude)]
        logger.info(f"Loaded {len(df)} identity values.")
        return df.reset_index(drop=True)

    @staticmethod
    def _load_pairs_sampled(
        identity_file: str,
        exclude: set[str],
        max_pairs: int,
        n_intervals: int,
    ) -> pd.DataFrame:
        """Stream through the TSV and return a stratified reservoir sample.

        The score axis [0, 1] is divided into ``n_intervals`` equal bins.
        Each bin keeps at most ``max_pairs // n_intervals`` rows using
        Algorithm R reservoir sampling, so the final DataFrame has at most
        ``max_pairs`` rows with a balanced score distribution.
        """

        logger.info(f"Loading {identity_file} identity values. Number of pairs: {max_pairs} | intervals: {n_intervals}")

        bucket_size = max_pairs // n_intervals
        # reservoirs[k] = list of (domain_i, domain_j, score) rows for bin k
        reservoirs: list[list[tuple]] = [[] for _ in range(n_intervals)]
        # counts[k] = total rows seen for bin k (including those not kept)
        counts = [0] * n_intervals

        dtype = {'domain_i': 'str', 'domain_j': 'str', 'score': 'float32'}
        reader = pd.read_csv(
            identity_file,
            delimiter='\t',
            header=None,
            index_col=None,
            names=['domain_i', 'domain_j', 'score'],
            dtype=dtype,
            comment='#',
            chunksize=_CHUNK_SIZE,
        )

        first_chunk = True
        total_seen = 0
        for chunk in reader:
            if first_chunk:
                chunk = _strip_header_row(chunk)
                first_chunk = False
            if exclude:
                chunk = chunk[~chunk['domain_i'].isin(exclude) & ~chunk['domain_j'].isin(exclude)]

            for row in chunk.itertuples(index=False):
                score = row.score
                if math.isnan(score):
                    continue
                k = min(int(score * n_intervals), n_intervals - 1)
                counts[k] += 1
                total_seen += 1
                reservoir = reservoirs[k]
                if len(reservoir) < bucket_size:
                    reservoir.append((row.domain_i, row.domain_j, score))
                else:
                    j = random.randrange(counts[k])
                    if j < bucket_size:
                        reservoir[j] = (row.domain_i, row.domain_j, score)

        logger.info(
            f"Reservoir sampling complete: {total_seen:,} pairs scanned, "
            + ", ".join(f"bin{k}={len(reservoirs[k])}" for k in range(n_intervals))
        )

        rows = [row for r in reservoirs for row in r]
        if not rows:
            return pd.DataFrame(columns=['domain_i', 'domain_j', 'score'])
        df = pd.DataFrame(rows, columns=['domain_i', 'domain_j', 'score'])
        df['score'] = df['score'].astype('float32')
        return df.reset_index(drop=True)

    def weights(self):
        return self.weighting_method(
            np.expand_dims(self.pairs['score'].to_numpy(), axis=1)
        )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        tokens_i = self._token_cache[row['domain_i']]
        tokens_j = self._token_cache[row['domain_j']]
        score = torch.tensor(
            self.score_method(row['score']),
            dtype=torch.float32,
        )
        return tokens_i, tokens_j, score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--identity_file', type=str, required=True)
    parser.add_argument('--fasta_file', type=str, required=True)
    args = parser.parse_args()

    dataset = SequenceIdentityDataset(
        fasta_file=args.fasta_file,
        identity_file=args.identity_file,
        score_method=fraction_score_of(100),
        weighting_method=tm_score_weights(10, 0.25)
    )

    dataloader = DataLoader(
        dataset,
        collate_fn=collate_sequence_pairs,
    )

    for seq_i, seq_j, score in dataloader:
        print(seq_i.shape, seq_j.shape, score)
