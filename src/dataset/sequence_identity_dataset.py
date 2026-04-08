import argparse
import logging

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from dataset.utils.tm_score_weight import binary_score, binary_weights, fraction_score, tm_score_weights, \
    fraction_score_of
from networks.sequence_autoencoder import tokenize_sequence, AA_PAD_IDX

d_type = np.float32

logger = logging.getLogger(__name__)


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


class SequenceIdentityDataset(Dataset):
    """Paired protein-sequence dataset with sequence identity labels.

    Reads sequences from a FASTA file and pairs + scores from a
    headerless 3-column TSV (``header_i \\t header_j \\t identity``).

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
    ):
        self.score_method = binary_score(self.BINARY_THR) if score_method is None else score_method
        self.weighting_method = binary_weights(self.BINARY_THR) if weighting_method is None else weighting_method

        self.sequences = parse_fasta(fasta_file)
        self.pairs = self._load_pairs(identity_file, exclude_ids_file)

        missing = (
            set(self.pairs['domain_i']) | set(self.pairs['domain_j'])
        ) - set(self.sequences)
        if missing:
            logger.warning(f"{len(missing)} identifiers in the pairs file have no sequence in the FASTA — dropping affected rows")
            self.pairs = self.pairs[
                self.pairs['domain_i'].isin(self.sequences) & self.pairs['domain_j'].isin(self.sequences)
            ].reset_index(drop=True)

        logger.info(f"Total pairs: {len(self.pairs)} | Unique sequences: {len(self.sequences)}")

        # Pre-tokenize all sequences for speed
        self._token_cache: dict[str, torch.Tensor] = {
            header: torch.tensor(tokenize_sequence(seq), dtype=torch.long)
            for header, seq in self.sequences.items()
        }

    @staticmethod
    def _load_pairs(identity_file: str, exclude_ids_file: str | None) -> pd.DataFrame:
        dtype = {
            'domain_i': 'str',
            'domain_j': 'str',
            'score': 'float32',
        }
        df = pd.read_csv(
            identity_file,
            delimiter='\t',
            header=None,
            index_col=None,
            names=['domain_i', 'domain_j', 'score'],
            dtype=dtype,
            comment='#',
        )
        # Skip a header row if the first value is non-numeric
        if len(df) > 0:
            try:
                float(df.iloc[0]['score'])
            except (ValueError, TypeError):
                df = df.iloc[1:].reset_index(drop=True)
                df['score'] = df['score'].astype('float32')

        if exclude_ids_file is not None:
            with open(exclude_ids_file) as f:
                exclude = {line.strip() for line in f}
            df = df[~df['domain_i'].isin(exclude) & ~df['domain_j'].isin(exclude)]

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
