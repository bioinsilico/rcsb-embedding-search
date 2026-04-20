import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from networks.layers import ResBlock


# Standard amino acid alphabet + special tokens
AA_TOKENS = {
    '<pad>': 0, '<bos>': 1, '<eos>': 2, '<unk>': 3,
    'A': 4, 'R': 5, 'N': 6, 'D': 7, 'C': 8, 'E': 9, 'Q': 10,
    'G': 11, 'H': 12, 'I': 13, 'L': 14, 'K': 15, 'M': 16, 'F': 17,
    'P': 18, 'S': 19, 'T': 20, 'W': 21, 'Y': 22, 'V': 23, 'X': 24,
    'B': 25, 'Z': 26, 'U': 27, 'O': 28,
}
AA_VOCAB_SIZE = len(AA_TOKENS)
AA_PAD_IDX = AA_TOKENS['<pad>']


def tokenize_sequence(seq: str) -> list[int]:
    """Convert an amino acid string to a list of token indices (no BOS/EOS)."""
    return [AA_TOKENS.get(aa, AA_TOKENS['<unk>']) for aa in seq]


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding added to token embeddings."""

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor of shape (B, L, d_model)."""
        return self.dropout(x + self.pe[:, :x.size(1)])


class SequenceAutoencoder(nn.Module):
    """Protein sequence autoencoder with similarity-preserving latent space.

    The encoder maps a variable-length amino acid sequence to a fixed-size
    L2-normalized latent vector.  The decoder reconstructs the sequence from
    that vector.  Because the latent is unit-normalized, the dot product
    between two latent vectors equals their cosine similarity, which is
    trained to approximate pairwise sequence identity.

    Architecture
    ------------
    Encoder:  AA embedding + positional encoding → TransformerEncoder
              → pooling (masked) → projection → L2 normalize

              Pooling strategy depends on ``res_block_layers``:
              - 0 (default): mean pooling → LayerNorm → Linear
              - >0: summation pooling → N × ResBlock → Linear

    Length:   latent → Linear → scalar log-length prediction.
              Trained with MSE on log(true_length).  At inference, call
              :meth:`predict_length` to obtain the integer length, then pass
              it to :meth:`decode` — no ground-truth length needed.

    Decoder:  latent → linear expansion to d_model → (1-token memory)
              Positional queries → TransformerDecoder cross-attending to memory
              → linear → vocab logits
    """

    dropout = 0.1

    def __init__(
        self,
        d_model: int = 512,
        latent_dim: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        max_seq_len: int = 2048,
        vocab_size: int = AA_VOCAB_SIZE,
        pad_idx: int = AA_PAD_IDX,
        res_block_layers: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.pad_idx = pad_idx
        self.res_block_layers = res_block_layers

        # --- Shared layers ---
        self.aa_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len, self.dropout)

        # --- Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=self.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        if res_block_layers == 0:
            # Mean pooling → LayerNorm → Linear
            self.to_latent = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, latent_dim),
            )
        else:
            # Summation pooling → ResBlocks → Linear
            layers = OrderedDict([
                (f'block{i}', ResBlock(d_model, d_model, self.dropout))
                for i in range(res_block_layers)
            ])
            layers['dropout'] = nn.Dropout(p=self.dropout)
            layers['linear'] = nn.Linear(d_model, latent_dim)
            layers['activation'] = nn.ReLU()
            self.to_latent = nn.Sequential(layers)

        # --- Length predictor ---
        # Predicts log(sequence_length) from the latent vector.
        # Log-space keeps gradients well-behaved across the wide protein length range (~20–2000).
        self.length_head = nn.Linear(latent_dim, 1)

        # --- Decoder ---
        self.from_latent = nn.Linear(latent_dim, d_model)
        # Learned position queries — the only input the decoder needs at inference
        # is the target length; no ground-truth tokens required.
        self.decoder_queries = nn.Embedding(max_seq_len, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=self.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def _pad_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """Boolean padding mask: True where token == pad_idx."""
        return tokens == self.pad_idx

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode token indices to an L2-normalized latent vector.

        Args:
            tokens: (B, L) long tensor of amino acid token indices.

        Returns:
            (B, latent_dim) unit-normalized embedding.
        """
        pad_mask = self._pad_mask(tokens)
        x = self.pos_encoding(self.aa_embedding(tokens))
        x = self.encoder(x, src_key_padding_mask=pad_mask)

        # Zero-out padding positions before pooling
        valid = (~pad_mask).unsqueeze(-1).float()  # (B, L, 1)
        if self.res_block_layers == 0:
            # Mean pooling
            pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        else:
            # Summation pooling
            pooled = (x * valid).sum(dim=1)

        latent = self.to_latent(pooled)
        return F.normalize(latent, dim=-1)

    def predict_length(self, latent: torch.Tensor) -> torch.Tensor:
        """Predict sequence length from a latent vector.

        Returns the predicted log-length as a ``(B,)`` float tensor.
        Use this during training to compute the length loss against
        ``torch.log(true_lengths.float())``.

        At inference, convert to an integer length with::

            seq_len = model.predict_length(latent).exp().round().long().clamp(min=1)
        """
        return self.length_head(latent).squeeze(-1)

    def decode(self, latent: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Decode a latent vector into per-position vocabulary logits.

        Fully non-autoregressive: the decoder uses learned position queries
        and cross-attends to the latent vector.  No ground-truth tokens are
        needed, so the same call works identically during training and inference.

        Args:
            latent: (B, latent_dim) L2-normalized embedding from :meth:`encode`.
            seq_len: number of positions to decode (typically the input length
                     during training; a desired output length at inference).

        Returns:
            (B, seq_len, vocab_size) logits over the amino acid vocabulary.
        """
        B = latent.size(0)

        # Memory: the latent as a single-token sequence for cross-attention
        memory = self.from_latent(latent).unsqueeze(1)  # (B, 1, d_model)

        # Position queries: (B, seq_len, d_model) — no target tokens needed
        positions = torch.arange(seq_len, device=latent.device)
        tgt = self.decoder_queries(positions).unsqueeze(0).expand(B, -1, -1)

        out = self.decoder(tgt, memory)
        return self.output_proj(out)

    def forward(
        self,
        tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass returning logits and latent embedding.

        Args:
            tokens: (B, L) amino acid token indices (padded).

        Returns:
            logits: (B, L, vocab_size) reconstruction logits.
            latent: (B, latent_dim) L2-normalized embedding.
        """
        latent = self.encode(tokens)
        logits = self.decode(latent, tokens.size(1))
        return logits, latent

    def embedding(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convenience alias for encode — returns the latent embedding."""
        return self.encode(tokens)

    def cosine_similarity(
        self,
        tokens_i: torch.Tensor,
        tokens_j: torch.Tensor,
    ) -> torch.Tensor:
        """Cosine similarity between two sequences' latent embeddings."""
        return F.cosine_similarity(self.encode(tokens_i), self.encode(tokens_j))
