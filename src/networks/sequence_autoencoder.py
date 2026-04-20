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

    Decoder:  Autoregressive TransformerDecoder with causal masking.

              Memory = latent → Linear → (B, 1, d_model).
              Target = token embeddings + positional encoding.

              Training uses teacher forcing: input ``[<bos>, A, …, V]`` →
              target ``[A, …, V, <eos>]``.  Inference generates one token
              at a time until ``<eos>`` or ``max_seq_len``.
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
        self.max_seq_len = max_seq_len
        self.eos_idx = AA_TOKENS['<eos>']

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

        # --- Decoder (autoregressive) ---
        self.from_latent = nn.Linear(latent_dim, d_model)
        self.decoder_pos_encoding = SinusoidalPositionalEncoding(d_model, max_seq_len, self.dropout)
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

    def decode(self, latent: torch.Tensor, tgt_tokens: torch.Tensor) -> torch.Tensor:
        """Teacher-forced decoding (training).

        Args:
            latent: (B, latent_dim) from :meth:`encode`.
            tgt_tokens: (B, L) decoder input tokens, e.g.
                ``[<bos>, A, R, …, V]`` (the target sequence shifted right
                by one with ``<bos>`` prepended).

        Returns:
            (B, L, vocab_size) logits.  The target for cross-entropy is the
            original un-shifted sequence ``[A, R, …, V, <eos>]``.
        """
        memory = self.from_latent(latent).unsqueeze(1)  # (B, 1, d_model)
        tgt = self.decoder_pos_encoding(self.aa_embedding(tgt_tokens))

        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            tgt_tokens.size(1), device=tgt.device,
        )
        tgt_pad_mask = (tgt_tokens == self.pad_idx)

        out = self.decoder(
            tgt, memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_pad_mask,
        )
        return self.output_proj(out)

    @torch.no_grad()
    def decode_sequence(self, latent: torch.Tensor) -> list[torch.Tensor]:
        """Autoregressive greedy decoding at inference.

        Generates one token at a time, feeding each prediction back as
        input, until every sequence in the batch has produced ``<eos>``
        or ``max_seq_len`` tokens have been generated.

        Args:
            latent: (B, latent_dim) embedding from :meth:`encode`.

        Returns:
            List of B 1-D long tensors, each containing the predicted amino
            acid token indices up to (but not including) ``<eos>``.
        """
        B = latent.size(0)
        memory = self.from_latent(latent).unsqueeze(1)  # (B, 1, d_model)

        generated = torch.full(
            (B, 1), AA_TOKENS['<bos>'], dtype=torch.long, device=latent.device,
        )
        finished = torch.zeros(B, dtype=torch.bool, device=latent.device)

        for _ in range(self.max_seq_len):
            tgt = self.decoder_pos_encoding(self.aa_embedding(generated))
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                generated.size(1), device=latent.device,
            )
            out = self.decoder(tgt, memory, tgt_mask=causal_mask)
            next_logits = self.output_proj(out[:, -1, :])      # (B, vocab)
            next_token = next_logits.argmax(dim=-1, keepdim=True)  # (B, 1)

            next_token = next_token.masked_fill(finished.unsqueeze(1), self.pad_idx)
            generated = torch.cat([generated, next_token], dim=1)

            finished = finished | (next_token.squeeze(1) == self.eos_idx)
            if finished.all():
                break

        # Strip leading <bos>, truncate each sequence at its first <eos>
        sequences = []
        for row in generated[:, 1:]:
            eos_positions = (row == self.eos_idx).nonzero(as_tuple=False)
            end = eos_positions[0].item() if len(eos_positions) > 0 else row.size(0)
            sequences.append(row[:end])
        return sequences

    def forward(
        self,
        tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass with teacher-forced decoding.

        Args:
            tokens: (B, L) amino acid token indices ending with ``<eos>``,
                padded with ``<pad>``.

        Returns:
            logits: (B, L, vocab_size) reconstruction logits whose target
                is ``tokens`` itself (the shifted input is built internally).
            latent: (B, latent_dim) L2-normalized embedding.
        """
        latent = self.encode(tokens)

        # Build shifted decoder input: [<bos>] + tokens[:, :-1]
        bos = torch.full(
            (tokens.size(0), 1), AA_TOKENS['<bos>'],
            dtype=torch.long, device=tokens.device,
        )
        decoder_input = torch.cat([bos, tokens[:, :-1]], dim=1)

        logits = self.decode(latent, decoder_input)
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
