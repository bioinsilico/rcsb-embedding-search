"""Heads that predict Foldseek 3Di states from frozen ESM3 residue embeddings.

Every head has the same shape - stem, body, output - so a new architecture is a
subclass implementing ``_body`` (plus its own ``__init__``), and switching between
them is a ``_target_`` change in the training config:

    head:
      _target_: networks.three_di_head.CnnThreeDiHead
      hidden: 512

The stem is fixed for all of them and matters:

* **LayerNorm on the input.** The stored embedding is the last block's output
  *before* ESM3's final LayerNorm, and it has massive-activation channels (dims 1361
  and 1489 reach ~16,000 while the median is ~100). Feeding that to a linear layer
  unnormalized would let two channels dominate every gradient. ESM3's own heads read
  the normalized output, so normalizing here also matches how the backbone is used.
* **Optional ss8 logits**, 11 numbers per residue from the same ESM3 pass, appended
  to the input when ``with_ss8`` is set (an ablation the brief asks for).

The body sees ``(B, L, hidden)`` and returns the same shape; the output layer maps it
to one logit per 3Di state. ``padding_mask`` is True where a position is padding, as
in ``nn.Transformer`` and the rest of this repo.
"""
from __future__ import annotations

import torch
from torch import nn

from dataset.utils.three_di_labels import THREE_DI

N_STATES = len(THREE_DI)
ESM3_DIM = 1536
SS8_DIM = 11


class ThreeDiHead(nn.Module):
    """Base class: input LayerNorm, projection, body, per-residue classifier."""

    def __init__(
            self,
            input_features: int = ESM3_DIM,
            hidden: int = 512,
            n_states: int = N_STATES,
            with_ss8: bool = False,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.with_ss8 = with_ss8
        self.n_states = n_states
        in_features = input_features + (SS8_DIM if with_ss8 else 0)
        self.norm = nn.LayerNorm(in_features)
        self.project = nn.Linear(in_features, hidden)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, n_states)

    def _body(self, x: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        """Map (B, L, hidden) to (B, L, hidden). Overridden by each architecture."""
        raise NotImplementedError

    def forward(
            self,
            embedding: torch.Tensor,
            padding_mask: torch.Tensor | None = None,
            ss8: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.with_ss8:
            if ss8 is None:
                raise ValueError("head was built with with_ss8=True but no ss8 logits were given")
            embedding = torch.cat((embedding, ss8), dim=-1)
        elif ss8 is not None:
            # Silently ignoring them would make an ss8 ablation report a result it never ran.
            raise ValueError("ss8 logits were given but the head was built with with_ss8=False")
        x = self.dropout(self.project(self.norm(embedding)))
        x = self._body(x, padding_mask)
        return self.classifier(x)

    def get_weights(self):
        return [(name, param) for name, param in self.named_parameters()]


class LinearThreeDiHead(ThreeDiHead):
    """Linear probe: how much 3Di is decodable from one residue's embedding alone.

    The floor every other head has to beat, and the quickest check that embeddings
    and labels are correctly aligned - a broken join scores near the 20% majority
    class, a working one well above it.
    """

    def __init__(self, input_features: int = ESM3_DIM, hidden: int = 512,
                 with_ss8: bool = False, dropout: float = 0.0):
        super().__init__(input_features=input_features, hidden=hidden,
                         with_ss8=with_ss8, dropout=dropout)

    def _body(self, x: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        return x


class CnnThreeDiHead(ThreeDiHead):
    """Two 1D convolutions - the architecture published 3Di heads use.

    A 3Di state describes local backbone geometry, and the sequence-local part of it
    spans a few residues, so two layers with kernel 5-7 (receptive field 9-13) cover
    it. Whatever is non-local was already mixed in by ESM3's 48 attention blocks.
    """

    def __init__(self, input_features: int = ESM3_DIM, hidden: int = 512,
                 kernel_size: int = 7, num_layers: int = 2, with_ss8: bool = False,
                 dropout: float = 0.1):
        super().__init__(input_features=input_features, hidden=hidden,
                         with_ss8=with_ss8, dropout=dropout)
        if kernel_size % 2 == 0:
            # padding=kernel_size//2 only preserves the length for odd kernels.
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden, hidden, kernel_size=kernel_size, padding=kernel_size // 2),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])

    def _body(self, x: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        for layer, norm in zip(self.layers, self.layer_norms):
            # Zero the padding before convolving: a kernel reaching past the end of a
            # domain must not pick up another sample's values or stale activations.
            if padding_mask is not None:
                x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
            x = norm(x + layer(x.transpose(1, 2)).transpose(1, 2))
        return x


class TransformerThreeDiHead(ThreeDiHead):
    """A small transformer encoder over the frozen embeddings.

    The ablation against the CNN: it can attend to a residue's structural partner,
    which may sit far away in sequence and which a convolution cannot reach - but it
    re-does work the frozen backbone already did, on far less data.
    """

    def __init__(self, input_features: int = ESM3_DIM, hidden: int = 256,
                 nhead: int = 8, num_layers: int = 2, dim_feedforward: int = 1024,
                 with_ss8: bool = False, dropout: float = 0.1):
        super().__init__(input_features=input_features, hidden=hidden,
                         with_ss8=with_ss8, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def _body(self, x: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        # Padded rows come back as NaN when a row is fully masked; keep them finite so
        # a masked loss never sees NaN gradients.
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        return x
