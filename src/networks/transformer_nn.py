import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

from networks.layers import ResBlock


def masked_pool(x, x_mask, mode="sum", norm=None):
    """Pool residue rows, ignoring padding (x_mask True at padded rows).

    In train mode the transformer leaves non-zero values in padded rows (only the eval-mode
    nested-tensor fast path zeroes them), so they are zeroed before pooling; inference in
    FoldMatch does the same.
    """
    if x_mask is None:
        s = x.sum(dim=x.dim() - 2)
        n = torch.full(s.shape[:-1] + (1,), x.shape[-2], dtype=x.dtype, device=x.device)
    else:
        x = x.masked_fill(x_mask.unsqueeze(-1), 0.0)
        s = x.sum(dim=-2)
        n = (~x_mask).sum(dim=-1, keepdim=True).clamp_min(1).to(x.dtype)
    if mode == "sum":
        return s
    if mode == "mean_ln":
        return norm(s / n)
    raise ValueError(f"unknown pooling {mode}")


class LinearEmbeddingCosine(nn.Module):

    def __init__(self, input_features=1280, hidden_layer=1280):
        super().__init__()
        self.embedding = nn.Sequential(OrderedDict([
            ('norm', nn.LayerNorm(input_features)),
            ('dropout', nn.Dropout(p=0.0)),
            ('linear', nn.Linear(input_features, hidden_layer)),
            ('activation', nn.ReLU())
        ]))

    def forward(self, x, x_mask, y, y_mask):
        return nn.functional.cosine_similarity(
            self.embedding(x.sum(dim=1)),
            self.embedding(y.sum(dim=1))
        )

    def get_weights(self):
        return [(name, param) for name, param in self.embedding.named_parameters()]


class TransformerEmbeddingCosine(nn.Module):
    """Transformer twin network; the defaults reproduce the production architecture.

    Options added for the stage-1 retraining study (all default to the original behaviour):
        output_dim: width of the final Linear (default ``hidden_layer``). The ResBlocks always
            stay at ``input_features`` wide, so the output can be narrowed without breaking them.
        output_activation: 'relu' (original) or 'none' (linear output).
        pooling: 'sum' (original) or 'mean_ln' (masked mean, then LayerNorm).
        norm_first: pre-norm encoder layers, with a final LayerNorm on the encoder output.
        input_norm: standardise every ESM3 dimension with fixed statistics (buffers
            ``input_mean`` / ``input_std``, filled from ``input_stats``, an .npz with ``mean``
            and ``std``, or from a checkpoint).
        length_feature: give the model the chain length, as log(number of residue rows); mean
            pooling otherwise discards the length that sum pooling keeps. Every length term is a
            zero-initialised Linear, so a model starts out exactly as without it.
              True / 'add': a learned Linear(1, input_features) added to the pooled vector (the same as
                  appending it as an extra input to the blocks after pooling);
              'film': 'add' plus a length-dependent scale of the pooled vector, p * (1 + g(len)) + b(len);
              'blocks': 'add' plus a separate additive length term before every ResBlock;
              'input': 'add' plus an additive length term on every residue row before the encoder, so
                  attention sees the length (a transformer without positions cannot count rows).
        length_rbf: encode log(length) as this many Gaussian bumps (centres evenly spaced over
            log 5 .. log 1500 residues, width one spacing) instead of the scalar log(length), so the
            length terms can depend on length non-monotonically (e.g. peptides apart); 0 = scalar.
    """
    dropout = 0.1

    def __init__(
        self,
        input_features=640,
        dim_feedforward=1280,
        hidden_layer=640,
        nhead=10,
        num_layers=6,
        res_block_layers=0,
        output_dim=None,
        output_activation="relu",
        pooling="sum",
        norm_first=False,
        input_norm=False,
        input_stats=None,
        length_feature=False,
        length_rbf=0
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_features,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=self.dropout,
            batch_first=True,
            norm_first=norm_first
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(input_features) if norm_first else None,
            enable_nested_tensor=not norm_first
        )
        self.pooling = pooling
        self.pool_norm = nn.LayerNorm(input_features) if pooling == "mean_ln" else None
        mode = "add" if length_feature is True else (length_feature or None)
        if mode not in (None, "add", "film", "blocks", "input"):
            raise ValueError(f"unknown length_feature {length_feature}")
        self.length_mode = mode
        self.length_proj = self.length_scale = self.length_input = None
        self.length_blocks = None
        if mode:
            k = int(length_rbf) or 1
            if length_rbf:
                centres = torch.linspace(np.log(5.0), np.log(1500.0), k)
                self.register_buffer("length_centres", centres, persistent=False)
                self.length_width = float(centres[1] - centres[0]) if k > 1 else 1.0
            self.length_proj = self._zero_linear(k, input_features)
            if mode == "film":
                self.length_scale = self._zero_linear(k, input_features)
            elif mode == "input":
                self.length_input = self._zero_linear(k, input_features)
            elif mode == "blocks":
                self.length_blocks = nn.ModuleList(self._zero_linear(k, input_features) for _ in range(res_block_layers))
        self.input_norm = input_norm
        if input_norm:
            self.register_buffer("input_mean", torch.zeros(input_features))
            self.register_buffer("input_std", torch.ones(input_features))
            if input_stats is not None:
                stats = np.load(str(input_stats))
                self.input_mean.copy_(torch.from_numpy(stats["mean"]).float())
                self.input_std.copy_(torch.from_numpy(stats["std"]).float().clamp_min(1e-6))
        out_features = hidden_layer if output_dim is None else output_dim
        activation = nn.ReLU() if output_activation == "relu" else nn.Identity()
        if res_block_layers == 0:
            self.embedding = nn.Sequential(OrderedDict([
                ('norm', nn.LayerNorm(input_features)),
                ('dropout', nn.Dropout(p=self.dropout)),
                ('linear', nn.Linear(input_features, out_features)),
                ('activation', activation)
            ]))
        else:
            res_block = OrderedDict([(
                f'block{i}',
                ResBlock(input_features, input_features, self.dropout)
            ) for i in range(res_block_layers)])
            res_block.update([
                ('dropout', nn.Dropout(p=self.dropout)),
                ('linear', nn.Linear(input_features, out_features)),
                ('activation', activation)
            ])
            self.embedding = nn.Sequential(res_block)

    @staticmethod
    def _zero_linear(i, o):
        layer = nn.Linear(i, o)
        nn.init.zeros_(layer.weight)
        nn.init.zeros_(layer.bias)
        return layer

    def length_code(self, x, x_mask):
        """log(rows) with the batch dims of x ([B, 1], or [1] for one unbatched [L, D] chain), or [..., K]
        Gaussian bumps of it with ``length_rbf``."""
        n = x.shape[-2] if x_mask is None else (~x_mask).sum(dim=-1)
        n = torch.as_tensor(n, dtype=x.dtype, device=x.device).expand(x.shape[:-2]).unsqueeze(-1)
        log_n = torch.log(n.clamp_min(1))
        if not hasattr(self, "length_centres"):
            return log_n
        return torch.exp(-0.5 * ((log_n - self.length_centres) / self.length_width) ** 2)

    def pool(self, x, x_mask, code=None):
        """Pooled vector (the input of the blocks after pooling): input standardisation, (length term
        on the rows), encoder, masked pooling, plus the pooled length terms when ``length_feature`` is on."""
        if self.input_norm:
            x = (x - self.input_mean) / self.input_std
        if self.length_mode and code is None:
            code = self.length_code(x, x_mask)
        if self.length_input is not None:
            x = x + self.length_input(code).unsqueeze(-2)
        p = masked_pool(self.transformer(x, src_key_padding_mask=x_mask), x_mask, self.pooling, self.pool_norm)
        if self.length_scale is not None:
            p = p * (1 + self.length_scale(code))
        if self.length_proj is not None:
            p = p + self.length_proj(code)
        return p

    def blocks(self, p, code=None):
        """The ResBlocks after pooling (with the per-block length terms in 'blocks' mode): h, the input of the
        final Linear. Evaluation code that needs h must go through this, not slice ``self.embedding``."""
        k = 0
        for layer in self.embedding:
            if not isinstance(layer, ResBlock):
                break
            if self.length_blocks is not None:
                p = p + self.length_blocks[k](code)
            p = layer(p)
            k += 1
        return p

    def head(self, h):
        """The layers after the ResBlocks (dropout, final Linear, activation)."""
        return self.embedding[sum(isinstance(layer, ResBlock) for layer in self.embedding):](h)

    def embedding_pooling(self, x, x_mask):
        code = self.length_code(x, x_mask) if self.length_mode else None
        return self.head(self.blocks(self.pool(x, x_mask, code), code))

    def forward(self, x, x_mask, y, y_mask):
        return nn.functional.cosine_similarity(
            self.embedding_pooling(x, x_mask),
            self.embedding_pooling(y, y_mask)
        )

    def get_weights(self):
        return [(name, param) for name, param in self.embedding.named_parameters()]


class MaskedTransformerEmbeddingCosine(nn.Module):
    """Transformer twin network with per-residue spatial attention masks.

    Same architecture as :class:`TransformerEmbeddingCosine` but the forward
    pass accepts an additional ``(B, L, L)`` boolean attention mask per protein
    that restricts each residue to attend only to its spatial neighbors (as
    produced by :func:`collate_masked_fn`).

    The mask is expanded to ``(B * nhead, L, L)`` before being passed to
    ``nn.TransformerEncoder`` as the ``mask`` argument.
    """
    dropout = 0.1

    def __init__(
        self,
        input_features=640,
        dim_feedforward=1280,
        hidden_layer=640,
        nhead=10,
        num_layers=6,
        res_block_layers=0
    ):
        super().__init__()
        self.nhead = nhead
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_features,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        if res_block_layers == 0:
            self.embedding = nn.Sequential(OrderedDict([
                ('norm', nn.LayerNorm(input_features)),
                ('dropout', nn.Dropout(p=self.dropout)),
                ('linear', nn.Linear(input_features, hidden_layer)),
                ('activation', nn.ReLU())
            ]))
        else:
            res_block = OrderedDict([(
                f'block{i}',
                ResBlock(input_features, hidden_layer, self.dropout)
            ) for i in range(res_block_layers)])
            res_block.update([
                ('dropout', nn.Dropout(p=self.dropout)),
                ('linear', nn.Linear(input_features, hidden_layer)),
                ('activation', nn.ReLU())
            ])
            self.embedding = nn.Sequential(res_block)

    def embedding_pooling(self, x, x_pad_mask, x_attn_mask):
        # Expand (B, L, L) → (B * nhead, L, L) for multi-head attention
        attn_mask = x_attn_mask.repeat_interleave(self.nhead, dim=0)
        return self.embedding(
            masked_pool(self.transformer(x, mask=attn_mask, src_key_padding_mask=x_pad_mask), x_pad_mask)
        )

    def forward(self, x, x_pad_mask, x_attn_mask, y, y_pad_mask, y_attn_mask):
        return nn.functional.cosine_similarity(
            self.embedding_pooling(x, x_pad_mask, x_attn_mask),
            self.embedding_pooling(y, y_pad_mask, y_attn_mask)
        )

    def get_weights(self):
        return [(name, param) for name, param in self.embedding.named_parameters()]
