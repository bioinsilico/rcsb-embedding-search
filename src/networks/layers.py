import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        if in_dim != out_dim:
            self.residual = nn.Linear(in_dim, out_dim)
        else:
            self.residual = nn.Identity()

        self.block = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(p=dropout),
            nn.Linear(in_dim, out_dim),
            nn.ReLU(),
            nn.LayerNorm(out_dim),
            nn.Dropout(p=dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.activate = nn.ReLU()

    def forward(self, x):
        residual = self.residual(x)
        x = self.block(x)
        x = self.activate(x + residual)
        return x


class MoELayer(nn.Module):
    """Mixture of Experts Layer with top-k gating."""

    def __init__(self, d_model, dim_feedforward, num_experts=8, top_k=2, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model

        # Gating network
        self.gate = nn.Linear(d_model, num_experts)

        # Expert networks (each is a 2-layer FFN)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model),
                nn.Dropout(dropout)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # (batch_size * seq_len, d_model)

        # Compute gating scores
        gate_logits = self.gate(x_flat)  # (batch_size * seq_len, num_experts)
        gate_scores = F.softmax(gate_logits, dim=-1)

        # Select top-k experts
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        top_k_scores = top_k_scores / top_k_scores.sum(dim=-1, keepdim=True)  # Renormalize

        # Compute expert outputs
        expert_outputs = torch.zeros(batch_size * seq_len, d_model, device=x.device)

        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            expert_weight = top_k_scores[:, i].unsqueeze(-1)

            # Process tokens by each expert
            for expert_id in range(self.num_experts):
                mask = expert_idx == expert_id
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    expert_outputs[mask] += expert_weight[mask] * expert_output

        return expert_outputs.view(batch_size, seq_len, d_model)


class MoETransformerEncoderLayer(nn.Module):
    """TransformerEncoderLayer with MoE replacing the feedforward network."""

    def __init__(self, d_model, nhead, dim_feedforward, num_experts=8, top_k=2, dropout=0.1, batch_first=True):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.moe = MoELayer(d_model, dim_feedforward, num_experts, top_k, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention block
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # MoE block
        src2 = self.moe(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src
