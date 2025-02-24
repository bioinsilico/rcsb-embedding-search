import torch.nn as nn


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
