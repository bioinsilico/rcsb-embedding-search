import torch
import torch.nn as nn


class SineGaussianActivation(nn.Module):
    """
    Activation function combining sine and Gaussian: f(x) = global_scale * sin(sine_scale * x) * exp(-gaussian_scale * (x + gaussian_bias)^2 / 2)

    This combines the periodic nature of sine with the decay of Gaussian,
    creating a smooth, bounded activation with oscillating behavior near zero.

    All parameters (sine_scale, gaussian_scale, gaussian_bias, global_scale) are learnable and will be optimized during training.
    """
    def __init__(self, sine_scale=1.0, gaussian_scale=1.0, gaussian_bias=0.0, global_scale=1.0):
        super().__init__()
        self.sine_scale = nn.Parameter(torch.tensor(sine_scale, dtype=torch.float32))
        self.gaussian_scale = nn.Parameter(torch.tensor(gaussian_scale, dtype=torch.float32))
        self.gaussian_bias = nn.Parameter(torch.tensor(gaussian_bias, dtype=torch.float32))
        self.global_scale = nn.Parameter(torch.tensor(global_scale, dtype=torch.float32))

    def forward(self, x):
        return self.global_scale * torch.sin(self.sine_scale * x) * torch.exp(-0.5 * self.gaussian_scale * (x + self.gaussian_bias)**2)


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
            SineGaussianActivation(),
            nn.LayerNorm(out_dim),
            nn.Dropout(p=dropout),
            nn.Linear(out_dim, out_dim),
        )
        self.activate = SineGaussianActivation()

    def forward(self, x):
        residual = self.residual(x)
        x = self.block(x)
        x = self.activate(x + residual)
        return x
