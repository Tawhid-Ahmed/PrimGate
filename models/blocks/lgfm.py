"""Local-Global Feature Mixing (LGFM) from MambaLiteUNet."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .mamba_layers import DepthwiseConvolution


class LocalGlobalFeatureMixing(nn.Module):
    def __init__(self, input_dim: int, num_heads: int = 8):
        super().__init__()
        self.local_conv = nn.Conv1d(
            input_dim, input_dim, kernel_size=3, padding=1, groups=input_dim
        )
        self.global_attn = nn.MultiheadAttention(
            embed_dim=input_dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(input_dim)
        self.fc = nn.Linear(input_dim * 2, input_dim)
        self.act = nn.GELU()
        self.dwconv = DepthwiseConvolution(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        x_local = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        x_global, _ = self.global_attn(x, x, x)
        x_out = self.act(self.norm(self.fc(torch.cat([x_local, x_global], dim=-1))))

        B, N, C = x_out.shape
        H = W = int(math.sqrt(N))
        if H * W != N:
            raise ValueError(f"Feature map is not square. Got N={N}.")
        return self.dwconv(x_out, H, W)
