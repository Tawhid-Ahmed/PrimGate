"""Reusable Mamba SSM wrappers extracted from MambaLiteUNet (CVPR 2026)."""

from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

try:
    from mamba_ssm import Mamba

    MAMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - Windows/CUDA wheels often missing
    Mamba = None
    MAMBA_AVAILABLE = False
    warnings.warn(
        "mamba_ssm is not installed. MambaSSMLayer will use a Conv1d fallback "
        "for scaffolding/smoke tests only. Install mamba_ssm for real PrimGate runs.",
        RuntimeWarning,
        stacklevel=1,
    )


class DepthwiseConvolution(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim
        )
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        if N != H * W:
            raise ValueError(f"DepthwiseConvolution expects N=H*W, got N={N}, HxW={H}x{W}")
        x = x.transpose(1, 2).view(B, C, H, W)
        x = F.layer_norm(x, [H, W])
        x = self.dwconv(x)
        return x.flatten(2).transpose(1, 2)


class _ConvMambaFallback(nn.Module):
    """Temporary stand-in when mamba_ssm is unavailable (dev only)."""

    def __init__(self, d_model: int, d_conv: int = 4, expand: int = 2):
        super().__init__()
        hidden = d_model * expand
        self.proj_in = nn.Linear(d_model, hidden * 2)
        self.dw = nn.Conv1d(
            hidden, hidden, kernel_size=d_conv, padding=d_conv - 1, groups=hidden
        )
        self.proj_out = nn.Linear(hidden, d_model)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        u, v = self.proj_in(x).chunk(2, dim=-1)
        v = self.dw(v.transpose(1, 2))[..., : x.size(1)].transpose(1, 2)
        return self.proj_out(self.act(u) * v)


class MambaSSMLayer(nn.Module):
    """
    Thin wrapper around Mamba SSM.
    Accepts (B, C, H, W) or (B, N, C) and returns (B, N, C).
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        if MAMBA_AVAILABLE:
            self.mamba = Mamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
        else:
            self.mamba = _ConvMambaFallback(d_model=d_model, d_conv=d_conv, expand=expand)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.view(B, C, H * W).transpose(1, 2)
        elif x.dim() != 3:
            raise ValueError(f"MambaSSMLayer expects 3 or 4 dims, got {x.dim()}")
        return self.mamba(x)
