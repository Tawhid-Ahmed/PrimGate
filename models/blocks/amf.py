"""Adaptive Multi-Branch Mamba Feature Fusion (AMF) from MambaLiteUNet."""

from __future__ import annotations

import torch
import torch.nn as nn

from .lgfm import LocalGlobalFeatureMixing
from .mamba_layers import MambaSSMLayer


class AdaptiveMultiBranchMambaFeatureFusion(nn.Module):
    """
    Input (B, C, H, W) -> multi-branch Mamba + adaptive gating + LGFM
    -> output (B, C_out, H, W).
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_branches: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_branches = num_branches
        self.norm = nn.LayerNorm(input_dim)

        if input_dim % num_branches != 0:
            raise ValueError(
                f"input_dim({input_dim}) must be divisible by num_branches({num_branches})"
            )

        self.mamba_layers = nn.ModuleList(
            [
                MambaSSMLayer(
                    d_model=input_dim // num_branches,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(num_branches)
            ]
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

        self.conv_q_dw = nn.Conv1d(
            input_dim, num_branches, kernel_size=3, padding=1, groups=num_branches
        )
        self.conv_q_pw = nn.Conv1d(num_branches, num_branches, kernel_size=1)
        self.conv_r_dw = nn.Conv1d(
            input_dim, input_dim, kernel_size=3, padding=1, groups=input_dim
        )
        self.conv_r_pw = nn.Conv1d(input_dim, input_dim, kernel_size=1)
        self.dw_conv_final_dw = nn.Conv1d(
            2 * input_dim, 2 * input_dim, kernel_size=3, padding=1, groups=2 * input_dim
        )
        self.dw_conv_final_pw = nn.Conv1d(2 * input_dim, input_dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.local_global_mixing = LocalGlobalFeatureMixing(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        x = x.view(B, C, N).transpose(1, 2)
        x_norm = self.norm(x)
        x_chunks = torch.chunk(x_norm, self.num_branches, dim=2)

        x_mamba_chunks = [
            self.mamba_layers[i](x_chunk) + self.skip_scale * x_chunk
            for i, x_chunk in enumerate(x_chunks)
        ]
        x_concat = torch.cat(x_mamba_chunks, dim=2)

        Q = self.conv_q_pw(self.conv_q_dw(x_concat.transpose(1, 2)))
        g = self.sigmoid(Q)
        g_chunks = torch.chunk(g, self.num_branches, dim=1)
        x_gated = torch.cat(
            [
                g_chunk.permute(0, 2, 1) * x_mamba_chunk
                for g_chunk, x_mamba_chunk in zip(g_chunks, x_mamba_chunks)
            ],
            dim=2,
        )

        x_gated_t = x_gated.transpose(1, 2)
        R = self.conv_r_pw(self.conv_r_dw(x_gated_t)) + x_gated_t
        R = R.transpose(1, 2)

        concat_NR = torch.cat([x, R], dim=2)
        concat_NR_t = concat_NR.transpose(1, 2)
        x_mamba_t = self.dw_conv_final_pw(self.dw_conv_final_dw(concat_NR_t))
        x_mamba_t = x_mamba_t + concat_NR_t[:, :C, :]
        x_mamba = x_mamba_t.transpose(1, 2)

        x_mixed = self.local_global_mixing(x_mamba)
        x_mixed = self.proj(self.norm(x_mixed))
        return x_mixed.transpose(1, 2).view(B, self.output_dim, H, W)
