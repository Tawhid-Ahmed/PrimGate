"""
Original Cross-Gated Attention Gate (CGA) from MambaLiteUNet.

Kept as reference only. PrimGate classification uses CGSF
(models/blocks/cgsf.py), which adapts the cross-gate idea for
symptom/context evidence filtering without segmentation skips.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .mamba_layers import MambaSSMLayer


class CrossGatedAttentionGate(nn.Module):
    def __init__(
        self,
        F_g: int,
        F_l: int,
        F_int: int,
        num_branches: int = 4,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.num_branches = num_branches

        if F_g % num_branches != 0 or F_l % num_branches != 0:
            raise ValueError("F_g and F_l must be divisible by num_branches")

        self.mamba_g = nn.ModuleList(
            [
                MambaSSMLayer(F_g // num_branches, d_state, d_conv, expand)
                for _ in range(num_branches)
            ]
        )
        self.mamba_x = nn.ModuleList(
            [
                MambaSSMLayer(F_l // num_branches, d_state, d_conv, expand)
                for _ in range(num_branches)
            ]
        )
        self.conv_g_dw = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        F_g // num_branches,
                        F_g // num_branches,
                        3,
                        1,
                        1,
                        groups=F_g // num_branches,
                    ),
                    nn.BatchNorm2d(F_g // num_branches),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_branches)
            ]
        )
        self.conv_x_dw = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        F_l // num_branches,
                        F_l // num_branches,
                        3,
                        1,
                        1,
                        groups=F_l // num_branches,
                    ),
                    nn.BatchNorm2d(F_l // num_branches),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_branches)
            ]
        )
        self.branch_proj = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        F_g // num_branches,
                        F_g // num_branches,
                        3,
                        1,
                        1,
                        groups=F_g // num_branches,
                    ),
                    nn.BatchNorm2d(F_g // num_branches),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(F_g // num_branches, F_int, 1),
                    nn.BatchNorm2d(F_int),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_branches)
            ]
        )
        self.combine = nn.Sequential(
            nn.Conv2d(F_int * num_branches, F_int, 3, 1, 1),
            nn.BatchNorm2d(F_int),
            nn.ReLU(inplace=True),
            nn.Conv2d(F_int, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = g.shape
        g_chunks = torch.chunk(g, self.num_branches, dim=1)
        x_chunks = torch.chunk(x, self.num_branches, dim=1)
        cross_branch_list = []

        for i in range(self.num_branches):
            g_i_mamba = self.mamba_g[i](g_chunks[i])
            _, n_g, c_g = g_i_mamba.shape
            if n_g != H * W:
                raise ValueError(f"G branch mismatch: N={n_g}, H*W={H * W}")
            g_i_mamba = g_i_mamba.transpose(1, 2).view(B, c_g, H, W)

            x_i_mamba = self.mamba_x[i](x_chunks[i])
            _, n_x, c_x = x_i_mamba.shape
            if n_x != H * W:
                raise ValueError(f"X branch mismatch: N={n_x}, H*W={H * W}")
            x_i_mamba = x_i_mamba.transpose(1, 2).view(B, c_x, H, W)

            g_i_conv = self.conv_g_dw[i](g_i_mamba)
            x_i_conv = self.conv_x_dw[i](x_i_mamba)
            cross_i = x_i_mamba * torch.sigmoid(g_i_conv) + g_i_mamba * torch.sigmoid(x_i_conv)
            cross_branch_list.append(self.branch_proj[i](cross_i))

        psi = self.combine(torch.cat(cross_branch_list, dim=1))
        return x * psi
