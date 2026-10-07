"""
Cross-Gated Symptom Filtering (CGSF) for PrimGate classification.

Adapted from MambaLiteUNet CGA principle, reformulated for symptom/context
evidence streams (not encoder-decoder skip masks):

    s' = s ⊙ σ(DWConv(c))
    c' = c ⊙ σ(DWConv(s))
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossGatedSymptomFiltering(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.gate_from_context = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=channels,
                bias=True,
            ),
            nn.Sigmoid(),
        )
        self.gate_from_symptom = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=channels,
                bias=True,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self, symptom: torch.Tensor, context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            symptom: (B, C, H, W)
            context: (B, C, H, W)
        Returns:
            gated_symptom s', gated_context c'
        """
        if symptom.shape != context.shape:
            raise ValueError(
                f"CGSF expects matching shapes, got {tuple(symptom.shape)} vs {tuple(context.shape)}"
            )
        s_prime = symptom * self.gate_from_context(context)
        c_prime = context * self.gate_from_symptom(symptom)
        return s_prime, c_prime
