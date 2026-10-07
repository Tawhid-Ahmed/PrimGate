"""Frequency-balanced primitive occupancy loss (SPF) — Phase 3."""

from __future__ import annotations

import torch
import torch.nn as nn


class PrimitiveBalanceLoss(nn.Module):
    """L_bal = sum_k (E_head[alpha_k] - E_tail[alpha_k])^2"""

    def forward(
        self,
        alpha: torch.Tensor,
        is_head: torch.Tensor,
        is_tail: torch.Tensor,
    ) -> torch.Tensor:
        if alpha.ndim != 2:
            raise ValueError("alpha must be (B, K)")
        if is_head.any() and is_tail.any():
            head_mean = alpha[is_head].mean(dim=0)
            tail_mean = alpha[is_tail].mean(dim=0)
            return ((head_mean - tail_mean) ** 2).sum()
        return alpha.new_zeros(())
