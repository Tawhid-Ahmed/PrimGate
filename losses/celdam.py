"""CE / LDAM losses — full LDAM implementation in Phase 1."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None):
        super().__init__()
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, weight=self.weight)


class LDAMLoss(nn.Module):
    """Placeholder LDAM (Cao et al., NeurIPS 2019). Implemented in Phase 1."""

    def __init__(self, cls_num_list: list[int], max_m: float = 0.5, s: float = 30.0):
        super().__init__()
        self.cls_num_list = cls_num_list
        self.max_m = max_m
        self.s = s
        # margins filled in Phase 1
        self.register_buffer("m_list", torch.zeros(len(cls_num_list)))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets)
