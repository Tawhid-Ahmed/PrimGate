"""CE, LDAM, and Balanced Softmax losses for long-tailed classification."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    def __init__(self, weight: torch.Tensor | None = None):
        super().__init__()
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if weight is not None and weight.device != logits.device:
            weight = weight.to(logits.device)
        return F.cross_entropy(logits, targets, weight=weight)


class LDAMLoss(nn.Module):
    """Label-Distribution-Aware Margin loss (Cao et al., NeurIPS 2019)."""

    def __init__(
        self,
        cls_num_list: list[int],
        max_m: float = 0.5,
        s: float = 30.0,
        weight: torch.Tensor | None = None,
    ):
        super().__init__()
        cls_num = np.asarray(cls_num_list, dtype=np.float64)
        m_list = 1.0 / np.sqrt(np.sqrt(np.clip(cls_num, 1, None)))
        m_list = m_list * (max_m / m_list.max())
        self.register_buffer("m_list", torch.tensor(m_list, dtype=torch.float32))
        self.s = s
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, targets.view(-1, 1), True)
        batch_m = torch.matmul(self.m_list[None, :], index.float().transpose(0, 1)).view(-1, 1)
        output = torch.where(index, logits - batch_m, logits)
        weight = self.weight
        if weight is not None and weight.device != logits.device:
            weight = weight.to(logits.device)
        return F.cross_entropy(self.s * output, targets, weight=weight)


class BalancedSoftmaxLoss(nn.Module):
    """Balanced Softmax (Ren et al., NeurIPS 2020)."""

    def __init__(self, cls_num_list: list[int]):
        super().__init__()
        priors = torch.tensor(cls_num_list, dtype=torch.float32).clamp(min=1)
        self.register_buffer("log_prior", torch.log(priors))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits + self.log_prior.unsqueeze(0), targets)


def build_loss(
    name: str,
    cls_num_list: list[int] | None = None,
) -> nn.Module:
    key = name.lower().replace("-", "_")
    if key in {"ce", "cross_entropy"}:
        return CrossEntropyLoss()
    if cls_num_list is None:
        raise ValueError(f"cls_num_list required for loss={name}")
    if key == "ldam":
        return LDAMLoss(cls_num_list)
    if key in {"balanced_softmax", "bs", "bal_softmax"}:
        return BalancedSoftmaxLoss(cls_num_list)
    raise ValueError(f"Unknown loss: {name}")
