"""Sibling-Aware Tail Contrastive (SATC)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiblingAwareTailContrastive(nn.Module):
    """
    Push embeddings of different classes in the same sibling group apart (margin).
    Vectorized pairwise formulation for batch efficiency.
    """

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        class_to_group: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: (B, D)
            labels: (B,)
            class_to_group: (num_classes,) int group id per class
        """
        if embeddings.size(0) < 2:
            return embeddings.new_zeros(())

        z = F.normalize(embeddings, dim=1)
        groups = class_to_group[labels]
        dist = torch.cdist(z, z, p=2)
        same_group = groups.unsqueeze(0) == groups.unsqueeze(1)
        diff_class = labels.unsqueeze(0) != labels.unsqueeze(1)
        eye = torch.eye(labels.size(0), device=labels.device, dtype=torch.bool)
        mask = same_group & diff_class & (~eye)
        if not mask.any():
            return embeddings.new_zeros(())
        return F.relu(self.margin - dist)[mask].mean()
