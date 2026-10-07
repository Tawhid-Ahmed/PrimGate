"""Sibling-Aware Tail Contrastive (SATC) — Phase 3."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiblingAwareTailContrastive(nn.Module):
    """Push embeddings of same-sibling-group samples apart within a batch (margin)."""

    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        sibling_of: dict[int, int],
    ) -> torch.Tensor:
        # sibling_of: class_id -> group_id
        if embeddings.size(0) < 2:
            return embeddings.new_zeros(())
        z = F.normalize(embeddings, dim=1)
        loss = embeddings.new_zeros(())
        n_pairs = 0
        for i in range(z.size(0)):
            for j in range(i + 1, z.size(0)):
                gi = sibling_of.get(int(labels[i]), -1)
                gj = sibling_of.get(int(labels[j]), -1)
                if gi == gj and gi >= 0 and int(labels[i]) != int(labels[j]):
                    dist = torch.norm(z[i] - z[j], p=2)
                    loss = loss + F.relu(self.margin - dist)
                    n_pairs += 1
        if n_pairs == 0:
            return embeddings.new_zeros(())
        return loss / n_pairs
