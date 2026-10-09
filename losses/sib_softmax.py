"""Sibling-conditional softmax: CE renormalized inside the true sibling group."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SiblingConditionalSoftmax(nn.Module):
    """
    For each sample, restrict logits to the sibling group of the true label
    and apply CE in that subspace. Directly targets within-group confusion
    (e.g. Brown vs Grey blight).
    """

    def __init__(self, class_to_group: torch.Tensor, num_classes: int = 9):
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer("class_to_group", class_to_group.long())
        # Precompute group membership masks: (G, C)
        groups = class_to_group.unique(sorted=True)
        masks = []
        for g in groups.tolist():
            masks.append((class_to_group == g).float())
        self.register_buffer("group_masks", torch.stack(masks, dim=0))  # (G, C)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.size(0) == 0:
            return logits.new_zeros(())
        groups = self.class_to_group[targets]  # (B,)
        # Gather per-sample mask (B, C)
        masks = self.group_masks[groups]
        # Masked logits: large negative outside group
        neg = torch.finfo(logits.dtype).min / 4
        masked = torch.where(masks.bool(), logits, logits.new_full((), neg))
        log_probs = F.log_softmax(masked, dim=1)
        return F.nll_loss(log_probs, targets)
