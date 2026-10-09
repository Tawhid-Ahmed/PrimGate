"""Composite loss for PrimGate-SC."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .celdam import build_loss
from .sib_softmax import SiblingConditionalSoftmax


class PrimGateSCCriterion(nn.Module):
    def __init__(
        self,
        cls_num_list: list[int],
        class_to_group: torch.Tensor,
        base_loss: str = "balanced_softmax",
        lambda_sib: float = 0.5,
        lambda_gate: float = 0.05,
        use_sib: bool = True,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.cls_loss = build_loss(
            base_loss, cls_num_list=cls_num_list, label_smoothing=label_smoothing
        )
        self.sib_loss = SiblingConditionalSoftmax(
            class_to_group=class_to_group, num_classes=len(cls_num_list)
        )
        self.lambda_sib = lambda_sib
        self.lambda_gate = lambda_gate
        self.use_sib = use_sib

    def forward(self, outputs: dict[str, torch.Tensor], targets: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = outputs["logits"]
        s = outputs["symptom"]
        s_raw = outputs["symptom_raw"]

        l_cls = self.cls_loss(logits, targets)
        if self.use_sib and self.lambda_sib > 0:
            l_sib = self.sib_loss(logits, targets)
        else:
            l_sib = logits.new_zeros(())

        l_gate = F.mse_loss(s.mean(dim=(2, 3)), s_raw.detach().mean(dim=(2, 3)))
        # Cap gate term relative to classification to avoid late explosion
        l_gate = torch.clamp(l_gate, max=2.0)

        total = l_cls + self.lambda_sib * l_sib + self.lambda_gate * l_gate
        return {
            "loss": total,
            "loss_cls": l_cls.detach(),
            "loss_sib": l_sib.detach(),
            "loss_gate": l_gate.detach(),
        }
