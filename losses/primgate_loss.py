"""Composite PrimGate training loss."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .celdam import build_loss
from .satc import SiblingAwareTailContrastive
from .spf_bal import PrimitiveBalanceLoss


class PrimGateCriterion(nn.Module):
    def __init__(
        self,
        cls_num_list: list[int],
        head_ids: list[int],
        few_ids: list[int],
        class_to_group: torch.Tensor,
        base_loss: str = "balanced_softmax",
        lambda_bal: float = 0.5,
        lambda_satc: float = 0.2,
        lambda_gate: float = 0.05,
        use_satc: bool = True,
        satc_margin: float = 0.5,
    ):
        super().__init__()
        self.cls_loss = build_loss(base_loss, cls_num_list=cls_num_list)
        self.bal_loss = PrimitiveBalanceLoss()
        self.satc = SiblingAwareTailContrastive(margin=satc_margin)
        self.lambda_bal = lambda_bal
        self.lambda_satc = lambda_satc
        self.lambda_gate = lambda_gate
        self.use_satc = use_satc
        self.register_buffer("head_ids", torch.tensor(head_ids, dtype=torch.long))
        self.register_buffer("few_ids", torch.tensor(few_ids, dtype=torch.long))
        self.register_buffer("class_to_group", class_to_group.long())

    def forward(self, outputs: dict[str, torch.Tensor], targets: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = outputs["logits"]
        alpha = outputs["alpha"]
        emb = outputs["embedding"]
        s = outputs["symptom"]
        s_raw = outputs["symptom_raw"]

        l_cls = self.cls_loss(logits, targets)

        is_head = torch.isin(targets, self.head_ids)
        is_tail = torch.isin(targets, self.few_ids)
        l_bal = self.bal_loss(alpha, is_head, is_tail)

        if self.use_satc:
            l_satc = self.satc(emb, targets, self.class_to_group)
        else:
            l_satc = logits.new_zeros(())

        # Keep gated symptom features from collapsing vs raw symptom stream
        l_gate = F.mse_loss(s.mean(dim=(2, 3)), s_raw.detach().mean(dim=(2, 3)))

        total = (
            l_cls
            + self.lambda_bal * l_bal
            + self.lambda_satc * l_satc
            + self.lambda_gate * l_gate
        )
        return {
            "loss": total,
            "loss_cls": l_cls.detach(),
            "loss_bal": l_bal.detach(),
            "loss_satc": l_satc.detach(),
            "loss_gate": l_gate.detach(),
        }
