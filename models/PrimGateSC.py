"""
PrimGate-SC: Sibling-Conditional Evidence Model.

Keeps dual-stream + CGSF; replaces global SPF/SATC with:
  - Group-conditional primitive banks (per sibling group)
  - Embedding refined only within the sample's sibling bank
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from loader.tea_loader import SIBLING_GROUPS

from .blocks import CrossGatedSymptomFiltering
from .PrimGate import ResNetDualEncoder


class GroupConditionalProtos(nn.Module):
    """Per-sibling-group prototype banks; occupancy stays inside the group."""

    def __init__(
        self,
        feat_dim: int,
        group_members: list[list[int]],
        num_classes: int = 9,
        protos_per_group: int = 8,
        temperature: float = 0.2,
    ):
        super().__init__()
        self.num_groups = len(group_members)
        self.protos_per_group = protos_per_group
        self.temperature = temperature
        self.banks = nn.ParameterList(
            [
                nn.Parameter(torch.randn(protos_per_group, feat_dim) * 0.02)
                for _ in range(self.num_groups)
            ]
        )
        self.residual = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim),
        )
        ctg = torch.zeros(num_classes, dtype=torch.long)
        for gid, members in enumerate(group_members):
            for c in members:
                ctg[c] = gid
        self.register_buffer("class_to_group", ctg)

    def _refine_with_bank(
        self, h: torch.Tensor, bank: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        proto = F.normalize(bank, dim=1)
        hn = F.normalize(h, dim=1)
        logits = (hn @ proto.t()) / max(self.temperature, 1e-6)
        alpha = F.softmax(logits, dim=1)
        shared = alpha @ bank
        z = shared + self.residual(h)
        return z, alpha

    def forward(
        self,
        h: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            z: refined embedding (B, D)
            alpha: occupancy (B, K_g) for the selected bank (padded layout per-sample)
            group_ids: (B,)
        """
        B = h.size(0)
        if labels is not None:
            group_ids = self.class_to_group[labels]
        else:
            # Soft group routing at inference: pick bank with highest mean cosine sim
            scores = []
            hn = F.normalize(h, dim=1)
            for bank in self.banks:
                proto = F.normalize(bank, dim=1)
                scores.append((hn @ proto.t()).mean(dim=1))
            group_ids = torch.stack(scores, dim=1).argmax(dim=1)

        # Run bank refine in fp32 to avoid AMP index_put dtype mismatches
        z = torch.zeros(h.shape, device=h.device, dtype=torch.float32)
        alpha = torch.zeros(B, self.protos_per_group, device=h.device, dtype=torch.float32)
        h32 = h.float()
        ctx = (
            torch.amp.autocast("cuda", enabled=False)
            if h.is_cuda
            else torch.amp.autocast("cpu", enabled=False)
        )
        with ctx:
            for g in range(self.num_groups):
                mask = group_ids == g
                if not mask.any():
                    continue
                zg, ag = self._refine_with_bank(h32[mask], self.banks[g].float())
                z[mask] = zg.float()
                alpha[mask] = ag.float()
        return z.to(dtype=h.dtype), alpha.to(dtype=h.dtype), group_ids


class PrimGateSC(nn.Module):
    def __init__(
        self,
        num_classes: int = 9,
        feat_dim: int = 256,
        protos_per_group: int = 8,
        use_cgsf: bool = True,
        use_group_proto: bool = True,
        temperature: float = 0.2,
        backbone: str = "resnet50",
        pretrained: bool = True,
        use_amf: bool = True,
        class_to_group: torch.Tensor | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_cgsf = use_cgsf
        self.use_group_proto = use_group_proto
        self.backbone_name = backbone.lower().replace("-", "_")

        if self.backbone_name not in {"resnet50", "resnet_50", "r50"}:
            raise ValueError("PrimGate-SC currently supports backbone=resnet50 only")

        self.encoder = ResNetDualEncoder(
            feat_dim=feat_dim, pretrained=pretrained, use_amf=use_amf
        )
        self.cgsf = CrossGatedSymptomFiltering(channels=feat_dim)

        group_members = list(SIBLING_GROUPS.values())
        self.group_proto = GroupConditionalProtos(
            feat_dim=feat_dim,
            group_members=group_members,
            num_classes=num_classes,
            protos_per_group=protos_per_group,
            temperature=temperature,
        )
        if class_to_group is not None:
            self.group_proto.class_to_group.copy_(class_to_group.long())

        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        s, c = self.encoder(x)
        if self.use_cgsf:
            s_prime, c_prime = self.cgsf(s, c)
        else:
            s_prime, c_prime = s, c

        pooled = s_prime.mean(dim=(2, 3))
        if self.use_group_proto:
            z, alpha, group_ids = self.group_proto(pooled, labels=labels)
        else:
            z = pooled
            alpha = pooled.new_zeros((pooled.size(0), self.group_proto.protos_per_group))
            group_ids = (
                self.group_proto.class_to_group[labels]
                if labels is not None
                else pooled.new_zeros(pooled.size(0), dtype=torch.long)
            )

        logits = self.classifier(z)
        return {
            "logits": logits,
            "embedding": z,
            "pooled_symptom": pooled,
            "symptom": s_prime,
            "context": c_prime,
            "symptom_raw": s,
            "context_raw": c,
            "alpha": alpha,
            "group_ids": group_ids,
        }
