"""
PrimGate classifier scaffold (Phase 0).

Full dual-encoder + SPF + SATC wiring lands in Phase 3.
This stub verifies CGSF + lightweight symptom/context stems end-to-end.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import AdaptiveMultiBranchMambaFeatureFusion, CrossGatedSymptomFiltering


class ContextEncoder(nn.Module):
    """Lightweight CNN stem for leaf lamina / background context."""

    def __init__(self, in_ch: int = 3, out_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SymptomEncoder(nn.Module):
    """Mamba-inspired symptom stream (AMF block over a CNN stem)."""

    def __init__(self, in_ch: int = 3, mid_ch: int = 64, out_ch: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, mid_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
        )
        self.amf = AdaptiveMultiBranchMambaFeatureFusion(
            input_dim=mid_ch,
            output_dim=out_ch,
            num_branches=4,
            d_state=16,
            d_conv=4,
            expand=2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.amf(self.stem(x))


class PrimGate(nn.Module):
    """
    Phase-0 PrimGate: dual streams + CGSF + GAP classifier.

    SPF / SATC hooks are placeholders for Phase 3.
    """

    def __init__(self, num_classes: int = 9, feat_dim: int = 64):
        super().__init__()
        self.symptom_encoder = SymptomEncoder(out_ch=feat_dim)
        self.context_encoder = ContextEncoder(out_ch=feat_dim)
        self.cgsf = CrossGatedSymptomFiltering(channels=feat_dim)
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        s = self.symptom_encoder(x)
        c = self.context_encoder(x)
        if s.shape[-2:] != c.shape[-2:]:
            c = F.interpolate(c, size=s.shape[-2:], mode="bilinear", align_corners=False)
        s_prime, c_prime = self.cgsf(s, c)
        pooled = s_prime.mean(dim=(2, 3))
        logits = self.classifier(pooled)
        return {
            "logits": logits,
            "symptom": s_prime,
            "context": c_prime,
            "embedding": pooled,
        }
