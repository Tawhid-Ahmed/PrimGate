"""
PrimGate: Cross-Gated Symptom Primitive Factorization classifier.

Modules:
  1. Symptom–Context Dual Encoder (lite CNN or capacity-matched ResNet-50)
  2. Cross-Gated Symptom Filtering (CGSF)
  3. Symptom Primitive Factorization (SPF)
  Classifier head (+ SATC applied in loss on embeddings)
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from .blocks import AdaptiveMultiBranchMambaFeatureFusion, CrossGatedSymptomFiltering


class ContextEncoder(nn.Module):
    """Lightweight CNN stem for leaf lamina / background context.

    Three stride-2 stages → 28×28 @ 224 input so CGSF/AMF stay in 16GB VRAM
    (LGFM global attention is O(N²) in spatial tokens).
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SymptomEncoder(nn.Module):
    """Mamba-inspired symptom stream (CNN stem + AMF)."""

    def __init__(self, in_ch: int = 3, mid_ch: int = 128, out_ch: int = 128):
        super().__init__()
        # 224 → 28×28 tokens before AMF/LGFM (avoids attention OOM at 56×56)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, stride=2, padding=1),
            nn.BatchNorm2d(mid_ch),
            nn.GELU(),
            nn.Conv2d(mid_ch, mid_ch, 3, stride=2, padding=1),
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


class ResNetDualEncoder(nn.Module):
    """Capacity-matched dual stream: shared ResNet-50 stem + dual layer4.

    Spatial maps stay 7×7 @ 224, so AMF/LGFM (and Conv1d Mamba fallback) fit in 16GB.
    """

    def __init__(
        self,
        feat_dim: int = 256,
        pretrained: bool = True,
        use_amf: bool = True,
    ):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        net = models.resnet50(weights=weights)
        self.stem = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
            net.layer1,
            net.layer2,
            net.layer3,
        )
        self.symptom_layer4 = net.layer4
        self.context_layer4 = copy.deepcopy(net.layer4)
        self.symptom_proj = nn.Sequential(
            nn.Conv2d(2048, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.GELU(),
        )
        self.context_proj = nn.Sequential(
            nn.Conv2d(2048, feat_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.GELU(),
        )
        self.use_amf = use_amf
        self.amf = (
            AdaptiveMultiBranchMambaFeatureFusion(
                input_dim=feat_dim,
                output_dim=feat_dim,
                num_branches=4,
                d_state=16,
                d_conv=4,
                expand=2,
            )
            if use_amf
            else None
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.stem(x)
        s = self.symptom_proj(self.symptom_layer4(h))
        c = self.context_proj(self.context_layer4(h))
        if self.amf is not None:
            s = self.amf(s)
        return s, c


class SymptomPrimitiveFactorization(nn.Module):
    """Soft occupancy over K shared primitives + residual refinement."""

    def __init__(self, feat_dim: int, num_primitives: int = 16, temperature: float = 0.2):
        super().__init__()
        self.num_primitives = num_primitives
        self.temperature = temperature
        self.prototypes = nn.Parameter(torch.randn(num_primitives, feat_dim) * 0.02)
        self.residual = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, feat_dim),
        )
        self.class_bias = nn.Parameter(torch.zeros(1, feat_dim))  # unused placeholder for ablations

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # h: (B, D)
        proto = F.normalize(self.prototypes, dim=1)
        hn = F.normalize(h, dim=1)
        logits = (hn @ proto.t()) / max(self.temperature, 1e-6)
        alpha = F.softmax(logits, dim=1)
        shared = alpha @ self.prototypes
        z = shared + self.residual(h)
        return z, alpha


class PrimGate(nn.Module):
    def __init__(
        self,
        num_classes: int = 9,
        feat_dim: int = 128,
        num_primitives: int = 16,
        use_cgsf: bool = True,
        use_spf: bool = True,
        temperature: float = 0.2,
        backbone: str = "lite",
        pretrained: bool = True,
        use_amf: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_cgsf = use_cgsf
        self.use_spf = use_spf
        self.backbone_name = backbone.lower().replace("-", "_")

        if self.backbone_name in {"resnet50", "resnet_50", "r50"}:
            # Match Phase-1 ResNet-50 capacity; default wider feats than lite stem
            if feat_dim == 128:
                feat_dim = 256
                self.feat_dim = feat_dim
            self.encoder = ResNetDualEncoder(
                feat_dim=feat_dim, pretrained=pretrained, use_amf=use_amf
            )
            self.symptom_encoder = None
            self.context_encoder = None
        elif self.backbone_name in {"lite", "cnn", "scratch"}:
            self.encoder = None
            self.symptom_encoder = SymptomEncoder(out_ch=feat_dim)
            self.context_encoder = ContextEncoder(out_ch=feat_dim)
        else:
            raise ValueError(f"Unknown PrimGate backbone: {backbone}")

        self.cgsf = CrossGatedSymptomFiltering(channels=feat_dim)
        self.spf = SymptomPrimitiveFactorization(
            feat_dim=feat_dim,
            num_primitives=num_primitives,
            temperature=temperature,
        )
        self.classifier = nn.Linear(feat_dim, num_classes)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.encoder is not None:
            return self.encoder(x)
        s = self.symptom_encoder(x)
        c = self.context_encoder(x)
        if s.shape[-2:] != c.shape[-2:]:
            c = F.interpolate(c, size=s.shape[-2:], mode="bilinear", align_corners=False)
        return s, c

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        s, c = self.encode(x)

        if self.use_cgsf:
            s_prime, c_prime = self.cgsf(s, c)
        else:
            s_prime, c_prime = s, c

        pooled = s_prime.mean(dim=(2, 3))
        if self.use_spf:
            z, alpha = self.spf(pooled)
        else:
            z, alpha = pooled, pooled.new_zeros((pooled.size(0), self.spf.num_primitives))

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
        }
