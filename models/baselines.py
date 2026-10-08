"""Baseline classifiers for Phase 1."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


class ClassifierWrapper(nn.Module):
    """Wrap a backbone so train/eval can always read out['logits']."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        logits = self.backbone(x)
        return {"logits": logits, "embedding": logits}


def build_resnet50(num_classes: int = 9, pretrained: bool = True) -> ClassifierWrapper:
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    net = models.resnet50(weights=weights)
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    return ClassifierWrapper(net)


def build_baseline(name: str, num_classes: int = 9, pretrained: bool = True) -> nn.Module:
    key = name.lower().replace("-", "_")
    if key in {"resnet50", "resnet_50"}:
        return build_resnet50(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(f"Unknown baseline model: {name}")
