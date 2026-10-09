"""Baseline classifiers for Phase 1."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import ResNet


class ClassifierWrapper(nn.Module):
    """Wrap a backbone so train/eval can read logits and penultimate embedding."""

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        net = self.backbone
        if isinstance(net, ResNet):
            x = net.conv1(x)
            x = net.bn1(x)
            x = net.relu(x)
            x = net.maxpool(x)
            x = net.layer1(x)
            x = net.layer2(x)
            x = net.layer3(x)
            x = net.layer4(x)
            x = net.avgpool(x)
            return torch.flatten(x, 1)
        # Generic fallback: embedding == logits (weak)
        return net(x)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.forward_features(x)
        if isinstance(self.backbone, ResNet):
            logits = self.backbone.fc(feats)
        else:
            logits = feats
        return {"logits": logits, "embedding": feats}


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
