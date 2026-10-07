"""CIFAR-100-LT config placeholder (Phase 4)."""

from __future__ import annotations


class CIFAR100LTConfig:
    num_classes = 100
    imbalance_factor = 100  # freeze one protocol in Phase 4
    image_size = 32
    batch_size = 128
    lr = 0.1
    epochs = 200
    seed = 42
