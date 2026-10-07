"""Tea dataset experiment config."""

from __future__ import annotations

from pathlib import Path


class TeaConfig:
    project_root = Path(r"f:\Shakil thesis\PrimGate")
    thesis_root = Path(r"f:\Shakil thesis")
    data_root = thesis_root / "processed_dataset"
    split_path = project_root / "splits" / "tea_split_seed42.json"

    num_classes = 9
    image_size = 224
    batch_size = 16
    lr = 1e-4
    weight_decay = 1e-4
    epochs = 50
    seed = 42
    train_ratio = 0.70
    val_ratio = 0.15
    test_ratio = 0.15

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # head / med / few thresholds by count (set after split stats in Phase 1)
    many_threshold = 500
    few_threshold = 100
