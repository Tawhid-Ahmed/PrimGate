"""Tea leaf dataset loader with frozen stratified splits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from configs.config_tea import TeaConfig

# Canonical class order (folder names under processed_dataset)
TEA_CLASSES = [
    "Black Rot - Corticium theae Bernard",
    "Blister Blight - Exobasidium vexans",
    "Brown Blight - Colletotrichum spp",
    "Grey Blight - Pestalotiopsis spp",
    "Healthy Leaf",
    "No damage",
    "Pest damage (leaf chewing)",
    "Red Rust - cephaleuros",
    "Red Spider",
]

CLASS_TO_IDX = {name: i for i, name in enumerate(TEA_CLASSES)}

# Visual sibling groups for SATC (proposal §5.6)
SIBLING_GROUPS = {
    "G1_blight": [0, 1, 2, 3],
    "G2_healthy_like": [4, 5],
    "G3_rare": [6, 7, 8],
}

CLASS_TO_SIBLING = {
    cls_id: gid
    for gid, (_, members) in enumerate(SIBLING_GROUPS.items())
    for cls_id in members
}


class TeaLeafDataset(Dataset):
    """
    ImageFolder-style tea dataset.

    If `samples` is provided (list of (path, label)), uses that split.
    Otherwise scans `root` class folders.
    """

    def __init__(
        self,
        root: str | Path,
        samples: list[tuple[str, int]] | None = None,
        transform: Callable | None = None,
    ):
        self.root = Path(root)
        self.transform = transform
        if samples is not None:
            self.samples = samples
        else:
            self.samples = self._scan_root(self.root)

    @staticmethod
    def _scan_root(root: Path) -> list[tuple[str, int]]:
        samples: list[tuple[str, int]] = []
        for class_name, idx in CLASS_TO_IDX.items():
            class_dir = root / class_name
            if not class_dir.is_dir():
                continue
            for path in sorted(class_dir.rglob("*")):
                if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    samples.append((str(path), idx))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def count_per_class(samples: list[tuple[str, int]], num_classes: int = 9) -> list[int]:
    counts = [0] * num_classes
    for _, y in samples:
        counts[y] += 1
    return counts


def load_split_json(split_path: str | Path) -> dict:
    path = Path(split_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Split file not found: {path}. Run: python scripts/make_splits.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def samples_from_split(
    split_data: dict,
    split: str,
    data_root: str | Path | None = None,
) -> list[tuple[str, int]]:
    root = Path(data_root) if data_root is not None else Path(split_data["data_root"])
    entries = split_data["splits"][split]
    return [(str(root / e["path"]), int(e["label"])) for e in entries]


def build_transforms(train: bool, image_size: int = TeaConfig.image_size) -> transforms.Compose:
    normalize = transforms.Normalize(mean=TeaConfig.mean, std=TeaConfig.std)
    if train:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


def make_tea_datasets(
    split_path: str | Path | None = None,
    data_root: str | Path | None = None,
) -> tuple[TeaLeafDataset, TeaLeafDataset, TeaLeafDataset, dict]:
    split_path = Path(split_path) if split_path else TeaConfig.split_path
    split_data = load_split_json(split_path)
    root = Path(data_root) if data_root else Path(split_data["data_root"])
    train_ds = TeaLeafDataset(
        root, samples_from_split(split_data, "train", root), build_transforms(True)
    )
    val_ds = TeaLeafDataset(
        root, samples_from_split(split_data, "val", root), build_transforms(False)
    )
    test_ds = TeaLeafDataset(
        root, samples_from_split(split_data, "test", root), build_transforms(False)
    )
    return train_ds, val_ds, test_ds, split_data
