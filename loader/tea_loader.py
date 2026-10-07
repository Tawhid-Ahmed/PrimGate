"""Tea leaf dataset loader (Phase 0 stub; splits wired in Phase 1)."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from torch.utils.data import Dataset
from PIL import Image

# Canonical class order (alphabetical by folder name under processed_dataset)
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
    "G1_blight": [0, 1, 2, 3],  # Black Rot, Blister, Brown, Grey
    "G2_healthy_like": [4, 5],  # Healthy Leaf, No damage
    "G3_rare": [6, 7, 8],  # Pest, Red Rust, Red Spider
}


class TeaLeafDataset(Dataset):
    """
    ImageFolder-style tea dataset.

    If `samples` is provided (list of (path, label)), uses that split.
    Otherwise scans `root` class folders (full dataset; Phase 0 smoke only).
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
