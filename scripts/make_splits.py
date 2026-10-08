"""Create frozen stratified tea splits (70/15/15) and write JSON."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.config_tea import TeaConfig
from loader.tea_loader import CLASS_TO_IDX, TEA_CLASSES, TeaLeafDataset, count_per_class


def stratified_split(
    samples: list[tuple[str, int]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[tuple[str, int]]]:
    rng = np.random.default_rng(seed)
    by_class: dict[int, list[str]] = defaultdict(list)
    for path, label in samples:
        by_class[label].append(path)

    train, val, test = [], [], []
    for label in sorted(by_class.keys()):
        paths = by_class[label]
        rng.shuffle(paths)
        n = len(paths)
        if n < 3:
            # Extreme edge case: put what we can in train/val/test without empty train
            n_train = max(1, n - 2) if n >= 2 else n
            n_val = 1 if n >= 2 else 0
            n_test = n - n_train - n_val
        else:
            n_train = int(round(n * train_ratio))
            n_val = int(round(n * val_ratio))
            # Keep at least one sample in each split when possible
            n_train = min(max(n_train, 1), n - 2)
            n_val = min(max(n_val, 1), n - n_train - 1)
            n_test = n - n_train - n_val
            if n_test < 1:
                n_test = 1
                if n_val > 1:
                    n_val -= 1
                else:
                    n_train -= 1

        train.extend((p, label) for p in paths[:n_train])
        val.extend((p, label) for p in paths[n_train : n_train + n_val])
        test.extend((p, label) for p in paths[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return {"train": train, "val": val, "test": test}


def to_relative(samples: list[tuple[str, int]], data_root: Path) -> list[dict]:
    root = data_root.resolve()
    out = []
    for path, label in samples:
        rel = Path(path).resolve().relative_to(root).as_posix()
        out.append({"path": rel, "label": int(label), "class": TEA_CLASSES[label]})
    return out


def frequency_buckets(train_counts: list[int], many_th: int, few_th: int) -> dict[str, list[int]]:
    many, med, few = [], [], []
    for i, c in enumerate(train_counts):
        if c >= many_th:
            many.append(i)
        elif c < few_th:
            few.append(i)
        else:
            med.append(i)
    return {"many": many, "medium": med, "few": few}


def parse_args():
    p = argparse.ArgumentParser(description="Make stratified tea dataset splits")
    p.add_argument("--data-root", type=str, default=str(TeaConfig.data_root))
    p.add_argument("--out", type=str, default=str(TeaConfig.split_path))
    p.add_argument("--seed", type=int, default=TeaConfig.seed)
    p.add_argument("--train-ratio", type=float, default=TeaConfig.train_ratio)
    p.add_argument("--val-ratio", type=float, default=TeaConfig.val_ratio)
    return p.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    dataset = TeaLeafDataset(data_root)
    if len(dataset) == 0:
        raise FileNotFoundError(f"No images under {data_root}")

    splits = stratified_split(
        dataset.samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    train_counts = count_per_class(splits["train"], num_classes=len(TEA_CLASSES))
    buckets = frequency_buckets(
        train_counts, TeaConfig.many_threshold, TeaConfig.few_threshold
    )

    payload = {
        "seed": args.seed,
        "data_root": str(data_root.resolve()),
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": round(1.0 - args.train_ratio - args.val_ratio, 4),
        },
        "classes": TEA_CLASSES,
        "class_to_idx": CLASS_TO_IDX,
        "train_counts": train_counts,
        "val_counts": count_per_class(splits["val"], len(TEA_CLASSES)),
        "test_counts": count_per_class(splits["test"], len(TEA_CLASSES)),
        "frequency_buckets": buckets,
        "many_threshold": TeaConfig.many_threshold,
        "few_threshold": TeaConfig.few_threshold,
        "splits": {
            "train": to_relative(splits["train"], data_root),
            "val": to_relative(splits["val"], data_root),
            "test": to_relative(splits["test"], data_root),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"wrote {out_path}")
    print(f"total={len(dataset)} train={len(splits['train'])} val={len(splits['val'])} test={len(splits['test'])}")
    print("train_counts:", train_counts)
    print("frequency_buckets:", buckets)
    # Leak check
    sets = {s: {x["path"] for x in payload["splits"][s]} for s in ("train", "val", "test")}
    assert sets["train"].isdisjoint(sets["val"])
    assert sets["train"].isdisjoint(sets["test"])
    assert sets["val"].isdisjoint(sets["test"])
    print("leak_check_ok")


if __name__ == "__main__":
    main()
