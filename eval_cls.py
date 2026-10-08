"""Evaluate a checkpoint on val/test with Phase-1 metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.local_cache import setup_local_caches

setup_local_caches()

import torch
from torch.utils.data import DataLoader

from configs.config_tea import TeaConfig
from engine_cls import evaluate
from loader import TEA_CLASSES, make_tea_datasets
from losses import build_loss
from models import PrimGate
from models.baselines import build_baseline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "primgate"])
    p.add_argument("--loss", type=str, default="ce", choices=["ce", "ldam", "balanced_softmax"])
    p.add_argument("--split", type=str, default=str(TeaConfig.split_path))
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--split-name", type=str, default="test", choices=["train", "val", "test"])
    p.add_argument("--batch-size", type=int, default=TeaConfig.batch_size)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    train_ds, val_ds, test_ds, split_data = make_tea_datasets(args.split, args.data_root)
    ds = {"train": train_ds, "val": val_ds, "test": test_ds}[args.split_name]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    if args.model == "primgate":
        model = PrimGate(num_classes=TeaConfig.num_classes)
    else:
        model = build_baseline(args.model, num_classes=TeaConfig.num_classes, pretrained=False)
    model = model.to(device)

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    criterion = build_loss(args.loss, cls_num_list=split_data["train_counts"]).to(device)

    metrics = evaluate(
        model,
        loader,
        device,
        criterion,
        frequency_buckets=split_data["frequency_buckets"],
        class_names=TEA_CLASSES,
    )
    print(metrics["summary"])
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump({k: v for k, v in metrics.items() if k != "summary"}, f, indent=2)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
