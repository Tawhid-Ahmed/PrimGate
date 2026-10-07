"""Training entrypoint — Phase 1 wires configs, splits, and baselines."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from configs.config_tea import TeaConfig
from engine_cls import evaluate, train_one_epoch
from loader import TeaLeafDataset
from losses import CrossEntropyLoss
from models import PrimGate


def parse_args():
    p = argparse.ArgumentParser(description="PrimGate / baseline classification training")
    p.add_argument("--data-root", type=str, default=str(TeaConfig.data_root))
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=TeaConfig.batch_size)
    p.add_argument("--lr", type=float, default=TeaConfig.lr)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    tfm = transforms.Compose(
        [
            transforms.Resize((TeaConfig.image_size, TeaConfig.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=TeaConfig.mean, std=TeaConfig.std),
        ]
    )
    dataset = TeaLeafDataset(args.data_root, transform=tfm)
    if len(dataset) == 0:
        raise FileNotFoundError(f"No images found under {args.data_root}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    model = PrimGate(num_classes=TeaConfig.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = CrossEntropyLoss()

    print(f"device={device} samples={len(dataset)} classes={TeaConfig.num_classes}")
    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(model, loader, optimizer, criterion, device)
        val_metrics = evaluate(model, loader, device, criterion)
        print(
            f"epoch {epoch+1}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.3f} "
            f"eval_acc={val_metrics['acc']:.3f}"
        )

    out_dir = Path("experiments/primgate")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "phase0_smoke.pt"
    torch.save({"model": model.state_dict()}, ckpt)
    print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
