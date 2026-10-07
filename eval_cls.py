"""Evaluation entrypoint — Phase 1 adds Macro-F1 / Many-Med-Few / sibling error."""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from configs.config_tea import TeaConfig
from engine_cls import evaluate
from loader import TeaLeafDataset
from models import PrimGate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=str(TeaConfig.data_root))
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--batch-size", type=int, default=TeaConfig.batch_size)
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
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    model = PrimGate(num_classes=TeaConfig.num_classes).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"])
    metrics = evaluate(model, loader, device)
    print(metrics)


if __name__ == "__main__":
    main()
