"""Classification train/eval engine — Phase 1 fills metrics fully."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: Callable | None = None,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        out = model(images)
        logits = out["logits"] if isinstance(out, dict) else out
        if criterion is not None:
            total_loss += float(criterion(logits, labels).item()) * labels.size(0)
        pred = logits.argmax(dim=1)
        correct += int((pred == labels).sum().item())
        total += labels.size(0)
    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        out = model(images)
        logits = out["logits"] if isinstance(out, dict) else out
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += labels.size(0)
    return {
        "loss": total_loss / max(total, 1),
        "acc": correct / max(total, 1),
    }
