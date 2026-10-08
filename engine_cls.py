"""Classification train/eval engine with long-tail metrics."""

from __future__ import annotations

from typing import Callable, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics_cls import compute_classification_metrics, format_metrics_line


def _logits_from_output(out) -> torch.Tensor:
    return out["logits"] if isinstance(out, dict) else out


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: Callable | None = None,
    frequency_buckets: dict[str, Sequence[int]] | None = None,
    class_names: Sequence[str] | None = None,
    use_amp: bool = False,
) -> dict:
    model.eval()
    total_loss = 0.0
    total = 0
    all_true: list[int] = []
    all_pred: list[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = _logits_from_output(model(images))
                loss = criterion(logits, labels) if criterion is not None else None
        else:
            logits = _logits_from_output(model(images))
            loss = criterion(logits, labels) if criterion is not None else None
        if loss is not None:
            total_loss += float(loss.item()) * labels.size(0)
        pred = logits.argmax(dim=1)
        all_true.extend(labels.cpu().tolist())
        all_pred.extend(pred.cpu().tolist())
        total += labels.size(0)

    metrics = compute_classification_metrics(
        all_true,
        all_pred,
        frequency_buckets=frequency_buckets,
        class_names=class_names,
    )
    metrics["loss"] = total_loss / max(total, 1)
    metrics["summary"] = format_metrics_line(metrics)
    return metrics


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: Callable,
    device: torch.device,
    use_amp: bool = False,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    if use_amp and device.type == "cuda" and scaler is None:
        scaler = torch.amp.GradScaler("cuda")

    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                logits = _logits_from_output(model(images))
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = _logits_from_output(model(images))
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
