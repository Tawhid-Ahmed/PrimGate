"""Train PrimGate or Phase-1 baselines on frozen tea splits."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

# Keep Torch/HF downloads on this project drive (not C:)
from utils.local_cache import setup_local_caches

setup_local_caches()

import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.config_tea import TeaConfig
from engine_cls import evaluate, train_one_epoch
from loader import TEA_CLASSES, make_tea_datasets
from losses import build_loss
from models import PrimGate
from models.baselines import build_baseline


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="PrimGate / baseline classification training")
    p.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "primgate"])
    p.add_argument("--loss", type=str, default="ce", choices=["ce", "ldam", "balanced_softmax"])
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--split", type=str, default=str(TeaConfig.split_path))
    p.add_argument("--epochs", type=int, default=TeaConfig.epochs)
    p.add_argument("--batch-size", type=int, default=TeaConfig.batch_size)
    p.add_argument("--lr", type=float, default=TeaConfig.lr)
    p.add_argument("--weight-decay", type=float, default=TeaConfig.weight_decay)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=TeaConfig.seed)
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--eval-test", action="store_true", help="Evaluate best ckpt on test after training")
    p.add_argument("--amp", action="store_true", help="Use CUDA mixed precision (faster)")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument(
        "--scale-lr",
        action="store_true",
        help="Scale LR linearly with batch size vs TeaConfig.batch_size",
    )
    return p.parse_args()


def build_model(name: str, num_classes: int, pretrained: bool):
    if name == "primgate":
        return PrimGate(num_classes=num_classes)
    return build_baseline(name, num_classes=num_classes, pretrained=pretrained)


def append_results_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    pretrained = False if args.no_pretrained else args.pretrained
    use_amp = (args.amp or device.type == "cuda") and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_ds, val_ds, test_ds, split_data = make_tea_datasets(args.split, args.data_root)
    buckets = split_data["frequency_buckets"]
    cls_num_list = split_data["train_counts"]

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
        "prefetch_factor": 2 if args.num_workers > 0 else None,
    }
    # DataLoader rejects prefetch_factor=None on some versions
    if loader_kwargs["prefetch_factor"] is None:
        loader_kwargs.pop("prefetch_factor")
        loader_kwargs.pop("persistent_workers")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    model = build_model(args.model, TeaConfig.num_classes, pretrained).to(device)
    criterion = build_loss(args.loss, cls_num_list=cls_num_list).to(device)
    lr = args.lr
    if args.scale_lr and args.batch_size != TeaConfig.batch_size:
        lr = args.lr * (args.batch_size / TeaConfig.batch_size)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda") if use_amp and device.type == "cuda" else None

    run_name = f"{args.model}_{args.loss}_seed{args.seed}"
    out_dir = Path(args.out_dir) if args.out_dir else Path("experiments/baselines") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"run={run_name} device={device} amp={use_amp} "
        f"batch={args.batch_size} workers={args.num_workers} lr={lr:g} "
        f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}"
    )

    best_f1 = -1.0
    best_path = out_dir / "best.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            use_amp=use_amp,
            scaler=scaler,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            criterion,
            frequency_buckets=buckets,
            class_names=TEA_CLASSES,
            use_amp=use_amp,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["acc"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_balanced_acc": val_metrics["balanced_acc"],
            "val_acc_many": val_metrics.get("acc_many"),
            "val_acc_medium": val_metrics.get("acc_medium"),
            "val_acc_few": val_metrics.get("acc_few"),
            "val_sibling_error": val_metrics.get("sibling_error_among_errors"),
        }
        history.append(row)
        print(
            f"epoch {epoch}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.3f} "
            f"val {val_metrics['summary']}"
        )

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_macro_f1": best_f1,
                    "args": vars(args),
                    "cls_num_list": cls_num_list,
                },
                best_path,
            )

    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    result = {
        "run": run_name,
        "model": args.model,
        "loss": args.loss,
        "seed": args.seed,
        "best_val_macro_f1": best_f1,
        "checkpoint": str(best_path),
    }

    if args.eval_test and best_path.exists():
        state = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        test_metrics = evaluate(
            model,
            test_loader,
            device,
            criterion,
            frequency_buckets=buckets,
            class_names=TEA_CLASSES,
            use_amp=use_amp,
        )
        result.update(
            {
                "test_acc": test_metrics["acc"],
                "test_macro_f1": test_metrics["macro_f1"],
                "test_balanced_acc": test_metrics["balanced_acc"],
                "test_acc_many": test_metrics.get("acc_many"),
                "test_acc_medium": test_metrics.get("acc_medium"),
                "test_acc_few": test_metrics.get("acc_few"),
                "test_sibling_error": test_metrics.get("sibling_error_among_errors"),
            }
        )
        with (out_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
            # drop huge confusion for csv; keep full json
            json.dump(
                {k: v for k, v in test_metrics.items() if k != "summary"},
                f,
                indent=2,
            )
        print(f"test {test_metrics['summary']}")

    append_results_row(Path("experiments/baselines/tea_results.csv"), result)
    with (out_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"saved best Macro-F1={best_f1:.4f} -> {best_path}")


if __name__ == "__main__":
    main()
