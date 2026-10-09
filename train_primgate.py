"""Train PrimGate (full method + ablations) on frozen tea splits."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from utils.local_cache import setup_local_caches

setup_local_caches()

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config_tea import TeaConfig
from engine_cls import evaluate
from loader import CLASS_TO_SIBLING, TEA_CLASSES, make_tea_datasets
from losses.primgate_loss import PrimGateCriterion
from models import PrimGate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="Train PrimGate")
    p.add_argument("--split", type=str, default=str(TeaConfig.split_path))
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=TeaConfig.seed)
    p.add_argument("--feat-dim", type=int, default=128)
    p.add_argument("--num-primitives", type=int, default=16)
    p.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        choices=["resnet50", "lite"],
        help="resnet50 = capacity-matched dual stream; lite = scratch CNN",
    )
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-amf", action="store_true", help="Disable AMF on ResNet symptom stream")
    p.add_argument("--base-loss", type=str, default="balanced_softmax")
    p.add_argument("--lambda-bal", type=float, default=0.5)
    p.add_argument("--lambda-satc", type=float, default=0.3)
    p.add_argument("--lambda-gate", type=float, default=0.05)
    p.add_argument("--no-cgsf", action="store_true")
    p.add_argument("--no-spf", action="store_true")
    p.add_argument("--no-satc", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--eval-test", action="store_true")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--run-name", type=str, default=None)
    return p.parse_args()


def train_one_epoch_primgate(
    model,
    loader,
    optimizer,
    criterion,
    device,
    use_amp,
    scaler,
):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                out = model(images)
                losses = criterion(out, labels)
                loss = losses["loss"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(images)
            losses = criterion(out, labels)
            loss = losses["loss"]
            loss.backward()
            optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        correct += int((out["logits"].argmax(dim=1) == labels).sum().item())
        total += labels.size(0)
    return {"loss": total_loss / max(total, 1), "acc": correct / max(total, 1)}


def append_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    use_amp = args.amp and not args.no_amp and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_ds, val_ds, test_ds, split_data = make_tea_datasets(args.split, args.data_root)
    buckets = split_data["frequency_buckets"]
    cls_num_list = split_data["train_counts"]
    class_to_group = torch.tensor(
        [CLASS_TO_SIBLING[i] for i in range(TeaConfig.num_classes)], dtype=torch.long
    )

    loader_kw = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_kw["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)

    feat_dim = args.feat_dim
    if args.backbone == "resnet50" and feat_dim == 128:
        feat_dim = 256

    model = PrimGate(
        num_classes=TeaConfig.num_classes,
        feat_dim=feat_dim,
        num_primitives=args.num_primitives,
        use_cgsf=not args.no_cgsf,
        use_spf=not args.no_spf,
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        use_amf=not args.no_amf,
    ).to(device)

    criterion = PrimGateCriterion(
        cls_num_list=cls_num_list,
        head_ids=buckets["many"],
        few_ids=buckets["few"],
        class_to_group=class_to_group,
        base_loss=args.base_loss,
        lambda_bal=0.0 if args.no_spf else args.lambda_bal,
        lambda_satc=0.0 if args.no_satc else args.lambda_satc,
        lambda_gate=args.lambda_gate,
        use_satc=not args.no_satc,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    tags = []
    if args.no_cgsf:
        tags.append("noCGSF")
    if args.no_spf:
        tags.append("noSPF")
    if args.no_satc:
        tags.append("noSATC")
    prefix = "primgate_r50" if args.backbone == "resnet50" else "primgate"
    run_name = args.run_name or (
        f"{prefix}_full" if not tags else f"{prefix}_" + "_".join(tags)
    )
    run_name = f"{run_name}_seed{args.seed}"
    out_dir = Path(args.out_dir) if args.out_dir else Path("experiments/primgate") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"run={run_name} backbone={args.backbone} device={device} amp={use_amp} "
        f"params={n_params:,} feat_dim={feat_dim} "
        f"cgsf={not args.no_cgsf} spf={not args.no_spf} satc={not args.no_satc} "
        f"amf={not args.no_amf}"
    )

    best_f1 = -1.0
    best_path = out_dir / "best.pt"
    history = []

    for epoch in range(1, args.epochs + 1):
        train_m = train_one_epoch_primgate(
            model, train_loader, optimizer, criterion, device, use_amp, scaler
        )
        # Eval uses CE on logits for reporting (metrics only)
        val_m = evaluate(
            model,
            val_loader,
            device,
            criterion=None,
            frequency_buckets=buckets,
            class_names=TEA_CLASSES,
            use_amp=use_amp,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "train_acc": train_m["acc"],
            "val_acc": val_m["acc"],
            "val_macro_f1": val_m["macro_f1"],
            "val_balanced_acc": val_m["balanced_acc"],
            "val_acc_many": val_m.get("acc_many"),
            "val_acc_medium": val_m.get("acc_medium"),
            "val_acc_few": val_m.get("acc_few"),
            "val_sibling_error": val_m.get("sibling_error_among_errors"),
        }
        history.append(row)
        print(
            f"epoch {epoch}/{args.epochs} "
            f"train_loss={train_m['loss']:.4f} train_acc={train_m['acc']:.3f} "
            f"val {val_m['summary']}"
        )
        if val_m["macro_f1"] > best_f1:
            best_f1 = val_m["macro_f1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_macro_f1": best_f1,
                    "args": vars(args),
                },
                best_path,
            )

    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    result = {
        "run": run_name,
        "model": "primgate",
        "backbone": args.backbone,
        "feat_dim": feat_dim,
        "best_val_macro_f1": best_f1,
        "checkpoint": str(best_path),
        "use_cgsf": not args.no_cgsf,
        "use_spf": not args.no_spf,
        "use_satc": not args.no_satc,
        "use_amf": not args.no_amf,
        "lambda_satc": 0.0 if args.no_satc else args.lambda_satc,
        "lambda_bal": 0.0 if args.no_spf else args.lambda_bal,
    }

    if args.eval_test and best_path.exists():
        state = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        test_m = evaluate(
            model,
            test_loader,
            device,
            criterion=None,
            frequency_buckets=buckets,
            class_names=TEA_CLASSES,
            use_amp=use_amp,
        )
        result.update(
            {
                "test_acc": test_m["acc"],
                "test_macro_f1": test_m["macro_f1"],
                "test_balanced_acc": test_m["balanced_acc"],
                "test_acc_many": test_m.get("acc_many"),
                "test_acc_medium": test_m.get("acc_medium"),
                "test_acc_few": test_m.get("acc_few"),
                "test_sibling_error": test_m.get("sibling_error_among_errors"),
            }
        )
        with (out_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
            json.dump({k: v for k, v in test_m.items() if k != "summary"}, f, indent=2)
        print(f"test {test_m['summary']}")

    append_row(Path("experiments/primgate/tea_results.csv"), result)
    with (out_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"saved best Macro-F1={best_f1:.4f} -> {best_path}")


if __name__ == "__main__":
    main()
