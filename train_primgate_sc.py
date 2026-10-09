"""Train PrimGate-SC (sibling-conditional) on frozen tea splits."""

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
from losses.primgate_sc_loss import PrimGateSCCriterion
from models import PrimGateSC


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(description="Train PrimGate-SC")
    p.add_argument("--split", type=str, default=str(TeaConfig.split_path))
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=TeaConfig.seed)
    p.add_argument("--feat-dim", type=int, default=256)
    p.add_argument("--protos-per-group", type=int, default=8)
    p.add_argument("--base-loss", type=str, default="balanced_softmax")
    p.add_argument("--lambda-sib", type=float, default=0.5)
    p.add_argument("--lambda-gate", type=float, default=0.05)
    p.add_argument(
        "--selection",
        type=str,
        default="composite",
        choices=["f1", "composite"],
        help="Checkpoint selection: f1 = Macro-F1 only; composite = F1 - beta*G1",
    )
    p.add_argument("--sel-beta", type=float, default=0.35, help="G1 weight in composite selection")
    p.add_argument("--no-cgsf", action="store_true")
    p.add_argument("--no-group-proto", action="store_true")
    p.add_argument("--no-sib", action="store_true")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--no-amf", action="store_true")
    p.add_argument("--patience", type=int, default=12, help="Early stop patience on selection score")
    p.add_argument("--min-epochs", type=int, default=8, help="Do not early-stop before this epoch")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--eval-test", action="store_true")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--run-name", type=str, default=None)
    return p.parse_args()


def selection_score(val_m: dict, mode: str, beta: float) -> float:
    f1 = float(val_m["macro_f1"])
    g1 = float(val_m.get("sibling_error_G1_blight") or 0.0)
    if mode == "composite":
        return f1 - beta * g1
    return f1


def train_one_epoch(model, loader, optimizer, criterion, device, use_amp, scaler):
    model.train()
    total_loss = 0.0
    sum_cls = sum_sib = sum_gate = 0.0
    correct = 0
    total = 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        if use_amp and device.type == "cuda":
            with torch.amp.autocast("cuda"):
                out = model(images, labels=labels)
                losses = criterion(out, labels)
                loss = losses["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(images, labels=labels)
            losses = criterion(out, labels)
            loss = losses["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        bs = labels.size(0)
        total_loss += float(loss.item()) * bs
        sum_cls += float(losses["loss_cls"].item()) * bs
        sum_sib += float(losses["loss_sib"].item()) * bs
        sum_gate += float(losses["loss_gate"].item()) * bs
        correct += int((out["logits"].argmax(dim=1) == labels).sum().item())
        total += bs
    n = max(total, 1)
    return {
        "loss": total_loss / n,
        "loss_cls": sum_cls / n,
        "loss_sib": sum_sib / n,
        "loss_gate": sum_gate / n,
        "acc": correct / n,
    }


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

    model = PrimGateSC(
        num_classes=TeaConfig.num_classes,
        feat_dim=args.feat_dim,
        protos_per_group=args.protos_per_group,
        use_cgsf=not args.no_cgsf,
        use_group_proto=not args.no_group_proto,
        backbone="resnet50",
        pretrained=not args.no_pretrained,
        use_amf=not args.no_amf,
        class_to_group=class_to_group,
    ).to(device)

    criterion = PrimGateSCCriterion(
        cls_num_list=cls_num_list,
        class_to_group=class_to_group,
        base_loss=args.base_loss,
        lambda_sib=0.0 if args.no_sib else args.lambda_sib,
        lambda_gate=args.lambda_gate,
        use_sib=not args.no_sib,
        label_smoothing=args.label_smoothing,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    tags = []
    if args.no_cgsf:
        tags.append("noCGSF")
    if args.no_group_proto:
        tags.append("noGroupProto")
    if args.no_sib:
        tags.append("noSibSoftmax")
    run_name = args.run_name or (
        "primgate_sc_full" if not tags else "primgate_sc_" + "_".join(tags)
    )
    run_name = f"{run_name}_seed{args.seed}"
    out_dir = Path(args.out_dir) if args.out_dir else Path("experiments/primgate_sc") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"run={run_name} device={device} amp={use_amp} params={n_params:,} "
        f"cgsf={not args.no_cgsf} group_proto={not args.no_group_proto} "
        f"sib={not args.no_sib} lr={args.lr} wd={args.weight_decay} "
        f"ls={args.label_smoothing} sel={args.selection} beta={args.sel_beta} "
        f"patience={args.patience}"
    )

    best_score = float("-inf")
    best_f1 = -1.0
    best_g1 = float("nan")
    best_epoch = 0
    best_path = out_dir / "best.pt"
    history = []
    stale = 0

    for epoch in range(1, args.epochs + 1):
        train_m = train_one_epoch(
            model, train_loader, optimizer, criterion, device, use_amp, scaler
        )
        val_m = evaluate(
            model,
            val_loader,
            device,
            criterion=None,
            frequency_buckets=buckets,
            class_names=TEA_CLASSES,
            use_amp=use_amp,
        )
        scheduler.step()
        score = selection_score(val_m, args.selection, args.sel_beta)
        row = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "train_loss": train_m["loss"],
            "train_loss_cls": train_m["loss_cls"],
            "train_loss_sib": train_m["loss_sib"],
            "train_loss_gate": train_m["loss_gate"],
            "train_acc": train_m["acc"],
            "val_acc": val_m["acc"],
            "val_macro_f1": val_m["macro_f1"],
            "val_balanced_acc": val_m["balanced_acc"],
            "val_acc_many": val_m.get("acc_many"),
            "val_acc_medium": val_m.get("acc_medium"),
            "val_acc_few": val_m.get("acc_few"),
            "val_G1": val_m.get("sibling_error_G1_blight"),
            "val_sib_among": val_m.get("sibling_error_among_errors"),
            "val_sel_score": score,
        }
        history.append(row)
        print(
            f"epoch {epoch}/{args.epochs} lr={row['lr']:.2e} "
            f"loss={train_m['loss']:.4f} (cls={train_m['loss_cls']:.3f} "
            f"sib={train_m['loss_sib']:.3f} gate={train_m['loss_gate']:.3f}) "
            f"train_acc={train_m['acc']:.3f} sel={score:.4f} val {val_m['summary']}"
        )
        if score > best_score:
            best_score = score
            best_f1 = float(val_m["macro_f1"])
            best_g1 = float(val_m.get("sibling_error_G1_blight") or float("nan"))
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "val_macro_f1": best_f1,
                    "val_G1": best_g1,
                    "val_sel_score": best_score,
                    "args": vars(args),
                },
                best_path,
            )
        else:
            stale += 1
            if epoch >= args.min_epochs and stale >= args.patience:
                print(f"early stop at epoch {epoch} (patience={args.patience})")
                break

    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    result = {
        "run": run_name,
        "model": "primgate_sc",
        "backbone": "resnet50",
        "feat_dim": args.feat_dim,
        "best_val_macro_f1": best_f1,
        "best_val_G1": best_g1,
        "best_val_sel_score": best_score,
        "best_epoch": best_epoch,
        "checkpoint": str(best_path),
        "use_cgsf": not args.no_cgsf,
        "use_group_proto": not args.no_group_proto,
        "use_sib": not args.no_sib,
        "lambda_sib": 0.0 if args.no_sib else args.lambda_sib,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "selection": args.selection,
        "sel_beta": args.sel_beta,
        "stopped_epoch": history[-1]["epoch"] if history else 0,
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
                "test_G1": test_m.get("sibling_error_G1_blight"),
                "test_sib_among": test_m.get("sibling_error_among_errors"),
                "test_Healthy_f1": test_m.get("per_class_f1", {}).get("Healthy Leaf")
                if isinstance(test_m.get("per_class_f1"), dict)
                else None,
            }
        )
        # per_class_f1 may be nested in metrics
        if "per_class_f1" in test_m:
            result["test_Healthy_f1"] = test_m["per_class_f1"].get("Healthy Leaf")
        with (out_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
            json.dump({k: v for k, v in test_m.items() if k != "summary"}, f, indent=2)
        print(
            f"test F1={test_m['macro_f1']:.4f} G1={test_m.get('sibling_error_G1_blight'):.4f} "
            f"few={test_m.get('acc_few'):.4f} {test_m['summary']}"
        )

    append_row(Path("experiments/primgate_sc/tea_results.csv"), result)
    with (out_dir / "result.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"saved best Macro-F1={best_f1:.4f} -> {best_path}")


if __name__ == "__main__":
    main()
