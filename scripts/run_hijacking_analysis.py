"""Phase 2 / RQ1: run primitive hijacking analysis on trained ResNet baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:  # pragma: no cover
    HAS_MPL = False
    plt = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.local_cache import setup_local_caches

setup_local_caches()

from analysis.hijacking import (
    analyze_hijacking,
    sibling_confusion_rates,
    summary_metrics,
)
from configs.config_tea import TeaConfig
from loader import SIBLING_GROUPS, TEA_CLASSES, make_tea_datasets
from models.baselines import build_baseline


@torch.no_grad()
def extract_feats_preds(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    feats, labels, preds = [], [], []
    for images, y in tqdm(loader, desc="extract", leave=False):
        images = images.to(device, non_blocking=True)
        out = model(images)
        logits = out["logits"]
        emb = out["embedding"]
        feats.append(emb.float().cpu().numpy())
        labels.append(y.numpy())
        preds.append(logits.argmax(dim=1).cpu().numpy())
    return (
        np.concatenate(feats, axis=0),
        np.concatenate(labels, axis=0),
        np.concatenate(preds, axis=0),
    )


def plot_occupancy_heatmap(
    class_means: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    if not HAS_MPL:
        np.savetxt(out_path.with_suffix(".csv"), class_means, delimiter=",")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(class_means, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_yticks(range(len(TEA_CLASSES)))
    ax.set_yticklabels([c.split(" - ")[0][:22] for c in TEA_CLASSES], fontsize=8)
    ax.set_xlabel("Primitive index")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_head_tail_gap(
    head_mean: np.ndarray,
    tail_mean: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    if not HAS_MPL:
        np.savetxt(
            out_path.with_suffix(".csv"),
            np.vstack([head_mean, tail_mean]),
            delimiter=",",
            header="row0=head,row1=tail",
            comments="",
        )
        return
    K = len(head_mean)
    x = np.arange(K)
    width = 0.4
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width / 2, head_mean, width, label="Head (many)")
    ax.bar(x + width / 2, tail_mean, width, label="Tail (few)")
    ax.set_xlabel("Primitive")
    ax.set_ylabel("Mean soft occupancy")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_confusion(cm: list[list[int]], out_path: Path, title: str) -> None:
    arr = np.asarray(cm)
    if not HAS_MPL:
        np.savetxt(out_path.with_suffix(".csv"), arr, delimiter=",", fmt="%d")
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks(range(len(TEA_CLASSES)))
    ax.set_yticks(range(len(TEA_CLASSES)))
    short = [c.split(" - ")[0][:14] for c in TEA_CLASSES]
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(short, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to Phase-1 best.pt",
    )
    p.add_argument("--model", type=str, default="resnet50")
    p.add_argument("--split", type=str, default=str(TeaConfig.split_path))
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    ckpt_path = Path(args.checkpoint)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else ROOT / "experiments" / "analysis" / f"hijack_{ckpt_path.parent.name}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, split_data = make_tea_datasets(args.split)
    buckets = split_data["frequency_buckets"]
    head_ids = buckets["many"]
    few_ids = buckets["few"]

    loader_kw = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    if args.num_workers > 0:
        loader_kw["prefetch_factor"] = 2
    else:
        loader_kw.pop("persistent_workers", None)

    train_loader = DataLoader(train_ds, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)

    model = build_baseline(args.model, num_classes=TeaConfig.num_classes, pretrained=False)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model = model.to(device)

    print(f"checkpoint={ckpt_path}")
    print(f"head_ids={head_ids} few_ids={few_ids}")

    train_feats, train_labels, _ = extract_feats_preds(model, train_loader, device)
    test_feats, test_labels, test_preds = extract_feats_preds(model, test_loader, device)

    result, km = analyze_hijacking(
        train_feats,
        train_labels,
        test_feats,
        test_labels,
        test_preds,
        head_ids=head_ids,
        few_ids=few_ids,
        num_classes=TeaConfig.num_classes,
        K=args.K,
        temperature=args.temperature,
        seed=TeaConfig.seed,
    )
    cls_metrics = summary_metrics(test_labels, test_preds, TeaConfig.num_classes)
    sib = sibling_confusion_rates(test_labels, test_preds, SIBLING_GROUPS)

    payload = {
        "checkpoint": str(ckpt_path),
        "K": result.K,
        "temperature": result.temperature,
        "hijack_l2": result.hijack_l2,
        "hijack_l1": result.hijack_l1,
        "primitive_entropy_head": result.primitive_entropy_head,
        "primitive_entropy_tail": result.primitive_entropy_tail,
        "corr_error_vs_head_overlap_sample": result.corr_error_vs_head_overlap,
        "corr_error_vs_head_overlap_class": result.corr_error_vs_head_overlap_class,
        "head_dominance": result.head_dominance.tolist(),
        "n_head_dominated_primitives": int((result.head_dominance >= 0.6).sum()),
        "head_mean_occupancy": result.head_mean_occupancy.tolist(),
        "tail_mean_occupancy": result.tail_mean_occupancy.tolist(),
        "class_mean_occupancy": result.class_mean_occupancy.tolist(),
        "per_class_error": {TEA_CLASSES[k]: v for k, v in result.per_class_macro_f1_proxy.items()},
        "sibling_confusion": sib,
        "test_metrics": {k: v for k, v in cls_metrics.items() if k != "confusion"},
        "frequency_buckets": buckets,
        "gate_b": {
            "hijack_l2_positive": result.hijack_l2 > 0.01,
            "corr_positive": (
                result.corr_error_vs_head_overlap == result.corr_error_vs_head_overlap
                and result.corr_error_vs_head_overlap > 0.05
            ),
            "sibling_still_high": any(v == v and v > 0.2 for v in sib.values()),
        },
    }

    with (out_dir / "hijacking_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    np.savez_compressed(
        out_dir / "hijacking_arrays.npz",
        alphas=result.alphas,
        labels=result.labels,
        preds=result.preds,
        centers=km.cluster_centers_,
        class_mean_occupancy=result.class_mean_occupancy,
        head_dominance=result.head_dominance,
    )

    plot_occupancy_heatmap(
        result.class_mean_occupancy,
        out_dir / "class_primitive_occupancy.png",
        f"Class × primitive occupancy ({ckpt_path.parent.name})",
    )
    plot_head_tail_gap(
        result.head_mean_occupancy,
        result.tail_mean_occupancy,
        out_dir / "head_vs_tail_occupancy.png",
        f"Head vs tail primitive occupancy (L2 gap={result.hijack_l2:.4f})",
    )
    plot_confusion(
        cls_metrics["confusion"],
        out_dir / "test_confusion.png",
        f"Test confusion ({ckpt_path.parent.name})",
    )

    print(
        f"hijack_l2={result.hijack_l2:.4f} hijack_l1={result.hijack_l1:.4f} "
        f"corr_sample={result.corr_error_vs_head_overlap:.4f} "
        f"corr_class={result.corr_error_vs_head_overlap_class:.4f}"
    )
    print(f"sibling_confusion={sib}")
    print(f"gate_b={payload['gate_b']}")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
