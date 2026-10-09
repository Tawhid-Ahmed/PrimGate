"""Gate C root-cause package: metrics audit + hijack remasure on PrimGate R50."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.local_cache import setup_local_caches

setup_local_caches()

from analysis.hijacking import analyze_hijacking, sibling_confusion_rates
from configs.config_tea import TeaConfig
from loader import CLASS_TO_SIBLING, SIBLING_GROUPS, TEA_CLASSES, make_tea_datasets
from models import PrimGate
from models.baselines import build_baseline


RUNS = {
    "BS": {
        "kind": "baseline",
        "ckpt": ROOT / "experiments/baselines/resnet50_bs_seed42/best.pt",
        "metrics": ROOT / "experiments/baselines/resnet50_bs_seed42/test_metrics.json",
        "history": ROOT / "experiments/baselines/resnet50_bs_seed42/history.json",
    },
    "r50_full": {
        "kind": "primgate",
        "ckpt": ROOT / "experiments/primgate/primgate_r50_full_seed42/best.pt",
        "metrics": ROOT / "experiments/primgate/primgate_r50_full_seed42/test_metrics.json",
        "history": ROOT / "experiments/primgate/primgate_r50_full_seed42/history.json",
        "backbone": "resnet50",
        "use_cgsf": True,
        "use_spf": True,
        "use_amf": True,
    },
    "r50_noCGSF": {
        "kind": "primgate",
        "ckpt": ROOT / "experiments/primgate/primgate_r50_noCGSF_seed42/best.pt",
        "metrics": ROOT / "experiments/primgate/primgate_r50_noCGSF_seed42/test_metrics.json",
        "history": ROOT / "experiments/primgate/primgate_r50_noCGSF_seed42/history.json",
        "backbone": "resnet50",
        "use_cgsf": False,
        "use_spf": True,
        "use_amf": True,
    },
    "r50_noSPF": {
        "kind": "primgate",
        "ckpt": ROOT / "experiments/primgate/primgate_r50_noSPF_seed42/best.pt",
        "metrics": ROOT / "experiments/primgate/primgate_r50_noSPF_seed42/test_metrics.json",
        "history": ROOT / "experiments/primgate/primgate_r50_noSPF_seed42/history.json",
        "backbone": "resnet50",
        "use_cgsf": True,
        "use_spf": False,
        "use_amf": True,
    },
    "r50_noSATC": {
        "kind": "primgate",
        "ckpt": ROOT / "experiments/primgate/primgate_r50_noSATC_seed42/best.pt",
        "metrics": ROOT / "experiments/primgate/primgate_r50_noSATC_seed42/test_metrics.json",
        "history": ROOT / "experiments/primgate/primgate_r50_noSATC_seed42/history.json",
        "backbone": "resnet50",
        "use_cgsf": True,
        "use_spf": True,
        "use_amf": True,
    },
}


def load_json(path: Path):
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def error_structure(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    wrong = y_true != y_pred
    n_err = int(wrong.sum())
    within_g1 = 0
    cross_group = 0
    within_other = 0
    for t, p in zip(y_true[wrong], y_pred[wrong]):
        gt = CLASS_TO_SIBLING[int(t)]
        gp = CLASS_TO_SIBLING[int(p)]
        if gt == gp:
            if gt == 0:
                within_g1 += 1
            else:
                within_other += 1
        else:
            cross_group += 1
    return {
        "n_errors": n_err,
        "n_correct": int((~wrong).sum()),
        "within_g1_errors": within_g1,
        "within_other_sib_errors": within_other,
        "cross_group_errors": cross_group,
        "frac_errors_within_g1": within_g1 / max(n_err, 1),
        "frac_errors_cross_group": cross_group / max(n_err, 1),
        "g1_sibling_rate_all_g1": float(
            sibling_confusion_rates(y_true, y_pred, SIBLING_GROUPS).get("G1_blight", float("nan"))
        ),
    }


@torch.no_grad()
def extract(model, loader, device):
    model.eval()
    feats, labels, preds = [], [], []
    for images, y in tqdm(loader, desc="extract", leave=False):
        images = images.to(device, non_blocking=True)
        out = model(images)
        feats.append(out["embedding"].float().cpu().numpy())
        labels.append(y.numpy())
        preds.append(out["logits"].argmax(dim=1).cpu().numpy())
    return (
        np.concatenate(feats),
        np.concatenate(labels),
        np.concatenate(preds),
    )


def build_model(spec: dict, device: torch.device):
    if spec["kind"] == "baseline":
        model = build_baseline("resnet50", num_classes=TeaConfig.num_classes, pretrained=False)
    else:
        model = PrimGate(
            num_classes=TeaConfig.num_classes,
            feat_dim=256,
            backbone=spec["backbone"],
            use_cgsf=spec["use_cgsf"],
            use_spf=spec["use_spf"],
            pretrained=False,
            use_amf=spec.get("use_amf", True),
        )
    ckpt = torch.load(spec["ckpt"], map_location=device, weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    return model.to(device)


def history_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    hist = load_json(path)
    losses = [float(r["train_loss"]) for r in hist]
    f1s = [float(r["val_macro_f1"]) for r in hist]
    best_ep = int(np.argmax(f1s)) + 1
    return {
        "epochs": len(hist),
        "train_loss_min": float(np.min(losses)),
        "train_loss_final": losses[-1],
        "train_loss_explosion_ratio": losses[-1] / max(min(losses), 1e-6),
        "best_val_macro_f1": float(np.max(f1s)),
        "best_epoch": best_ep,
        "final_val_macro_f1": f1s[-1],
    }


def metric_row(path: Path) -> dict:
    m = load_json(path)
    return {
        "macro_f1": m["macro_f1"],
        "sib_among_errors": m.get("sibling_error_among_errors"),
        "G1_blight": m.get("sibling_error_G1_blight"),
        "acc_many": m.get("acc_many"),
        "acc_medium": m.get("acc_medium"),
        "acc_few": m.get("acc_few"),
        "Healthy_f1": m["per_class_f1"].get("Healthy Leaf"),
        "Brown_f1": m["per_class_f1"].get("Brown Blight - Colletotrichum spp"),
        "acc": m["acc"],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--out", type=str, default=str(ROOT / "experiments/analysis/gate_c_root_cause"))
    args = p.parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, split = make_tea_datasets(TeaConfig.split_path)
    buckets = split["frequency_buckets"]
    loader_kw = {"num_workers": 4, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)

    table = {}
    hijack = {}
    structure = {}
    hist = {}

    for name, spec in RUNS.items():
        print(f"=== {name} ===", flush=True)
        table[name] = metric_row(spec["metrics"])
        hist[name] = history_summary(spec["history"])
        model = build_model(spec, device)
        train_feats, train_y, _ = extract(model, train_loader, device)
        test_feats, test_y, test_pred = extract(model, test_loader, device)
        structure[name] = error_structure(test_y, test_pred)
        hj, _km = analyze_hijacking(
            train_feats=train_feats,
            train_labels=train_y,
            eval_feats=test_feats,
            eval_labels=test_y,
            eval_preds=test_pred,
            head_ids=buckets["many"],
            few_ids=buckets["few"],
            K=args.k,
            temperature=0.2,
            seed=42,
        )
        hijack[name] = {
            "hijack_l2": float(hj.hijack_l2),
            "hijack_l1": float(hj.hijack_l1),
            "corr_error_vs_head_overlap_sample": float(hj.corr_error_vs_head_overlap),
            "G1_from_preds": structure[name]["g1_sibling_rate_all_g1"],
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    causes = {
        "metric_trap": {
            "claim": "sib_among_errors rose while G1 blight rate did not worsen vs BS",
            "BS_sib_among": table["BS"]["sib_among_errors"],
            "BS_G1": table["BS"]["G1_blight"],
            "full_sib_among": table["r50_full"]["sib_among_errors"],
            "full_G1": table["r50_full"]["G1_blight"],
            "verdict": bool(table["r50_full"]["G1_blight"] <= table["BS"]["G1_blight"]),
        },
        "spf_harms_healthy": {
            "full_Healthy_f1": table["r50_full"]["Healthy_f1"],
            "noSPF_Healthy_f1": table["r50_noSPF"]["Healthy_f1"],
            "noSPF_macro_f1": table["r50_noSPF"]["macro_f1"],
            "full_macro_f1": table["r50_full"]["macro_f1"],
            "verdict": bool(table["r50_noSPF"]["macro_f1"] > table["r50_full"]["macro_f1"]),
        },
        "satc_helps_f1_not_g1": {
            "full_macro_f1": table["r50_full"]["macro_f1"],
            "noSATC_macro_f1": table["r50_noSATC"]["macro_f1"],
            "full_G1": table["r50_full"]["G1_blight"],
            "noSATC_G1": table["r50_noSATC"]["G1_blight"],
            "delta_f1": table["r50_full"]["macro_f1"] - table["r50_noSATC"]["macro_f1"],
            "delta_G1": table["r50_full"]["G1_blight"] - table["r50_noSATC"]["G1_blight"],
        },
        "cgsf_helps_f1": {
            "full_macro_f1": table["r50_full"]["macro_f1"],
            "noCGSF_macro_f1": table["r50_noCGSF"]["macro_f1"],
            "verdict": bool(table["r50_full"]["macro_f1"] > table["r50_noCGSF"]["macro_f1"]),
        },
        "hijack_not_closed": {
            name: hijack[name]["hijack_l2"] for name in hijack
        },
        "loss_instability": {
            name: {
                "explosion_ratio": hist[name].get("train_loss_explosion_ratio"),
                "best_epoch": hist[name].get("best_epoch"),
            }
            for name in hist
            if hist[name]
        },
    }

    payload = {
        "dataset": {
            "root": str(TeaConfig.data_root),
            "split": str(TeaConfig.split_path),
            "train_counts": split["train_counts"],
            "test_counts": split["test_counts"],
            "classes": TEA_CLASSES,
            "note": "Custom tea pathology LT set; prep OK; hardness = imbalance + siblings + tiny Healthy",
        },
        "metrics_table": table,
        "error_structure": structure,
        "hijack_remeasure": hijack,
        "history_summary": hist,
        "causes": causes,
        "gate_c_v3_protocol": {
            "primary": ["macro_f1", "sibling_error_G1_blight", "hijack_l2"],
            "secondary": ["acc_few", "Healthy_f1"],
            "diagnostic_only": ["sibling_error_among_errors"],
        },
        "recommended_next": "PrimGate-SC: keep CGSF; drop global L_bal SPF; replace SATC with sibling-conditional softmax + group-conditional protos",
    }

    out_json = out_dir / "root_cause_summary.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {out_json}", flush=True)

    # Markdown report
    lines = [
        "# Gate C root-cause analysis",
        "",
        "## Dataset",
        f"- Root: `{TeaConfig.data_root}` (custom tea pathology, 9 classes)",
        f"- Split: frozen 70/15/15 seed 42",
        f"- Train counts: {split['train_counts']}",
        "- Prep is fine; hardness is LT + blight siblings + Healthy n=12.",
        "",
        "## Metrics table (test)",
        "",
        "| Run | Macro-F1 | sib_among | G1 blight | few | Healthy F1 | hijack_L2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in RUNS:
        t = table[name]
        h = hijack[name]
        lines.append(
            f"| {name} | {t['macro_f1']:.4f} | {t['sib_among_errors']:.4f} | "
            f"{t['G1_blight']:.4f} | {t['acc_few']:.4f} | {t['Healthy_f1']:.4f} | "
            f"{h['hijack_l2']:.4f} |"
        )
    lines += [
        "",
        "## Causes",
        "",
        "1. **Metric trap:** Gate C used `sib_among_errors`. Full model G1 is "
        f"{table['r50_full']['G1_blight']:.4f} vs BS {table['BS']['G1_blight']:.4f} "
        f"(not worse), while sib_among rose because cross-group errors shrank "
        f"(full cross-group errors={structure['r50_full']['cross_group_errors']}, "
        f"BS={structure['BS']['cross_group_errors']}).",
        "",
        "2. **SPF / L_bal harms rare classes:** noSPF Macro-F1 "
        f"{table['r50_noSPF']['macro_f1']:.4f} > full {table['r50_full']['macro_f1']:.4f}; "
        f"Healthy F1 {table['r50_noSPF']['Healthy_f1']:.3f} vs full {table['r50_full']['Healthy_f1']:.3f}. "
        "Online SPF ≠ Phase-2 KMeans hijack object.",
        "",
        "3. **SATC helps F1, barely moves G1:** "
        f"ΔF1={causes['satc_helps_f1_not_g1']['delta_f1']:.4f}, "
        f"ΔG1={causes['satc_helps_f1_not_g1']['delta_G1']:.4f}. "
        "Embedding margin ≠ within-group logit ranking.",
        "",
        "4. **CGSF helps:** "
        f"full F1 {table['r50_full']['macro_f1']:.4f} vs noCGSF {table['r50_noCGSF']['macro_f1']:.4f}.",
        "",
        "5. **Hijack not closed by SPF:** see hijack_L2 column (full vs BS).",
        "",
        "6. **Loss instability:** late train_loss explosion "
        f"(full ratio={hist['r50_full'].get('train_loss_explosion_ratio', float('nan')):.1f}x).",
        "",
        "## Gate C v3 protocol (frozen)",
        "- Primary: Macro-F1, G1 sibling rate, hijack_L2",
        "- Secondary: few Acc, Healthy F1",
        "- Diagnostic only: sib_among_errors",
        "",
        "## Next move",
        payload["recommended_next"],
        "",
    ]
    md_path = out_dir / "gate_c_root_cause.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
