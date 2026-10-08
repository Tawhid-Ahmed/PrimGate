"""Long-tailed classification metrics for PrimGate Phase 1."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

from loader.tea_loader import CLASS_TO_SIBLING, SIBLING_GROUPS, TEA_CLASSES


def sibling_error_rate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    Among misclassified samples, fraction whose prediction is in the same
    sibling group as the true label. Also report per-group rates.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    wrong = y_true != y_pred
    out: dict[str, float] = {}

    def rate(mask: np.ndarray) -> float:
        if mask.sum() == 0:
            return float("nan")
        sibling_wrong = 0
        for t, p in zip(y_true[mask], y_pred[mask]):
            gt = CLASS_TO_SIBLING.get(int(t), -1)
            gp = CLASS_TO_SIBLING.get(int(p), -2)
            if gt == gp and int(t) != int(p):
                sibling_wrong += 1
        return sibling_wrong / float(mask.sum())

    out["sibling_error_among_errors"] = rate(wrong)
    # Conditional: P(pred in same group, pred != true | true in group)
    for name, members in SIBLING_GROUPS.items():
        member_set = set(members)
        in_group = np.array([int(t) in member_set for t in y_true])
        if in_group.sum() == 0:
            out[f"sibling_error_{name}"] = float("nan")
            continue
        sw = 0
        for t, p in zip(y_true[in_group], y_pred[in_group]):
            if int(t) != int(p) and int(p) in member_set:
                sw += 1
        out[f"sibling_error_{name}"] = sw / float(in_group.sum())
    return out


def many_med_few_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    buckets: dict[str, Sequence[int]],
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out: dict[str, float] = {}
    for name in ("many", "medium", "few"):
        ids = set(buckets.get(name, []))
        if not ids:
            out[f"acc_{name}"] = float("nan")
            continue
        mask = np.array([int(t) in ids for t in y_true])
        if mask.sum() == 0:
            out[f"acc_{name}"] = float("nan")
        else:
            out[f"acc_{name}"] = float(accuracy_score(y_true[mask], y_pred[mask]))
    return out


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    frequency_buckets: dict[str, Sequence[int]] | None = None,
    class_names: Sequence[str] | None = None,
) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    names = list(class_names) if class_names is not None else TEA_CLASSES
    n_classes = len(names)

    per_class_f1 = f1_score(
        y_true, y_pred, labels=list(range(n_classes)), average=None, zero_division=0
    )
    per_class_recall = []
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    for i in range(n_classes):
        support = cm[i].sum()
        per_class_recall.append(float(cm[i, i] / support) if support > 0 else float("nan"))

    metrics: dict = {
        "acc": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=list(range(n_classes)), average="macro", zero_division=0)
        ),
        "per_class_f1": {names[i]: float(per_class_f1[i]) for i in range(n_classes)},
        "per_class_recall": {names[i]: per_class_recall[i] for i in range(n_classes)},
        "confusion_matrix": cm.tolist(),
    }
    metrics.update(sibling_error_rate(y_true, y_pred))
    if frequency_buckets is not None:
        metrics.update(many_med_few_accuracy(y_true, y_pred, frequency_buckets))
    return metrics


def format_metrics_line(metrics: dict) -> str:
    parts = [
        f"acc={metrics.get('acc', float('nan')):.4f}",
        f"macro_f1={metrics.get('macro_f1', float('nan')):.4f}",
        f"bal_acc={metrics.get('balanced_acc', float('nan')):.4f}",
    ]
    for k in ("acc_many", "acc_medium", "acc_few"):
        if k in metrics and metrics[k] == metrics[k]:  # not NaN
            parts.append(f"{k}={metrics[k]:.4f}")
    if "sibling_error_among_errors" in metrics:
        v = metrics["sibling_error_among_errors"]
        if v == v:
            parts.append(f"sib_err={v:.4f}")
    return " ".join(parts)
