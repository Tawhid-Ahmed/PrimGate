"""
Primitive hijacking analysis (RQ1 / Phase 2).

Pipeline:
1. Extract penultimate features from a trained classifier.
2. Fit K shared visual primitives (k-means centroids on train features).
3. Soft occupancy alpha via temperature-scaled similarity to centroids.
4. Score head vs tail occupancy gap (hijacking) and correlate with errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import normalize


@dataclass
class HijackingResult:
    K: int
    temperature: float
    hijack_l2: float
    hijack_l1: float
    head_mean_occupancy: np.ndarray
    tail_mean_occupancy: np.ndarray
    class_mean_occupancy: np.ndarray
    head_dominance: np.ndarray  # per-primitive: head_mass / (head+tail)
    primitive_entropy_head: float
    primitive_entropy_tail: float
    per_class_macro_f1_proxy: dict[int, float]
    corr_error_vs_head_overlap: float  # sample-level
    corr_error_vs_head_overlap_class: float
    labels: np.ndarray
    preds: np.ndarray
    alphas: np.ndarray


def _softmax_neg_dist(dist: np.ndarray, temperature: float) -> np.ndarray:
    # dist: (N, K)
    z = -dist / max(temperature, 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)


def _entropy(p: np.ndarray, axis: int = -1) -> float:
    p = np.clip(p, 1e-12, 1.0)
    h = -(p * np.log(p)).sum(axis=axis)
    return float(np.mean(h))


def fit_primitives(
    train_feats: np.ndarray,
    K: int = 16,
    seed: int = 42,
) -> MiniBatchKMeans:
    x = normalize(train_feats, norm="l2")
    km = MiniBatchKMeans(
        n_clusters=K,
        random_state=seed,
        batch_size=min(2048, max(256, len(x))),
        n_init=10,
        max_iter=200,
    )
    km.fit(x)
    return km


def soft_occupancy(
    feats: np.ndarray,
    centers: np.ndarray,
    temperature: float = 0.2,
) -> np.ndarray:
    x = normalize(feats, norm="l2")
    c = normalize(centers, norm="l2")
    # cosine distance = 1 - cosine similarity
    sim = x @ c.T
    dist = 1.0 - sim
    return _softmax_neg_dist(dist, temperature=temperature)


def class_mean_occupancy(alphas: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((num_classes, alphas.shape[1]), dtype=np.float64)
    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            out[c] = alphas[mask].mean(axis=0)
    return out


def compute_hijacking_score(
    alphas: np.ndarray,
    labels: np.ndarray,
    head_ids: Sequence[int],
    few_ids: Sequence[int],
    num_classes: int = 9,
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    head_mask = np.isin(labels, list(head_ids))
    tail_mask = np.isin(labels, list(few_ids))
    if not head_mask.any() or not tail_mask.any():
        raise ValueError("Need both head and tail samples for hijacking score")

    head_mean = alphas[head_mask].mean(axis=0)
    tail_mean = alphas[tail_mask].mean(axis=0)
    gap = head_mean - tail_mean
    hijack_l2 = float(np.sum(gap**2))
    hijack_l1 = float(np.sum(np.abs(gap)))
    class_means = class_mean_occupancy(alphas, labels, num_classes)
    return hijack_l2, hijack_l1, head_mean, tail_mean, class_means


def head_dominance_per_primitive(
    alphas: np.ndarray,
    labels: np.ndarray,
    head_ids: Sequence[int],
    few_ids: Sequence[int],
) -> np.ndarray:
    """
    Frequency-normalized dominance using mean occupancy (not raw mass),
    so head class size does not automatically dominate every primitive.
    """
    head_mask = np.isin(labels, list(head_ids))
    tail_mask = np.isin(labels, list(few_ids))
    head_mean = alphas[head_mask].mean(axis=0)
    tail_mean = alphas[tail_mask].mean(axis=0)
    return head_mean / np.clip(head_mean + tail_mean, 1e-12, None)


def correlate_errors_with_head_overlap(
    alphas: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
    head_dominance: np.ndarray,
) -> tuple[float, dict[int, float], float]:
    """
    Returns:
      - sample-level Pearson corr(is_error, occupancy on head-dominated primitives)
      - per-class error rates
      - class-level Pearson corr(error_rate, mean head-overlap) when variance allows
    """
    headish = head_dominance >= 0.6
    if not np.any(headish):
        # top-quartile head-dominated primitives
        thr = float(np.quantile(head_dominance, 0.75))
        headish = head_dominance >= thr

    sample_overlap = alphas[:, headish].sum(axis=1)
    is_error = (preds != labels).astype(np.float64)
    if float(np.std(sample_overlap)) < 1e-8 or float(np.std(is_error)) < 1e-8:
        sample_corr = float("nan")
    else:
        sample_corr = float(np.corrcoef(sample_overlap, is_error)[0, 1])

    classes = sorted(set(labels.tolist()))
    err_rates = []
    overlaps = []
    per_class: dict[int, float] = {}
    for c in classes:
        mask = labels == c
        if not mask.any():
            continue
        err = float((preds[mask] != labels[mask]).mean())
        overlap = float(sample_overlap[mask].mean())
        err_rates.append(err)
        overlaps.append(overlap)
        per_class[int(c)] = err

    if len(err_rates) < 3 or float(np.std(overlaps)) < 1e-8 or float(np.std(err_rates)) < 1e-8:
        class_corr = float("nan")
    else:
        class_corr = float(np.corrcoef(overlaps, err_rates)[0, 1])
    return sample_corr, per_class, class_corr


def analyze_hijacking(
    train_feats: np.ndarray,
    train_labels: np.ndarray,
    eval_feats: np.ndarray,
    eval_labels: np.ndarray,
    eval_preds: np.ndarray,
    head_ids: Sequence[int],
    few_ids: Sequence[int],
    num_classes: int = 9,
    K: int = 16,
    temperature: float = 0.2,
    seed: int = 42,
) -> tuple[HijackingResult, MiniBatchKMeans]:
    km = fit_primitives(train_feats, K=K, seed=seed)
    alphas = soft_occupancy(eval_feats, km.cluster_centers_, temperature=temperature)
    hijack_l2, hijack_l1, head_mean, tail_mean, class_means = compute_hijacking_score(
        alphas, eval_labels, head_ids, few_ids, num_classes=num_classes
    )
    dominance = head_dominance_per_primitive(alphas, eval_labels, head_ids, few_ids)
    sample_corr, per_class_err, class_corr = correlate_errors_with_head_overlap(
        alphas, eval_labels, eval_preds, dominance
    )

    head_mask = np.isin(eval_labels, list(head_ids))
    tail_mask = np.isin(eval_labels, list(few_ids))

    result = HijackingResult(
        K=K,
        temperature=temperature,
        hijack_l2=hijack_l2,
        hijack_l1=hijack_l1,
        head_mean_occupancy=head_mean,
        tail_mean_occupancy=tail_mean,
        class_mean_occupancy=class_means,
        head_dominance=dominance,
        primitive_entropy_head=_entropy(alphas[head_mask]),
        primitive_entropy_tail=_entropy(alphas[tail_mask]),
        per_class_macro_f1_proxy=per_class_err,
        corr_error_vs_head_overlap=sample_corr,
        corr_error_vs_head_overlap_class=class_corr,
        labels=eval_labels,
        preds=eval_preds,
        alphas=alphas,
    )
    return result, km


def sibling_confusion_rates(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sibling_groups: dict[str, list[int]],
) -> dict[str, float]:
    out = {}
    for name, members in sibling_groups.items():
        s = set(members)
        mask = np.array([int(t) in s for t in y_true])
        if not mask.any():
            out[name] = float("nan")
            continue
        sw = 0
        for t, p in zip(y_true[mask], y_pred[mask]):
            if int(t) != int(p) and int(p) in s:
                sw += 1
        out[name] = sw / float(mask.sum())
    return out


def summary_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 9) -> dict:
    return {
        "acc": float((y_true == y_pred).mean()),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=list(range(num_classes)),
                average="macro",
                zero_division=0,
            )
        ),
        "confusion": confusion_matrix(y_true, y_pred, labels=list(range(num_classes))).tolist(),
    }
