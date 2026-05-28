"""Common evaluation metrics — thin wrappers around sklearn for consistency."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute the standard suite of binary classification metrics."""
    y_pred = (y_prob >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold": threshold,
    }


def best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    """Threshold that maximises F1; returns (threshold, f1)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # Avoid divide-by-zero
    denom = precision + recall
    f1 = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    best_idx = int(np.argmax(f1[:-1]))  # last point has no threshold
    return float(thresholds[best_idx]), float(f1[best_idx])
