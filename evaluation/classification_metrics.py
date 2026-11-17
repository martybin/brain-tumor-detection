"""Classification metrics. Rows with a missing label are counted and excluded."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from preprocessing.errors import MissingLabelError


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    labels = np.asarray(y_true)
    probs = np.asarray(y_prob, dtype=float)
    valid = labels >= 0
    n_missing = int((~valid).sum())
    if not np.any(valid):
        raise MissingLabelError("No classification labels are available.")
    y = labels[valid].astype(int)
    p = probs[valid]
    if p.ndim == 1:
        pred = (p >= 0.5).astype(int)
        score = p
        average = "binary"
    else:
        pred = p.argmax(axis=1)
        score = p
        average = "macro"
    result: dict = {
        "n": int(len(y)),
        "n_missing": n_missing,
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, average=average, zero_division=0)),
        "recall": float(recall_score(y, pred, average=average, zero_division=0)),
        "f1": float(f1_score(y, pred, average=average, zero_division=0)),
        "confusion_matrix": confusion_matrix(
            y,
            pred,
            labels=list(range(p.shape[1])) if p.ndim == 2 else [0, 1],
        ).tolist(),
    }
    if len(np.unique(y)) < 2:
        result["roc_auc"] = None
        result["pr_auc"] = None
        result["auc_reason"] = "ROC-AUC and PR-AUC are undefined when only one class is present."
        return result
    if p.ndim == 1:
        result["roc_auc"] = float(roc_auc_score(y, score))
        result["pr_auc"] = float(average_precision_score(y, score))
    else:
        result["roc_auc"] = float(roc_auc_score(y, score, multi_class="ovr", average="macro"))
        result["pr_auc"] = float(_macro_average_precision(y, score))
    return result


def _macro_average_precision(y: np.ndarray, probs: np.ndarray) -> float:
    classes = np.unique(y)
    scores = []
    for index, class_id in enumerate(range(probs.shape[1])):
        if class_id not in classes:
            continue
        scores.append(average_precision_score((y == class_id).astype(int), probs[:, index]))
    return float(np.mean(scores))
