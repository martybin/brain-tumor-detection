"""Uncertainty and calibration. Inputs are probabilities, not invented labels."""

from __future__ import annotations

import numpy as np


def predictive_entropy(probs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = np.clip(np.asarray(probs, dtype=float), eps, 1.0)
    values = values / values.sum(axis=-1, keepdims=True)
    return -np.sum(values * np.log(values), axis=-1)


def bernoulli_entropy(probs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=float), eps, 1.0 - eps)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def brier_score(probs: np.ndarray, targets: np.ndarray) -> float:
    p = np.asarray(probs, dtype=float)
    y = np.asarray(targets, dtype=float)
    if p.shape != y.shape:
        raise ValueError(f"Brier inputs differ: {p.shape} vs {y.shape}")
    return float(np.mean(np.sum((p - y) ** 2, axis=-1)))


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> dict:
    conf = np.asarray(confidence, dtype=float).reshape(-1)
    hit = np.asarray(correct, dtype=float).reshape(-1)
    if conf.shape != hit.shape:
        raise ValueError("Confidence and correctness must have the same shape.")
    if len(conf) == 0:
        raise ValueError("Calibration received no samples.")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    error = 0.0
    bins = []
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        if index == n_bins - 1:
            selected = (conf >= lower) & (conf <= upper)
        else:
            selected = (conf >= lower) & (conf < upper)
        if not np.any(selected):
            continue
        accuracy = float(hit[selected].mean())
        mean_conf = float(conf[selected].mean())
        fraction = float(selected.mean())
        error += fraction * abs(accuracy - mean_conf)
        bins.append({"lower": float(lower), "upper": float(upper), "accuracy": accuracy, "confidence": mean_conf, "n": int(selected.sum())})
    return {"ece": float(error), "bins": bins}
