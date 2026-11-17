"""Patient-level statistics. These functions do not rank unpublished models."""

from __future__ import annotations

import numpy as np

from preprocessing.errors import SplitLeakageError
from preprocessing.splitting import assert_no_overlap


def bootstrap_mean_ci(values: np.ndarray, n_resamples: int = 1000, seed: int = 0, alpha: float = 0.05) -> dict[str, float]:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        raise ValueError("No finite values to summarize.")
    rng = np.random.default_rng(seed)
    means = np.array(
        [data[rng.integers(0, len(data), len(data))].mean() for _ in range(n_resamples)],
        dtype=float,
    )
    low, high = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(data.mean()), "ci_low": float(low), "ci_high": float(high), "n": int(len(data))}


def paired_wilcoxon(left: np.ndarray, right: np.ndarray) -> dict[str, float | str | int]:
    from scipy.stats import wilcoxon

    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Paired scores must have the same shape.")
    valid = np.isfinite(a) & np.isfinite(b)
    if not np.any(valid):
        raise ValueError("No finite paired scores.")
    n = int(valid.sum())
    if np.allclose(a[valid], b[valid]):
        return {"statistic": 0.0, "pvalue": 1.0, "n": n, "note": "Scores are identical."}
    statistic, pvalue = wilcoxon(a[valid], b[valid])
    return {"statistic": float(statistic), "pvalue": float(pvalue), "n": n}


def assert_external_cohort(train_ids: list[str], external_ids: list[str]) -> None:
    overlap = sorted(set(train_ids).intersection(external_ids))
    if overlap:
        raise SplitLeakageError(f"External patients also appear in training: {overlap[:8]}")
    assert_no_overlap({"train": train_ids, "external": external_ids})
