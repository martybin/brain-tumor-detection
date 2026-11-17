"""Per-modality intensity normalization. Background zeros stay zero."""

from __future__ import annotations

import numpy as np


def nonzero_zscore(volume: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    data = volume.astype(np.float32, copy=False)
    mask = data != 0
    if not np.any(mask):
        raise ValueError("Cannot normalize an empty volume.")
    values = data[mask]
    mean = float(values.mean())
    std = float(values.std())
    out = np.zeros_like(data, dtype=np.float32)
    out[mask] = (values - mean) / (std + eps)
    return out


def percentile_clip(volume: np.ndarray, low: float = 0.5, high: float = 99.5) -> np.ndarray:
    data = volume.astype(np.float32, copy=False)
    mask = data != 0
    if not np.any(mask):
        raise ValueError("Cannot clip an empty volume.")
    lo, hi = np.percentile(data[mask], [low, high])
    clipped = np.clip(data, lo, hi)
    clipped[~mask] = 0
    return clipped.astype(np.float32)
