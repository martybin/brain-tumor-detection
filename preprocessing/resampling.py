"""Spacing resampling. Labels use nearest-neighbor. Images use linear."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom

from preprocessing.io_utils import spacing_from_affine


def resample_array(
    data: np.ndarray,
    affine: np.ndarray,
    target_spacing: tuple[float, float, float],
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    current = np.array(spacing_from_affine(affine), dtype=np.float64)
    target = np.array(target_spacing, dtype=np.float64)
    if np.any(current <= 0) or np.any(target <= 0):
        raise ValueError(f"Spacing must be positive, got {current} -> {target}")
    if np.allclose(current, target, rtol=1e-3, atol=1e-3):
        return data, affine
    factor = current / target
    if data.dtype.kind in "iu" or order == 0:
        resampled = zoom(data, factor, order=0, prefilter=False)
    else:
        resampled = zoom(data.astype(np.float32), factor, order=order, prefilter=True)
    new_affine = np.array(affine, dtype=np.float64, copy=True)
    scale = target / current
    new_affine[:3, :3] = affine[:3, :3] @ np.diag(scale)
    return resampled, new_affine


def match_shape(data: np.ndarray, shape: tuple[int, int, int], order: int) -> np.ndarray:
    """Crop or pad so stacked modalities share one grid after resampling."""
    out = np.zeros(shape, dtype=data.dtype)
    slices_src = []
    slices_dst = []
    for src, dst in zip(data.shape, shape):
        n = min(src, dst)
        slices_src.append(slice(0, n))
        slices_dst.append(slice(0, n))
    out[tuple(slices_dst)] = data[tuple(slices_src)]
    if order > 0 and data.shape != shape:
        return out.astype(np.float32)
    return out
