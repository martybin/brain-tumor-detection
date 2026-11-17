"""PNG slices for the viewer. Display only; files on disk are not rewritten."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image


def slice_png(
    volume: np.ndarray,
    axis: str,
    index: int,
    overlay: np.ndarray | None = None,
    low: float = 1.0,
    high: float = 99.0,
) -> bytes:
    plane = _plane(volume, axis, index)
    image = _window(plane, low, high)
    rgb = np.stack([image, image, image], axis=-1)
    if overlay is not None:
        mask = _plane(overlay.astype(np.float32), axis, index) > 0
        color = np.array([214, 92, 68], dtype=np.float32)
        rgb[mask] = rgb[mask] * 0.45 + color * 0.55
    buffer = io.BytesIO()
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB").save(buffer, format="PNG")
    return buffer.getvalue()


def flat_png(payload: bytes) -> bytes:
    image = Image.open(io.BytesIO(payload)).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _plane(volume: np.ndarray, axis: str, index: int) -> np.ndarray:
    if volume.ndim != 3:
        raise ValueError("Expected a 3D volume.")
    axis_index = {"sagittal": 0, "coronal": 1, "axial": 2}[axis]
    length = volume.shape[axis_index]
    if index < 0 or index >= length:
        raise IndexError(f"Slice {index} is outside 0..{length - 1}.")
    plane = np.take(volume, index, axis=axis_index)
    return np.flipud(np.asarray(plane).T)


def _window(plane: np.ndarray, low: float, high: float) -> np.ndarray:
    finite = plane[np.isfinite(plane)]
    if finite.size == 0:
        return np.zeros(plane.shape, dtype=np.float32)
    lo, hi = np.percentile(finite, [low, high])
    if hi <= lo:
        hi = lo + 1.0
    scaled = np.clip((plane - lo) / (hi - lo), 0, 1)
    return (scaled * 255).astype(np.float32)
