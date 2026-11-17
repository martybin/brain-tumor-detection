"""Sliding-window inference. The input array is not written back to disk."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def patch_weight(patch_size: tuple[int, int, int]) -> np.ndarray:
    axes = []
    for length in patch_size:
        if length <= 1:
            axes.append(np.ones(length, dtype=np.float32))
        else:
            window = np.hanning(length).astype(np.float32)
            axes.append(np.clip(window, 0.1, None))
    weight = axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    return weight.astype(np.float32)


def sliding_window_probabilities(
    model: nn.Module,
    volume: np.ndarray,
    patch_size: tuple[int, int, int],
    overlap: float,
    device: torch.device,
) -> np.ndarray:
    """Return sigmoid probabilities shaped (channels, D, H, W)."""
    if volume.ndim != 4:
        raise ValueError(f"Expected (modalities, D, H, W), got {volume.shape}")
    model.eval()
    spatial = volume.shape[1:]
    weight = patch_weight(patch_size)
    stride = tuple(max(1, int(length * (1.0 - overlap))) for length in patch_size)
    pads = [max(0, patch - length) for patch, length in zip(patch_size, spatial)]
    padded = np.pad(volume, [(0, 0), *[(0, pad) for pad in pads]])
    channels = int(getattr(model, "seg_channels"))
    accumulator = np.zeros((channels, *padded.shape[1:]), dtype=np.float32)
    normalizer = np.zeros(padded.shape[1:], dtype=np.float32)
    starts = [_axis_starts(padded.shape[axis + 1], patch, step) for axis, (patch, step) in enumerate(zip(patch_size, stride))]
    from training.amp import autocast

    with torch.no_grad():
        for z in starts[0]:
            for y in starts[1]:
                for x in starts[2]:
                    crop = padded[:, z : z + patch_size[0], y : y + patch_size[1], x : x + patch_size[2]]
                    tensor = torch.from_numpy(np.ascontiguousarray(crop))[None].to(device)
                    with autocast(device, enabled=True):
                        logits = model(tensor)["seg"][0]
                    probs = torch.sigmoid(logits).float().cpu().numpy()
                    accumulator[:, z : z + patch_size[0], y : y + patch_size[1], x : x + patch_size[2]] += probs * weight
                    normalizer[z : z + patch_size[0], y : y + patch_size[1], x : x + patch_size[2]] += weight
    averaged = accumulator / np.clip(normalizer, 1e-6, None)
    return averaged[:, : spatial[0], : spatial[1], : spatial[2]]


def _axis_starts(length: int, patch: int, stride: int) -> list[int]:
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    if starts[-1] != length - patch:
        starts.append(length - patch)
    return starts
