"""Spatial flips and rotations applied to a patch and its mask together."""

from __future__ import annotations

import numpy as np


def augment_pair(
    image: np.ndarray,
    label: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    for axis in (1, 2, 3):
        if rng.random() < 0.5:
            image = np.flip(image, axis=axis)
            label = np.flip(label, axis=axis)
    turns = int(rng.integers(0, 4))
    if turns:
        image = np.rot90(image, turns, axes=(2, 3))
        label = np.rot90(label, turns, axes=(2, 3))
    return np.ascontiguousarray(image), np.ascontiguousarray(label)
