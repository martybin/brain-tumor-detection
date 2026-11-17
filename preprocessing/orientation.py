"""Reorient volumes to RAS without writing the source file."""

from __future__ import annotations

import numpy as np


def to_ras(data: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import nibabel as nib

    image = nib.Nifti1Image(np.asanyarray(data), np.asarray(affine, dtype=np.float64))
    canonical = nib.as_closest_canonical(image)
    return np.asanyarray(canonical.dataobj), np.asarray(canonical.affine, dtype=np.float64)
