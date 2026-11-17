"""MRI volume checks. Failures are raised, not dropped."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from preprocessing.io_utils import load_nifti, orientation_codes, spacing_from_affine


def header_report(path: Path) -> dict:
    import nibabel as nib

    image = nib.load(str(path))
    return {
        "path": str(path),
        "shape": [int(v) for v in image.shape],
        "spacing": [float(v) for v in image.header.get_zooms()[:3]],
        "orientation": "".join(nib.aff2axcodes(image.affine)),
        "dtype": str(image.get_data_dtype()),
    }


def inspect_volume(path: Path) -> dict:
    data, affine = load_nifti(path)
    finite = np.isfinite(data)
    return {
        "path": str(path),
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "spacing": spacing_from_affine(affine),
        "orientation": orientation_codes(affine),
        "min": float(np.min(data[finite])) if finite.any() else None,
        "max": float(np.max(data[finite])) if finite.any() else None,
        "nonfinite": int((~finite).sum()),
        "nonzero": int(np.count_nonzero(data)),
    }


def validate_volume(path: Path) -> dict:
    report = inspect_volume(path)
    if report["nonfinite"]:
        raise ValueError(f"Non-finite voxels in {path}: {report['nonfinite']}")
    if report["nonzero"] == 0:
        raise ValueError(f"Volume is empty: {path}")
    return report


def label_values(path: Path) -> list[int]:
    data, _ = load_nifti(path)
    values = np.unique(data.astype(np.int32))
    return [int(v) for v in values]
