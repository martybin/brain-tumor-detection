"""NIfTI helpers. Writes under datasets/raw are refused."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from preprocessing.errors import RawDataProtectionError

CANONICAL_MODALITIES = ("t1", "t1ce", "t2", "flair")


def assert_not_raw(path: Path, raw_root: Path) -> None:
    target = path.resolve()
    raw = raw_root.resolve()
    if target == raw or raw in target.parents:
        raise RawDataProtectionError(
            f"Refusing to modify raw medical data: {target}"
        )


def load_nifti(path: Path, expected_ndim: tuple[int, ...] = (3,)) -> tuple[np.ndarray, np.ndarray]:
    import nibabel as nib

    image = nib.load(str(path))
    data = np.asanyarray(image.dataobj)
    if data.ndim not in expected_ndim:
        raise ValueError(
            f"Expected ndim in {expected_ndim}, got shape {data.shape} at {path}"
        )
    affine = np.asarray(image.affine, dtype=np.float64)
    return data, affine


def read_dicom_series(files: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Read one series to a 3D array. The source files are not modified."""
    import SimpleITK as sitk

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(files)
    image = reader.Execute()
    array_zyx = sitk.GetArrayFromImage(image).astype(np.float32)
    data = np.transpose(array_zyx, (2, 1, 0))
    spacing = np.array(image.GetSpacing(), dtype=np.float64)
    direction = np.array(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = direction @ np.diag(spacing)
    affine[:3, 3] = np.array(image.GetOrigin(), dtype=np.float64)
    return data, affine


def validate_array(data: np.ndarray, name: str) -> None:
    if data.ndim != 3:
        raise ValueError(f"{name} must be 3D, got shape {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError(f"{name} contains non-finite values")
    if int(np.count_nonzero(data)) == 0:
        raise ValueError(f"{name} is empty")


def save_nifti(path: Path, data: np.ndarray, affine: np.ndarray, raw_root: Path) -> None:
    import nibabel as nib

    assert_not_raw(path, raw_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(np.asarray(data), np.asarray(affine, dtype=np.float64))
    nib.save(image, str(path))


def spacing_from_affine(affine: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(v) for v in np.sqrt((affine[:3, :3] ** 2).sum(axis=0)))


def orientation_codes(affine: np.ndarray) -> str:
    import nibabel as nib

    return "".join(nib.aff2axcodes(affine))
