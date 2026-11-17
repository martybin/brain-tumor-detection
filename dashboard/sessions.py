"""Session files live under outputs/ and never under datasets/raw."""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path

import numpy as np

from preprocessing.io_utils import load_nifti
from training.config import project_root

SESSION_ROOT = project_root() / "outputs" / "dashboard_sessions"
ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def new_session() -> Path:
    SESSION_ROOT.mkdir(parents=True, exist_ok=True)
    folder = SESSION_ROOT / uuid.uuid4().hex
    (folder / "original").mkdir(parents=True)
    return folder


def read_meta(session_id: str) -> tuple[Path, dict]:
    if not ID_PATTERN.match(session_id):
        raise FileNotFoundError(session_id)
    folder = SESSION_ROOT / session_id
    path = folder / "meta.json"
    if not path.is_file():
        raise FileNotFoundError(session_id)
    return folder, json.loads(path.read_text(encoding="utf-8"))


def write_meta(folder: Path, meta: dict) -> None:
    (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def study_dir(folder: Path) -> Path:
    path = folder / "original" / folder.name
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_nifti_upload(folder: Path, modality: str, filename: str, payload: bytes) -> None:
    lower = filename.lower()
    if not (lower.endswith(".nii") or lower.endswith(".nii.gz")):
        raise ValueError(f"{modality} باید NIfTI باشد، نه {filename}.")
    suffix = ".nii.gz" if lower.endswith(".nii.gz") else ".nii"
    destination = study_dir(folder) / f"{folder.name}_{modality}{suffix}"
    destination.write_bytes(payload)
    data, _ = load_nifti(destination)
    if data.ndim != 3:
        destination.unlink(missing_ok=True)
        raise ValueError(f"{modality} سه‌بعدی نیست.")


def save_flat_image(folder: Path, filename: str, payload: bytes) -> None:
    lower = filename.lower()
    if not lower.endswith((".png", ".jpg", ".jpeg", ".webp")):
        raise ValueError("عکس دوبعدی باید PNG یا JPEG باشد.")
    (folder / "original" / "flat_image").write_bytes(payload)


def volume_shape(folder: Path, meta: dict) -> list[int] | None:
    modalities = meta.get("modalities") or {}
    if not modalities:
        return None
    modality = next(iter(modalities))
    data, _ = load_nifti(folder / "original" / folder.name / modalities[modality])
    return [int(value) for value in data.shape]


def load_original(folder: Path, meta: dict, modality: str, space: str = "original") -> np.ndarray:
    if space == "aligned":
        path = folder / "aligned" / meta["id"] / f"{modality}.nii.gz"
    else:
        name = (meta.get("modalities") or {}).get(modality)
        if not name:
            raise FileNotFoundError(modality)
        path = folder / "original" / folder.name / name
    data, _ = load_nifti(path)
    return np.asarray(data, dtype=np.float32)


def checkpoint_path() -> Path | None:
    chosen = os.environ.get("CHECKPOINT")
    if chosen and Path(chosen).is_file():
        return Path(chosen)
    default = project_root() / "checkpoints" / "BraTS" / "hybrid" / "best.pt"
    if default.is_file():
        return default
    root = project_root() / "checkpoints"
    if not root.exists():
        return None
    found = sorted(root.glob("**/best.pt"))
    return found[-1] if found else None


def write_sample(folder: Path) -> dict:
    import nibabel as nib

    shape = (72, 72, 48)
    grid = np.indices(shape)
    center = np.array(shape) / 2
    radius = ((grid - center[:, None, None, None]) ** 2 / np.array([28, 30, 18])[:, None, None, None] ** 2).sum(0)
    brain = np.clip(1.1 - radius, 0, 1).astype(np.float32)
    ventricles = (((grid[0] - center[0]) / 6) ** 2 + ((grid[1] - center[1]) / 8) ** 2 + ((grid[2] - center[2]) / 5) ** 2) < 1
    brain[ventricles] *= 0.35
    modalities = {}
    for modality, gain in (("t1", 70), ("t1ce", 95), ("t2", 80), ("flair", 88)):
        volume = brain * gain
        path = study_dir(folder) / f"{folder.name}_{modality}.nii.gz"
        nib.save(nib.Nifti1Image(volume.astype(np.float32), np.eye(4)), str(path))
        modalities[modality] = path.name
    meta = {
        "id": folder.name,
        "modalities": modalities,
        "image": None,
        "result": None,
        "sample": True,
        "shape": list(shape),
    }
    write_meta(folder, meta)
    return meta
