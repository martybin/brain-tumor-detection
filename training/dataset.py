"""Patch sampling from prepared NIfTI. Raw images are never opened here."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from preprocessing.augmentation import augment_pair
from preprocessing.io_utils import CANONICAL_MODALITIES, load_nifti


class PatchDataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        patient_ids: list[str],
        patch_size: tuple[int, int, int],
        samples_per_volume: int,
        pos_fraction: float,
        seed: int,
        augment: bool,
    ) -> None:
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        wanted = set(patient_ids)
        self.cases = [case for case in payload["cases"] if case["patient_id"] in wanted]
        missing = wanted.difference(case["patient_id"] for case in self.cases)
        if missing:
            raise FileNotFoundError(f"Split patients missing from the manifest: {sorted(missing)}")
        if not self.cases:
            raise FileNotFoundError("The split has no prepared cases.")
        for case in self.cases:
            if case.get("segmentation_status") != "confirmed" or "regions" not in case:
                raise FileNotFoundError(
                    f"{case['patient_id']} has no confirmed region labels. "
                    "Segmentation training will not invent them."
                )
        self.patch_size = tuple(int(v) for v in patch_size)
        self.samples_per_volume = int(samples_per_volume)
        self.pos_fraction = float(pos_fraction)
        self.seed = int(seed)
        self.augment = augment
        self.region_order = self.cases[0]["region_order"]

    def __len__(self) -> int:
        return len(self.cases) * self.samples_per_volume

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        case = self.cases[index // self.samples_per_volume]
        rng = np.random.default_rng(self.seed + index)
        image = _stack_modalities(case)
        regions, _ = load_nifti(Path(case["regions"]), expected_ndim=(4,))
        regions = np.moveaxis(np.asarray(regions), -1, 0).astype(np.float32)
        image, regions = _crop_pair(image, regions, self.patch_size, rng, self.pos_fraction)
        if self.augment:
            image, regions = augment_pair(image, regions, rng)
        return {
            "image": torch.from_numpy(image),
            "target": torch.from_numpy(regions),
            "patient_id": case["patient_id"],
        }


def _stack_modalities(case: dict) -> np.ndarray:
    volumes = []
    for modality in CANONICAL_MODALITIES:
        path = case["modalities"].get(modality)
        if not path:
            raise FileNotFoundError(f"{case['patient_id']} is missing prepared modality {modality}.")
        data, _ = load_nifti(Path(path))
        volumes.append(np.asarray(data, dtype=np.float32))
    return np.stack(volumes, axis=0)


def _crop_pair(
    image: np.ndarray,
    regions: np.ndarray,
    patch_size: tuple[int, int, int],
    rng: np.random.Generator,
    pos_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    spatial = image.shape[1:]
    starts = []
    foreground = np.argwhere(regions.reshape(regions.shape[0], -1).any(axis=0))
    use_foreground = regions.any() and float(rng.random()) < pos_fraction and len(foreground)
    if use_foreground:
        flat = int(foreground[int(rng.integers(len(foreground)))])
        center = np.unravel_index(flat, spatial)
    else:
        center = None
    for axis, (length, patch) in enumerate(zip(spatial, patch_size)):
        if length <= patch:
            starts.append(0)
            continue
        if center is None:
            starts.append(int(rng.integers(0, length - patch + 1)))
        else:
            start = int(center[axis] - patch // 2)
            starts.append(min(max(start, 0), length - patch))
    image_crop = _take(image, starts, patch_size, image.shape[1:])
    region_crop = _take(regions, starts, patch_size, regions.shape[1:])
    return image_crop, region_crop


def _take(array: np.ndarray, starts: list[int], patch_size: tuple[int, int, int], spatial: tuple[int, ...]) -> np.ndarray:
    slices = [slice(None)]
    pads = []
    for start, patch, length in zip(starts, patch_size, spatial):
        if length >= patch:
            slices.append(slice(start, start + patch))
            pads.append((0, 0))
        else:
            slices.append(slice(0, length))
            pads.append((0, patch - length))
    crop = array[tuple(slices)]
    if any(pad != (0, 0) for pad in pads):
        crop = np.pad(crop, [(0, 0), *pads])
    return np.ascontiguousarray(crop)
