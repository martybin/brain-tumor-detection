#!/usr/bin/env python3
"""Split Medical Segmentation Decathlon Task01 into per-modality NIfTI files.

The official BraTS 2021 package is a 142 GB Aspera download from TCIA.
This importer uses the public Decathlon brain-tumour archive instead and
records that source in the schema. Label names are read from dataset.json.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import numpy as np
import yaml

import bootstrap  # noqa: F401

from training.config import project_root

ARCHIVE = project_root() / "datasets" / "raw" / "_archives" / "Task01_BrainTumour.tar"
EXTRACTED = project_root() / "datasets" / "raw" / "_archives" / "Task01_BrainTumour"
OUTPUT = project_root() / "datasets" / "raw" / "MSD-BrainTumour"
SCHEMA_PATH = project_root() / "configs" / "datasets" / "msd_brain.yml"

MODALITY_NAMES = {
    "flair": "flair",
    "t1w": "t1",
    "t1": "t1",
    "t1gd": "t1ce",
    "t1ce": "t1ce",
    "t2w": "t2",
    "t2": "t2",
}
REGION_MEMBERSHIP = {
    "edema": ("WT",),
    "oedema": ("WT",),
    "non-enhancing tumor": ("WT", "TC"),
    "non-enhancing tumour": ("WT", "TC"),
    "enhancing tumor": ("WT", "TC", "ET"),
    "enhancing tumour": ("WT", "TC", "ET"),
}


def main() -> None:
    extracted = _extract()
    description = json.loads((extracted / "dataset.json").read_text(encoding="utf-8"))
    modality_order = _modality_order(description)
    label_values, regions = _labels(description)
    training = description.get("training") or []
    if not training:
        raise RuntimeError("dataset.json has no training cases.")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(training, start=1):
        image_path = extracted / case["image"]
        label_path = extracted / case["label"]
        patient_id = image_path.name.replace(".nii.gz", "").replace(".nii", "")
        destination = OUTPUT / patient_id
        if (destination / f"{patient_id}_seg.nii.gz").is_file():
            continue
        _split_case(image_path, label_path, destination, patient_id, modality_order)
        if index % 25 == 0 or index == len(training):
            print(f"converted {index}/{len(training)}")
    _write_schema(label_values, regions, len(training))
    print(f"cases: {len(training)}")
    print(f"schema: {SCHEMA_PATH}")


def _extract() -> Path:
    if (EXTRACTED / "dataset.json").is_file():
        return EXTRACTED
    if not ARCHIVE.is_file():
        raise FileNotFoundError(ARCHIVE)
    EXTRACTED.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE) as archive:
        archive.extractall(EXTRACTED.parent)
    if not (EXTRACTED / "dataset.json").is_file():
        raise FileNotFoundError("The archive did not contain Task01_BrainTumour/dataset.json.")
    return EXTRACTED


def _modality_order(description: dict) -> list[str]:
    raw = description.get("modality") or {}
    order = []
    for key in sorted(raw, key=lambda item: int(item)):
        name = str(raw[key]).strip().lower()
        canonical = MODALITY_NAMES.get(name)
        if canonical is None:
            raise RuntimeError(f"Unrecognized modality name in dataset.json: {raw[key]}")
        order.append(canonical)
    if order != ["flair", "t1", "t1ce", "t2"]:
        raise RuntimeError(f"Unexpected modality order: {order}")
    return order


def _labels(description: dict) -> tuple[dict[int, str], dict[str, list[int]]]:
    raw = description.get("labels") or {}
    values: dict[int, str] = {}
    regions: dict[str, list[int]] = {"WT": [], "TC": [], "ET": []}
    for key, name in raw.items():
        code = int(key)
        label_name = str(name).strip()
        values[code] = label_name
        if label_name.lower() == "background":
            continue
        membership = REGION_MEMBERSHIP.get(label_name.lower())
        if membership is None:
            raise RuntimeError(f"Unrecognized label name in dataset.json: {label_name}")
        for region in membership:
            regions[region].append(code)
    if not all(regions.values()):
        raise RuntimeError(f"Label map did not fill WT, TC, and ET: {regions}")
    return values, regions


def _split_case(image_path: Path, label_path: Path, destination: Path, patient_id: str, modality_order: list[str]) -> None:
    import nibabel as nib

    image = nib.load(str(image_path))
    data = np.asanyarray(image.dataobj)
    if data.ndim != 4 or data.shape[-1] != len(modality_order):
        raise RuntimeError(f"{image_path.name} has shape {data.shape}, expected channels last.")
    label = nib.load(str(label_path))
    labels = np.asanyarray(label.dataobj)
    if labels.shape != data.shape[:-1]:
        raise RuntimeError(f"Label shape {labels.shape} does not match {data.shape[:-1]}.")
    destination.mkdir(parents=True, exist_ok=True)
    affine = image.affine
    for channel, modality in enumerate(modality_order):
        nib.save(
            nib.Nifti1Image(np.asarray(data[..., channel]), affine),
            str(destination / f"{patient_id}_{modality}.nii.gz"),
        )
    nib.save(nib.Nifti1Image(np.asarray(labels), label.affine), str(destination / f"{patient_id}_seg.nii.gz"))


def _write_schema(values: dict[int, str], regions: dict[str, list[int]], count: int) -> None:
    schema = {
        "name": "MSD-BrainTumour",
        "root": "datasets/raw/MSD-BrainTumour",
        "patient_id_from": "parent_directory",
        "confirmed_layout": True,
        "source": {
            "name": "Medical Segmentation Decathlon Task01_BrainTumour",
            "url": "http://medicaldecathlon.com/",
            "download": "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar",
            "note": "Public BraTS-derived training set. This is not the BraTS 2021 challenge package from TCIA.",
        },
        "modalities": {
            "t1": {"required": True, "patterns": ["*_t1.nii.gz"]},
            "t1ce": {"required": True, "patterns": ["*_t1ce.nii.gz"]},
            "t2": {"required": True, "patterns": ["*_t2.nii.gz"]},
            "flair": {"required": True, "patterns": ["*_flair.nii.gz"]},
        },
        "labels": {
            "segmentation": {
                "confirmed": True,
                "convention": "Integers and names copied from Task01 dataset.json. Regions follow those names: WT includes edema, non-enhancing tumor, and enhancing tumor.",
                "patterns": ["*_seg.nii.gz"],
                "values": {str(key): value for key, value in sorted(values.items())},
                "regions": regions,
            },
            "classification": {"confirmed": False},
            "molecular": {"confirmed": False},
            "survival": {"confirmed": False},
        },
        "tasks": {
            "segmentation": True,
            "classification": False,
            "molecular": False,
            "survival": False,
        },
        "notes": f"{count} training cases were split from the Decathlon 4-channel volumes. Validation images in the archive have no labels and were not imported.",
    }
    SCHEMA_PATH.write_text(yaml.safe_dump(schema, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
