"""Preprocess confirmed cases into datasets/processed. Raw files are read-only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from preprocessing.adapters.discovery import CaseRecord, DiscoveryReport, require_complete
from preprocessing.errors import DatasetLayoutError, LabelSchemaError, MissingLabelError, MissingModalityError
from preprocessing.harmonization import task_enabled
from preprocessing.io_utils import load_nifti, read_dicom_series, save_nifti, validate_array
from preprocessing.normalization import nonzero_zscore, percentile_clip
from preprocessing.orientation import to_ras
from preprocessing.resampling import match_shape, resample_array
from preprocessing.validation import validate_volume
from training.config import project_root, resolve_path

TARGET_SPACING = (1.0, 1.0, 1.0)


def prepare_cases(
    schema: dict[str, Any],
    report: DiscoveryReport,
    processed_root: Path | None = None,
    spacing: tuple[float, float, float] = TARGET_SPACING,
    allow_incomplete: bool = False,
) -> Path:
    if allow_incomplete:
        cases = report.complete_cases
        if not cases:
            raise MissingModalityError(
                f"Every {report.dataset} patient is missing required modalities: {report.incomplete}"
            )
    else:
        cases = require_complete(report)
    raw_root = report.root
    out_root = processed_root or (project_root() / "datasets" / "processed" / schema["name"])
    out_root = resolve_path(out_root)
    seg_confirmed = task_enabled(schema, "segmentation")
    regions: dict[str, list[int]] = {}
    if seg_confirmed:
        regions = schema["labels"]["segmentation"].get("regions") or {}
        if not regions:
            raise LabelSchemaError("Segmentation is confirmed but no regions are defined.")

    manifest_cases = [
        _prepare_one(case, raw_root, out_root, spacing, seg_confirmed, regions) for case in cases
    ]
    payload = {
        "dataset": schema["name"],
        "spacing": list(spacing),
        "excluded_incomplete": report.incomplete if allow_incomplete else [],
        "unmatched_files": report.unmatched_files,
        "cases": manifest_cases,
    }
    manifest_path = out_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def _load_source(case: CaseRecord, modality: str, path: Path) -> tuple[np.ndarray, np.ndarray]:
    dicom_files = case.extra.get("dicom_files", {}).get(modality)
    if dicom_files:
        data, affine = read_dicom_series(list(dicom_files))
    else:
        validate_volume(path)
        data, affine = load_nifti(path)
    validate_array(data, f"{case.patient_id}:{modality}")
    return to_ras(data, affine)


def _prepare_one(
    case: CaseRecord,
    raw_root: Path,
    out_root: Path,
    spacing: tuple[float, float, float],
    seg_confirmed: bool,
    regions: dict[str, list[int]],
) -> dict[str, Any]:
    loaded = {
        modality: _load_source(case, modality, path) for modality, path in case.modalities.items()
    }
    _assert_same_grid(case.patient_id, loaded)
    reference_affine = None
    reference_shape = None
    written: dict[str, str] = {}
    patient_dir = out_root / case.patient_id
    for modality, (data, affine) in loaded.items():
        data, affine = resample_array(data, affine, spacing, order=1)
        data = nonzero_zscore(percentile_clip(data))
        if reference_shape is None:
            reference_shape = data.shape
            reference_affine = affine
        else:
            data = match_shape(data, reference_shape, order=1)
        dest = patient_dir / f"{modality}.nii.gz"
        save_nifti(dest, data.astype(np.float32), reference_affine, raw_root)
        written[modality] = str(dest)

    record: dict[str, Any] = {
        "patient_id": case.patient_id,
        "modalities": written,
        "segmentation": None,
        "region_order": [],
        "segmentation_status": "absent",
    }
    if case.segmentation is None:
        if seg_confirmed:
            raise MissingLabelError(
                f"{case.patient_id} has no segmentation file and {case.dataset} "
                "requires confirmed segmentation labels."
            )
        return record
    if not seg_confirmed:
        record["segmentation_status"] = "file_present_labels_unconfirmed"
        record["segmentation_source"] = str(case.segmentation)
        return record

    labels, affine = load_nifti(case.segmentation)
    labels, affine = to_ras(labels, affine)
    declared = {0, *[int(value) for group in regions.values() for value in group]}
    found = {int(value) for value in np.unique(labels)}
    unknown = sorted(found - declared)
    if unknown:
        raise LabelSchemaError(
            f"{case.patient_id} has label values {unknown} that are absent from the confirmed schema. "
            "They are not dropped."
        )
    _assert_same_grid(case.patient_id, {"labels": (labels, affine), **loaded})
    labels, _ = resample_array(labels, affine, spacing, order=0)
    labels = match_shape(labels, reference_shape, order=0)
    dest = patient_dir / "seg.nii.gz"
    save_nifti(dest, labels.astype(np.int16), reference_affine, raw_root)
    from preprocessing.harmonization import regions_from_label

    region_vol = regions_from_label(labels, regions)
    save_nifti(
        patient_dir / "regions.nii.gz",
        np.moveaxis(region_vol, 0, -1).astype(np.uint8),
        reference_affine,
        raw_root,
    )
    record["segmentation"] = str(dest)
    record["regions"] = str(patient_dir / "regions.nii.gz")
    record["region_order"] = list(regions)
    record["segmentation_status"] = "confirmed"
    return record


def _assert_same_grid(patient_id: str, volumes: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    items = list(volumes.items())
    _, (reference, affine) = items[0]
    for name, (data, other) in items[1:]:
        if data.shape != reference.shape or not np.allclose(affine, other, atol=1e-3):
            raise DatasetLayoutError(
                f"{patient_id}: {name} is not on the same grid as the reference modality. "
                "This pipeline does not invent a registration."
            )
