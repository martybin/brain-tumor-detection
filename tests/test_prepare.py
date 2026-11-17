"""Discovery and preprocessing against synthetic NIfTI, not a downloaded cohort."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

nib = pytest.importorskip("nibabel")

from preprocessing.adapters.discovery import discover, require_complete
from preprocessing.adapters.registry import get_schema
from preprocessing.errors import DatasetLayoutError, LabelSchemaError, MissingModalityError
from preprocessing.io_utils import load_nifti
from preprocessing.pipeline import prepare_cases


def _write(path: Path, data: np.ndarray, affine: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(np.asarray(data), np.eye(4) if affine is None else affine), str(path))


def _case(root: Path, patient: str, missing: str | None = None, label_value: int = 4) -> None:
    image = np.zeros((16, 16, 16), dtype=np.float32)
    image[4:12, 4:12, 4:12] = 80
    for modality, suffix in (("t1", "t1"), ("t1ce", "t1ce"), ("t2", "t2"), ("flair", "flair")):
        if modality == missing:
            continue
        _write(root / patient / f"{patient}_{suffix}.nii.gz", image + {"t1": 0, "t1ce": 5, "t2": 9, "flair": 3}[modality])
    seg = np.zeros((16, 16, 16), dtype=np.int16)
    seg[5:11, 5:11, 5:11] = 2
    seg[7:9, 7:9, 7:9] = 1
    seg[8, 8, 8] = label_value
    _write(root / patient / f"{patient}_seg.nii.gz", seg)


def test_discovery_reports_a_missing_modality(tmp_path: Path) -> None:
    _case(tmp_path, "BraTS_001")
    _case(tmp_path, "BraTS_002", missing="flair")
    schema = get_schema("BraTS")
    schema["root"] = str(tmp_path)
    report = discover(schema)
    assert {case.patient_id for case in report.cases} == {"BraTS_001", "BraTS_002"}
    assert report.cases[0].modalities["t1ce"].name.endswith("t1ce.nii.gz")
    assert report.incomplete == [{"patient_id": "BraTS_002", "missing_modalities": ["flair"]}]
    with pytest.raises(MissingModalityError):
        require_complete(report)


def test_prepare_leaves_raw_bytes_unchanged(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _case(raw, "BraTS_001")
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in raw.rglob("*.nii.gz")}
    schema = get_schema("BraTS")
    schema["root"] = str(raw)
    report = discover(schema)
    manifest = prepare_cases(schema, report, processed_root=tmp_path / "processed")
    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in raw.rglob("*.nii.gz")}
    assert before == after
    regions, _ = load_nifti(manifest.parent / "BraTS_001" / "regions.nii.gz", expected_ndim=(4,))
    stacked = np.moveaxis(np.asarray(regions), -1, 0)
    whole, core, enhancing = stacked
    assert int(enhancing.sum()) == 1
    assert int(core.sum()) > int(enhancing.sum())
    assert int(whole.sum()) > int(core.sum())


def test_unknown_label_and_grid_mismatch_are_errors(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _case(raw, "BraTS_001", label_value=7)
    schema = get_schema("BraTS")
    schema["root"] = str(raw)
    with pytest.raises(LabelSchemaError):
        prepare_cases(schema, discover(schema), processed_root=tmp_path / "processed")

    raw2 = tmp_path / "raw2"
    _case(raw2, "BraTS_009")
    shifted = np.eye(4)
    shifted[0, 3] = 5
    image = np.zeros((16, 16, 16), dtype=np.float32)
    image[4:12, 4:12, 4:12] = 40
    _write(raw2 / "BraTS_009" / "BraTS_009_t2.nii.gz", image, shifted)
    schema["root"] = str(raw2)
    with pytest.raises(DatasetLayoutError):
        prepare_cases(schema, discover(schema), processed_root=tmp_path / "processed2")
