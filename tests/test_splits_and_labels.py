"""Patient splits, labels, and raw-data protection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from preprocessing.adapters.registry import get_schema
from preprocessing.errors import LabelSchemaError, RawDataProtectionError, SplitLeakageError
from preprocessing.harmonization import regions_from_label, require_task, task_enabled
from preprocessing.io_utils import save_nifti
from preprocessing.splitting import assert_no_overlap, patient_level_split
from training.callbacks import EarlyStopping


def test_patient_split_has_no_overlap() -> None:
    patients = [f"p{i:03d}" for i in range(30)]
    splits = patient_level_split(patients, {"train": 0.7, "val": 0.15, "test": 0.15}, seed=7)
    assert_no_overlap(splits)
    assert set(patients) == set(splits["train"] + splits["val"] + splits["test"])
    assert splits["val"] and splits["test"]


def test_split_overlap_is_an_error() -> None:
    with pytest.raises(SplitLeakageError):
        assert_no_overlap({"train": ["a"], "test": ["a"]})


def test_brats_tasks_follow_confirmed_labels() -> None:
    schema = get_schema("BraTS")
    assert task_enabled(schema, "segmentation")
    assert not task_enabled(schema, "classification")
    assert not task_enabled(schema, "survival")
    ucsf = get_schema("UCSF-PDGM")
    assert not task_enabled(ucsf, "segmentation")
    with pytest.raises(LabelSchemaError):
        require_task(ucsf, "molecular")


def test_regions_follow_the_declared_map() -> None:
    label = np.zeros((6, 6, 6), dtype=np.int16)
    label[1:5, 1:5, 1:5] = 2
    label[2:4, 2:4, 2:4] = 1
    label[3, 3, 3] = 4
    regions = regions_from_label(label, {"WT": [1, 2, 4], "TC": [1, 4], "ET": [4]})
    assert regions.shape == (3, 6, 6, 6)
    assert regions[0, 1, 1, 1] == 1
    assert regions[1, 1, 1, 1] == 0
    assert regions[2, 3, 3, 3] == 1
    assert regions[1, 3, 3, 3] == 1


def test_raw_tree_is_not_writable(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    with pytest.raises(RawDataProtectionError):
        save_nifti(raw / "t1.nii.gz", np.zeros((2, 2, 2), np.float32), np.eye(4), raw)


def test_early_stopping_uses_the_metric() -> None:
    stopper = EarlyStopping(patience=1, mode="max")
    assert stopper.step(0.4) is False
    assert stopper.step(0.2) is True
