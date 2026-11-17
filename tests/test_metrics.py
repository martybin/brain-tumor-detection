"""Metric behavior for empty masks, missing labels, and calibration."""

from __future__ import annotations

import numpy as np
import pytest

from evaluation.classification_metrics import classification_metrics
from evaluation.experiments import apply_perturbation, build_ablation_configs
from evaluation.segmentation_metrics import dice_score, hausdorff95, iou_score, sensitivity
from evaluation.statistics import assert_external_cohort, bootstrap_mean_ci, paired_wilcoxon
from evaluation.uncertainty import bernoulli_entropy, brier_score, expected_calibration_error, predictive_entropy
from preprocessing.errors import MissingLabelError, SplitLeakageError


def test_empty_masks() -> None:
    empty = np.zeros((8, 8, 8), dtype=bool)
    lesion = np.zeros_like(empty)
    lesion[2:5, 2:5, 2:5] = True
    assert dice_score(empty, empty) == 1.0
    assert iou_score(empty, empty) == 1.0
    assert dice_score(empty, lesion) == 0.0
    assert np.isnan(hausdorff95(empty, empty))
    assert np.isnan(hausdorff95(empty, lesion))
    assert np.isnan(sensitivity(empty, empty))
    assert hausdorff95(lesion, lesion) == 0.0
    assert dice_score(lesion, lesion) == 1.0


def test_missing_classification_labels_are_counted() -> None:
    with pytest.raises(MissingLabelError):
        classification_metrics(np.array([-1, -1]), np.array([0.2, 0.8]))
    metrics = classification_metrics(np.array([0, 1, -1]), np.array([0.1, 0.9, 0.4]))
    assert metrics["n"] == 2
    assert metrics["n_missing"] == 1
    assert metrics["accuracy"] == 1.0
    single = classification_metrics(np.array([1, 1]), np.array([0.6, 0.7]))
    assert single["roc_auc"] is None


def test_calibration_and_entropy() -> None:
    assert expected_calibration_error(np.ones(4), np.ones(4))["ece"] == 0.0
    assert brier_score(np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[1.0, 0.0], [0.0, 1.0]])) == 0.0
    peaked = predictive_entropy(np.array([0.99, 0.01]))
    uniform = predictive_entropy(np.array([0.5, 0.5]))
    assert uniform > peaked
    assert bernoulli_entropy(np.array(0.5)) > bernoulli_entropy(np.array(0.01))


def test_external_cohort_cannot_repeat_training_patients() -> None:
    with pytest.raises(SplitLeakageError):
        assert_external_cohort(["a", "b"], ["b", "c"])
    summary = bootstrap_mean_ci(np.array([0.5, 0.7, 0.9]), n_resamples=50, seed=1)
    assert summary["n"] == 3
    paired = paired_wilcoxon(np.array([0.2, 0.4]), np.array([0.2, 0.4]))
    assert paired["pvalue"] == 1.0


def test_ablation_and_robustness_do_not_invent_scores() -> None:
    variants = build_ablation_configs("configs/experiments/ablation.yaml")
    names = {item["ablation_name"] for item in variants}
    assert {"cnn_only", "unet3d", "swin_unetr", "nnunet"}.issubset(names)
    volume = np.ones((4, 6, 6, 6), dtype=np.float32)
    dropped = apply_perturbation(volume, {"name": "modality_dropout", "drop": ["flair"]}, np.random.default_rng(0))
    assert dropped[3].sum() == 0
    assert dropped[0].sum() > 0
