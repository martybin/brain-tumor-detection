"""Volume-level evaluation. Metrics are written only after the split is scored."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from evaluation.segmentation_metrics import nanmean_patients, patient_region_scores
from evaluation.statistics import bootstrap_mean_ci
from models.factory import build_model
from preprocessing.io_utils import CANONICAL_MODALITIES, load_nifti
from training.checkpoint import load_checkpoint, read_checkpoint
from training.config import project_root, resolve_path
from training.inference import sliding_window_probabilities


def evaluate_checkpoint(
    checkpoint: Path,
    split_name: str = "test",
    overlap: float = 0.5,
    output: Path | None = None,
) -> dict:
    payload = read_checkpoint(checkpoint)
    model = build_model(payload["model_config"])
    load_checkpoint(checkpoint, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    config = payload["config"]
    dataset = payload["dataset"]
    manifest_path = resolve_path(config["data"]["processed_root"]) / dataset / "manifest.json"
    split_path = resolve_path(config["data"]["splits_root"]) / dataset / "splits.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = json.loads(split_path.read_text(encoding="utf-8"))
    if split_name not in splits:
        raise KeyError(f"Split '{split_name}' is not in {split_path}.")
    wanted = set(splits[split_name])
    cases = [case for case in manifest["cases"] if case["patient_id"] in wanted]
    if len(cases) != len(wanted):
        raise FileNotFoundError("The split names patients that are absent from the manifest.")
    region_order = list(payload["region_order"])
    patch = tuple(int(v) for v in config["data"]["patch_size"])
    patient_scores = []
    for case in cases:
        if case.get("segmentation_status") != "confirmed":
            raise FileNotFoundError(f"{case['patient_id']} has no confirmed segmentation labels.")
        volume = _stack(case)
        regions, _ = load_nifti(Path(case["regions"]), expected_ndim=(4,))
        regions = np.moveaxis(np.asarray(regions), -1, 0)
        probs = sliding_window_probabilities(model, volume, patch, overlap, device)
        patient_scores.append(patient_region_scores(probs >= 0.5, regions > 0, region_order))
    summary = nanmean_patients(patient_scores)
    for region in region_order:
        dice_values = [scores[region]["dice"] for scores in patient_scores]
        summary[region]["dice_bootstrap"] = bootstrap_mean_ci(np.array(dice_values), seed=int(config["seed"]))
    result = {
        "dataset": dataset,
        "split": split_name,
        "n_patients": len(cases),
        "checkpoint": str(checkpoint),
        "regions": summary,
        "clinical_validity": False,
    }
    destination = output or (project_root() / "outputs" / "metrics" / f"{dataset}_{split_name}.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _stack(case: dict) -> np.ndarray:
    volumes = []
    for modality in CANONICAL_MODALITIES:
        path = case["modalities"].get(modality)
        if not path:
            raise FileNotFoundError(f"{case['patient_id']} is missing {modality}.")
        data, _ = load_nifti(Path(path))
        volumes.append(np.asarray(data, dtype=np.float32))
    return np.stack(volumes, axis=0)
