#!/usr/bin/env python3
"""Predict region masks for one patient folder. The input folder is not modified."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import bootstrap  # noqa: F401


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a checkpoint on one patient.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="BraTS")
    parser.add_argument("--output", default="outputs/predictions")
    args = parser.parse_args()
    source = Path(args.input)
    checkpoint = Path(args.checkpoint)
    if not source.exists() or not checkpoint.is_file():
        print("Input path or checkpoint is missing.")
        return 1
    try:
        import torch

        from evaluation.uncertainty import bernoulli_entropy
        from explainability.visualization import save_overlay, save_uncertainty_slice
        from models.factory import build_model
        from preprocessing.adapters.discovery import discover
        from preprocessing.adapters.registry import get_schema
        from preprocessing.io_utils import save_nifti
        from preprocessing.pipeline import _prepare_one
        from training.checkpoint import load_checkpoint, read_checkpoint
        from training.inference import sliding_window_probabilities
    except ImportError as exc:
        print(f"Prediction dependencies are missing: {exc}")
        return 1

    schema = get_schema(args.dataset)
    schema = dict(schema)
    schema["root"] = str(source if _contains_volume(source) else source)
    if source.is_dir() and any(source.glob("*.nii*")):
        schema["root"] = str(source)
    try:
        report = discover(schema)
    except Exception as exc:  # noqa: BLE001
        print(exc)
        return 1
    if len(report.cases) != 1:
        print(f"Expected one patient in {source}, found {len(report.cases)}.")
        return 1
    case = report.complete_cases[0] if not report.incomplete else None
    if case is None:
        print(f"Missing modalities: {report.incomplete}")
        return 1
    payload = read_checkpoint(checkpoint)
    out_dir = Path(args.output) / case.patient_id
    record = _prepare_one(
        case,
        report.root,
        out_dir / "_preprocessed",
        (1.0, 1.0, 1.0),
        seg_confirmed=False,
        regions={},
    )
    volume = _load_prepared(record)
    model = build_model(payload["model_config"])
    load_checkpoint(checkpoint, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    patch = tuple(int(v) for v in payload["config"]["data"]["patch_size"])
    probs = sliding_window_probabilities(model, volume, patch, overlap=0.5, device=device)
    masks = (probs >= 0.5).astype(np.uint8)
    affine = _affine(Path(record["modalities"]["flair"]))
    save_nifti(out_dir / "region_masks.nii.gz", np.moveaxis(masks, 0, -1), affine, report.root)
    region_order = list(payload["region_order"])
    (out_dir / "prediction.json").write_text(
        json.dumps(
            {
                "patient_id": case.patient_id,
                "region_order": region_order,
                "clinical_validity": False,
                "note": "Research output. Not a diagnostic device.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    slice_index = int(volume.shape[-1] // 2)
    background = volume[1, :, :, slice_index]
    overlay_mask = masks[region_order.index("ET")][:, :, slice_index] if "ET" in region_order else masks[0, :, :, slice_index]
    save_overlay(background, overlay_mask, out_dir / "overlay.png", title="Research overlay")
    entropy = bernoulli_entropy(probs).mean(axis=0)[:, :, slice_index]
    save_uncertainty_slice(entropy, out_dir / "entropy.png")
    print(f"prediction: {out_dir}")
    return 0


def _contains_volume(path: Path) -> bool:
    return path.is_dir()


def _load_prepared(record: dict) -> np.ndarray:
    from preprocessing.io_utils import CANONICAL_MODALITIES, load_nifti

    volumes = [np.asarray(load_nifti(Path(record["modalities"][name]))[0], dtype=np.float32) for name in CANONICAL_MODALITIES]
    return np.stack(volumes, axis=0)


def _affine(path: Path) -> np.ndarray:
    from preprocessing.io_utils import load_nifti

    _, affine = load_nifti(path)
    return affine


if __name__ == "__main__":
    sys.exit(main())
