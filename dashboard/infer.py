"""Run a trained checkpoint on one uploaded study. No checkpoint, no prediction."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dashboard.sessions import SESSION_ROOT
from preprocessing.adapters.discovery import discover, require_complete
from preprocessing.adapters.registry import get_schema
from preprocessing.io_utils import load_nifti, save_nifti
from preprocessing.pipeline import _prepare_one
from training.checkpoint import load_checkpoint, read_checkpoint


def run_case(folder: Path, meta: dict, checkpoint: Path) -> dict:
    import torch

    from models.factory import build_model
    from training.inference import sliding_window_probabilities

    schema = dict(get_schema("BraTS"))
    schema["root"] = str(folder / "original")
    schema["tasks"] = dict(schema.get("tasks") or {})
    schema["tasks"]["segmentation"] = False
    schema["labels"] = dict(schema.get("labels") or {})
    report = discover(schema)
    case = require_complete(report)[0]
    raw_root = SESSION_ROOT
    record = _prepare_one(case, raw_root, folder / "aligned", (1.0, 1.0, 1.0), False, {})
    volume = np.stack(
        [np.asarray(load_nifti(Path(record["modalities"][name]))[0], dtype=np.float32) for name in ("t1", "t1ce", "t2", "flair")],
        axis=0,
    )
    payload = read_checkpoint(checkpoint)
    model = build_model(payload["model_config"])
    load_checkpoint(checkpoint, model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    patch = tuple(int(value) for value in payload["config"]["data"]["patch_size"])
    probabilities = sliding_window_probabilities(model, volume, patch, overlap=0.5, device=device)
    order = list(payload.get("region_order") or [f"region_{index}" for index in range(probabilities.shape[0])])
    masks = probabilities >= 0.5
    regions = {}
    for name, channel in zip(order, masks):
        regions[name] = {"voxels": int(channel.sum())}
    signal = np.any(masks, axis=0)
    aligned_dir = folder / "aligned" / meta["id"]
    reference = Path(record["modalities"]["flair"])
    _, affine = load_nifti(reference)
    save_nifti(aligned_dir / "signal.nii.gz", signal.astype(np.uint8), affine, raw_root)
    marked = bool(signal.any())
    if marked:
        message = "مدل پژوهشی در این حجم ناحیه مشکوک علامت گذاشت. این به‌معنی تشخیص تومور برای شما نیست."
    else:
        message = "مدل پژوهشی ناحیه‌ای بالاتر از آستانه علامت نزد. این به‌معنی رد تومور برای شما نیست."
    return {
        "available": True,
        "marked": marked,
        "clinical_validity": False,
        "regions": regions,
        "message": message,
        "disclaimer": "این خروجی پژوهشی است و تشخیص پزشکی نیست. تصمیم بالینی با پزشک و رادیولوژیست است.",
        "mask_file": f"aligned/{meta['id']}/signal.nii.gz",
        "space": "aligned",
        "shape": [int(value) for value in signal.shape],
        "checkpoint": str(checkpoint),
    }
