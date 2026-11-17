"""Segmentation metrics.

Both-empty Dice and IoU are 1. HD95 is undefined when either mask is empty
and is returned as NaN. Missing values stay missing.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = np.asarray(pred).astype(bool)
    gt_b = np.asarray(gt).astype(bool)
    tp = int(np.logical_and(pred_b, gt_b).sum())
    fp = int(np.logical_and(pred_b, ~gt_b).sum())
    fn = int(np.logical_and(~pred_b, gt_b).sum())
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    return (2 * tp) / (2 * tp + fp + fn)


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = np.asarray(pred).astype(bool)
    gt_b = np.asarray(gt).astype(bool)
    tp = int(np.logical_and(pred_b, gt_b).sum())
    fp = int(np.logical_and(pred_b, ~gt_b).sum())
    fn = int(np.logical_and(~pred_b, gt_b).sum())
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    return tp / (tp + fp + fn)


def sensitivity(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = np.asarray(pred).astype(bool)
    gt_b = np.asarray(gt).astype(bool)
    positive = int(gt_b.sum())
    if positive == 0:
        return float("nan")
    return float(np.logical_and(pred_b, gt_b).sum() / positive)


def specificity(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = np.asarray(pred).astype(bool)
    gt_b = np.asarray(gt).astype(bool)
    negative = int((~gt_b).sum())
    if negative == 0:
        return float("nan")
    tn = int(np.logical_and(~pred_b, ~gt_b).sum())
    return tn / negative


def hausdorff95(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    pred_b = np.asarray(pred).astype(bool)
    gt_b = np.asarray(gt).astype(bool)
    if not pred_b.any() or not gt_b.any():
        return float("nan")
    pred_surface = _surface(pred_b)
    gt_surface = _surface(gt_b)
    pred_distance = ndimage.distance_transform_edt(~pred_b, sampling=spacing)
    gt_distance = ndimage.distance_transform_edt(~gt_b, sampling=spacing)
    distances = np.concatenate([gt_distance[pred_surface], pred_distance[gt_surface]])
    return float(np.percentile(distances, 95))


def lesion_sensitivity(pred: np.ndarray, gt: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred_b = np.asarray(pred).astype(bool)
    gt_b = np.asarray(gt).astype(bool)
    labeled, count = ndimage.label(gt_b)
    if count == 0:
        return {"n_lesions": 0, "mean_overlap": float("nan"), "detected_fraction": float("nan")}
    overlaps = []
    detected = 0
    for index in range(1, count + 1):
        component = labeled == index
        overlap = float(np.logical_and(pred_b, component).sum() / component.sum())
        overlaps.append(overlap)
        if overlap >= threshold:
            detected += 1
    return {
        "n_lesions": int(count),
        "mean_overlap": float(np.mean(overlaps)),
        "detected_fraction": detected / count,
    }


def patient_region_scores(
    pred: np.ndarray,
    gt: np.ndarray,
    region_names: list[str],
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict[str, dict[str, float]]:
    if pred.shape != gt.shape:
        raise ValueError(f"Prediction and label shapes differ: {pred.shape} vs {gt.shape}")
    if pred.shape[0] != len(region_names):
        raise ValueError("Region names do not match the channel count.")
    scores = {}
    for name, pred_c, gt_c in zip(region_names, pred, gt):
        scores[name] = {
            "dice": dice_score(pred_c, gt_c),
            "iou": iou_score(pred_c, gt_c),
            "hd95": hausdorff95(pred_c, gt_c, spacing),
            "sensitivity": sensitivity(pred_c, gt_c),
            "specificity": specificity(pred_c, gt_c),
            **{f"lesion_{key}": value for key, value in lesion_sensitivity(pred_c, gt_c).items()},
        }
    return scores


def nanmean_patients(patient_scores: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    if not patient_scores:
        raise ValueError("No patient scores to aggregate.")
    summary: dict[str, dict[str, float]] = {}
    regions = patient_scores[0]
    for region in regions:
        summary[region] = {}
        keys = regions[region]
        for key in keys:
            values = np.array([scores[region][key] for scores in patient_scores], dtype=float)
            finite = values[np.isfinite(values)]
            summary[region][key] = float(finite.mean()) if len(finite) else float("nan")
            summary[region][f"{key}_n"] = int(len(finite))
    return summary


def _surface(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    eroded = ndimage.binary_erosion(mask, structure=ndimage.generate_binary_structure(mask.ndim, 1), border_value=0)
    surface = np.logical_and(mask, ~eroded)
    if not surface.any():
        return mask
    return surface
