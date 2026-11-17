"""Ablation configs and robustness copies. No metric is filled in here."""

from __future__ import annotations

import copy
from typing import Any

import numpy as np

from training.config import load_yaml

MODALITY_INDEX = {"t1": 0, "t1ce": 1, "t2": 2, "flair": 3}


def build_ablation_configs(path: str) -> list[dict[str, Any]]:
    spec = load_yaml(path)
    base = load_yaml(spec["base_model"])
    variants = []
    for variant in spec["variants"]:
        model = copy.deepcopy(base)
        model.update(variant.get("overrides") or {})
        model["ablation_name"] = variant["name"]
        variants.append(model)
    for baseline_path in spec.get("baselines") or []:
        baseline = load_yaml(baseline_path)
        baseline["ablation_name"] = baseline["name"]
        variants.append(baseline)
    return variants


def apply_perturbation(volume: np.ndarray, spec: dict[str, Any], rng: np.random.Generator) -> np.ndarray:
    name = spec["name"]
    out = np.array(volume, copy=True)
    if name == "identity":
        return out
    if name == "gaussian_noise":
        noise = rng.normal(0.0, float(spec["std"]), size=out.shape).astype(out.dtype)
        return out + noise
    if name == "intensity_scale":
        return out * float(spec["factor"])
    if name == "modality_dropout":
        for modality in spec["drop"]:
            if modality not in MODALITY_INDEX:
                raise KeyError(f"Unknown modality in robustness config: {modality}")
            index = MODALITY_INDEX[modality]
            if index >= out.shape[0]:
                raise IndexError(f"Volume has no channel for {modality}.")
            out[index] = 0
        return out
    raise KeyError(f"Unknown perturbation: {name}")
