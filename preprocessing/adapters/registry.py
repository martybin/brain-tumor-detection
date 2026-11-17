"""Map dataset names to schema files. Adapters do not share a fake common label set."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from training.config import load_yaml, project_root

SCHEMAS: dict[str, str] = {
    "BraTS": "configs/datasets/brats.yaml",
    "UCSF-PDGM": "configs/datasets/ucsf_pdgm.yaml",
    "UPENN-GBM": "configs/datasets/upenn_gbm.yaml",
    "REMBRANDT": "configs/datasets/rembrandt.yaml",
    "BraTS-PEDs": "configs/datasets/brats_peds.yaml",
}


def list_datasets() -> list[str]:
    return list(SCHEMAS)


def get_schema(name: str) -> dict[str, Any]:
    key = _canonical(name)
    if key not in SCHEMAS:
        known = ", ".join(SCHEMAS)
        raise KeyError(f"Unknown dataset '{name}'. Known datasets: {known}.")
    return load_yaml(project_root() / SCHEMAS[key])


def schema_path(name: str) -> Path:
    return project_root() / SCHEMAS[_canonical(name)]


def _canonical(name: str) -> str:
    lookup = {key.lower(): key for key in SCHEMAS}
    found = lookup.get(name.lower())
    if found is None:
        known = ", ".join(SCHEMAS)
        raise KeyError(f"Unknown dataset '{name}'. Known datasets: {known}.")
    return found
