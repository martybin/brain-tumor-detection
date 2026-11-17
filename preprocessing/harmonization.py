"""Dataset-specific label maps. Unconfirmed maps do not invent classes."""

from __future__ import annotations

from typing import Any

import numpy as np

from preprocessing.errors import LabelSchemaError


def segmentation_spec(schema: dict[str, Any]) -> dict[str, Any]:
    spec = schema.get("labels", {}).get("segmentation", {})
    if not spec.get("confirmed", False):
        raise LabelSchemaError(
            f"{schema.get('name')} segmentation labels are not confirmed. "
            "Inspect unique values, set labels.segmentation.values, and set confirmed: true."
        )
    if not spec.get("regions") and not spec.get("values"):
        raise LabelSchemaError("Confirmed segmentation schema has no values or regions.")
    return spec


def regions_from_label(label: np.ndarray, regions: dict[str, list[int]]) -> np.ndarray:
    """Stack binary region masks in the declared region order."""
    channels = []
    for name in regions:
        values = regions[name]
        mask = np.zeros(label.shape, dtype=np.float32)
        for value in values:
            mask[label == int(value)] = 1.0
        channels.append(mask)
    return np.stack(channels, axis=0)


def task_enabled(schema: dict[str, Any], task: str) -> bool:
    """A task is on only when the schema both requests it and confirms labels."""
    if not schema.get("tasks", {}).get(task, False):
        return False
    if task == "segmentation":
        return bool(schema.get("labels", {}).get("segmentation", {}).get("confirmed"))
    block = schema.get("labels", {}).get(task, {})
    return bool(block.get("confirmed"))


def require_task(schema: dict[str, Any], task: str) -> None:
    if not task_enabled(schema, task):
        raise LabelSchemaError(
            f"Task '{task}' is inactive for {schema.get('name')}. "
            "Compatible confirmed labels are required."
        )
