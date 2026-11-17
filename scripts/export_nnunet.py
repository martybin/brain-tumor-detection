#!/usr/bin/env python3
"""Write an nnU-Net dataset.json from a confirmed prepared manifest.

This does not train nnU-Net and does not create a score.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bootstrap  # noqa: F401

from preprocessing.adapters.registry import get_schema, list_datasets
from preprocessing.errors import LabelSchemaError
from preprocessing.harmonization import segmentation_spec
from training.config import project_root, resolve_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a confirmed manifest for external nnU-Net.")
    parser.add_argument("--dataset", required=True, choices=list_datasets())
    args = parser.parse_args()
    schema = get_schema(args.dataset)
    try:
        spec = segmentation_spec(schema)
    except LabelSchemaError as exc:
        print(exc)
        return 1
    manifest_path = project_root() / "datasets" / "processed" / schema["name"] / "manifest.json"
    if not manifest_path.is_file():
        print(f"Prepared manifest not found: {manifest_path}")
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    labels = {"background": 0}
    for value, name in (spec.get("values") or {}).items():
        labels[str(name)] = int(value)
    payload = {
        "name": schema["name"],
        "channel_names": {"0": "t1", "1": "t1ce", "2": "t2", "3": "flair"},
        "labels": labels,
        "file_ending": ".nii.gz",
        "numTraining": len(manifest["cases"]),
        "regions": spec.get("regions") or {},
        "note": "Run nnU-Net separately. This file is not a result.",
    }
    destination = resolve_path(f"experiments/nnunet/{schema['name']}/dataset.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"dataset_json: {destination}")
    print("nnU-Net is external. No metric was computed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
