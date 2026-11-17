#!/usr/bin/env python3
"""Inspect a dataset root. This does not modify files and does not invent labels."""

from __future__ import annotations

import argparse
import json
import sys

import bootstrap  # noqa: F401

from preprocessing.adapters.discovery import discover, list_dicom_series
from preprocessing.adapters.registry import get_schema, list_datasets
from preprocessing.errors import DatasetLayoutError, DatasetNotFoundError
from preprocessing.harmonization import task_enabled
from preprocessing.validation import header_report, label_values


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a configured MRI dataset.")
    parser.add_argument("--dataset", required=True, choices=list_datasets())
    parser.add_argument("--stats", action="store_true", help="Read voxel values for label histograms.")
    args = parser.parse_args()
    schema = get_schema(args.dataset)
    print(f"dataset: {schema['name']}")
    print(f"root: {schema['root']}")
    print(f"layout_confirmed: {schema.get('confirmed_layout', schema['name'] == 'BraTS')}")
    for task in ("segmentation", "classification", "molecular", "survival"):
        print(f"task.{task}: {task_enabled(schema, task)}")
    if schema.get("format") == "dicom":
        try:
            series = list_dicom_series(_root(schema))
        except (DatasetNotFoundError, DatasetLayoutError) as exc:
            print(exc)
            return 1
        descriptions = sorted({item["series_description"] or "<empty>" for item in series})
        print(f"dicom_series: {len(series)}")
        print("series_descriptions:")
        for description in descriptions:
            print(f"  - {description}")
        if not schema.get("series_description_map"):
            print("series_description_map is empty. prepare_dataset.py will refuse this dataset.")
            return 1
    try:
        report = discover(schema)
    except (DatasetNotFoundError, DatasetLayoutError) as exc:
        print(exc)
        return 1
    print(f"patients: {len(report.cases)}")
    print(f"incomplete: {len(report.incomplete)}")
    for item in report.incomplete:
        print(f"  missing {item['patient_id']}: {item['missing_modalities']}")
    if report.unmatched_files:
        print(f"unmatched: {len(report.unmatched_files)}")
        for name in report.unmatched_files[:10]:
            print(f"  - {name}")
    for case in report.cases[:2]:
        print(f"case: {case.patient_id}")
        for modality, path in case.modalities.items():
            if path.is_file() and path.name.lower().endswith((".nii", ".nii.gz")):
                print(f"  {modality}: {json.dumps(header_report(path))}")
        if args.stats and case.segmentation is not None and case.segmentation.is_file():
            print(f"  segmentation_values: {label_values(case.segmentation)}")
    return 0


def _root(schema):
    from training.config import project_root, resolve_path

    root = resolve_path(schema["root"], project_root())
    if not root.exists():
        raise DatasetNotFoundError(f"{schema['name']} root does not exist: {root}")
    return root


if __name__ == "__main__":
    sys.exit(main())
