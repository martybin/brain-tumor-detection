#!/usr/bin/env python3
"""Preprocess a dataset into datasets/processed. Raw files are only read."""

from __future__ import annotations

import argparse
import sys

import bootstrap  # noqa: F401

from preprocessing.adapters.discovery import discover
from preprocessing.adapters.registry import get_schema, list_datasets
from preprocessing.errors import FrameworkError
from preprocessing.pipeline import prepare_cases
from training.config import resolve_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a dataset without modifying raw images.")
    parser.add_argument("--dataset", required=True, choices=list_datasets())
    parser.add_argument("--spacing", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    parser.add_argument("--processed-root", default=None)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Prepare complete patients and record the incomplete ones in the manifest.",
    )
    args = parser.parse_args()
    schema = get_schema(args.dataset)
    try:
        report = discover(schema)
        manifest = prepare_cases(
            schema,
            report,
            processed_root=resolve_path(args.processed_root) if args.processed_root else None,
            spacing=tuple(args.spacing),
            allow_incomplete=args.allow_incomplete,
        )
    except FrameworkError as exc:
        print(exc)
        return 1
    print(f"manifest: {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
