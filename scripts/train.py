#!/usr/bin/env python3
"""Train from a YAML config. Refuses to start when labels or data are missing."""

from __future__ import annotations

import argparse
import sys

import bootstrap  # noqa: F401


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a configured brain-tumor model.")
    parser.add_argument("--config", default="configs/training.yaml")
    args = parser.parse_args()
    try:
        from training.engine import run_training
    except ImportError as exc:
        print(f"Training dependencies are missing: {exc}")
        return 1
    try:
        checkpoint = run_training(args.config)
    except Exception as exc:  # noqa: BLE001 - CLI reports the framework error and returns non-zero.
        print(exc)
        return 1
    print(f"checkpoint: {checkpoint}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
