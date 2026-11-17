#!/usr/bin/env python3
"""Resolve ablation configs. Training starts only with --run and a prepared dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

import bootstrap  # noqa: F401

from evaluation.experiments import build_ablation_configs
from training.config import project_root


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize ablation configs.")
    parser.add_argument("--config", default="configs/experiments/ablation.yaml")
    parser.add_argument("--run", action="store_true", help="Train each in-repository variant. Requires prepared data.")
    args = parser.parse_args()
    variants = build_ablation_configs(args.config)
    out_dir = project_root() / "experiments" / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for variant in variants:
        if variant.get("implementation") == "external":
            print(f"external baseline not launched: {variant['name']}")
            continue
        path = out_dir / f"{variant['ablation_name']}.yaml"
        path.write_text(yaml.safe_dump(variant, sort_keys=False), encoding="utf-8")
        written.append(path)
        print(f"config: {path}")
    if not args.run:
        print("Configs were written. No training was run and no metric was created.")
        return 0
    from training.engine import run_training

    for path in written:
        print(f"training: {path}")
        print(run_training(_training_config(path)))
    return 0


def _training_config(model_config: Path) -> Path:
    base = yaml.safe_load((project_root() / "configs" / "training.yaml").read_text(encoding="utf-8"))
    base["model_config"] = str(model_config.relative_to(project_root()))
    destination = model_config.with_name(model_config.stem + "_training.yaml")
    destination.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    return destination


if __name__ == "__main__":
    sys.exit(main())
