"""YAML configuration loading. Hyperparameters live in configs/, not in code."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from preprocessing.errors import ConfigurationError


def load_yaml(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.is_file():
        raise ConfigurationError(f"Config file not found: {file_path}")
    with file_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ConfigurationError(f"Config must be a mapping: {file_path}")
    return data


def require(config: dict[str, Any], *keys: str) -> Any:
    """Return a nested value or raise ConfigurationError."""
    node: Any = config
    walked: list[str] = []
    for key in keys:
        walked.append(key)
        if not isinstance(node, dict) or key not in node:
            raise ConfigurationError(f"Missing config key: {'.'.join(walked)}")
        node = node[key]
    return node


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (root or project_root()) / candidate
