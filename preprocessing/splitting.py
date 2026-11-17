"""Patient-level splits. Image-level splitting is not offered."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from preprocessing.errors import SplitLeakageError


def patient_level_split(
    patient_ids: Sequence[str],
    ratios: dict[str, float],
    seed: int,
    stratify: Sequence[str] | None = None,
) -> dict[str, list[str]]:
    ids = [str(p) for p in patient_ids]
    if len(ids) != len(set(ids)):
        raise SplitLeakageError("Duplicate patient ids in the cohort.")
    if not ids:
        raise ValueError("Cannot split an empty patient list.")
    names = list(ratios)
    weights = np.array([float(ratios[name]) for name in names], dtype=np.float64)
    if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError(f"Split ratios must be non-negative and sum to 1, got {ratios}")

    rng = np.random.default_rng(seed)
    if stratify is None:
        order = ids.copy()
        rng.shuffle(order)
        return _chunk(order, names, weights)

    labels = [str(s) for s in stratify]
    if len(labels) != len(ids):
        raise ValueError("stratify must have one label per patient.")
    groups: dict[str, list[str]] = {}
    for patient, label in zip(ids, labels):
        groups.setdefault(label, []).append(patient)
    folded: dict[str, list[str]] = {name: [] for name in names}
    for label in sorted(groups):
        group = groups[label]
        rng.shuffle(group)
        part = _chunk(group, names, weights)
        for name in names:
            folded[name].extend(part[name])
    assert_no_overlap(folded)
    return folded


def assert_no_overlap(splits: dict[str, Sequence[str]]) -> None:
    seen: dict[str, str] = {}
    for name, patients in splits.items():
        for patient in patients:
            if patient in seen:
                raise SplitLeakageError(
                    f"Patient {patient} is in both {seen[patient]} and {name}."
                )
            seen[patient] = name


def _chunk(ids: list[str], names: list[str], weights: np.ndarray) -> dict[str, list[str]]:
    n = len(ids)
    counts = np.floor(weights * n).astype(int)
    # Give leftover patients to the largest split, then the next, so small
    # cohorts still fill val and test when n is large enough.
    leftover = n - int(counts.sum())
    order = np.argsort(-weights)
    for i in range(leftover):
        counts[order[i % len(counts)]] += 1
    if n >= len(names) and np.any(counts == 0):
        # Keep every requested split non-empty when there is one patient each.
        for idx in np.where(counts == 0)[0]:
            donor = int(np.argmax(counts))
            if counts[donor] > 1:
                counts[donor] -= 1
                counts[idx] += 1
    out: dict[str, list[str]] = {}
    start = 0
    for name, count in zip(names, counts):
        out[name] = ids[start : start + int(count)]
        start += int(count)
    return out
