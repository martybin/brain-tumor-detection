"""Early stopping on a metric computed by the trainer. It does not invent scores."""

from __future__ import annotations

import math


class EarlyStopping:
    def __init__(self, patience: int, mode: str = "max") -> None:
        if mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'.")
        self.patience = int(patience)
        self.mode = mode
        self.best: float | None = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        if not math.isfinite(value):
            raise ValueError("Early stopping received a non-finite metric.")
        improved = self.best is None or (value > self.best if self.mode == "max" else value < self.best)
        if improved:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience
