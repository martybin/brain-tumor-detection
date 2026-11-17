"""Optimizers and epoch-wise learning-rate values."""

from __future__ import annotations

import math

import torch
from torch import nn


def build_optimizer(model: nn.Module, name: str, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    if name.lower() != "adamw":
        raise ValueError(f"Unsupported optimizer: {name}")
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def cosine_lr(epoch: int, total_epochs: int, warmup_epochs: int, base_lr: float) -> float:
    if epoch < warmup_epochs:
        return base_lr * float(epoch + 1) / float(max(warmup_epochs, 1))
    progress = (epoch - warmup_epochs) / float(max(total_epochs - warmup_epochs, 1))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr
