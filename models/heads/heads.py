"""Task heads. A head exists only when the caller enables that task."""

from __future__ import annotations

import torch
from torch import nn


class SegmentationHead(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        if out_channels <= 0:
            raise ValueError("Segmentation head needs at least one output channel.")
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ClassificationHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        if num_classes is None or int(num_classes) < 2:
            raise ValueError("Classification requires num_classes >= 2 and confirmed labels.")
        self.fc = nn.Linear(in_channels, int(num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.flatten(2).mean(dim=-1)
        return self.fc(pooled)


class SurvivalHead(nn.Module):
    """Log-risk score for a Cox partial likelihood. It is not a survival time."""

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.flatten(2).mean(dim=-1)
        return self.fc(pooled).squeeze(-1)
