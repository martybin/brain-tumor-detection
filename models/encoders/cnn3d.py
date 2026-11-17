"""Weight-shared 3D CNN encoder. One forward pass is a single modality."""

from __future__ import annotations

import torch
from torch import nn

from models.encoders.blocks import ConvBlock3d


class CNN3DEncoder(nn.Module):
    """Hierarchical residual-style encoder.

    Returns feature maps from high resolution to low resolution.
    """

    def __init__(self, in_channels: int, channels: tuple[int, ...], dropout: float = 0.0) -> None:
        super().__init__()
        if not channels:
            raise ValueError("Encoder channels must not be empty.")
        self.channels = tuple(channels)
        self.stem = ConvBlock3d(in_channels, channels[0], dropout=dropout)
        downs: list[nn.Module] = []
        for src, dst in zip(channels[:-1], channels[1:]):
            downs.append(
                nn.Sequential(
                    nn.Conv3d(src, dst, kernel_size=3, stride=2, padding=1, bias=False),
                    nn.InstanceNorm3d(dst, affine=True),
                    nn.LeakyReLU(0.01, inplace=True),
                    ConvBlock3d(dst, dst, dropout=dropout),
                )
            )
        self.downs = nn.ModuleList(downs)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = [self.stem(x)]
        for down in self.downs:
            features.append(down(features[-1]))
        return features
