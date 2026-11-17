"""3D decoder. Upsampling follows the skip resolution rather than assuming even shapes."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from models.encoders.blocks import ConvBlock3d


class UpBlock3d(nn.Module):
    def __init__(self, low_channels: int, skip_channels: int, dropout: float) -> None:
        super().__init__()
        self.conv = ConvBlock3d(low_channels + skip_channels, skip_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode="trilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class UNetDecoder3D(nn.Module):
    def __init__(self, channels: tuple[int, ...], dropout: float = 0.0) -> None:
        super().__init__()
        pairs = list(zip(channels[1:][::-1], channels[:-1][::-1]))
        self.ups = nn.ModuleList(UpBlock3d(low, skip, dropout) for low, skip in pairs)
        self.cam_layer = self.ups[-1].conv.out_conv if self.ups else None

    def forward(self, skips: list[torch.Tensor]) -> torch.Tensor:
        current = skips[-1]
        for index, up in enumerate(self.ups):
            current = up(current, skips[-(index + 2)])
        return current
