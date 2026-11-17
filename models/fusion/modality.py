"""Modality dropout, adaptive weights, and cross-attention fusion."""

from __future__ import annotations

import torch
from torch import nn

from models.encoders.cnn3d import group_norm


class ModalityDropout(nn.Module):
    """Drop whole modalities. At least one modality is kept."""

    def __init__(self, probability: float) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("modality dropout probability must be in [0, 1).")
        self.probability = probability

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if (not self.training) or self.probability == 0:
            return x, None
        batch, modalities = x.shape[:2]
        keep = torch.rand(batch, modalities, device=x.device) >= self.probability
        missing = ~keep.any(dim=1)
        if missing.any():
            keep = keep.clone()
            keep[missing, 0] = True
        return x * keep.view(batch, modalities, 1, 1, 1).to(x.dtype), keep


class AdaptiveModalityWeighting(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.score = nn.Linear(channels, 1)

    def forward(
        self,
        features: list[torch.Tensor],
        keep: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = [self.score(self.pool(feat).flatten(1)) for feat in features]
        logits = torch.cat(scores, dim=1)
        if keep is not None:
            logits = logits.masked_fill(~keep, -1e4)
        weights = torch.softmax(logits, dim=1)
        stacked = torch.stack(features, dim=1)
        fused = (stacked * weights[:, :, None, None, None, None]).sum(dim=1)
        return fused, weights


class CrossAttentionFusion(nn.Module):
    """Fuse modalities with weights plus a cross-attention channel bias.

    Full spatial attention across four 3D maps does not fit in 8 GB.
    Each modality is one token. The attention matrix is stored for visualization.
    """

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        if channels % num_heads != 0:
            num_heads = 1
        self.heads = num_heads
        self.weighting = AdaptiveModalityWeighting(channels)
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.out = nn.Linear(channels, channels)
        self.gate = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))
        self.last_attention: torch.Tensor | None = None

    def forward(
        self,
        features: list[torch.Tensor],
        keep: torch.Tensor | None,
    ) -> torch.Tensor:
        fused, _weights = self.weighting(features, keep)
        tokens = torch.stack([feat.mean(dim=(2, 3, 4)) for feat in features], dim=1)
        tokens = self.norm(tokens)
        batch, modalities, channels = tokens.shape
        qkv = self.qkv(tokens).reshape(batch, modalities, 3, self.heads, channels // self.heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        scale = (channels // self.heads) ** -0.5
        logits = query @ key.transpose(-2, -1) * scale
        if keep is not None:
            logits = logits.masked_fill(~keep[:, None, None, :], -1e4)
        attention = torch.softmax(logits, dim=-1)
        self.last_attention = attention.detach()
        mixed = (attention @ value).transpose(1, 2).reshape(batch, modalities, channels)
        bias = self.out(mixed).mean(dim=1)
        return fused + self.gate * bias[:, :, None, None, None]


class ConcatFusion(nn.Module):
    """Project stacked modalities back to the encoder width. Used by ablations."""

    def __init__(self, channels: int, n_modalities: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(channels * n_modalities, channels, kernel_size=1, bias=False),
            group_norm(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: list[torch.Tensor], keep: torch.Tensor | None) -> torch.Tensor:
        del keep
        return self.proj(torch.cat(features, dim=1))
