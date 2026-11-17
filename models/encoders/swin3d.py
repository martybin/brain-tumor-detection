"""Windowed self-attention bottleneck.

This is not a full Swin UNETR. The Swin UNETR baseline is the MONAI model
selected by configs/models/swin_unetr.yaml. The block here is the hybrid
bottleneck and is small enough for an 8 GB GPU.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class WindowAttention3D(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}.")
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.last_attention: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, depth, height, width = x.shape
        window = self.window_size
        pad_d = (window - depth % window) % window
        pad_h = (window - height % window) % window
        pad_w = (window - width % window) % window
        padded = F.pad(x, (0, pad_w, 0, pad_h, 0, pad_d))
        _, _, depth_p, height_p, width_p = padded.shape
        windows = padded.view(
            batch,
            channels,
            depth_p // window,
            window,
            height_p // window,
            window,
            width_p // window,
            window,
        )
        windows = windows.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
        tokens = windows.view(-1, window ** 3, channels)
        tokens = self.norm(tokens)
        qkv = self.qkv(tokens).reshape(tokens.shape[0], tokens.shape[1], 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attention = (query * self.scale) @ key.transpose(-2, -1)
        attention = attention.softmax(dim=-1)
        self.last_attention = attention.detach()
        mixed = (attention @ value).transpose(1, 2).reshape(tokens.shape[0], tokens.shape[1], channels)
        mixed = self.proj(mixed)
        mixed = mixed.view(
            batch,
            depth_p // window,
            height_p // window,
            width_p // window,
            window,
            window,
            window,
            channels,
        )
        mixed = mixed.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        mixed = mixed.view(batch, channels, depth_p, height_p, width_p)
        mixed = mixed[:, :, :depth, :height, :width]
        return x + mixed
