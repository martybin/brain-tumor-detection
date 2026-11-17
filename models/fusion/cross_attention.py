"""Cross-modality attention and adaptive weighting."""

from __future__ import annotations

import torch
from torch import nn


def attention_heads(channels: int, requested: int) -> int:
    heads = max(1, min(requested, channels))
    while channels % heads != 0 and heads > 1:
        heads -= 1
    return heads


class CrossModalAttention(nn.Module):
    """Attention across modality tokens. Spatial maps stay outside this module."""

    def __init__(self, channels: int, num_heads: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.last_weights: torch.Tensor | None = None

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(tokens)
        mixed, weights = self.attention(normalized, normalized, normalized, need_weights=True, average_attn_weights=True)
        self.last_weights = weights.detach()
        return mixed


class ModalityFusion(nn.Module):
    """Fuse a stack of modality features shaped (batch, modalities, channels, D, H, W)."""

    def __init__(self, channels: int, mode: str, num_heads: int) -> None:
        super().__init__()
        if mode not in {"weighted_sum", "cross_attention"}:
            raise ValueError(f"Unknown fusion mode: {mode}")
        self.mode = mode
        hidden = max(channels // 2, 1)
        self.weight_mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Linear(hidden, 1),
        )
        self.cross = CrossModalAttention(channels, attention_heads(channels, num_heads)) if mode == "cross_attention" else None
        self.gate = nn.Linear(channels, channels) if mode == "cross_attention" else None
        self.last_modality_weights: torch.Tensor | None = None

    def forward(self, stack: torch.Tensor, modality_mask: torch.Tensor) -> torch.Tensor:
        pooled = stack.flatten(3).mean(dim=-1)
        logits = self.weight_mlp(pooled).squeeze(-1)
        logits = logits.masked_fill(modality_mask <= 0, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        self.last_modality_weights = weights.detach()
        fused_stack = stack
        if self.cross is not None and self.gate is not None:
            context = self.cross(pooled)
            gates = torch.sigmoid(self.gate(context))
            fused_stack = stack * gates[:, :, :, None, None, None]
        return (fused_stack * weights[:, :, None, None, None, None]).sum(dim=1)


def apply_modality_dropout(modality_mask: torch.Tensor, probability: float) -> torch.Tensor:
    """Drop whole modalities. At least one present modality is kept."""
    if probability <= 0:
        return modality_mask
    keep = torch.bernoulli(torch.full_like(modality_mask, 1.0 - probability))
    present = modality_mask > 0
    keep = keep * present.to(keep.dtype)
    missing_all = keep.sum(dim=1) == 0
    for index in torch.where(missing_all)[0].tolist():
        choices = torch.where(present[index])[0]
        if len(choices) == 0:
            continue
        chosen = int(choices[torch.randint(len(choices), (1,), device=choices.device)])
        keep[index, chosen] = 1.0
    return keep
