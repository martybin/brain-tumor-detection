"""Model factory. nnU-Net stays an external baseline."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from models.hybrid import HybridCNNTransformer
from preprocessing.errors import ConfigurationError


class BaselineWrapper(nn.Module):
    """MONAI networks take stacked channels shaped (batch, modalities, D, H, W)."""

    def __init__(self, network: nn.Module, seg_channels: int) -> None:
        super().__init__()
        self.network = network
        self.seg_channels = seg_channels

    def forward(self, x: torch.Tensor, modality_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if modality_mask is not None:
            x = x * modality_mask[:, :, None, None, None].to(dtype=x.dtype)
        return {"seg": self.network(x)}


def build_model(config: dict[str, Any]) -> nn.Module:
    name = config.get("name")
    if name == "hybrid":
        classification = config.get("classification") or {}
        molecular = config.get("molecular") or {}
        survival = config.get("survival") or {}
        channels = tuple(int(v) for v in config.get("encoder_channels") or [8, 16, 32, 64])
        return HybridCNNTransformer(
            in_modalities=int(config.get("in_modalities", 4)),
            channels=channels,
            seg_channels=int(config.get("seg_channels", 3)),
            use_swin_bottleneck=bool(config.get("use_swin_bottleneck", True)),
            swin_heads=int(config.get("swin_heads", 4)),
            swin_window=int(config.get("swin_window", 4)),
            fusion=str(config.get("fusion", "cross_attention")),
            modality_dropout=float(config.get("modality_dropout", 0.0)),
            feature_dropout=float(config.get("feature_dropout", 0.0)),
            num_classes=classification.get("num_classes") if classification.get("enabled") else None,
            molecular_classes=molecular.get("num_classes") if molecular.get("enabled") else None,
            survival=bool(survival.get("enabled", False)),
        )
    if name == "unet3d":
        return BaselineWrapper(_monai_unet(config), int(config["out_channels"]))
    if name == "swin_unetr":
        return BaselineWrapper(_monai_swin(config), int(config["out_channels"]))
    if name == "nnunet":
        raise ConfigurationError(
            "nnU-Net is an external baseline. Use scripts/export_nnunet.py and run nnU-Net itself. "
            "This repository does not invent nnU-Net metrics."
        )
    raise ConfigurationError(f"Unknown model name: {name}")


def _monai_unet(config: dict[str, Any]) -> nn.Module:
    try:
        from monai.networks.nets import UNet
    except ImportError as exc:
        raise ConfigurationError("MONAI is required for the 3D U-Net baseline.") from exc
    return UNet(
        spatial_dims=3,
        in_channels=int(config["in_channels"]),
        out_channels=int(config["out_channels"]),
        channels=tuple(config["channels"]),
        strides=tuple(config["strides"]),
        num_res_units=int(config.get("num_res_units", 2)),
        dropout=float(config.get("dropout", 0.0)),
        norm=config.get("norm", "batch"),
    )


def _monai_swin(config: dict[str, Any]) -> nn.Module:
    try:
        from monai.networks.nets import SwinUNETR
    except ImportError as exc:
        raise ConfigurationError("MONAI is required for the Swin UNETR baseline.") from exc
    return SwinUNETR(
        in_channels=int(config["in_channels"]),
        out_channels=int(config["out_channels"]),
        feature_size=int(config.get("feature_size", 24)),
        depths=tuple(config.get("depths", [2, 2, 2, 2])),
        num_heads=tuple(config.get("num_heads", [3, 6, 12, 24])),
        use_checkpoint=bool(config.get("use_checkpoint", True)),
        spatial_dims=int(config.get("spatial_dims", 3)),
    )
