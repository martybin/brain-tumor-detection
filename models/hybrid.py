"""Hybrid CNN encoder, window-attention bottleneck, and multimodal fusion."""

from __future__ import annotations

import torch
from torch import nn

from models.decoders.unet_decoder import UNetDecoder3D
from models.encoders.cnn3d import CNN3DEncoder
from models.encoders.swin3d import WindowAttention3D
from models.fusion.cross_attention import ModalityFusion, apply_modality_dropout
from models.heads.heads import ClassificationHead, SegmentationHead, SurvivalHead


class HybridCNNTransformer(nn.Module):
    def __init__(
        self,
        in_modalities: int = 4,
        channels: tuple[int, ...] = (8, 16, 32, 64),
        seg_channels: int = 3,
        use_swin_bottleneck: bool = True,
        swin_heads: int = 4,
        swin_window: int = 4,
        fusion: str = "cross_attention",
        modality_dropout: float = 0.1,
        feature_dropout: float = 0.1,
        num_classes: int | None = None,
        molecular_classes: int | None = None,
        survival: bool = False,
    ) -> None:
        super().__init__()
        self.seg_channels = seg_channels
        self.modality_dropout_p = float(modality_dropout)
        self.encoder = CNN3DEncoder(1, channels, dropout=feature_dropout)
        self.fusions = nn.ModuleList(
            ModalityFusion(count, fusion, swin_heads) for count in channels
        )
        self.swin = (
            WindowAttention3D(channels[-1], swin_heads, swin_window) if use_swin_bottleneck else None
        )
        self.decoder = UNetDecoder3D(channels, dropout=feature_dropout)
        self.seg_head = SegmentationHead(channels[0], seg_channels)
        self.cls_head = ClassificationHead(channels[-1], num_classes) if num_classes else None
        self.molecular_head = ClassificationHead(channels[-1], molecular_classes) if molecular_classes else None
        self.survival_head = SurvivalHead(channels[-1]) if survival else None

    def forward(self, x: torch.Tensor, modality_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if x.ndim != 5:
            raise ValueError(f"Expected (batch, modalities, D, H, W), got {tuple(x.shape)}")
        batch, modalities, _, _, _ = x.shape
        if modality_mask is None:
            modality_mask = torch.ones(batch, modalities, device=x.device, dtype=x.dtype)
        else:
            modality_mask = modality_mask.to(device=x.device, dtype=x.dtype)
        if self.training:
            modality_mask = apply_modality_dropout(modality_mask, self.modality_dropout_p)

        per_modality: list[list[torch.Tensor]] = []
        for index in range(modalities):
            volume = x[:, index : index + 1] * modality_mask[:, index, None, None, None, None]
            per_modality.append(self.encoder(volume))

        fused: list[torch.Tensor] = []
        for scale, fusion in enumerate(self.fusions):
            stack = torch.stack([features[scale] for features in per_modality], dim=1)
            stack = stack * modality_mask[:, :, None, None, None, None]
            fused.append(fusion(stack, modality_mask))
        if self.swin is not None:
            fused[-1] = self.swin(fused[-1])

        decoded = self.decoder(fused)
        outputs: dict[str, torch.Tensor] = {"seg": self.seg_head(decoded)}
        if self.cls_head is not None:
            outputs["cls"] = self.cls_head(fused[-1])
        if self.molecular_head is not None:
            outputs["molecular"] = self.molecular_head(fused[-1])
        if self.survival_head is not None:
            outputs["survival"] = self.survival_head(fused[-1])
        return outputs

    def attention_maps(self) -> list[torch.Tensor]:
        maps = []
        for fusion in self.fusions:
            if fusion.cross is not None and fusion.cross.last_weights is not None:
                maps.append(fusion.cross.last_weights)
        return maps
