"""Segmentation and survival losses.

Region targets overlap, so segmentation uses sigmoid Dice and binary
cross-entropy. Softmax is not used on whole-tumor, tumor-core, and
enhancing-tumor channels.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from preprocessing.errors import MissingLabelError


def soft_dice_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    dice_weight: float = 1.0,
    ce_weight: float = 1.0,
    eps: float = 1.0,
) -> torch.Tensor:
    if logits.shape != target.shape:
        raise ValueError(f"Segmentation shape mismatch: {tuple(logits.shape)} vs {tuple(target.shape)}")
    probs = torch.sigmoid(logits)
    dims = tuple(range(2, probs.ndim))
    intersection = (probs * target).sum(dim=dims)
    denom = probs.sum(dim=dims) + target.sum(dim=dims)
    # smooth=1 keeps an empty target and a near-zero prediction at Dice 1.
    dice = (2 * intersection + eps) / (denom + eps)
    dice_loss = 1.0 - dice.mean()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    return dice_weight * dice_loss + ce_weight * bce


def masked_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = target >= 0
    if not torch.any(valid):
        raise MissingLabelError("No classification labels in this batch.")
    return F.cross_entropy(logits[valid], target[valid].long())


def cox_ph_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """Negative Breslow partial log-likelihood. Ties share the risk set."""
    observed = event > 0
    if not torch.any(observed):
        raise MissingLabelError("No observed survival events in this batch.")
    order = torch.argsort(time, descending=True)
    ordered_risk = risk[order]
    ordered_event = event[order].float()
    log_risk_set = torch.logcumsumexp(ordered_risk, dim=0)
    partial = ordered_risk - log_risk_set
    return -(partial * ordered_event).sum() / ordered_event.sum().clamp_min(1.0)


class MultiTaskLoss(nn.Module):
    def __init__(self, dice_weight: float = 1.0, ce_weight: float = 1.0) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        total = torch.zeros((), device=_device(outputs), dtype=torch.float32)
        parts: dict[str, float] = {}
        if "seg" in outputs:
            if "target" not in batch:
                raise MissingLabelError("Segmentation logits have no target.")
            seg = soft_dice_bce(outputs["seg"], batch["target"], self.dice_weight, self.ce_weight)
            total = total + seg
            parts["seg"] = float(seg.detach())
        if "cls" in outputs:
            cls = masked_cross_entropy(outputs["cls"], batch["cls_label"])
            total = total + cls
            parts["cls"] = float(cls.detach())
        if "molecular" in outputs:
            molecular = masked_cross_entropy(outputs["molecular"], batch["molecular_label"])
            total = total + molecular
            parts["molecular"] = float(molecular.detach())
        if "survival" in outputs:
            survival = cox_ph_loss(outputs["survival"], batch["survival_time"], batch["survival_event"])
            total = total + survival
            parts["survival"] = float(survival.detach())
        if not parts:
            raise MissingLabelError("The model returned no trainable task.")
        return total, parts


def _device(outputs: dict[str, torch.Tensor]) -> torch.device:
    return next(iter(outputs.values())).device
