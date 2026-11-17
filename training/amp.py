"""Mixed-precision helpers that tolerate older and newer PyTorch AMP APIs."""

from __future__ import annotations

import torch


def autocast(device: torch.device, enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")
    return torch.cuda.amp.autocast(enabled=enabled and device.type == "cuda")


def make_scaler(enabled: bool):
    use = enabled and torch.cuda.is_available()
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=use)
        except TypeError:
            return torch.cuda.amp.GradScaler(enabled=use)
    return torch.cuda.amp.GradScaler(enabled=use)
