"""Grad-CAM for a 3D convolution. The target layer must be a spatial conv."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from torch import nn
from scipy.ndimage import zoom


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()

    def _save_activation(self, _module, _inputs, output) -> None:
        self.activations = output

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0]

    def __call__(self, image: torch.Tensor, score_fn: Callable[[torch.Tensor], torch.Tensor]) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        outputs = self.model(image)
        score = score_fn(outputs["seg"])
        score.backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM did not capture activations. The target layer did not run.")
        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1))[0]
        cam_np = cam.detach().cpu().numpy()
        spatial = tuple(int(v) for v in image.shape[-3:])
        factors = [dst / src for dst, src in zip(spatial, cam_np.shape)]
        resized = zoom(cam_np, factors, order=1)
        peak = float(resized.max())
        if peak > 0:
            resized = resized / peak
        return resized.astype(np.float32)
