"""Input attribution and figure writers. Figures go to the output path, never to raw data."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


def integrated_gradients(
    model: nn.Module,
    image: torch.Tensor,
    target_channel: int,
    steps: int = 8,
) -> np.ndarray:
    """Attribute the summed logit of one segmentation channel to the input modalities."""
    baseline = torch.zeros_like(image)
    total = torch.zeros_like(image)
    for alpha in torch.linspace(0.0, 1.0, steps, device=image.device):
        point = (baseline + alpha * (image - baseline)).detach().requires_grad_(True)
        logits = model(point)["seg"]
        score = logits[:, target_channel].sum()
        grad = torch.autograd.grad(score, point)[0]
        total = total + grad
    attribution = (image - baseline) * (total / steps)
    return attribution.detach().cpu().numpy()


def save_overlay(image2d: np.ndarray, mask2d: np.ndarray | None, path: Path, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5, 5), dpi=150)
    axis.imshow(np.asarray(image2d).T, cmap="gray", origin="lower")
    if mask2d is not None and np.any(mask2d):
        shown = np.ma.masked_where(np.asarray(mask2d) <= 0, np.asarray(mask2d))
        axis.imshow(shown.T, cmap="autumn", alpha=0.45, origin="lower")
    axis.set_axis_off()
    if title:
        axis.set_title(title)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def save_matrix(matrix: np.ndarray, labels: list[str], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(4.5, 4), dpi=150)
    image = axis.imshow(matrix, cmap="magma")
    axis.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    axis.set_yticks(range(len(labels)), labels)
    axis.set_title(title)
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)


def save_uncertainty_slice(entropy2d: np.ndarray, path: Path, title: str = "Predictive entropy") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(5, 5), dpi=150)
    image = axis.imshow(np.asarray(entropy2d).T, cmap="viridis", origin="lower")
    axis.set_axis_off()
    axis.set_title(title)
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path)
    plt.close(figure)
