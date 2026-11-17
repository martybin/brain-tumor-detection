"""Monte Carlo dropout probabilities. Batch-norm stays in eval."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def mc_dropout_segmentation(model: nn.Module, image: torch.Tensor, passes: int) -> np.ndarray:
    if passes < 2:
        raise ValueError("Uncertainty from dropout needs at least two passes.")
    was_training = model.training
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout3d)):
            module.train()
    samples = []
    with torch.no_grad():
        for _ in range(passes):
            logits = model(image)["seg"]
            samples.append(torch.sigmoid(logits).cpu().numpy())
    model.train(was_training)
    return np.stack(samples, axis=0)
