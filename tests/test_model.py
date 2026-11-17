"""Model forward and one CPU training step. Skipped when PyTorch is absent."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from models.factory import build_model
from preprocessing.errors import ConfigurationError, MissingLabelError
from training.engine import run_epoch
from training.losses import MultiTaskLoss, soft_dice_bce
from explainability.gradcam import GradCAM


class _Loader:
    def __init__(self, batches: list[dict]) -> None:
        self.batches = batches

    def __len__(self) -> int:
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


def _model() -> torch.nn.Module:
    return build_model(
        {
            "name": "hybrid",
            "encoder_channels": [4, 8],
            "seg_channels": 3,
            "use_swin_bottleneck": True,
            "swin_heads": 4,
            "swin_window": 4,
            "fusion": "cross_attention",
            "modality_dropout": 0.0,
            "feature_dropout": 0.0,
            "classification": {"enabled": False},
            "molecular": {"enabled": False},
            "survival": {"enabled": False},
        }
    )


def test_hybrid_forward_and_one_step() -> None:
    torch.manual_seed(0)
    model = _model()
    image = torch.randn(1, 4, 8, 8, 8)
    target = torch.zeros(1, 3, 8, 8, 8)
    target[:, 0, 2:5, 2:5, 2:5] = 1
    outputs = model(image)
    assert outputs["seg"].shape == (1, 3, 8, 8, 8)
    assert "cls" not in outputs
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = MultiTaskLoss()
    loader = _Loader([{"image": image, "target": target}])
    loss = run_epoch(
        model,
        loader,  # type: ignore[arg-type]
        loss_fn,
        optimizer,
        torch.device("cpu"),
        train=True,
        accum_steps=1,
        grad_clip=1.0,
        scaler=__import__("training.amp", fromlist=["make_scaler"]).make_scaler(False),
        amp=False,
    )
    assert loss > 0


def test_empty_region_loss_is_small_and_missing_class_label_errors() -> None:
    logits = torch.full((1, 1, 4, 4, 4), -20.0)
    target = torch.zeros_like(logits)
    assert float(soft_dice_bce(logits, target)) < 0.05
    model = _model()
    model.cls_head = torch.nn.Linear(8, 2)
    # The segmentation loss path is covered above. A batch with no class ids must fail.
    with pytest.raises(MissingLabelError):
        MultiTaskLoss()({"cls": torch.zeros(2, 2)}, {"cls_label": torch.tensor([-1, -1])})


def test_nnunet_is_not_a_local_network() -> None:
    with pytest.raises(ConfigurationError):
        build_model({"name": "nnunet"})


def test_gradcam_returns_an_input_sized_map() -> None:
    torch.manual_seed(0)
    model = _model()
    model.eval()
    image = torch.randn(1, 4, 8, 8, 8)
    layer = model.decoder.cam_layer
    cam = GradCAM(model, layer)
    heatmap = cam(image, lambda seg: seg[:, 0].sum())
    cam.close()
    assert heatmap.shape == (8, 8, 8)
    assert np_isfinite(heatmap)


def np_isfinite(values) -> bool:
    import numpy as np

    return bool(np.isfinite(values).all())
