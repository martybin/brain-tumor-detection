"""Training and validation loops.

Validation during training reports patch Dice. Volume Dice is computed by
scripts/evaluate.py. Neither number is written unless the loader ran.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluation.segmentation_metrics import dice_score
from models.factory import build_model
from preprocessing.adapters.registry import get_schema
from preprocessing.errors import ConfigurationError, LabelSchemaError
from preprocessing.harmonization import segmentation_spec, task_enabled
from preprocessing.splitting import assert_no_overlap, patient_level_split
from training.amp import autocast, make_scaler
from training.callbacks import EarlyStopping
from training.checkpoint import save_checkpoint
from training.config import load_yaml, project_root, require, resolve_path
from training.dataset import PatchDataset
from training.log import get_logger
from training.losses import MultiTaskLoss
from training.optim import build_optimizer, cosine_lr, set_lr
from training.seed import set_seed

LOGGER = get_logger("train")


def run_training(config_path: str | Path) -> Path:
    config = load_yaml(config_path)
    seed = int(require(config, "seed"))
    set_seed(seed)
    schema = get_schema(str(require(config, "dataset")))
    model_config = load_yaml(resolve_path(require(config, "model_config")))
    _assert_tasks(schema, model_config)
    if not task_enabled(schema, "segmentation"):
        raise LabelSchemaError(
            f"{schema['name']} has no confirmed segmentation labels. Training is refused."
        )
    regions = list(segmentation_spec(schema)["regions"])
    if len(regions) != int(model_config.get("seg_channels", model_config.get("out_channels", -1))):
        raise ConfigurationError(
            f"Model segmentation channels do not match confirmed regions {regions}."
        )

    manifest = resolve_path(config["data"]["processed_root"]) / schema["name"] / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"Prepared manifest not found: {manifest}. Run scripts/prepare_dataset.py first."
        )
    splits = _load_or_make_splits(config, schema["name"], manifest, seed)
    patch = tuple(int(v) for v in config["data"]["patch_size"])
    train_ds = PatchDataset(
        manifest,
        splits["train"],
        patch,
        int(config["data"]["samples_per_volume"]),
        float(config["data"]["pos_fraction"]),
        seed,
        augment=True,
    )
    val_ds = PatchDataset(
        manifest,
        splits["val"],
        patch,
        samples_per_volume=1,
        pos_fraction=1.0,
        seed=seed,
        augment=False,
    )
    loader_kwargs = {
        "batch_size": int(config["data"]["batch_size"]),
        "num_workers": int(config["data"].get("num_workers", 0)),
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)

    device = _device(str(config.get("device", "auto")))
    model = build_model(model_config).to(device)
    optimizer = build_optimizer(
        model,
        str(config["optim"]["name"]),
        float(config["optim"]["lr"]),
        float(config["optim"]["weight_decay"]),
    )
    criterion = MultiTaskLoss(float(config["loss"]["dice_weight"]), float(config["loss"]["ce_weight"]))
    stopper = EarlyStopping(int(config["early_stopping"]["patience"]), str(config["early_stopping"]["mode"]))
    scaler = make_scaler(bool(config["optim"]["amp"]))
    writer = _start_logger(config)
    checkpoint_dir = resolve_path(config["checkpoint"]["dir"]) / schema["name"] / model_config["name"]
    best_path = checkpoint_dir / "best.pt"
    best_metric = None

    epochs = int(require(config, "epochs"))
    for epoch in range(epochs):
        lr = cosine_lr(epoch, epochs, int(config["scheduler"].get("warmup_epochs", 0)), float(config["optim"]["lr"]))
        set_lr(optimizer, lr)
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            train=True,
            accum_steps=int(config["optim"]["accum_steps"]),
            grad_clip=float(config["optim"]["grad_clip"]),
            scaler=scaler,
            amp=bool(config["optim"]["amp"]),
        )
        val_loss, patch_dice = run_epoch(
            model,
            val_loader,
            criterion,
            optimizer,
            device,
            train=False,
            accum_steps=1,
            grad_clip=0.0,
            scaler=scaler,
            amp=bool(config["optim"]["amp"]),
        )
        record = {"epoch": epoch, "lr": lr, "train_loss": train_loss, "val_loss": val_loss, "patch_dice": patch_dice}
        LOGGER.info("epoch %s train_loss=%.4f val_loss=%.4f patch_dice=%.4f", epoch, train_loss, val_loss, patch_dice)
        if writer is not None:
            for key, value in record.items():
                writer.add_scalar(key, value, epoch)
        improved = best_metric is None or patch_dice > best_metric
        if improved:
            best_metric = patch_dice
            save_checkpoint(
                best_path,
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "metrics": {"patch_dice": patch_dice, "val_loss": val_loss},
                    "model_config": model_config,
                    "dataset": schema["name"],
                    "region_order": regions,
                    "seed": seed,
                    "config": config,
                },
            )
        save_checkpoint(checkpoint_dir / "last.pt", {"model_state": model.state_dict(), "epoch": epoch, "model_config": model_config, "dataset": schema["name"], "region_order": regions, "seed": seed, "metrics": record, "config": config})
        if stopper.step(patch_dice):
            LOGGER.info("Early stopping at epoch %s. Best patch_dice=%.4f", epoch, best_metric)
            break
    if writer is not None:
        writer.close()
    if best_metric is None or not best_path.is_file():
        raise RuntimeError("Training finished without a computed validation metric.")
    return best_path


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: MultiTaskLoss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
    accum_steps: int,
    grad_clip: float,
    scaler: torch.amp.GradScaler,
    amp: bool,
) -> tuple[float, float] | float:
    model.train(train)
    if len(loader) == 0:
        raise RuntimeError("The data loader is empty.")
    losses: list[float] = []
    dice_values: list[float] = []
    if train and optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        image = batch["image"].to(device)
        target = batch["target"].to(device)
        with autocast(device, amp):
            outputs = model(image)
            loss, _ = criterion(outputs, {"target": target})
            scaled = loss / accum_steps if train else loss
        if train and optimizer is not None:
            scaler.scale(scaled).backward()
            if (step + 1) % accum_steps == 0 or step + 1 == len(loader):
                scaler.unscale_(optimizer)
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach()))
        if not train:
            probs = torch.sigmoid(outputs["seg"]).detach().cpu().numpy()
            truth = target.detach().cpu().numpy()
            dice_values.append(_batch_dice(probs, truth))
    mean_loss = float(np.mean(losses))
    if train:
        return mean_loss
    return mean_loss, float(np.mean(dice_values))


def _batch_dice(probs: np.ndarray, target: np.ndarray) -> float:
    scores = []
    pred = probs >= 0.5
    for item_pred, item_true in zip(pred, target):
        for channel_pred, channel_true in zip(item_pred, item_true):
            scores.append(dice_score(channel_pred, channel_true > 0.5))
    finite = [score for score in scores if np.isfinite(score)]
    if not finite:
        raise RuntimeError("Validation produced no finite Dice scores.")
    return float(np.mean(finite))


def _assert_tasks(schema: dict[str, Any], model_config: dict[str, Any]) -> None:
    mapping = {
        "classification": "classification",
        "molecular": "molecular",
        "survival": "survival",
    }
    for task, key in mapping.items():
        enabled = bool((model_config.get(key) or {}).get("enabled", False))
        available = task_enabled(schema, task)
        if enabled != available:
            raise LabelSchemaError(
                f"Model {key}.enabled={enabled} but {schema['name']} confirmed {task} labels={available}."
            )


def _load_or_make_splits(config: dict[str, Any], dataset: str, manifest: Path, seed: int) -> dict[str, list[str]]:
    path = resolve_path(config["data"]["splits_root"]) / dataset / "splits.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        splits = {key: list(payload[key]) for key in ("train", "val", "test")}
        assert_no_overlap(splits)
        return splits
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    patients = [case["patient_id"] for case in payload["cases"]]
    splits = patient_level_split(
        patients,
        {name: float(config["split"][name]) for name in ("train", "val", "test")},
        seed,
    )
    assert_no_overlap(splits)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"dataset": dataset, "seed": seed, **splits}, indent=2), encoding="utf-8")
    LOGGER.info("Wrote patient-level split %s", path)
    return splits


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _start_logger(config: dict[str, Any]):
    backend = config.get("log", {}).get("backend", "tensorboard")
    log_dir = resolve_path(config.get("log", {}).get("dir", "runs"))
    if backend == "tensorboard":
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            LOGGER.warning("TensorBoard is not installed. Metrics stay in the console and checkpoints.")
            return None
        return SummaryWriter(log_dir=str(log_dir))
    if backend == "mlflow":
        try:
            import mlflow
        except ImportError:
            LOGGER.warning("MLflow is not installed. Metrics stay in the console and checkpoints.")
            return None
        mlflow.set_tracking_uri(str(log_dir / "mlruns"))
        mlflow.start_run()
        return _MLflowWriter()
    raise ConfigurationError(f"Unknown log backend: {backend}")


class _MLflowWriter:
    def add_scalar(self, key: str, value: float, step: int) -> None:
        import mlflow

        mlflow.log_metric(key, value, step=step)

    def close(self) -> None:
        import mlflow

        mlflow.end_run()
