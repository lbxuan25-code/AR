"""Training utilities for the pairing+transport surrogate."""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict
from pathlib import Path

import autograd.numpy as anp
import numpy as np
from autograd import grad

from .config import TrainConfig
from .models import forward_residual_mlp, initialize_residual_mlp, spec_from_params


def load_dataset(npz_path: Path) -> dict[str, np.ndarray]:
    payload = np.load(npz_path, allow_pickle=False)
    return {key: np.asarray(payload[key]) for key in payload.files}


def _normalize(train: np.ndarray, full: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    std = np.where(std < 1.0e-8, 1.0, std)
    return (full - mean) / std, mean, std


def _bias_weights(bias: np.ndarray, config: TrainConfig) -> np.ndarray:
    bias_abs = np.abs(np.asarray(bias, dtype=np.float64))
    weights = np.ones_like(bias_abs)
    weights += config.low_bias_weight * (bias_abs <= 5.0)
    weights += config.peak_region_weight * ((bias_abs >= 5.0) & (bias_abs <= 20.0))
    return np.asarray(weights / np.mean(weights), dtype=np.float64)


def _loss_components(
    params: list[np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    bias_weights: np.ndarray,
    derivative_weight: float,
    weight_decay: float,
) -> tuple[anp.ndarray, anp.ndarray, anp.ndarray]:
    prediction = forward_residual_mlp(params, x)
    diff = prediction - y
    weighted_mse = anp.mean(diff**2 * bias_weights[None, :])
    derivative_mse = anp.mean((anp.diff(prediction, axis=1) - anp.diff(y, axis=1)) ** 2)
    reg = weight_decay * anp.sum(anp.concatenate([p.ravel() for p in params]) ** 2)
    total = weighted_mse + derivative_weight * derivative_mse + reg
    return total, weighted_mse, derivative_mse


def _evaluate(
    params: list[np.ndarray],
    x: np.ndarray,
    y: np.ndarray,
    bias_weights: np.ndarray,
    config: TrainConfig,
) -> tuple[float, dict[str, float]]:
    total, weighted_mse, derivative_mse = _loss_components(
        params,
        x,
        y,
        bias_weights,
        config.derivative_weight,
        config.weight_decay,
    )
    return float(total), {
        "weighted_mse": float(weighted_mse),
        "derivative_mse": float(derivative_mse),
        "total_loss": float(total),
    }


def _adam_update(
    params: list[np.ndarray],
    grads: list[np.ndarray],
    m: list[np.ndarray],
    v: list[np.ndarray],
    step: int,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1.0e-8,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    new_params: list[np.ndarray] = []
    for index, (param, grad_value) in enumerate(zip(params, grads, strict=True)):
        m[index] = beta1 * m[index] + (1.0 - beta1) * grad_value
        v[index] = beta2 * v[index] + (1.0 - beta2) * (grad_value**2)
        m_hat = m[index] / (1.0 - beta1**step)
        v_hat = v[index] / (1.0 - beta2**step)
        new_params.append(param - learning_rate * m_hat / (np.sqrt(v_hat) + eps))
    return new_params, m, v


def train_surrogate(
    dataset_path: Path,
    output_dir: Path,
    config: TrainConfig,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_dataset(dataset_path)

    x = np.asarray(data["features"], dtype=np.float64)
    y = np.asarray(data["spectra"], dtype=np.float64)
    bias = np.asarray(data["bias"], dtype=np.float64)
    train_idx = np.asarray(data["train_idx"], dtype=np.int64)
    val_idx = np.asarray(data["val_idx"], dtype=np.int64)
    test_idx = np.asarray(data["test_idx"], dtype=np.int64)

    x_norm, x_mean, x_std = _normalize(x[train_idx], x)
    y_norm, y_mean, y_std = _normalize(y[train_idx], y)
    bias_weights = _bias_weights(bias, config)

    params = initialize_residual_mlp(
        in_features=x.shape[1],
        out_features=y.shape[1],
        hidden_dim=config.hidden_dim,
        num_blocks=config.num_blocks,
        seed=config.seed,
    )
    m = [np.zeros_like(param) for param in params]
    v = [np.zeros_like(param) for param in params]

    grad_fn = grad(lambda p, xb, yb: _loss_components(p, xb, yb, bias_weights, config.derivative_weight, config.weight_decay)[0])

    best_val = float("inf")
    best_epoch = -1
    best_state = [np.array(param, copy=True) for param in params]
    patience_left = config.patience
    history: list[dict[str, float]] = []

    for epoch in range(config.max_epochs):
        grads = grad_fn(params, x_norm[train_idx], y_norm[train_idx])
        params, m, v = _adam_update(params, grads, m, v, epoch + 1, config.learning_rate)

        train_loss, _ = _evaluate(params, x_norm[train_idx], y_norm[train_idx], bias_weights, config)
        val_loss, val_metrics = _evaluate(params, x_norm[val_idx], y_norm[val_idx], bias_weights, config)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, **val_metrics})

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = [np.array(param, copy=True) for param in params]
            patience_left = config.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    params = best_state
    test_loss, test_metrics = _evaluate(params, x_norm[test_idx], y_norm[test_idx], bias_weights, config)
    checkpoint = {
        "params": params,
        "spec": asdict(spec_from_params(params)),
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "bias": bias,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "config": config.to_dict(),
    }

    checkpoint_path = output_dir / "surrogate_checkpoint.pkl"
    with checkpoint_path.open("wb") as handle:
        pickle.dump(checkpoint, handle)

    log_path = output_dir / "train_log.json"
    log_path.write_text(
        json.dumps(
            {
                "config": config.to_dict(),
                "best_epoch": best_epoch,
                "best_val_loss": best_val,
                "test_loss": test_loss,
                "test_metrics": test_metrics,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return checkpoint_path, log_path
