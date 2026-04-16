"""Evaluation helpers for surrogate checkpoints."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .models import forward_residual_mlp
from .train import load_dataset


def load_checkpoint(checkpoint_path: Path) -> dict[str, object]:
    with checkpoint_path.open("rb") as handle:
        return pickle.load(handle)


def predict_from_checkpoint(checkpoint: dict[str, object], features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float64)
    x_norm = (x - checkpoint["x_mean"]) / checkpoint["x_std"]
    prediction = np.asarray(forward_residual_mlp(checkpoint["params"], x_norm), dtype=np.float64)
    return prediction * checkpoint["y_std"] + checkpoint["y_mean"]


def _peak_positions(curves: np.ndarray, bias: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positive_mask = bias >= 0.0
    negative_mask = bias <= 0.0
    pos_bias = bias[positive_mask]
    neg_bias = bias[negative_mask]
    pos_index = np.argmax(curves[:, positive_mask], axis=1)
    neg_index = np.argmax(curves[:, negative_mask], axis=1)
    return pos_bias[pos_index], neg_bias[neg_index]


def evaluate_checkpoint(dataset_path: Path, checkpoint_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_dataset(dataset_path)
    checkpoint = load_checkpoint(checkpoint_path)

    features = np.asarray(data["features"], dtype=np.float64)
    targets = np.asarray(data["spectra"], dtype=np.float64)
    bias = np.asarray(data["bias"], dtype=np.float64)
    test_idx = np.asarray(data["test_idx"], dtype=np.int64)
    categories = np.asarray(data["categories"])

    prediction = predict_from_checkpoint(checkpoint, features[test_idx])
    truth = targets[test_idx]
    diff = prediction - truth

    mse = float(np.mean(diff**2))
    derivative_mse = float(np.mean((np.diff(prediction, axis=1) - np.diff(truth, axis=1)) ** 2))
    zero_index = int(np.argmin(np.abs(bias)))
    zero_bias_error = float(np.mean(np.abs(prediction[:, zero_index] - truth[:, zero_index])))
    pred_pos_peak, pred_neg_peak = _peak_positions(prediction, bias)
    true_pos_peak, true_neg_peak = _peak_positions(truth, bias)
    peak_position_error = float(
        np.mean(np.abs(pred_pos_peak - true_pos_peak) + np.abs(pred_neg_peak - true_neg_peak)) / 2.0
    )
    dynamic_range_error = float(
        np.mean(np.abs((prediction.max(axis=1) - prediction.min(axis=1)) - (truth.max(axis=1) - truth.min(axis=1))))
    )
    low_bias_mask = np.abs(bias) <= 5.0
    peak_region_mask = (np.abs(bias) >= 5.0) & (np.abs(bias) <= 20.0)
    low_bias_mse = float(np.mean((prediction[:, low_bias_mask] - truth[:, low_bias_mask]) ** 2))
    peak_region_mse = float(np.mean((prediction[:, peak_region_mask] - truth[:, peak_region_mask]) ** 2))

    per_sample_mse = np.mean(diff**2, axis=1)
    worst_local = int(np.argmax(per_sample_mse))
    median_local = int(np.argsort(per_sample_mse)[len(per_sample_mse) // 2])

    def _plot_pair(local_index: int, output_path: Path, title: str) -> None:
        figure, axis = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
        axis.plot(bias, truth[local_index], label="physics", linewidth=1.7)
        axis.plot(bias, prediction[local_index], label="surrogate", linewidth=1.5)
        axis.set_title(title)
        axis.set_xlabel("Bias (meV)")
        axis.set_ylabel("Conductance")
        axis.grid(alpha=0.2)
        axis.legend()
        figure.savefig(output_path, dpi=160)
        plt.close(figure)

    _plot_pair(worst_local, output_dir / "worst_sample.png", "Worst Test Sample")
    _plot_pair(median_local, output_dir / "representative_sample.png", "Representative Test Sample")

    heldout_mask = categories[test_idx] == "A_luo_anchor"
    heldout_metrics = {
        "num_samples": int(np.count_nonzero(heldout_mask)),
        "mse": None,
        "low_bias_mse": None,
    }
    if np.any(heldout_mask):
        heldout_metrics["mse"] = float(np.mean((prediction[heldout_mask] - truth[heldout_mask]) ** 2))
        heldout_metrics["low_bias_mse"] = float(
            np.mean((prediction[heldout_mask][:, low_bias_mask] - truth[heldout_mask][:, low_bias_mask]) ** 2)
        )

    report = {
        "test_mse": mse,
        "derivative_mse": derivative_mse,
        "zero_bias_error": zero_bias_error,
        "peak_position_error": peak_position_error,
        "dynamic_range_error": dynamic_range_error,
        "low_bias_mse": low_bias_mse,
        "peak_region_mse": peak_region_mse,
        "luo_source_like_heldout": heldout_metrics,
    }
    report_path = output_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path
