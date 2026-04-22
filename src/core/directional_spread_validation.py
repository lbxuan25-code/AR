"""Validation for Task-P narrow directional-spread primitive."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import math

import matplotlib.pyplot as plt
import numpy as np

from forward import (
    BiasGrid,
    DirectionalSpread,
    FitLayerSpectrumRequest,
    generate_spread_spectrum_from_fit_layer,
    transport_with_direction_mode,
)


DEFAULT_SPREAD_WIDTHS: tuple[float, ...] = (0.0, float(math.pi / 128.0), float(math.pi / 64.0), float(math.pi / 32.0))
DEFAULT_SPREAD_MODES: tuple[str, ...] = ("inplane_100", "inplane_110")
DEFAULT_SPREAD_BARRIERS: tuple[float, ...] = (0.5, 1.0)
DEFAULT_PAIRING_STATES: dict[str, dict[str, float]] = {
    "formal_baseline": {},
    "fit_layer_shifted": {"delta_zz_s": 0.25, "delta_perp_x": -0.1},
}
SPREAD_SMOOTHNESS_MAX_ABS_STEP = 0.25


@dataclass(frozen=True, slots=True)
class DirectionalSpreadValidationArtifacts:
    """Generated Task-P directional-spread validation artifact paths."""

    summary_path: Path
    metrics_csv_path: Path
    plot_path: Path | None


def _json_ready(value: object) -> object:
    if isinstance(value, complex):
        return {"re": float(np.real(value)), "im": float(np.imag(value))}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _curve_features(curve: np.ndarray, bias: np.ndarray) -> dict[str, float]:
    values = np.asarray(curve, dtype=np.float64)
    energies = np.asarray(bias, dtype=np.float64)
    zero_index = int(np.argmin(np.abs(energies)))
    return {
        "zero_bias_conductance": float(values[zero_index]),
        "min_conductance": float(np.min(values)),
        "max_conductance": float(np.max(values)),
        "mean_conductance": float(np.mean(values)),
        "dynamic_range": float(np.max(values) - np.min(values)),
    }


def _row_key(row: dict[str, object]) -> tuple[str, str, float]:
    return (str(row["direction_mode"]), str(row["pairing_state"]), float(row["barrier_z"]))


def _compute_width_smoothness(rows: list[dict[str, object]]) -> None:
    grouped: dict[tuple[str, str, float], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(_row_key(row), []).append(row)
    for group_rows in grouped.values():
        group_rows.sort(key=lambda item: float(item["half_width"]))
        previous_curve: np.ndarray | None = None
        previous_width: float | None = None
        for row in group_rows:
            curve = np.asarray(row["conductance"], dtype=np.float64)
            if previous_curve is None:
                row["previous_half_width"] = None
                row["width_step_mse"] = 0.0
                row["width_step_max_abs"] = 0.0
            else:
                diff = curve - previous_curve
                row["previous_half_width"] = previous_width
                row["width_step_mse"] = float(np.mean(np.square(diff)))
                row["width_step_max_abs"] = float(np.max(np.abs(diff)))
            previous_curve = curve
            previous_width = float(row["half_width"])


def _write_metrics_csv(rows: list[dict[str, object]], path: Path) -> Path:
    fieldnames = [
        "direction_mode",
        "pairing_state",
        "barrier_z",
        "half_width",
        "half_width_pi",
        "num_samples",
        "width_step_mse",
        "width_step_max_abs",
        "zero_bias_conductance",
        "dynamic_range",
        "mean_mismatch_penalty",
        "sample_angle_min",
        "sample_angle_max",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            features = row["spectrum_features"]
            writer.writerow(
                {
                    "direction_mode": row["direction_mode"],
                    "pairing_state": row["pairing_state"],
                    "barrier_z": row["barrier_z"],
                    "half_width": row["half_width"],
                    "half_width_pi": row["half_width_pi"],
                    "num_samples": row["num_samples"],
                    "width_step_mse": row["width_step_mse"],
                    "width_step_max_abs": row["width_step_max_abs"],
                    "zero_bias_conductance": features["zero_bias_conductance"],
                    "dynamic_range": features["dynamic_range"],
                    "mean_mismatch_penalty": row["mean_mismatch_penalty"],
                    "sample_angle_min": row["sample_angle_min"],
                    "sample_angle_max": row["sample_angle_max"],
                }
            )
    return path


def _plot_spread_validation(rows: list[dict[str, object]], output_path: Path) -> Path | None:
    if not rows:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    representative = [
        row
        for row in rows
        if row["direction_mode"] == "inplane_110" and row["pairing_state"] == "formal_baseline" and np.isclose(row["barrier_z"], 0.5)
    ]
    representative.sort(key=lambda item: float(item["half_width"]))
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), constrained_layout=True)
    if representative:
        bias = np.asarray(representative[0]["bias_mev"], dtype=np.float64)
        for row in representative:
            axes[0].plot(bias, row["conductance"], label=f"{row['half_width_pi']:.5f}π")
    axes[0].set_xlabel("Bias (meV)")
    axes[0].set_ylabel("Normalized conductance")
    axes[0].set_title("Spread spectra: inplane_110, baseline, Z=0.5")
    axes[0].legend(title="half width", fontsize=8)

    for mode in sorted({str(row["direction_mode"]) for row in rows}):
        mode_rows = [row for row in rows if row["direction_mode"] == mode]
        widths = np.asarray([row["half_width_pi"] for row in mode_rows], dtype=np.float64)
        steps = np.asarray([row["width_step_max_abs"] for row in mode_rows], dtype=np.float64)
        axes[1].scatter(widths, steps, label=mode, s=28)
    axes[1].axhline(SPREAD_SMOOTHNESS_MAX_ABS_STEP, color="tab:red", linestyle=":", linewidth=1.0)
    axes[1].set_xlabel("half width / pi")
    axes[1].set_ylabel("max |G(width)-G(previous width)|")
    axes[1].set_title("Width-step smoothness")
    axes[1].legend(fontsize=8)

    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def run_directional_spread_validation(
    *,
    output_dir: Path = Path("outputs/core/directional_spread_validation"),
    direction_modes: tuple[str, ...] = DEFAULT_SPREAD_MODES,
    half_widths: tuple[float, ...] = DEFAULT_SPREAD_WIDTHS,
    barriers: tuple[float, ...] = DEFAULT_SPREAD_BARRIERS,
    pairing_states: dict[str, dict[str, float]] = DEFAULT_PAIRING_STATES,
    num_spread_samples: int = 5,
    nk: int = 31,
    bias_max: float = 40.0,
    num_bias: int = 161,
    gamma: float = 1.0,
    temperature: float = 3.0,
) -> tuple[dict[str, object], DirectionalSpreadValidationArtifacts]:
    """Validate narrow directional-spread spectra over widths, barriers, and states."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bias_grid = BiasGrid(bias_min_mev=-float(bias_max), bias_max_mev=float(bias_max), num_bias=int(num_bias))
    rows: list[dict[str, object]] = []
    for direction_mode in direction_modes:
        for pairing_state, controls in pairing_states.items():
            for barrier_z in barriers:
                for half_width in half_widths:
                    spread = DirectionalSpread(
                        direction_mode=direction_mode,
                        half_width=float(half_width),
                        num_samples=1 if half_width == 0.0 else int(num_spread_samples),
                    )
                    request = FitLayerSpectrumRequest(
                        pairing_controls=dict(controls),
                        transport=transport_with_direction_mode(
                            direction_mode,
                            barrier_z=float(barrier_z),
                            gamma=float(gamma),
                            temperature_kelvin=float(temperature),
                            nk=int(nk),
                        ),
                        bias_grid=bias_grid,
                        request_label=f"{pairing_state}_{direction_mode}_spread",
                    )
                    result = generate_spread_spectrum_from_fit_layer(request, spread)
                    samples = result.transport_summary["directional_spread_samples"]
                    angles = [float(sample["interface_angle"]) for sample in samples]
                    curve = np.asarray(result.conductance, dtype=np.float64)
                    rows.append(
                        {
                            "direction_mode": direction_mode,
                            "pairing_state": pairing_state,
                            "pairing_controls": dict(controls),
                            "barrier_z": float(barrier_z),
                            "half_width": float(half_width),
                            "half_width_pi": float(half_width / np.pi),
                            "num_samples": int(spread.num_samples),
                            "sample_angle_min": float(min(angles)),
                            "sample_angle_max": float(max(angles)),
                            "mean_mismatch_penalty": float(result.transport_summary["mean_mismatch_penalty"]),
                            "bias_mev": list(result.bias_mev),
                            "conductance": [float(value) for value in curve],
                            "spectrum_features": _curve_features(curve, np.asarray(result.bias_mev, dtype=np.float64)),
                        }
                    )

    _compute_width_smoothness(rows)
    metrics_csv_path = _write_metrics_csv(rows, output_dir / "directional_spread_metrics.csv")
    plot_path = _plot_spread_validation(rows, output_dir / "directional_spread_validation.png")
    max_width_step = float(max(row["width_step_max_abs"] for row in rows)) if rows else 0.0
    smooth = bool(max_width_step <= SPREAD_SMOOTHNESS_MAX_ABS_STEP)
    summary: dict[str, object] = {
        "task": "Task P",
        "spread_definition": {
            "central_direction": "supported named in-plane mode",
            "half_width": "symmetric angular half width in radians",
            "averaging_rule": "uniform_symmetric arithmetic average of normalized spectra",
            "max_half_width": float(math.pi / 32.0),
            "not_a_mixture_fit": True,
        },
        "validation_config": {
            "direction_modes": list(direction_modes),
            "half_widths": [float(value) for value in half_widths],
            "barriers": [float(value) for value in barriers],
            "pairing_states": pairing_states,
            "num_spread_samples": int(num_spread_samples),
            "nk": int(nk),
            "bias_min_mev": float(-bias_max),
            "bias_max_mev": float(bias_max),
            "num_bias": int(num_bias),
            "gamma_mev": float(gamma),
            "temperature_kelvin": float(temperature),
        },
        "smoothness_threshold": {
            "max_abs_width_step": float(SPREAD_SMOOTHNESS_MAX_ABS_STEP),
        },
        "max_width_step_observed": max_width_step,
        "spectra_smooth_under_width_variation": smooth,
        "num_cases": len(rows),
        "metrics": rows,
        "final_verdict": (
            "Narrow directional spread is available as a reproducible uniform symmetric average around supported "
            "in-plane named modes. It is safe as a forward approximation only within the documented narrow half-width "
            "contract and is not an experiment-side directional mixture fit."
        ),
    }
    summary_path = output_dir / "directional_spread_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(summary), handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary, DirectionalSpreadValidationArtifacts(
        summary_path=summary_path,
        metrics_csv_path=metrics_csv_path,
        plot_path=plot_path,
    )
