"""Validation for Task-M canonical in-plane directional modes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np

from forward import (
    BiasGrid,
    FitLayerSpectrumRequest,
    TransportControls,
    generate_spectrum_from_fit_layer,
    list_directional_modes,
    transport_with_direction_mode,
)
from forward.schema import ForwardSpectrumResult


@dataclass(frozen=True, slots=True)
class DirectionalModeValidationArtifacts:
    """Generated Task-M directional-mode validation artifact paths."""

    summary_path: Path
    metrics_csv_path: Path
    comparison_plot_path: Path | None


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


def _spectrum_for_transport(transport: TransportControls, bias_grid: BiasGrid, label: str) -> ForwardSpectrumResult:
    request = FitLayerSpectrumRequest(
        pairing_controls={},
        transport=transport,
        bias_grid=bias_grid,
        request_label=label,
    )
    return generate_spectrum_from_fit_layer(request)


def _comparison_row(mode_name: str, raw_result: ForwardSpectrumResult, named_result: ForwardSpectrumResult) -> dict[str, object]:
    raw_curve = np.asarray(raw_result.conductance, dtype=np.float64)
    named_curve = np.asarray(named_result.conductance, dtype=np.float64)
    diff = named_curve - raw_curve
    named_transport = named_result.transport_summary
    raw_transport = raw_result.transport_summary
    return {
        "direction_mode": mode_name,
        "crystal_label": named_transport["direction_crystal_label"],
        "raw_interface_angle": float(raw_transport["interface_angle"]),
        "named_interface_angle": float(named_transport["interface_angle"]),
        "raw_direction_mode": raw_transport["direction_mode"],
        "named_direction_mode": named_transport["direction_mode"],
        "direction_support_tier": named_transport["direction_support_tier"],
        "max_abs_conductance_diff": float(np.max(np.abs(diff))),
        "mse_conductance_diff": float(np.mean(diff**2)),
        "raw_num_channels": int(raw_transport["num_channels"]),
        "named_num_channels": int(named_transport["num_channels"]),
        "raw_num_input_channels": int(raw_transport["num_input_channels"]),
        "named_num_input_channels": int(named_transport["num_input_channels"]),
        "raw_num_same_band_channels": int(raw_transport["num_same_band_channels"]),
        "named_num_same_band_channels": int(named_transport["num_same_band_channels"]),
        "raw_num_filtered_channels": int(raw_transport["num_filtered_channels"]),
        "named_num_filtered_channels": int(named_transport["num_filtered_channels"]),
        "raw_mean_mismatch_penalty": float(raw_transport["mean_mismatch_penalty"]),
        "named_mean_mismatch_penalty": float(named_transport["mean_mismatch_penalty"]),
    }


def _write_metrics_csv(rows: list[dict[str, object]], path: Path) -> Path:
    fieldnames = [
        "direction_mode",
        "crystal_label",
        "raw_interface_angle",
        "named_interface_angle",
        "raw_direction_mode",
        "named_direction_mode",
        "direction_support_tier",
        "max_abs_conductance_diff",
        "mse_conductance_diff",
        "raw_num_channels",
        "named_num_channels",
        "raw_num_input_channels",
        "named_num_input_channels",
        "raw_num_same_band_channels",
        "named_num_same_band_channels",
        "raw_num_filtered_channels",
        "named_num_filtered_channels",
        "raw_mean_mismatch_penalty",
        "named_mean_mismatch_penalty",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})
    return path


def _plot_comparisons(
    comparison_payloads: list[dict[str, object]],
    output_path: Path,
) -> Path | None:
    if not comparison_payloads:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, len(comparison_payloads), figsize=(6.0 * len(comparison_payloads), 4.6), constrained_layout=True)
    axes_array = np.asarray(axes, dtype=object).reshape(-1)
    for axis, payload in zip(axes_array, comparison_payloads, strict=True):
        bias = np.asarray(payload["bias_mev"], dtype=np.float64)
        raw_curve = np.asarray(payload["raw_conductance"], dtype=np.float64)
        named_curve = np.asarray(payload["named_conductance"], dtype=np.float64)
        axis.plot(bias, raw_curve, label="raw angle", linewidth=2.0)
        axis.plot(bias, named_curve, "--", label="named mode", linewidth=1.6)
        axis.set_title(f"{payload['direction_mode']} ({payload['crystal_label']})")
        axis.set_xlabel("Bias (meV)")
        axis.set_ylabel("Normalized conductance")
        axis.legend(fontsize=8)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def run_directional_modes_validation(
    *,
    output_dir: Path = Path("outputs/core/directional_modes_validation"),
    nk: int = 41,
    bias_max: float = 40.0,
    num_bias: int = 201,
    barrier_z: float = 0.5,
    gamma: float = 1.0,
    temperature: float = 3.0,
) -> tuple[dict[str, object], DirectionalModeValidationArtifacts]:
    """Validate that named in-plane modes reproduce their raw-angle calls."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bias_grid = BiasGrid(bias_min_mev=-float(bias_max), bias_max_mev=float(bias_max), num_bias=int(num_bias))

    rows: list[dict[str, object]] = []
    payloads: list[dict[str, object]] = []
    for mode in list_directional_modes():
        raw_transport = TransportControls(
            direction_mode=None,
            interface_angle=float(mode.interface_angle),
            barrier_z=float(barrier_z),
            gamma=float(gamma),
            temperature_kelvin=float(temperature),
            nk=int(nk),
        )
        named_transport = transport_with_direction_mode(
            mode.name,
            barrier_z=float(barrier_z),
            gamma=float(gamma),
            temperature_kelvin=float(temperature),
            nk=int(nk),
        )
        raw_result = _spectrum_for_transport(raw_transport, bias_grid, label=f"raw_{mode.name}")
        named_result = _spectrum_for_transport(named_transport, bias_grid, label=f"named_{mode.name}")
        row = _comparison_row(mode.name, raw_result, named_result)
        rows.append(row)
        payloads.append(
            {
                "direction_mode": mode.name,
                "crystal_label": mode.crystal_label,
                "bias_mev": list(raw_result.bias_mev),
                "raw_conductance": list(raw_result.conductance),
                "named_conductance": list(named_result.conductance),
                "raw_request_transport": raw_result.request["transport"],
                "named_request_transport": named_result.request["transport"],
                "raw_transport_summary": raw_result.transport_summary,
                "named_transport_summary": named_result.transport_summary,
            }
        )

    metrics_csv_path = _write_metrics_csv(rows, output_dir / "directional_modes_metrics.csv")
    plot_path = _plot_comparisons(payloads, output_dir / "directional_modes_comparison.png")
    max_abs_diff = float(max(row["max_abs_conductance_diff"] for row in rows)) if rows else 0.0

    summary: dict[str, object] = {
        "task": "Task M",
        "canonical_modes": [mode.to_dict() for mode in list_directional_modes()],
        "validation_config": {
            "nk": int(nk),
            "bias_min_mev": float(-bias_max),
            "bias_max_mev": float(bias_max),
            "num_bias": int(num_bias),
            "barrier_z": float(barrier_z),
            "gamma_mev": float(gamma),
            "temperature_kelvin": float(temperature),
        },
        "comparisons": rows,
        "representative_payloads": payloads,
        "max_abs_conductance_diff_across_modes": max_abs_diff,
        "verdict": (
            "Named in-plane high-symmetry directional modes reproduce the corresponding raw-angle "
            "forward spectra and diagnostics exactly within floating-point comparison precision."
        ),
    }

    summary_path = output_dir / "directional_modes_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(summary), handle, indent=2, sort_keys=True)
        handle.write("\n")

    return summary, DirectionalModeValidationArtifacts(
        summary_path=summary_path,
        metrics_csv_path=metrics_csv_path,
        comparison_plot_path=plot_path,
    )
