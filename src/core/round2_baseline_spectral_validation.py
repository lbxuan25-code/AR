"""Spectral validation diagnostics for the formal round-2 baseline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np

from core.parameters import ModelParams, PhysicalPairingChannels
from core.pipeline import SpectroscopyPipeline
from core.presets import (
    base_normal_state_params,
    base_physical_pairing_channels,
    compatibility_physical_pairing_channels,
)
from core.simulation_model import SimulationModel

from source.luo_loader import load_luo_samples
from source.round2_projection import (
    DEFAULT_ROUND2_PROJECTION_CONFIG,
    Round2ProjectionConfig,
    project_luo_sample_to_round2_channels,
)

SCAN_ORDER: tuple[str, ...] = ("interface_angle", "barrier_z", "gamma", "temperature")
DEFAULT_SCAN_VALUES: dict[str, tuple[float, ...]] = {
    "interface_angle": (0.0, float(np.pi / 8.0), float(np.pi / 4.0)),
    "barrier_z": (0.25, 0.5, 1.0),
    "gamma": (0.5, 1.0, 2.0),
    "temperature": (0.5, 3.0, 8.0),
}
DEFAULT_TRANSPORT_POINT: dict[str, float] = {
    "interface_angle": 0.0,
    "barrier_z": 0.5,
    "gamma": 1.0,
    "temperature": 3.0,
}
FEATURE_KEYS: tuple[str, ...] = (
    "mse",
    "max_abs_diff",
    "low_bias_mse",
    "peak_region_mse",
    "zero_bias_shift",
    "positive_peak_bias_shift",
    "negative_peak_bias_shift",
    "positive_peak_height_shift",
    "negative_peak_height_shift",
    "dynamic_range_shift",
)


@dataclass(slots=True)
class SpectralValidationArtifacts:
    summary_path: Path
    metrics_csv_path: Path
    model_scan_plot_path: Path | None
    channel_sensitivity_plot_path: Path | None


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


def _real_stats(values: list[float] | np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "p95": float(np.percentile(arr, 95.0)),
    }


def _curve_features(curve: np.ndarray, bias: np.ndarray) -> dict[str, float]:
    values = np.asarray(curve, dtype=np.float64)
    energies = np.asarray(bias, dtype=np.float64)
    zero_index = int(np.argmin(np.abs(energies)))
    positive_mask = energies >= 0.0
    negative_mask = energies <= 0.0
    pos_bias = energies[positive_mask]
    neg_bias = energies[negative_mask]
    pos_curve = values[positive_mask]
    neg_curve = values[negative_mask]
    pos_peak_index = int(np.argmax(pos_curve))
    neg_peak_index = int(np.argmax(neg_curve))
    return {
        "zero_bias_conductance": float(values[zero_index]),
        "positive_peak_bias": float(pos_bias[pos_peak_index]),
        "negative_peak_bias": float(neg_bias[neg_peak_index]),
        "positive_peak_height": float(pos_curve[pos_peak_index]),
        "negative_peak_height": float(neg_curve[neg_peak_index]),
        "dynamic_range": float(np.max(values) - np.min(values)),
    }


def _curve_delta_metrics(reference_curve: np.ndarray, test_curve: np.ndarray, bias: np.ndarray) -> dict[str, float]:
    ref_curve = np.asarray(reference_curve, dtype=np.float64)
    cur_curve = np.asarray(test_curve, dtype=np.float64)
    diff = cur_curve - ref_curve
    low_bias_mask = np.abs(bias) <= 5.0
    peak_region_mask = (np.abs(bias) >= 5.0) & (np.abs(bias) <= 20.0)
    ref_features = _curve_features(ref_curve, bias)
    test_features = _curve_features(cur_curve, bias)
    return {
        "mse": float(np.mean(diff**2)),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "low_bias_mse": float(np.mean(diff[low_bias_mask] ** 2)),
        "peak_region_mse": float(np.mean(diff[peak_region_mask] ** 2)),
        "zero_bias_shift": float(test_features["zero_bias_conductance"] - ref_features["zero_bias_conductance"]),
        "positive_peak_bias_shift": float(test_features["positive_peak_bias"] - ref_features["positive_peak_bias"]),
        "negative_peak_bias_shift": float(test_features["negative_peak_bias"] - ref_features["negative_peak_bias"]),
        "positive_peak_height_shift": float(test_features["positive_peak_height"] - ref_features["positive_peak_height"]),
        "negative_peak_height_shift": float(test_features["negative_peak_height"] - ref_features["negative_peak_height"]),
        "dynamic_range_shift": float(test_features["dynamic_range"] - ref_features["dynamic_range"]),
    }


def _summarize_metric_rows(rows: list[dict[str, float]]) -> dict[str, object]:
    if not rows:
        raise ValueError("Expected at least one metric row.")
    summary: dict[str, object] = {"num_cases": len(rows)}
    for key in FEATURE_KEYS:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[key] = _real_stats(values)
        summary[f"abs_{key}"] = _real_stats(np.abs(values))
    return summary


def _transport_for_scan(scan_name: str, value: float) -> dict[str, float]:
    transport = dict(DEFAULT_TRANSPORT_POINT)
    transport[scan_name] = float(value)
    return transport


def _transport_label(scan_name: str, value: float) -> str:
    if scan_name == "interface_angle":
        return f"{scan_name}={value / np.pi:.3f}π"
    unit = "K" if scan_name == "temperature" else ""
    return f"{scan_name}={value:.3f}{unit}"


def _state_payload(channels: PhysicalPairingChannels, label: str) -> dict[str, object]:
    return {
        "label": label,
        "channels": channels.to_dict(),
        "channel_abs": {name: float(abs(value)) for name, value in channels.to_dict().items()},
    }


def _pairing_with_channel_zeroed(channels: PhysicalPairingChannels, channel_name: str) -> PhysicalPairingChannels:
    payload = channels.to_dict()
    payload[channel_name] = 0.0 + 0.0j
    return PhysicalPairingChannels(**payload)


def _model_from_channels(channels: PhysicalPairingChannels, name: str) -> SimulationModel:
    return SimulationModel(
        params=ModelParams(normal_state=base_normal_state_params(), pairing=channels),
        name=name,
    )


def _representative_projected_samples(
    max_samples: int | None = None,
    config: Round2ProjectionConfig = DEFAULT_ROUND2_PROJECTION_CONFIG,
) -> dict[str, dict[str, object]]:
    samples = load_luo_samples()
    if max_samples is not None:
        samples = samples[: int(max_samples)]
    ranked: list[dict[str, object]] = []
    for sample in samples:
        projected = project_luo_sample_to_round2_channels(sample, config=config)
        channels = projected.projected_physical_channels
        assert channels is not None
        ranked.append(
            {
                "sample": sample,
                "channels": channels,
                "retained_ratio_total": float(projected.round2_projection_metrics["retained_ratio_total"]),
                "residual_norm_total": float(projected.round2_projection_metrics["residual_norm_total"]),
            }
        )
    order = np.argsort(np.asarray([row["retained_ratio_total"] for row in ranked], dtype=np.float64))
    selections = {
        "best": ranked[int(order[-1])],
        "median": ranked[int(order[len(order) // 2])],
        "worst": ranked[int(order[0])],
    }
    return {
        name: {
            "sample_id": row["sample"].sample_id,
            "sample_kind": row["sample"].sample_kind,
            "coordinates": dict(row["sample"].coordinates),
            "retained_ratio_total": row["retained_ratio_total"],
            "residual_norm_total": row["residual_norm_total"],
            "channels": row["channels"],
        }
        for name, row in selections.items()
    }


def _compute_spectrum(
    channels: PhysicalPairingChannels,
    *,
    label: str,
    bias: np.ndarray,
    nk: int,
    transport: dict[str, float],
) -> dict[str, object]:
    pipeline = SpectroscopyPipeline(model=_model_from_channels(channels, name=label))
    result = pipeline.compute_multichannel_btk_conductance(
        interface_angle=float(transport["interface_angle"]),
        bias=np.asarray(bias, dtype=np.float64),
        barrier_z=float(transport["barrier_z"]),
        broadening_gamma=float(transport["gamma"]),
        temperature=float(transport["temperature"]),
        nk=int(nk),
    )
    curve = np.asarray(result.conductance, dtype=np.float64)
    return {
        "curve": curve,
        "features": _curve_features(curve, bias),
        "forward_summary": {
            "num_channels": int(result.num_channels),
            "num_input_channels": int(result.num_input_channels),
            "num_contours": int(result.num_contours),
            "mean_normal_transparency": float(result.mean_normal_transparency),
            "mean_mismatch_penalty": float(result.mean_mismatch_penalty),
        },
    }


def _plot_model_scan_comparisons(
    scan_cases: list[dict[str, object]],
    scan_values: dict[str, tuple[float, ...]],
    output_path: Path,
) -> Path | None:
    if not scan_cases:
        return None
    num_cols = max(len(values) for values in scan_values.values())
    figure, axes = plt.subplots(len(SCAN_ORDER), num_cols, figsize=(5.0 * num_cols, 12.0), constrained_layout=True)
    axes_array = np.asarray(axes, dtype=object)
    if axes_array.ndim == 1:
        axes_array = axes_array[:, np.newaxis]
    colors = {
        "compatibility_baseline": "#6b7280",
        "formal_round2_baseline": "#b91c1c",
        "representative_best": "#0f766e",
        "representative_median": "#2563eb",
        "representative_worst": "#d97706",
    }
    labels = {
        "compatibility_baseline": "compatibility",
        "formal_round2_baseline": "formal",
        "representative_best": "rep best",
        "representative_median": "rep median",
        "representative_worst": "rep worst",
    }
    case_map = {
        (case["scan_parameter"], float(case["scan_value"])): case
        for case in scan_cases
    }
    for row_index, scan_name in enumerate(SCAN_ORDER):
        values = tuple(scan_values[scan_name])
        for col_index in range(num_cols):
            axis = axes_array[row_index, col_index]
            if col_index >= len(values):
                axis.axis("off")
                continue
            value = values[col_index]
            case = case_map[(scan_name, float(value))]
            bias = np.asarray(case["bias"], dtype=np.float64)
            for model_name in (
                "compatibility_baseline",
                "formal_round2_baseline",
                "representative_best",
                "representative_median",
                "representative_worst",
            ):
                axis.plot(
                    bias,
                    np.asarray(case["model_curves"][model_name], dtype=np.float64),
                    label=labels[model_name],
                    linewidth=1.35,
                    color=colors[model_name],
                )
            axis.set_title(_transport_label(scan_name, float(value)))
            axis.set_xlabel("Bias (meV)")
            axis.set_ylabel("Conductance")
            axis.grid(alpha=0.18)
            if row_index == 0 and col_index == 0:
                axis.legend(loc="best", fontsize=8)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def _plot_channel_sensitivity(
    scan_cases: list[dict[str, object]],
    scan_values: dict[str, tuple[float, ...]],
    output_path: Path,
) -> Path | None:
    if not scan_cases:
        return None
    num_cols = max(len(values) for values in scan_values.values())
    figure, axes = plt.subplots(len(SCAN_ORDER), num_cols, figsize=(5.0 * num_cols, 12.0), constrained_layout=True)
    axes_array = np.asarray(axes, dtype=object)
    if axes_array.ndim == 1:
        axes_array = axes_array[:, np.newaxis]
    colors = {
        "formal_round2_baseline": "#111827",
        "delta_zz_d_zeroed": "#2563eb",
        "delta_zx_d_zeroed": "#dc2626",
        "delta_perp_x_zeroed": "#059669",
    }
    labels = {
        "formal_round2_baseline": "formal",
        "delta_zz_d_zeroed": "zero delta_zz_d",
        "delta_zx_d_zeroed": "zero delta_zx_d",
        "delta_perp_x_zeroed": "zero delta_perp_x",
    }
    case_map = {
        (case["scan_parameter"], float(case["scan_value"])): case
        for case in scan_cases
    }
    for row_index, scan_name in enumerate(SCAN_ORDER):
        values = tuple(scan_values[scan_name])
        for col_index in range(num_cols):
            axis = axes_array[row_index, col_index]
            if col_index >= len(values):
                axis.axis("off")
                continue
            value = values[col_index]
            case = case_map[(scan_name, float(value))]
            bias = np.asarray(case["bias"], dtype=np.float64)
            for model_name in (
                "formal_round2_baseline",
                "delta_zz_d_zeroed",
                "delta_zx_d_zeroed",
                "delta_perp_x_zeroed",
            ):
                axis.plot(
                    bias,
                    np.asarray(case["sensitivity_curves"][model_name], dtype=np.float64),
                    label=labels[model_name],
                    linewidth=1.35,
                    color=colors[model_name],
                )
            axis.set_title(_transport_label(scan_name, float(value)))
            axis.set_xlabel("Bias (meV)")
            axis.set_ylabel("Conductance")
            axis.grid(alpha=0.18)
            if row_index == 0 and col_index == 0:
                axis.legend(loc="best", fontsize=8)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def _write_metrics_csv(scan_cases: list[dict[str, object]], output_path: Path) -> Path:
    fieldnames = [
        "group",
        "scan_parameter",
        "scan_value",
        "comparison_name",
        *FEATURE_KEYS,
    ]
    rows: list[dict[str, object]] = []
    for case in scan_cases:
        prefix = {
            "scan_parameter": case["scan_parameter"],
            "scan_value": float(case["scan_value"]),
        }
        for name, metrics in case["formal_reference_comparisons"].items():
            rows.append({"group": "formal_reference", "comparison_name": name, **prefix, **metrics})
        for name, metrics in case["channel_sensitivity_metrics"].items():
            rows.append({"group": "channel_sensitivity", "comparison_name": name, **prefix, **metrics})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _build_verdict(summary: dict[str, object]) -> str:
    compat_peak_mse = float(summary["formal_vs_compatibility"]["all_cases"]["peak_region_mse"]["median"])
    channel_strength = {
        "delta_zz_d": float(summary["channel_sensitivity"]["delta_zz_d"]["all_cases"]["peak_region_mse"]["median"]),
        "delta_zx_d": float(summary["channel_sensitivity"]["delta_zx_d"]["all_cases"]["peak_region_mse"]["median"]),
        "delta_perp_x": float(summary["channel_sensitivity"]["delta_perp_x"]["all_cases"]["peak_region_mse"]["median"]),
    }
    strongest_channel = max(channel_strength, key=channel_strength.get)
    weakest_channel = min(channel_strength, key=channel_strength.get)
    return (
        "Formal round-2 baseline spectra are measurably different from the legacy-compatible baseline across the "
        f"Task-D scans (median peak-region MSE {compat_peak_mse:.3e} for compatibility-vs-formal). "
        f"Within the newly explicit round-2 channels, `{strongest_channel}` is the strongest spectral lever under this audit, "
        f"while `{weakest_channel}` is weakest at the current baseline amplitude. "
        "This is a spectral diagnostic statement, not a claim that the formal baseline is closer to full RMFT in every tensor component."
    )


def run_round2_baseline_spectral_validation(
    *,
    output_dir: Path,
    representative_selection_max_samples: int | None = None,
    scan_values: dict[str, tuple[float, ...]] | None = None,
    nk: int = 41,
    bias_max: float = 40.0,
    num_bias: int = 201,
    projection_config: Round2ProjectionConfig = DEFAULT_ROUND2_PROJECTION_CONFIG,
) -> tuple[dict[str, object], SpectralValidationArtifacts]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scan_values = scan_values or DEFAULT_SCAN_VALUES
    bias = np.linspace(-float(bias_max), float(bias_max), int(num_bias), dtype=np.float64)

    formal_channels = base_physical_pairing_channels()
    representative = _representative_projected_samples(
        max_samples=representative_selection_max_samples,
        config=projection_config,
    )

    model_states: dict[str, PhysicalPairingChannels] = {
        "compatibility_baseline": compatibility_physical_pairing_channels(),
        "formal_round2_baseline": formal_channels,
        "representative_best": representative["best"]["channels"],
        "representative_median": representative["median"]["channels"],
        "representative_worst": representative["worst"]["channels"],
    }
    sensitivity_states: dict[str, PhysicalPairingChannels] = {
        "formal_round2_baseline": formal_channels,
        "delta_zz_d_zeroed": _pairing_with_channel_zeroed(formal_channels, "delta_zz_d"),
        "delta_zx_d_zeroed": _pairing_with_channel_zeroed(formal_channels, "delta_zx_d"),
        "delta_perp_x_zeroed": _pairing_with_channel_zeroed(formal_channels, "delta_perp_x"),
    }

    spectrum_cache: dict[tuple[str, tuple[float, float, float, float]], dict[str, object]] = {}

    def cached_spectrum(
        label: str,
        channels: PhysicalPairingChannels,
        transport: dict[str, float],
    ) -> dict[str, object]:
        key = (
            label,
            (
                float(transport["interface_angle"]),
                float(transport["barrier_z"]),
                float(transport["gamma"]),
                float(transport["temperature"]),
            ),
        )
        if key not in spectrum_cache:
            spectrum_cache[key] = _compute_spectrum(
                channels,
                label=label,
                bias=bias,
                nk=nk,
                transport=transport,
            )
        return spectrum_cache[key]

    scan_cases: list[dict[str, object]] = []
    for scan_name in SCAN_ORDER:
        for value in scan_values[scan_name]:
            transport = _transport_for_scan(scan_name, float(value))
            model_outputs = {
                name: cached_spectrum(name, channels, transport)
                for name, channels in model_states.items()
            }
            sensitivity_outputs = {
                name: cached_spectrum(name, channels, transport)
                for name, channels in sensitivity_states.items()
            }
            formal_curve = np.asarray(model_outputs["formal_round2_baseline"]["curve"], dtype=np.float64)
            formal_reference_comparisons = {
                name: _curve_delta_metrics(formal_curve, np.asarray(payload["curve"], dtype=np.float64), bias)
                for name, payload in model_outputs.items()
                if name != "formal_round2_baseline"
            }
            formal_sensitivity_curve = np.asarray(sensitivity_outputs["formal_round2_baseline"]["curve"], dtype=np.float64)
            channel_sensitivity_metrics = {
                "delta_zz_d": _curve_delta_metrics(
                    formal_sensitivity_curve,
                    np.asarray(sensitivity_outputs["delta_zz_d_zeroed"]["curve"], dtype=np.float64),
                    bias,
                ),
                "delta_zx_d": _curve_delta_metrics(
                    formal_sensitivity_curve,
                    np.asarray(sensitivity_outputs["delta_zx_d_zeroed"]["curve"], dtype=np.float64),
                    bias,
                ),
                "delta_perp_x": _curve_delta_metrics(
                    formal_sensitivity_curve,
                    np.asarray(sensitivity_outputs["delta_perp_x_zeroed"]["curve"], dtype=np.float64),
                    bias,
                ),
            }
            scan_cases.append(
                {
                    "scan_parameter": scan_name,
                    "scan_value": float(value),
                    "transport": transport,
                    "bias": bias,
                    "model_curves": {name: payload["curve"] for name, payload in model_outputs.items()},
                    "model_features": {name: payload["features"] for name, payload in model_outputs.items()},
                    "formal_reference_comparisons": formal_reference_comparisons,
                    "sensitivity_curves": {name: payload["curve"] for name, payload in sensitivity_outputs.items()},
                    "channel_sensitivity_metrics": channel_sensitivity_metrics,
                }
            )

    formal_vs_compatibility_rows = [case["formal_reference_comparisons"]["compatibility_baseline"] for case in scan_cases]
    representative_rows = {
        "representative_best": [case["formal_reference_comparisons"]["representative_best"] for case in scan_cases],
        "representative_median": [case["formal_reference_comparisons"]["representative_median"] for case in scan_cases],
        "representative_worst": [case["formal_reference_comparisons"]["representative_worst"] for case in scan_cases],
    }
    channel_rows = {
        "delta_zz_d": [case["channel_sensitivity_metrics"]["delta_zz_d"] for case in scan_cases],
        "delta_zx_d": [case["channel_sensitivity_metrics"]["delta_zx_d"] for case in scan_cases],
        "delta_perp_x": [case["channel_sensitivity_metrics"]["delta_perp_x"] for case in scan_cases],
    }

    summary: dict[str, object] = {
        "scan_defaults": dict(DEFAULT_TRANSPORT_POINT),
        "scan_values": {name: [float(value) for value in scan_values[name]] for name in SCAN_ORDER},
        "bias_window_meV": {"min": float(np.min(bias)), "max": float(np.max(bias)), "num_bias": int(num_bias)},
        "nk": int(nk),
        "pairing_states": {
            "compatibility_baseline": _state_payload(model_states["compatibility_baseline"], "legacy-compatible round-2 translation"),
            "formal_round2_baseline": _state_payload(model_states["formal_round2_baseline"], "formal Stage-3 baseline"),
            "representative_samples": {
                name: {
                    "sample_id": representative[name]["sample_id"],
                    "sample_kind": representative[name]["sample_kind"],
                    "coordinates": representative[name]["coordinates"],
                    "retained_ratio_total": representative[name]["retained_ratio_total"],
                    "residual_norm_total": representative[name]["residual_norm_total"],
                    "channels": representative[name]["channels"].to_dict(),
                }
                for name in ("best", "median", "worst")
            },
        },
        "formal_vs_compatibility": {
            "all_cases": _summarize_metric_rows(formal_vs_compatibility_rows),
            "by_scan_parameter": {
                scan_name: _summarize_metric_rows(
                    [case["formal_reference_comparisons"]["compatibility_baseline"] for case in scan_cases if case["scan_parameter"] == scan_name]
                )
                for scan_name in SCAN_ORDER
            },
        },
        "representative_sample_differences": {
            name: {
                "all_cases": _summarize_metric_rows(rows),
                "by_scan_parameter": {
                    scan_name: _summarize_metric_rows(
                        [case["formal_reference_comparisons"][name] for case in scan_cases if case["scan_parameter"] == scan_name]
                    )
                    for scan_name in SCAN_ORDER
                },
            }
            for name, rows in representative_rows.items()
        },
        "channel_sensitivity": {
            channel_name: {
                "all_cases": _summarize_metric_rows(rows),
                "by_scan_parameter": {
                    scan_name: _summarize_metric_rows(
                        [case["channel_sensitivity_metrics"][channel_name] for case in scan_cases if case["scan_parameter"] == scan_name]
                    )
                    for scan_name in SCAN_ORDER
                },
            }
            for channel_name, rows in channel_rows.items()
        },
        "per_scan_cases": [
            {
                "scan_parameter": case["scan_parameter"],
                "scan_value": float(case["scan_value"]),
                "transport": dict(case["transport"]),
                "formal_reference_comparisons": case["formal_reference_comparisons"],
                "channel_sensitivity_metrics": case["channel_sensitivity_metrics"],
                "model_features": case["model_features"],
            }
            for case in scan_cases
        ],
    }
    summary["verdict"] = _build_verdict(summary)

    summary_path = output_dir / "round2_baseline_spectral_validation_summary.json"
    metrics_csv_path = output_dir / "round2_baseline_spectral_validation_metrics.csv"
    model_scan_plot_path = output_dir / "round2_baseline_scan_comparison.png"
    channel_sensitivity_plot_path = output_dir / "round2_channel_sensitivity_scan.png"

    summary_path.write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")
    _write_metrics_csv(scan_cases, metrics_csv_path)
    _plot_model_scan_comparisons(scan_cases, scan_values, model_scan_plot_path)
    _plot_channel_sensitivity(scan_cases, scan_values, channel_sensitivity_plot_path)

    return summary, SpectralValidationArtifacts(
        summary_path=summary_path,
        metrics_csv_path=metrics_csv_path,
        model_scan_plot_path=model_scan_plot_path,
        channel_sensitivity_plot_path=channel_sensitivity_plot_path,
    )
