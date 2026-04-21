"""AR validation comparing RMFT source tensors against round-2 projections."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from core.normal_state import h0_matrix
from core.parameters import ModelParams
from core.pipeline import SpectroscopyPipeline
from core.presets import base_normal_state_params, base_physical_pairing_channels
from core.round2_baseline_spectral_validation import DEFAULT_SCAN_VALUES, DEFAULT_TRANSPORT_POINT, SCAN_ORDER
from core.simulation_model import SimulationModel

from .luo_loader import load_luo_samples
from .round2_projection import (
    DEFAULT_ROUND2_PROJECTION_CONFIG,
    Round2ProjectionConfig,
    gauge_fix_source_tensors,
    project_luo_sample_to_round2_channels,
    source_pairing_tensors_meV,
)

METRIC_KEYS: tuple[str, ...] = (
    "mse",
    "max_abs_diff",
    "mean_abs_diff",
    "zero_bias_shift",
    "positive_peak_bias_shift",
    "negative_peak_bias_shift",
    "positive_peak_height_shift",
    "negative_peak_height_shift",
    "low_bias_mse",
    "shoulder_mse",
    "high_bias_mse",
)


@dataclass(frozen=True, slots=True)
class SourceTensorModel:
    """Diagnostic model using full gauge-fixed RMFT source pairing tensors."""

    params: ModelParams
    delta_x: np.ndarray
    delta_y: np.ndarray
    delta_z: np.ndarray
    name: str = "rmft_source_tensor_reference"

    def build_normal_state(self, kx: float, ky: float) -> np.ndarray:
        return h0_matrix(kx, ky, self.params.normal_state)

    def build_delta(self, kx: float, ky: float) -> np.ndarray:
        return (
            np.asarray(self.delta_x, dtype=np.complex128) * np.cos(kx)
            + np.asarray(self.delta_y, dtype=np.complex128) * np.cos(ky)
            + np.asarray(self.delta_z, dtype=np.complex128)
        )

    def build_bdg(self, kx: float, ky: float) -> np.ndarray:
        h0_k = self.build_normal_state(kx, ky)
        delta_k = self.build_delta(kx, ky)
        h0_minus_k = self.build_normal_state(-kx, -ky)
        return np.block([[h0_k, delta_k], [delta_k.conj().T, -h0_minus_k.T]]).astype(np.complex128)


@dataclass(slots=True)
class RMFTSourceARValidationArtifacts:
    summary_path: Path
    metrics_csv_path: Path
    plot_paths: list[Path]


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


def _features(curve: np.ndarray, bias: np.ndarray) -> dict[str, float]:
    values = np.asarray(curve, dtype=np.float64)
    energies = np.asarray(bias, dtype=np.float64)
    zero_index = int(np.argmin(np.abs(energies)))
    positive_mask = energies >= 0.0
    negative_mask = energies <= 0.0
    positive_bias = energies[positive_mask]
    negative_bias = energies[negative_mask]
    positive_curve = values[positive_mask]
    negative_curve = values[negative_mask]
    positive_peak_index = int(np.argmax(positive_curve))
    negative_peak_index = int(np.argmax(negative_curve))
    return {
        "zero_bias_conductance": float(values[zero_index]),
        "positive_peak_bias": float(positive_bias[positive_peak_index]),
        "negative_peak_bias": float(negative_bias[negative_peak_index]),
        "positive_peak_height": float(positive_curve[positive_peak_index]),
        "negative_peak_height": float(negative_curve[negative_peak_index]),
    }


def _metric_row(source_curve: np.ndarray, round2_curve: np.ndarray, bias: np.ndarray) -> dict[str, float]:
    source = np.asarray(source_curve, dtype=np.float64)
    projected = np.asarray(round2_curve, dtype=np.float64)
    diff = projected - source
    low_bias_mask = np.abs(bias) <= 5.0
    shoulder_mask = (np.abs(bias) > 5.0) & (np.abs(bias) <= 20.0)
    high_bias_mask = np.abs(bias) > 20.0
    source_features = _features(source, bias)
    projected_features = _features(projected, bias)
    return {
        "mse": float(np.mean(diff**2)),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "zero_bias_shift": float(projected_features["zero_bias_conductance"] - source_features["zero_bias_conductance"]),
        "positive_peak_bias_shift": float(projected_features["positive_peak_bias"] - source_features["positive_peak_bias"]),
        "negative_peak_bias_shift": float(projected_features["negative_peak_bias"] - source_features["negative_peak_bias"]),
        "positive_peak_height_shift": float(projected_features["positive_peak_height"] - source_features["positive_peak_height"]),
        "negative_peak_height_shift": float(projected_features["negative_peak_height"] - source_features["negative_peak_height"]),
        "low_bias_mse": float(np.mean(diff[low_bias_mask] ** 2)),
        "shoulder_mse": float(np.mean(diff[shoulder_mask] ** 2)),
        "high_bias_mse": float(np.mean(diff[high_bias_mask] ** 2)) if np.any(high_bias_mask) else 0.0,
    }


def _summarize_metrics(rows: list[dict[str, float]]) -> dict[str, object]:
    if not rows:
        raise ValueError("Expected at least one metrics row.")
    return {
        key: {
            "signed": _real_stats([row[key] for row in rows]),
            "abs": _real_stats([abs(row[key]) for row in rows]),
        }
        for key in METRIC_KEYS
    } | {"num_cases": len(rows)}


def _transport_for_scan(scan_name: str, value: float) -> dict[str, float]:
    transport = dict(DEFAULT_TRANSPORT_POINT)
    transport[scan_name] = float(value)
    return transport


def _transport_label(scan_name: str, value: float) -> str:
    if scan_name == "interface_angle":
        return f"{scan_name}={value / np.pi:.3f}π"
    if scan_name == "temperature":
        return f"{scan_name}={value:.3f}K"
    return f"{scan_name}={value:.3f}"


def _source_reference_model(sample, config: Round2ProjectionConfig) -> SourceTensorModel:
    delta_x, delta_y, delta_z = source_pairing_tensors_meV(sample)
    gauge_x, gauge_y, gauge_z, _ = gauge_fix_source_tensors(delta_x, delta_y, delta_z, config=config)
    return SourceTensorModel(
        params=ModelParams(normal_state=base_normal_state_params(), pairing=base_physical_pairing_channels()),
        delta_x=gauge_x,
        delta_y=gauge_y,
        delta_z=gauge_z,
        name=f"rmft_source::{sample.sample_id}",
    )


def _round2_projected_model(sample, config: Round2ProjectionConfig) -> tuple[SimulationModel, dict[str, object]]:
    projected = project_luo_sample_to_round2_channels(sample, config=config)
    assert projected.projected_physical_channels is not None
    return (
        SimulationModel(
            params=ModelParams(
                normal_state=base_normal_state_params(),
                pairing=projected.projected_physical_channels,
            ),
            name=f"round2::{sample.sample_id}",
        ),
        {
            "channels": projected.projected_physical_channels.to_dict(),
            "metrics": dict(projected.round2_projection_metrics),
            "metadata": dict(projected.round2_projection_metadata),
        },
    )


def _spectrum(model, bias: np.ndarray, nk: int, transport: dict[str, float]) -> np.ndarray:
    pipeline = SpectroscopyPipeline(model=model)
    result = pipeline.compute_multichannel_btk_conductance(
        interface_angle=float(transport["interface_angle"]),
        bias=np.asarray(bias, dtype=np.float64),
        barrier_z=float(transport["barrier_z"]),
        broadening_gamma=float(transport["gamma"]),
        temperature=float(transport["temperature"]),
        nk=int(nk),
    )
    return np.asarray(result.conductance, dtype=np.float64)


def _representative_rows(
    samples: list[object],
    config: Round2ProjectionConfig,
) -> dict[str, dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index, sample in enumerate(samples):
        projected = project_luo_sample_to_round2_channels(sample, config=config)
        assert projected.projected_physical_channels is not None
        rows.append(
            {
                "index": index,
                "sample": sample,
                "sample_id": sample.sample_id,
                "retained_ratio_total": float(projected.round2_projection_metrics["retained_ratio_total"]),
                "residual_norm_total": float(projected.round2_projection_metrics["residual_norm_total"]),
            }
        )
    retained = np.asarray([row["retained_ratio_total"] for row in rows], dtype=np.float64)
    ranking = np.argsort(retained)
    return {
        "best": rows[int(ranking[-1])],
        "median": rows[int(ranking[len(ranking) // 2])],
        "worst": rows[int(ranking[0])],
    }


def _plot_sample_scan(
    category: str,
    sample_id: str,
    scan_cases: list[dict[str, object]],
    scan_values: dict[str, tuple[float, ...]],
    output_path: Path,
) -> Path:
    num_cols = max(len(values) for values in scan_values.values())
    figure, axes = plt.subplots(len(SCAN_ORDER), num_cols, figsize=(5.0 * num_cols, 12.0), constrained_layout=True)
    axes_array = np.asarray(axes, dtype=object)
    if axes_array.ndim == 1:
        axes_array = axes_array[:, np.newaxis]
    for row_index, scan_name in enumerate(SCAN_ORDER):
        values = scan_values[scan_name]
        for col_index in range(num_cols):
            axis = axes_array[row_index, col_index]
            if col_index >= len(values):
                axis.axis("off")
                continue
            value = values[col_index]
            case = next(
                item
                for item in scan_cases
                if item["category"] == category
                and item["scan_parameter"] == scan_name
                and float(item["scan_value"]) == float(value)
            )
            bias = np.asarray(case["bias"], dtype=np.float64)
            axis.plot(bias, case["source_curve"], label="RMFT source", linewidth=1.5, color="#111827")
            axis.plot(bias, case["round2_curve"], label="round-2 projected", linewidth=1.35, color="#dc2626")
            axis.set_title(_transport_label(scan_name, float(value)))
            axis.set_xlabel("Bias (meV)")
            axis.set_ylabel("Conductance")
            axis.grid(alpha=0.18)
            if row_index == 0 and col_index == 0:
                axis.legend(loc="best", fontsize=8)
    figure.suptitle(f"{category}: {sample_id}", fontsize=12)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def write_rmft_source_ar_validation_outputs(
    output_dir: Path,
    summary: dict[str, object],
    scan_cases: list[dict[str, object]],
    scan_values: dict[str, tuple[float, ...]],
) -> RMFTSourceARValidationArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "rmft_source_vs_round2_ar_validation_summary.json"
    metrics_csv_path = output_dir / "rmft_source_vs_round2_ar_validation_metrics.csv"
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")

    fieldnames = ["category", "sample_id", "scan_parameter", "scan_value", *METRIC_KEYS]
    with metrics_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for case in scan_cases:
            writer.writerow(
                {
                    "category": case["category"],
                    "sample_id": case["sample_id"],
                    "scan_parameter": case["scan_parameter"],
                    "scan_value": case["scan_value"],
                    **case["metrics"],
                }
            )

    plot_paths = [
        _plot_sample_scan(
            category,
            summary["representative_samples"][category]["sample_id"],
            scan_cases,
            scan_values,
            output_dir / f"rmft_source_vs_round2_{category}_scan.png",
        )
        for category in ("best", "median", "worst")
    ]
    return RMFTSourceARValidationArtifacts(summary_path=summary_path, metrics_csv_path=metrics_csv_path, plot_paths=plot_paths)


def run_rmft_source_vs_round2_ar_validation(
    output_dir: Path,
    max_selection_samples: int | None = None,
    scan_values: dict[str, tuple[float, ...]] | None = None,
    nk: int = 41,
    bias_max: float = 40.0,
    num_bias: int = 201,
    config: Round2ProjectionConfig = DEFAULT_ROUND2_PROJECTION_CONFIG,
) -> tuple[dict[str, object], list[dict[str, object]], RMFTSourceARValidationArtifacts]:
    samples = load_luo_samples()
    selection_samples = samples if max_selection_samples is None else samples[: int(max_selection_samples)]
    representatives = _representative_rows(selection_samples, config=config)
    active_scan_values = scan_values or DEFAULT_SCAN_VALUES
    bias = np.linspace(-float(bias_max), float(bias_max), int(num_bias), dtype=np.float64)

    scan_cases: list[dict[str, object]] = []
    representative_payload: dict[str, dict[str, object]] = {}
    for category, row in representatives.items():
        sample = row["sample"]
        source_model = _source_reference_model(sample, config=config)
        round2_model, round2_payload = _round2_projected_model(sample, config=config)
        representative_payload[category] = {
            "sample_id": sample.sample_id,
            "sample_kind": sample.sample_kind,
            "coordinates": dict(sample.coordinates),
            "retained_ratio_total": row["retained_ratio_total"],
            "residual_norm_total": row["residual_norm_total"],
            "round2_projection": round2_payload,
        }
        for scan_name in SCAN_ORDER:
            for value in active_scan_values[scan_name]:
                transport = _transport_for_scan(scan_name, float(value))
                source_curve = _spectrum(source_model, bias=bias, nk=nk, transport=transport)
                round2_curve = _spectrum(round2_model, bias=bias, nk=nk, transport=transport)
                scan_cases.append(
                    {
                        "category": category,
                        "sample_id": sample.sample_id,
                        "scan_parameter": scan_name,
                        "scan_value": float(value),
                        "transport": transport,
                        "bias": bias,
                        "source_curve": source_curve,
                        "round2_curve": round2_curve,
                        "metrics": _metric_row(source_curve, round2_curve, bias),
                    }
                )

    all_metrics = [case["metrics"] for case in scan_cases]
    by_category = {
        category: _summarize_metrics([case["metrics"] for case in scan_cases if case["category"] == category])
        for category in ("best", "median", "worst")
    }
    by_scan = {
        scan_name: _summarize_metrics([case["metrics"] for case in scan_cases if case["scan_parameter"] == scan_name])
        for scan_name in SCAN_ORDER
    }
    median_mse = float(np.median(np.asarray([row["mse"] for row in all_metrics], dtype=np.float64)))
    median_zero = float(np.median(np.abs(np.asarray([row["zero_bias_shift"] for row in all_metrics], dtype=np.float64))))
    worst_case = max(scan_cases, key=lambda item: float(item["metrics"]["mse"]))
    verdict = (
        "Round-2 AR fidelity must be judged against the RMFT source reference. "
        f"Across representative scans, median source-vs-round2 MSE is {median_mse:.3e} and "
        f"median |zero-bias shift| is {median_zero:.3e}. "
        f"The largest discrepancy in this audit occurs for `{worst_case['category']}` under "
        f"`{worst_case['scan_parameter']}={worst_case['scan_value']}`. "
        "Use these source-reference AR metrics, not round-1-vs-round-2 metrics, as the main validation axis."
    )
    summary: dict[str, object] = {
        "validation_axis": "RMFT source-reference AR spectra vs round-2 projected-channel AR spectra",
        "source_reference_definition": (
            "Gauge-fixed Luo source pairing tensors are used directly as "
            "Delta(k)=delta_x*cos(kx)+delta_y*cos(ky)+delta_z after the existing single eV-to-meV conversion."
        ),
        "normal_state_policy": (
            "Both source-reference and round-2 projected AR spectra use the repository-local fixed normal-state "
            "baseline so the comparison isolates pairing-layer information."
        ),
        "num_representative_samples": 3,
        "nk": int(nk),
        "bias_window_meV": {"min": float(np.min(bias)), "max": float(np.max(bias)), "num_bias": int(num_bias)},
        "scan_values": {name: [float(value) for value in active_scan_values[name]] for name in SCAN_ORDER},
        "representative_samples": representative_payload,
        "overall_metrics": _summarize_metrics(all_metrics),
        "metrics_by_representative": by_category,
        "metrics_by_scan_parameter": by_scan,
        "worst_case": {
            "category": worst_case["category"],
            "sample_id": worst_case["sample_id"],
            "scan_parameter": worst_case["scan_parameter"],
            "scan_value": float(worst_case["scan_value"]),
            "metrics": worst_case["metrics"],
        },
        "conclusion": {
            "round2_sufficient_for_ar": median_mse < 1.0e-2 and median_zero < 0.1,
            "missing_source_structures_still_visible": bool(
                median_mse >= 1.0e-2 or median_zero >= 0.1 or float(worst_case["metrics"]["max_abs_diff"]) > 0.25
            ),
            "verdict": verdict,
        },
    }
    artifacts = write_rmft_source_ar_validation_outputs(output_dir, summary, scan_cases, active_scan_values)
    return summary, scan_cases, artifacts
