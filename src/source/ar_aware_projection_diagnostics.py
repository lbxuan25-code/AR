"""Comparison diagnostics for default vs AR-aware round-2 projection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np

from core.bdg import bdg_matrix
from core.normal_state import h0_matrix
from core.parameters import ModelParams
from core.pipeline import SpectroscopyPipeline
from core.presets import base_normal_state_params, base_physical_pairing_channels
from core.simulation_model import SimulationModel

from .luo_loader import load_luo_samples
from .round2_projection import (
    DEFAULT_ROUND2_PROJECTION_CONFIG,
    ROUND2_CHANNEL_NAMES,
    Round2ProjectionConfig,
    gauge_fix_source_tensors,
    project_luo_sample_to_round2_channels,
    source_pairing_tensors_meV,
)


@dataclass(slots=True)
class ARAwareComparisonArtifacts:
    summary_path: Path
    examples_csv_path: Path
    spectra_plot_path: Path | None


@dataclass(frozen=True, slots=True)
class SourceTensorModel:
    """Diagnostic model that uses the full gauge-fixed source pairing tensors."""

    params: ModelParams
    delta_x: np.ndarray
    delta_y: np.ndarray
    delta_z: np.ndarray
    name: str = "source_tensor_reference"

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


def _real_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "p95": float(np.percentile(arr, 95.0)),
    }


def _abs_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.abs(np.asarray(values, dtype=np.complex128).reshape(-1))
    return {
        "max_abs": float(np.max(arr)),
        "mean_abs": float(np.mean(arr)),
        "median_abs": float(np.median(arr)),
        "p95_abs": float(np.percentile(arr, 95.0)),
    }


def _representative_indices(per_sample: list[dict[str, object]]) -> dict[str, int]:
    retained = np.asarray([item["default_metrics"]["retained_ratio_total"] for item in per_sample], dtype=np.float64)
    ranking = np.argsort(retained)
    return {
        "best": int(ranking[-1]),
        "median": int(ranking[len(ranking) // 2]),
        "worst": int(ranking[0]),
    }


def _source_reference_model(sample, config: Round2ProjectionConfig) -> SourceTensorModel:
    source_x, source_y, source_z = source_pairing_tensors_meV(sample)
    gauge_x, gauge_y, gauge_z, _ = gauge_fix_source_tensors(source_x, source_y, source_z, config=config)
    return SourceTensorModel(
        params=ModelParams(normal_state=base_normal_state_params(), pairing=base_physical_pairing_channels()),
        delta_x=gauge_x,
        delta_y=gauge_y,
        delta_z=gauge_z,
        name=f"source_ref::{sample.sample_id}",
    )


def _pairing_model_from_projection(sample, config: Round2ProjectionConfig) -> tuple[SimulationModel, dict[str, object]]:
    projected = project_luo_sample_to_round2_channels(sample, config=config)
    assert projected.projected_physical_channels is not None
    model = SimulationModel(
        params=ModelParams(
            normal_state=base_normal_state_params(),
            pairing=projected.projected_physical_channels,
        ),
        name=f"{config.source_entry_weight_mode}::{sample.sample_id}",
    )
    return model, {
        "channels": projected.projected_physical_channels.to_dict(),
        "metrics": dict(projected.round2_projection_metrics),
        "metadata": dict(projected.round2_projection_metadata),
    }


def _spectrum_against_reference(
    sample,
    default_config: Round2ProjectionConfig,
    ar_config: Round2ProjectionConfig,
    bias: np.ndarray,
    nk: int,
    interface_angle: float,
    barrier_z: float,
    gamma: float,
    temperature: float,
) -> dict[str, object]:
    reference_model = _source_reference_model(sample, config=default_config)
    default_model, default_payload = _pairing_model_from_projection(sample, config=default_config)
    ar_model, ar_payload = _pairing_model_from_projection(sample, config=ar_config)

    ref_pipeline = SpectroscopyPipeline(model=reference_model)
    default_pipeline = SpectroscopyPipeline(model=default_model)
    ar_pipeline = SpectroscopyPipeline(model=ar_model)

    kwargs = {
        "interface_angle": float(interface_angle),
        "bias": np.asarray(bias, dtype=np.float64),
        "barrier_z": float(barrier_z),
        "broadening_gamma": float(gamma),
        "temperature": float(temperature),
        "nk": int(nk),
    }
    ref_result = ref_pipeline.compute_multichannel_btk_conductance(**kwargs)
    default_result = default_pipeline.compute_multichannel_btk_conductance(**kwargs)
    ar_result = ar_pipeline.compute_multichannel_btk_conductance(**kwargs)

    ref_curve = np.asarray(ref_result.conductance, dtype=np.float64)
    default_curve = np.asarray(default_result.conductance, dtype=np.float64)
    ar_curve = np.asarray(ar_result.conductance, dtype=np.float64)
    mse_default = float(np.mean((default_curve - ref_curve) ** 2))
    mse_ar = float(np.mean((ar_curve - ref_curve) ** 2))
    return {
        "sample_id": sample.sample_id,
        "default": default_payload,
        "ar_aware": ar_payload,
        "spectral_metrics": {
            "reference_vs_default_mse": mse_default,
            "reference_vs_ar_aware_mse": mse_ar,
            "ar_aware_mse_improvement": float(mse_default - mse_ar),
            "reference_vs_default_max_abs": float(np.max(np.abs(default_curve - ref_curve))),
            "reference_vs_ar_aware_max_abs": float(np.max(np.abs(ar_curve - ref_curve))),
        },
        "curves": {
            "bias": np.asarray(bias, dtype=np.float64),
            "reference": ref_curve,
            "default": default_curve,
            "ar_aware": ar_curve,
        },
    }


def summarize_ar_aware_projection_comparison(
    max_samples: int | None = None,
    default_config: Round2ProjectionConfig = DEFAULT_ROUND2_PROJECTION_CONFIG,
    ar_config: Round2ProjectionConfig | None = None,
    representative_spectrum_nk: int = 41,
    representative_num_bias: int = 201,
) -> tuple[list[dict[str, object]], dict[str, object], list[dict[str, object]]]:
    samples = load_luo_samples()
    if max_samples is not None:
        samples = samples[: int(max_samples)]
    ar_config = ar_config or Round2ProjectionConfig(source_entry_weight_mode="ar_aware")

    per_sample: list[dict[str, object]] = []
    for sample in samples:
        default_projected = project_luo_sample_to_round2_channels(sample, config=default_config)
        ar_projected = project_luo_sample_to_round2_channels(sample, config=ar_config)
        assert default_projected.projected_physical_channels is not None
        assert ar_projected.projected_physical_channels is not None
        channel_delta = {
            name: getattr(ar_projected.projected_physical_channels, name) - getattr(default_projected.projected_physical_channels, name)
            for name in ROUND2_CHANNEL_NAMES
        }
        default_channels = default_projected.projected_physical_channels.to_dict()
        ar_channels = ar_projected.projected_physical_channels.to_dict()
        default_norm = float(np.sqrt(sum(abs(default_channels[name]) ** 2 for name in ROUND2_CHANNEL_NAMES)))
        delta_norm = float(np.sqrt(sum(abs(channel_delta[name]) ** 2 for name in ROUND2_CHANNEL_NAMES)))
        per_sample.append(
            {
                "sample_id": sample.sample_id,
                "default_metrics": dict(default_projected.round2_projection_metrics),
                "ar_metrics": dict(ar_projected.round2_projection_metrics),
                "default_channels": default_channels,
                "ar_channels": ar_channels,
                "channel_delta": channel_delta,
                "channel_delta_norm": delta_norm,
                "channel_relative_delta_norm": float(delta_norm / default_norm) if default_norm > 0.0 else 0.0,
                "ar_channel_relevance_scores": dict(ar_projected.round2_projection_metadata.get("ar_channel_relevance_scores", {})),
            }
        )

    representative_indices = _representative_indices(per_sample)
    bias = np.linspace(-40.0, 40.0, int(representative_num_bias), dtype=np.float64)
    representative_samples = [
        _spectrum_against_reference(
            samples[index],
            default_config=default_config,
            ar_config=ar_config,
            bias=bias,
            nk=representative_spectrum_nk,
            interface_angle=0.0,
            barrier_z=0.5,
            gamma=1.0,
            temperature=3.0,
        )
        for index in representative_indices.values()
    ]

    spectral_improvements = np.asarray(
        [item["spectral_metrics"]["ar_aware_mse_improvement"] for item in representative_samples],
        dtype=np.float64,
    )
    retained_improvement = np.asarray(
        [item["ar_metrics"]["retained_ratio_total"] - item["default_metrics"]["retained_ratio_total"] for item in per_sample],
        dtype=np.float64,
    )
    residual_reduction = np.asarray(
        [item["default_metrics"]["residual_norm_total"] - item["ar_metrics"]["residual_norm_total"] for item in per_sample],
        dtype=np.float64,
    )
    summary = {
        "num_samples": len(per_sample),
        "default_config": default_config.to_dict(),
        "ar_aware_config": ar_config.to_dict(),
        "retained_ratio_total": {
            "default": _real_stats(np.asarray([item["default_metrics"]["retained_ratio_total"] for item in per_sample], dtype=np.float64)),
            "ar_aware": _real_stats(np.asarray([item["ar_metrics"]["retained_ratio_total"] for item in per_sample], dtype=np.float64)),
            "ar_aware_minus_default": _real_stats(retained_improvement),
        },
        "residual_norm_total": {
            "default": _real_stats(np.asarray([item["default_metrics"]["residual_norm_total"] for item in per_sample], dtype=np.float64)),
            "ar_aware": _real_stats(np.asarray([item["ar_metrics"]["residual_norm_total"] for item in per_sample], dtype=np.float64)),
            "default_minus_ar_aware": _real_stats(residual_reduction),
        },
        "projected_channel_stability": {
            "channel_delta_norm": _real_stats(np.asarray([item["channel_delta_norm"] for item in per_sample], dtype=np.float64)),
            "channel_relative_delta_norm": _real_stats(np.asarray([item["channel_relative_delta_norm"] for item in per_sample], dtype=np.float64)),
            "per_channel_abs_delta": {
                name: _abs_stats(np.asarray([item["channel_delta"][name] for item in per_sample], dtype=np.complex128))
                for name in ROUND2_CHANNEL_NAMES
            },
        },
        "representative_spectral_agreement": {
            item["sample_id"]: item["spectral_metrics"]
            for item in representative_samples
        },
        "spectral_mse_improvement": _real_stats(spectral_improvements),
        "verdict": (
            "AR-aware entry weighting measurably helps."
            if float(np.median(retained_improvement)) > 1.0e-4 and float(np.mean(spectral_improvements)) > 0.0
            else "AR-aware entry weighting does not materially help relative to the current default weighted-ridge path."
        ),
    }
    return per_sample, summary, representative_samples


def write_ar_aware_projection_outputs(
    output_dir: Path,
    per_sample: list[dict[str, object]],
    summary: dict[str, object],
    representative_samples: list[dict[str, object]],
    make_plots: bool = True,
) -> ARAwareComparisonArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "ar_aware_projection_comparison_summary.json"
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")

    examples_path = output_dir / "ar_aware_projection_examples.csv"
    ranked = sorted(per_sample, key=lambda item: item["ar_metrics"]["retained_ratio_total"] - item["default_metrics"]["retained_ratio_total"], reverse=True)
    candidate_rows = ranked[:3] + sorted(per_sample, key=lambda item: item["ar_metrics"]["retained_ratio_total"] - item["default_metrics"]["retained_ratio_total"])[:3]
    seen: set[str] = set()
    fieldnames = [
        "sample_id",
        "default_retained_ratio_total",
        "ar_aware_retained_ratio_total",
        "retained_ratio_delta",
        "default_residual_norm_total",
        "ar_aware_residual_norm_total",
        "residual_norm_reduction",
        "channel_delta_norm",
        "channel_relative_delta_norm",
    ]
    with examples_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in candidate_rows:
            if row["sample_id"] in seen:
                continue
            seen.add(row["sample_id"])
            writer.writerow(
                {
                    "sample_id": row["sample_id"],
                    "default_retained_ratio_total": row["default_metrics"]["retained_ratio_total"],
                    "ar_aware_retained_ratio_total": row["ar_metrics"]["retained_ratio_total"],
                    "retained_ratio_delta": row["ar_metrics"]["retained_ratio_total"] - row["default_metrics"]["retained_ratio_total"],
                    "default_residual_norm_total": row["default_metrics"]["residual_norm_total"],
                    "ar_aware_residual_norm_total": row["ar_metrics"]["residual_norm_total"],
                    "residual_norm_reduction": row["default_metrics"]["residual_norm_total"] - row["ar_metrics"]["residual_norm_total"],
                    "channel_delta_norm": row["channel_delta_norm"],
                    "channel_relative_delta_norm": row["channel_relative_delta_norm"],
                }
            )

    spectra_plot_path: Path | None = None
    if make_plots:
        spectra_plot_path = output_dir / "ar_aware_projection_representative_spectra.png"
        figure, axes = plt.subplots(len(representative_samples), 1, figsize=(8.0, 3.2 * len(representative_samples)), constrained_layout=True)
        if len(representative_samples) == 1:
            axes = [axes]
        for axis, row in zip(axes, representative_samples, strict=True):
            bias = np.asarray(row["curves"]["bias"], dtype=np.float64)
            axis.plot(bias, row["curves"]["reference"], label="source_ref", linewidth=1.8)
            axis.plot(bias, row["curves"]["default"], label="default_projection", linewidth=1.4)
            axis.plot(bias, row["curves"]["ar_aware"], label="ar_aware_projection", linewidth=1.4)
            axis.set_title(
                f"{row['sample_id']} | dMSE = {row['spectral_metrics']['ar_aware_mse_improvement']:.3e}"
            )
            axis.set_xlabel("Bias (meV)")
            axis.set_ylabel("Conductance")
            axis.grid(alpha=0.2)
            axis.legend(loc="best", fontsize=8)
        figure.savefig(spectra_plot_path, dpi=160)
        plt.close(figure)

    return ARAwareComparisonArtifacts(
        summary_path=summary_path,
        examples_csv_path=examples_path,
        spectra_plot_path=spectra_plot_path,
    )
