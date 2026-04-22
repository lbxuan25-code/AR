"""Direction-capability audit for the current forward truth chain.

This module intentionally audits the existing 2D in-plane ``interface_angle``
path. It does not add named direction modes or any c-axis physics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np

from core.interface_geometry import build_interface_segment_catalog, match_reflected_states_on_contour
from core.parameters import ModelParams
from core.pipeline import SpectroscopyPipeline
from core.presets import base_normal_state_params, base_physical_pairing_channels
from core.simulation_model import SimulationModel


DEFAULT_DIRECTION_AUDIT_ANGLES: tuple[tuple[str, float], ...] = (
    ("inplane_100_x_axis", 0.0),
    ("inplane_generic_pi_over_8", float(np.pi / 8.0)),
    ("inplane_110_diagonal", float(np.pi / 4.0)),
    ("inplane_generic_3pi_over_8", float(3.0 * np.pi / 8.0)),
    ("inplane_100_y_axis_equivalent", float(np.pi / 2.0)),
)


@dataclass(frozen=True, slots=True)
class DirectionAuditArtifacts:
    """Generated Task-L direction audit artifact paths."""

    summary_path: Path
    metrics_csv_path: Path
    representative_plot_path: Path | None


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


def _stats(values: list[float] | np.ndarray) -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {"min": None, "max": None, "mean": None, "median": None, "p95": None}
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95.0)),
    }


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


def _baseline_pipeline() -> SpectroscopyPipeline:
    channels = base_physical_pairing_channels()
    params = ModelParams(normal_state=base_normal_state_params(), pairing=channels)
    return SpectroscopyPipeline(model=SimulationModel(params=params, name="direction_capability_audit"))


def _angle_tier(label: str) -> tuple[str, str]:
    if "100" in label or "110" in label:
        return (
            "A",
            "Reliable current raw-angle in-plane high-symmetry path; not yet a named public direction mode.",
        )
    return (
        "B",
        "Computable continuous in-plane angle; generic-angle robustness is deferred to Task N.",
    )


def _match_metrics(
    pipeline: SpectroscopyPipeline,
    *,
    angle: float,
    nk: int,
    energy: float,
    dk: float,
    normal_velocity_tol: float,
    k_parallel_tol: float,
    match_distance_tol: float,
) -> dict[str, object]:
    gap_data = pipeline.gap_on_fermi_surface(nk=nk, energy=energy)
    segment_catalog = build_interface_segment_catalog(gap_data)

    incident_candidates = 0
    matched_count = 0
    same_band_count = 0
    cross_band_fallback_count = 0
    mismatches: list[float] = []

    for contour in gap_data:
        matched = match_reflected_states_on_contour(
            contour,
            segment_catalog,
            pipeline.model,
            angle=angle,
            dk=dk,
            normal_velocity_tol=normal_velocity_tol,
            k_parallel_tol=k_parallel_tol,
            match_distance_tol=match_distance_tol,
            allow_cross_band_fallback=False,
        )
        incident_candidates += int(matched.num_incident_candidates)
        matched_count += int(len(matched.k_in))
        same_band_count += int(np.count_nonzero(matched.matched_same_band))
        cross_band_fallback_count += int(np.count_nonzero(matched.used_cross_band_fallback))
        mismatches.extend(float(value) for value in np.asarray(matched.reflection_mismatch, dtype=np.float64))

    return {
        "incident_channel_count": int(incident_candidates),
        "matched_reflected_channel_count": int(matched_count),
        "unmatched_incident_channel_count": int(max(incident_candidates - matched_count, 0)),
        "matched_fraction_of_incident": float(matched_count / incident_candidates) if incident_candidates else 0.0,
        "same_band_match_fraction": float(same_band_count / matched_count) if matched_count else 0.0,
        "cross_band_fallback_count": int(cross_band_fallback_count),
        "reflection_mismatch": _stats(mismatches),
    }


def _spectrum_metrics(
    pipeline: SpectroscopyPipeline,
    *,
    angle: float,
    bias: np.ndarray,
    barrier_z: float,
    gamma: float,
    temperature: float,
    nk: int,
) -> dict[str, object]:
    result = pipeline.compute_multichannel_btk_conductance(
        interface_angle=float(angle),
        bias=np.asarray(bias, dtype=np.float64),
        barrier_z=float(barrier_z),
        broadening_gamma=float(gamma),
        temperature=float(temperature),
        nk=int(nk),
    )
    filtered_denominator = int(result.num_input_channels + result.num_filtered_channels)
    return {
        "num_transport_input_channels": int(result.num_input_channels),
        "num_transport_channels": int(result.num_channels),
        "num_filtered_channels": int(result.num_filtered_channels),
        "filtered_channel_fraction": float(result.num_filtered_channels / filtered_denominator)
        if filtered_denominator
        else 0.0,
        "num_same_band_transport_channels": int(result.num_same_band_channels),
        "num_contours": int(result.num_contours),
        "mean_normal_transparency": float(result.mean_normal_transparency),
        "mean_mismatch_penalty": float(result.mean_mismatch_penalty),
        "spectrum_features": _curve_features(np.asarray(result.conductance, dtype=np.float64), bias),
        "conductance": [float(value) for value in np.asarray(result.conductance, dtype=np.float64)],
        "conductance_unbroadened": [
            float(value) for value in np.asarray(result.conductance_unbroadened, dtype=np.float64)
        ],
        "approximation": result.approximation,
    }


def _write_metrics_csv(rows: list[dict[str, object]], path: Path) -> Path:
    fieldnames = [
        "angle_label",
        "interface_angle_rad",
        "interface_angle_pi",
        "tier",
        "tier_reason",
        "incident_channel_count",
        "matched_reflected_channel_count",
        "unmatched_incident_channel_count",
        "matched_fraction_of_incident",
        "same_band_match_fraction",
        "reflection_mismatch_mean",
        "reflection_mismatch_median",
        "reflection_mismatch_p95",
        "reflection_mismatch_max",
        "num_transport_input_channels",
        "num_transport_channels",
        "num_filtered_channels",
        "filtered_channel_fraction",
        "mean_normal_transparency",
        "mean_mismatch_penalty",
        "zero_bias_conductance",
        "max_conductance",
        "dynamic_range",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            mismatch = row["reflection_mismatch"]
            features = row["spectrum_features"]
            writer.writerow(
                {
                    "angle_label": row["angle_label"],
                    "interface_angle_rad": row["interface_angle_rad"],
                    "interface_angle_pi": row["interface_angle_pi"],
                    "tier": row["tier"],
                    "tier_reason": row["tier_reason"],
                    "incident_channel_count": row["incident_channel_count"],
                    "matched_reflected_channel_count": row["matched_reflected_channel_count"],
                    "unmatched_incident_channel_count": row["unmatched_incident_channel_count"],
                    "matched_fraction_of_incident": row["matched_fraction_of_incident"],
                    "same_band_match_fraction": row["same_band_match_fraction"],
                    "reflection_mismatch_mean": mismatch["mean"],
                    "reflection_mismatch_median": mismatch["median"],
                    "reflection_mismatch_p95": mismatch["p95"],
                    "reflection_mismatch_max": mismatch["max"],
                    "num_transport_input_channels": row["num_transport_input_channels"],
                    "num_transport_channels": row["num_transport_channels"],
                    "num_filtered_channels": row["num_filtered_channels"],
                    "filtered_channel_fraction": row["filtered_channel_fraction"],
                    "mean_normal_transparency": row["mean_normal_transparency"],
                    "mean_mismatch_penalty": row["mean_mismatch_penalty"],
                    "zero_bias_conductance": features["zero_bias_conductance"],
                    "max_conductance": features["max_conductance"],
                    "dynamic_range": features["dynamic_range"],
                }
            )
    return path


def _plot_representative_scan(rows: list[dict[str, object]], bias: np.ndarray, output_path: Path) -> Path | None:
    if not rows:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), constrained_layout=True)

    for row in rows:
        axes[0].plot(bias, row["conductance"], label=f"{row['angle_label']} ({row['interface_angle_pi']:.3f}π)")
    axes[0].set_xlabel("Bias (meV)")
    axes[0].set_ylabel("Normalized conductance")
    axes[0].set_title("Representative spectra")
    axes[0].legend(fontsize=8)

    labels = [str(row["angle_label"]) for row in rows]
    x = np.arange(len(rows), dtype=np.float64)
    axes[1].bar(x - 0.18, [float(row["matched_fraction_of_incident"]) for row in rows], width=0.36, label="matched / incident")
    axes[1].bar(x + 0.18, [float(row["same_band_match_fraction"]) for row in rows], width=0.36, label="same-band / matched")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35.0, ha="right")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("Fraction")
    axes[1].set_title("Reflected-state matching")
    axes[1].legend(fontsize=8)

    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def run_direction_capability_audit(
    *,
    output_dir: Path = Path("outputs/core/direction_capability_audit"),
    angles: tuple[tuple[str, float], ...] = DEFAULT_DIRECTION_AUDIT_ANGLES,
    nk: int = 41,
    bias_max: float = 40.0,
    num_bias: int = 201,
    barrier_z: float = 0.5,
    gamma: float = 1.0,
    temperature: float = 3.0,
    energy: float = 0.0,
    dk: float = 1.0e-4,
    normal_velocity_tol: float = 1.0e-4,
    k_parallel_tol: float = 5.0e-2,
    match_distance_tol: float = 1.5e-1,
) -> tuple[dict[str, object], DirectionAuditArtifacts]:
    """Run the Task-L representative direction-capability audit."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = _baseline_pipeline()
    bias = np.linspace(-float(bias_max), float(bias_max), int(num_bias), dtype=np.float64)

    rows: list[dict[str, object]] = []
    for label, angle in angles:
        tier, tier_reason = _angle_tier(label)
        match = _match_metrics(
            pipeline,
            angle=float(angle),
            nk=int(nk),
            energy=float(energy),
            dk=float(dk),
            normal_velocity_tol=float(normal_velocity_tol),
            k_parallel_tol=float(k_parallel_tol),
            match_distance_tol=float(match_distance_tol),
        )
        spectrum = _spectrum_metrics(
            pipeline,
            angle=float(angle),
            bias=bias,
            barrier_z=float(barrier_z),
            gamma=float(gamma),
            temperature=float(temperature),
            nk=int(nk),
        )
        rows.append(
            {
                "angle_label": label,
                "interface_angle_rad": float(angle),
                "interface_angle_pi": float(angle / np.pi),
                "normal_vector": [float(np.cos(angle)), float(np.sin(angle))],
                "tangent_vector": [float(-np.sin(angle)), float(np.cos(angle))],
                "tier": tier,
                "tier_reason": tier_reason,
                **match,
                **spectrum,
            }
        )

    metrics_csv_path = _write_metrics_csv(rows, output_dir / "direction_capability_metrics.csv")
    plot_path = _plot_representative_scan(rows, bias, output_dir / "direction_capability_representative_plots.png")

    tier_groups = {
        "Tier A": {
            "meaning": "Current reliable raw-angle, 2D in-plane high-symmetry forward paths.",
            "directions": [
                {
                    "label": "in-plane 100",
                    "supported_as": "raw interface_angle = 0 or pi/2",
                    "status": "supported as 2D in-plane high-symmetry shorthand; named mode deferred to Task M",
                },
                {
                    "label": "in-plane 110",
                    "supported_as": "raw interface_angle = pi/4",
                    "status": "supported as 2D in-plane high-symmetry shorthand; named mode deferred to Task M",
                },
            ],
        },
        "Tier B": {
            "meaning": "Computable continuous in-plane raw angles that require caution until dense Task-N validation.",
            "directions": [
                {
                    "label": "generic in-plane",
                    "supported_as": "raw interface_angle in the 2D kx-ky plane",
                    "status": "computable but not yet promoted to broadly reliable truth-mode directions",
                }
            ],
        },
        "Tier C": {
            "meaning": "Not physically implemented in the current forward model.",
            "directions": [
                {
                    "label": "c-axis",
                    "supported_as": None,
                    "status": "unsupported; the current interface geometry has no kz, out-of-plane velocity, or c-axis injection path",
                }
            ],
        },
    }

    summary: dict[str, object] = {
        "task": "Task L",
        "semantic_contract": {
            "interface_angle_definition": (
                "2D in-plane polar angle, in radians, of the interface normal in the model kx-ky plane."
            ),
            "normal_vector": "(cos(interface_angle), sin(interface_angle))",
            "tangent_vector": "(-sin(interface_angle), cos(interface_angle))",
            "reflection_rule": "k_out = k_in - 2 * (k_in dot normal) * normal, wrapped to the square Brillouin zone",
            "dimensionality": "strictly 2D in-plane in the current implementation",
            "not_implemented": [
                "general 3D crystal-direction interface requests",
                "true c-axis injection",
                "out-of-plane kz-resolved velocity matching",
                "named public direction modes",
            ],
        },
        "support_status": {
            "100": "Supported only as 2D in-plane shorthand for raw angles 0 and pi/2.",
            "110": "Supported only as 2D in-plane shorthand for raw angle pi/4.",
            "c-axis": "Not supported by the current forward model.",
        },
        "scan_config": {
            "nk": int(nk),
            "bias_min_mev": float(-bias_max),
            "bias_max_mev": float(bias_max),
            "num_bias": int(num_bias),
            "barrier_z": float(barrier_z),
            "gamma_mev": float(gamma),
            "temperature_kelvin": float(temperature),
            "energy_mev": float(energy),
            "normal_velocity_tol": float(normal_velocity_tol),
            "k_parallel_tol": float(k_parallel_tol),
            "match_distance_tol": float(match_distance_tol),
        },
        "bias_mev": [float(value) for value in bias],
        "angle_metrics": rows,
        "direction_tiers": tier_groups,
        "final_verdict": (
            "The current forward truth chain has a validated 2D in-plane raw-angle interface path. "
            "In-plane 100 and 110 labels are acceptable shorthand for high-symmetry raw angles, "
            "generic in-plane raw angles are computable but should be treated cautiously until Task N, "
            "and c-axis transport is not physically implemented."
        ),
    }

    summary_path = output_dir / "direction_capability_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(summary), handle, indent=2, sort_keys=True)
        handle.write("\n")

    return summary, DirectionAuditArtifacts(
        summary_path=summary_path,
        metrics_csv_path=metrics_csv_path,
        representative_plot_path=plot_path,
    )
