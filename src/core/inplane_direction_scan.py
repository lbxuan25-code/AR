"""Dense validation scan for generic 2D in-plane interface angles.

Task N asks whether the current continuous raw ``interface_angle`` path is
stable enough to treat generic non-high-symmetry angles as supported truth
modes. This module measures that boundary; it does not add new public modes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np

from core.interface_geometry import InterfaceSegmentCatalog, build_interface_segment_catalog, match_reflected_states_on_contour
from core.parameters import ModelParams
from core.pipeline import GapOnFermiSurface, SpectroscopyPipeline
from core.presets import base_normal_state_params, base_physical_pairing_channels
from core.simulation_model import SimulationModel


TOLERANCE_PROFILES: dict[str, dict[str, float]] = {
    "tight": {"k_parallel_tol": 2.5e-2, "match_distance_tol": 7.5e-2},
    "nominal": {"k_parallel_tol": 5.0e-2, "match_distance_tol": 1.5e-1},
    "loose": {"k_parallel_tol": 1.0e-1, "match_distance_tol": 3.0e-1},
}

SUPPORT_THRESHOLDS: dict[str, dict[str, float]] = {
    "robust": {
        "matched_fraction_min": 0.95,
        "same_band_fraction_min": 0.99,
        "mismatch_p95_max": 2.0e-2,
        "matched_fraction_tolerance_span_max": 0.10,
        "neighbor_max_abs_conductance_step_max": 0.25,
    },
    "caution": {
        "matched_fraction_min": 0.35,
        "same_band_fraction_min": 0.95,
        "mismatch_p95_max": 1.8e-1,
        "matched_fraction_tolerance_span_max": 0.70,
        "neighbor_max_abs_conductance_step_max": 0.75,
    },
}


@dataclass(frozen=True, slots=True)
class InplaneDirectionScanArtifacts:
    """Generated Task-N in-plane direction scan artifact paths."""

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


def _baseline_pipeline() -> SpectroscopyPipeline:
    channels = base_physical_pairing_channels()
    params = ModelParams(normal_state=base_normal_state_params(), pairing=channels)
    return SpectroscopyPipeline(model=SimulationModel(params=params, name="inplane_direction_scan"))


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


def _match_metrics_for_profile(
    gap_data: list[GapOnFermiSurface],
    segment_catalog: InterfaceSegmentCatalog,
    pipeline: SpectroscopyPipeline,
    *,
    angle: float,
    dk: float,
    normal_velocity_tol: float,
    k_parallel_tol: float,
    match_distance_tol: float,
) -> dict[str, object]:
    incident_candidates = 0
    matched_count = 0
    same_band_count = 0
    mismatches: list[float] = []

    for contour in gap_data:
        matched = match_reflected_states_on_contour(
            contour,
            segment_catalog,
            pipeline.model,
            angle=float(angle),
            dk=float(dk),
            normal_velocity_tol=float(normal_velocity_tol),
            k_parallel_tol=float(k_parallel_tol),
            match_distance_tol=float(match_distance_tol),
            allow_cross_band_fallback=False,
        )
        incident_candidates += int(matched.num_incident_candidates)
        matched_count += int(len(matched.k_in))
        same_band_count += int(np.count_nonzero(matched.matched_same_band))
        mismatches.extend(float(value) for value in np.asarray(matched.reflection_mismatch, dtype=np.float64))

    mismatch_stats = _stats(mismatches)
    return {
        "incident_channel_count": int(incident_candidates),
        "matched_reflected_channel_count": int(matched_count),
        "unmatched_incident_channel_count": int(max(incident_candidates - matched_count, 0)),
        "matched_fraction_of_incident": float(matched_count / incident_candidates) if incident_candidates else 0.0,
        "same_band_match_fraction": float(same_band_count / matched_count) if matched_count else 0.0,
        "reflection_mismatch": mismatch_stats,
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
    conductance = np.asarray(result.conductance, dtype=np.float64)
    return {
        "num_transport_input_channels": int(result.num_input_channels),
        "num_transport_channels": int(result.num_channels),
        "num_filtered_channels": int(result.num_filtered_channels),
        "filtered_channel_fraction": float(result.num_filtered_channels / filtered_denominator)
        if filtered_denominator
        else 0.0,
        "num_same_band_transport_channels": int(result.num_same_band_channels),
        "mean_mismatch_penalty": float(result.mean_mismatch_penalty),
        "conductance": [float(value) for value in conductance],
        "spectrum_features": _curve_features(conductance, bias),
    }


def _neighbor_smoothness(rows: list[dict[str, object]]) -> None:
    curves = [np.asarray(row["conductance"], dtype=np.float64) for row in rows]
    for index, row in enumerate(rows):
        neighbor_diffs: list[np.ndarray] = []
        if index > 0:
            neighbor_diffs.append(curves[index] - curves[index - 1])
        if index < len(rows) - 1:
            neighbor_diffs.append(curves[index] - curves[index + 1])
        if not neighbor_diffs:
            row["neighbor_mse_conductance_step"] = 0.0
            row["neighbor_max_abs_conductance_step"] = 0.0
            continue
        mse_values = [float(np.mean(np.square(diff))) for diff in neighbor_diffs]
        max_values = [float(np.max(np.abs(diff))) for diff in neighbor_diffs]
        row["neighbor_mse_conductance_step"] = float(max(mse_values))
        row["neighbor_max_abs_conductance_step"] = float(max(max_values))


def _classify_row(row: dict[str, object]) -> tuple[str, str]:
    nominal = row["match_profiles"]["nominal"]
    robust = SUPPORT_THRESHOLDS["robust"]
    caution = SUPPORT_THRESHOLDS["caution"]
    matched_fraction = float(nominal["matched_fraction_of_incident"])
    same_band_fraction = float(nominal["same_band_match_fraction"])
    mismatch_p95 = nominal["reflection_mismatch"]["p95"]
    mismatch_p95_value = float(mismatch_p95) if mismatch_p95 is not None else float("inf")
    tolerance_span = float(row["matched_fraction_tolerance_span"])
    neighbor_step = float(row["neighbor_max_abs_conductance_step"])

    robust_conditions = (
        matched_fraction >= robust["matched_fraction_min"]
        and same_band_fraction >= robust["same_band_fraction_min"]
        and mismatch_p95_value <= robust["mismatch_p95_max"]
        and tolerance_span <= robust["matched_fraction_tolerance_span_max"]
        and neighbor_step <= robust["neighbor_max_abs_conductance_step_max"]
    )
    if robust_conditions:
        return "robust", "meets all robust thresholds"

    caution_conditions = (
        matched_fraction >= caution["matched_fraction_min"]
        and same_band_fraction >= caution["same_band_fraction_min"]
        and mismatch_p95_value <= caution["mismatch_p95_max"]
        and tolerance_span <= caution["matched_fraction_tolerance_span_max"]
        and neighbor_step <= caution["neighbor_max_abs_conductance_step_max"]
    )
    if caution_conditions:
        return "caution", "computable but fails at least one robust threshold"

    return "unstable", "fails at least one caution threshold"


def _write_metrics_csv(rows: list[dict[str, object]], path: Path) -> Path:
    fieldnames = [
        "angle_index",
        "interface_angle_rad",
        "interface_angle_pi",
        "support_class",
        "support_reason",
        "nominal_incident_channel_count",
        "nominal_matched_reflected_channel_count",
        "nominal_matched_fraction",
        "nominal_same_band_fraction",
        "nominal_mismatch_mean",
        "nominal_mismatch_p95",
        "nominal_mismatch_max",
        "tight_matched_fraction",
        "loose_matched_fraction",
        "matched_fraction_tolerance_span",
        "mismatch_p95_tolerance_span",
        "neighbor_mse_conductance_step",
        "neighbor_max_abs_conductance_step",
        "num_transport_channels",
        "filtered_channel_fraction",
        "mean_mismatch_penalty",
        "zero_bias_conductance",
        "dynamic_range",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            nominal = row["match_profiles"]["nominal"]
            tight = row["match_profiles"]["tight"]
            loose = row["match_profiles"]["loose"]
            features = row["spectrum_features"]
            mismatch = nominal["reflection_mismatch"]
            writer.writerow(
                {
                    "angle_index": row["angle_index"],
                    "interface_angle_rad": row["interface_angle_rad"],
                    "interface_angle_pi": row["interface_angle_pi"],
                    "support_class": row["support_class"],
                    "support_reason": row["support_reason"],
                    "nominal_incident_channel_count": nominal["incident_channel_count"],
                    "nominal_matched_reflected_channel_count": nominal["matched_reflected_channel_count"],
                    "nominal_matched_fraction": nominal["matched_fraction_of_incident"],
                    "nominal_same_band_fraction": nominal["same_band_match_fraction"],
                    "nominal_mismatch_mean": mismatch["mean"],
                    "nominal_mismatch_p95": mismatch["p95"],
                    "nominal_mismatch_max": mismatch["max"],
                    "tight_matched_fraction": tight["matched_fraction_of_incident"],
                    "loose_matched_fraction": loose["matched_fraction_of_incident"],
                    "matched_fraction_tolerance_span": row["matched_fraction_tolerance_span"],
                    "mismatch_p95_tolerance_span": row["mismatch_p95_tolerance_span"],
                    "neighbor_mse_conductance_step": row["neighbor_mse_conductance_step"],
                    "neighbor_max_abs_conductance_step": row["neighbor_max_abs_conductance_step"],
                    "num_transport_channels": row["num_transport_channels"],
                    "filtered_channel_fraction": row["filtered_channel_fraction"],
                    "mean_mismatch_penalty": row["mean_mismatch_penalty"],
                    "zero_bias_conductance": features["zero_bias_conductance"],
                    "dynamic_range": features["dynamic_range"],
                }
            )
    return path


def _plot_scan(rows: list[dict[str, object]], output_path: Path) -> Path | None:
    if not rows:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    angles_pi = np.asarray([row["interface_angle_pi"] for row in rows], dtype=np.float64)
    nominal_match = np.asarray([row["match_profiles"]["nominal"]["matched_fraction_of_incident"] for row in rows], dtype=np.float64)
    tight_match = np.asarray([row["match_profiles"]["tight"]["matched_fraction_of_incident"] for row in rows], dtype=np.float64)
    loose_match = np.asarray([row["match_profiles"]["loose"]["matched_fraction_of_incident"] for row in rows], dtype=np.float64)
    mismatch_p95 = np.asarray([row["match_profiles"]["nominal"]["reflection_mismatch"]["p95"] or np.nan for row in rows], dtype=np.float64)
    smoothness = np.asarray([row["neighbor_max_abs_conductance_step"] for row in rows], dtype=np.float64)
    zero_bias = np.asarray([row["spectrum_features"]["zero_bias_conductance"] for row in rows], dtype=np.float64)

    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.5), constrained_layout=True)
    axes[0, 0].plot(angles_pi, nominal_match, label="nominal", linewidth=2.0)
    axes[0, 0].plot(angles_pi, tight_match, "--", label="tight", linewidth=1.3)
    axes[0, 0].plot(angles_pi, loose_match, ":", label="loose", linewidth=1.6)
    axes[0, 0].set_xlabel("interface_angle / pi")
    axes[0, 0].set_ylabel("matched / incident")
    axes[0, 0].set_title("Reflected-state match fraction")
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(angles_pi, mismatch_p95, color="tab:orange")
    axes[0, 1].axhline(SUPPORT_THRESHOLDS["robust"]["mismatch_p95_max"], color="tab:green", linestyle="--", linewidth=1.0)
    axes[0, 1].axhline(SUPPORT_THRESHOLDS["caution"]["mismatch_p95_max"], color="tab:red", linestyle=":", linewidth=1.0)
    axes[0, 1].set_xlabel("interface_angle / pi")
    axes[0, 1].set_ylabel("p95 mismatch")
    axes[0, 1].set_title("Nominal reflected mismatch")

    axes[1, 0].plot(angles_pi, smoothness, color="tab:purple")
    axes[1, 0].axhline(SUPPORT_THRESHOLDS["robust"]["neighbor_max_abs_conductance_step_max"], color="tab:green", linestyle="--", linewidth=1.0)
    axes[1, 0].axhline(SUPPORT_THRESHOLDS["caution"]["neighbor_max_abs_conductance_step_max"], color="tab:red", linestyle=":", linewidth=1.0)
    axes[1, 0].set_xlabel("interface_angle / pi")
    axes[1, 0].set_ylabel("max |G(theta)-G(neighbor)|")
    axes[1, 0].set_title("Angular spectrum smoothness")

    class_colors = {"robust": "tab:green", "caution": "tab:orange", "unstable": "tab:red"}
    for support_class, color in class_colors.items():
        mask = np.asarray([row["support_class"] == support_class for row in rows], dtype=np.bool_)
        axes[1, 1].scatter(angles_pi[mask], zero_bias[mask], color=color, label=support_class, s=32)
    axes[1, 1].set_xlabel("interface_angle / pi")
    axes[1, 1].set_ylabel("zero-bias conductance")
    axes[1, 1].set_title("Support class vs spectrum feature")
    axes[1, 1].legend(fontsize=8)

    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return output_path


def _support_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    return {name: int(sum(row["support_class"] == name for row in rows)) for name in ("robust", "caution", "unstable")}


def run_inplane_direction_scan(
    *,
    output_dir: Path = Path("outputs/core/inplane_direction_scan"),
    num_angles: int = 33,
    nk: int = 41,
    bias_max: float = 40.0,
    num_bias: int = 201,
    barrier_z: float = 0.5,
    gamma: float = 1.0,
    temperature: float = 3.0,
    energy: float = 0.0,
    dk: float = 1.0e-4,
    normal_velocity_tol: float = 1.0e-4,
) -> tuple[dict[str, object], InplaneDirectionScanArtifacts]:
    """Run the Task-N dense generic in-plane direction validation scan."""

    if num_angles < 5:
        raise ValueError("num_angles must be at least 5 for a meaningful in-plane scan.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = _baseline_pipeline()
    gap_data = pipeline.gap_on_fermi_surface(nk=int(nk), energy=float(energy))
    segment_catalog = build_interface_segment_catalog(gap_data)
    bias = np.linspace(-float(bias_max), float(bias_max), int(num_bias), dtype=np.float64)
    angles = np.linspace(0.0, 0.5 * np.pi, int(num_angles), dtype=np.float64)

    rows: list[dict[str, object]] = []
    for angle_index, angle in enumerate(angles):
        match_profiles: dict[str, object] = {}
        for profile_name, tolerances in TOLERANCE_PROFILES.items():
            match_profiles[profile_name] = _match_metrics_for_profile(
                gap_data,
                segment_catalog,
                pipeline,
                angle=float(angle),
                dk=float(dk),
                normal_velocity_tol=float(normal_velocity_tol),
                k_parallel_tol=float(tolerances["k_parallel_tol"]),
                match_distance_tol=float(tolerances["match_distance_tol"]),
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
        matched_fractions = [
            float(match_profiles[name]["matched_fraction_of_incident"]) for name in ("tight", "nominal", "loose")
        ]
        mismatch_p95_values = [
            match_profiles[name]["reflection_mismatch"]["p95"] for name in ("tight", "nominal", "loose")
        ]
        finite_mismatch_p95 = [float(value) for value in mismatch_p95_values if value is not None]
        rows.append(
            {
                "angle_index": int(angle_index),
                "interface_angle_rad": float(angle),
                "interface_angle_pi": float(angle / np.pi),
                "match_profiles": match_profiles,
                "matched_fraction_tolerance_span": float(max(matched_fractions) - min(matched_fractions)),
                "mismatch_p95_tolerance_span": float(max(finite_mismatch_p95) - min(finite_mismatch_p95))
                if finite_mismatch_p95
                else None,
                **spectrum,
            }
        )

    _neighbor_smoothness(rows)
    for row in rows:
        support_class, support_reason = _classify_row(row)
        row["support_class"] = support_class
        row["support_reason"] = support_reason

    metrics_csv_path = _write_metrics_csv(rows, output_dir / "metrics.csv")
    plot_path = _plot_scan(rows, output_dir / "inplane_direction_scan_plots.png")

    support_counts = _support_counts(rows)
    generic_rows = [
        row
        for row in rows
        if not np.isclose(row["interface_angle_rad"], 0.0)
        and not np.isclose(row["interface_angle_rad"], np.pi / 4.0)
        and not np.isclose(row["interface_angle_rad"], np.pi / 2.0)
    ]
    generic_counts = _support_counts(generic_rows)
    generic_supported = generic_counts["robust"] > 0
    if generic_supported:
        generic_decision = "partially_supported_with_restrictions"
    elif generic_counts["caution"] > 0:
        generic_decision = "diagnostic_or_caution_only"
    else:
        generic_decision = "not_supported_as_truth_mode"

    summary: dict[str, object] = {
        "task": "Task N",
        "scan_config": {
            "angle_domain": "[0, pi/2]",
            "num_angles": int(num_angles),
            "nk": int(nk),
            "bias_min_mev": float(-bias_max),
            "bias_max_mev": float(bias_max),
            "num_bias": int(num_bias),
            "barrier_z": float(barrier_z),
            "gamma_mev": float(gamma),
            "temperature_kelvin": float(temperature),
            "energy_mev": float(energy),
            "normal_velocity_tol": float(normal_velocity_tol),
            "tolerance_profiles": TOLERANCE_PROFILES,
        },
        "support_thresholds": SUPPORT_THRESHOLDS,
        "support_counts_all_angles": support_counts,
        "support_counts_generic_non_high_symmetry": generic_counts,
        "generic_inplane_support_decision": generic_decision,
        "restricted_supported_domain": {
            "truth_mode_recommendation": (
                "Use canonical in-plane high-symmetry modes as truth modes. Treat generic raw in-plane angles "
                "as diagnostic/caution-only unless an angle satisfies the robust thresholds in this scan."
            ),
            "currently_promoted_named_modes": ["inplane_100", "inplane_110"],
            "c_axis_status": "out_of_scope_for_task_n_and_still_unsupported",
        },
        "bias_mev": [float(value) for value in bias],
        "angle_metrics": rows,
        "final_verdict": (
            "Generic non-high-symmetry in-plane angles are not promoted to broadly supported truth modes by Task N. "
            "The dense scan records which raw angles are robust, caution-required, or unstable under explicit "
            "matching, smoothness, and tolerance-sensitivity thresholds."
        ),
    }

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(summary), handle, indent=2, sort_keys=True)
        handle.write("\n")

    return summary, InplaneDirectionScanArtifacts(
        summary_path=summary_path,
        metrics_csv_path=metrics_csv_path,
        plot_path=plot_path,
    )
