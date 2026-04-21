"""Diagnostics, baseline selection, and comparisons for the round-2 truth layer."""

from __future__ import annotations

from pathlib import Path
import csv
import json

import numpy as np

from core.conventions import CORE_PHYSICAL_PAIRING_CHANNELS, OPTIONAL_PHYSICAL_PAIRING_CHANNELS
from core.formal_baseline import AUTHORITATIVE_ROUND2_BASELINE_RECORD
from core.parameters import PhysicalPairingChannels

from .luo_loader import load_luo_samples
from .luo_projection import project_luo_sample_to_pairing
from .projection_metrics import build_projection_metric_bundle
from .round2_projection import (
    DEFAULT_ROUND2_PROJECTION_CONFIG,
    ROUND2_CHANNEL_NAMES,
    Round2ProjectionConfig,
    project_luo_sample_to_round2_channels,
    reconstruct_source_tensors_from_channels,
    source_pairing_tensors_meV,
)
from .schema import LuoSample

PROJECT_ROOT = AUTHORITATIVE_ROUND2_BASELINE_RECORD.parents[2]


def _repo_relative_path(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT))


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


def _round1_reconstruction_triplet(sample: LuoSample) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    projected = project_luo_sample_to_pairing(sample).projected_pairing_params
    assert projected is not None

    recon_x = np.zeros((4, 4), dtype=np.complex128)
    recon_y = np.zeros((4, 4), dtype=np.complex128)
    recon_z = np.zeros((4, 4), dtype=np.complex128)
    recon_x[0, 0] = recon_x[2, 2] = projected.eta_z_s
    recon_y[0, 0] = recon_y[2, 2] = projected.eta_z_s
    recon_x[1, 1] = recon_x[3, 3] = projected.eta_x_s + projected.eta_x_d
    recon_y[1, 1] = recon_y[3, 3] = projected.eta_x_s - projected.eta_x_d
    recon_z[0, 2] = recon_z[2, 0] = projected.eta_z_perp
    return recon_x, recon_y, recon_z


def _round1_reconstruction_metrics(sample: LuoSample) -> dict[str, float]:
    delta_x, delta_y, delta_z = source_pairing_tensors_meV(sample)
    recon_x, recon_y, recon_z = _round1_reconstruction_triplet(sample)
    return build_projection_metric_bundle(delta_x, delta_y, delta_z, recon_x, recon_y, recon_z)


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


def _mean_p(sample: LuoSample) -> float | None:
    if "p" not in sample.coordinates:
        return None
    values = np.asarray(sample.coordinates["p"], dtype=np.float64).reshape(-1)
    if values.size == 0:
        return None
    return float(np.mean(values))


def select_round2_baseline_cluster(samples: list[LuoSample], cluster_size: int = 8) -> list[LuoSample]:
    """Return the Stage-3 low-temperature reference cluster.

    The formal baseline is defined from the charge-balanced low-temperature
    branch in the temperature sweep dataset, which is the cleanest source-side
    route to a stable round-2 truth-state reference.
    """

    candidates = [
        sample
        for sample in samples
        if sample.sample_kind == "temperature sweep RMFT pairing data"
        and abs(_mean_p(sample) or 0.0) < 1.0e-8
        and float(sample.coordinates.get("temperature_eV", 1.0e9)) <= 1.0e-3
    ]
    candidates.sort(key=lambda sample: float(sample.coordinates.get("temperature_eV", 1.0e9)))
    if len(candidates) < cluster_size:
        raise RuntimeError(
            f"Expected at least {cluster_size} low-temperature charge-balanced samples for the round-2 baseline, "
            f"found {len(candidates)}."
        )
    return candidates[:cluster_size]


def _componentwise_complex_median(values: list[complex]) -> complex:
    arr = np.asarray(values, dtype=np.complex128)
    return complex(np.median(arr.real) + 1.0j * np.median(arr.imag))


def build_round2_baseline_summary(
    samples: list[LuoSample],
    config: Round2ProjectionConfig = DEFAULT_ROUND2_PROJECTION_CONFIG,
    cluster_size: int = 8,
) -> tuple[PhysicalPairingChannels, dict[str, object]]:
    cluster = select_round2_baseline_cluster(samples, cluster_size=cluster_size)
    projected_cluster = [project_luo_sample_to_round2_channels(sample, config=config) for sample in cluster]

    channel_values = {
        name: [getattr(sample.projected_physical_channels, name) for sample in projected_cluster]
        for name in ROUND2_CHANNEL_NAMES
    }
    channels = PhysicalPairingChannels(
        **{name: _componentwise_complex_median(channel_values[name]) for name in ROUND2_CHANNEL_NAMES}
    )
    if config.freeze_optional_weak_channel_by_default and config.optional_weak_channel_name == "delta_zx_s":
        channel_payload = channels.to_dict()
        channel_payload["delta_zx_s"] = 0.0 + 0.0j
        channels = PhysicalPairingChannels(**channel_payload)

    temperatures = np.asarray([float(sample.coordinates["temperature_eV"]) for sample in cluster], dtype=np.float64)
    spread = {
        name: _abs_stats(
            np.asarray(channel_values[name], dtype=np.complex128) - getattr(channels, name)
        )
        for name in ROUND2_CHANNEL_NAMES
    }
    return channels, {
        "record_role": "authoritative_formal_round2_baseline",
        "authoritative_record_path": _repo_relative_path(AUTHORITATIVE_ROUND2_BASELINE_RECORD),
        "selection_rule": (
            "temperature sweep RMFT pairing data, charge-balanced p≈0 branch, "
            "temperature_eV <= 1.0e-3, first 8 samples sorted by temperature"
        ),
        "generation_method": "source-side default round-2 projection, componentwise complex median",
        "weak_channel_policy": {
            "channel": "delta_zx_s",
            "default_task_c_freeze_applied": bool(
                config.freeze_optional_weak_channel_by_default
                and config.optional_weak_channel_name == "delta_zx_s"
            ),
            "default_value": 0.0 + 0.0j,
        },
        "num_samples": len(cluster),
        "sample_ids": [sample.sample_id for sample in cluster],
        "temperature_range_eV": {
            "min": float(np.min(temperatures)),
            "max": float(np.max(temperatures)),
        },
        "pairing_channels": channels.to_dict(),
        "channel_spread_about_median": spread,
    }


def summarize_round2_projection(
    max_samples: int | None = None,
    config: Round2ProjectionConfig = DEFAULT_ROUND2_PROJECTION_CONFIG,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    samples = load_luo_samples()
    baseline_channels, baseline_summary = build_round2_baseline_summary(samples, config=config)
    eval_samples = samples[: int(max_samples)] if max_samples is not None else samples

    per_sample: list[dict[str, object]] = []
    for sample in eval_samples:
        round2_sample = project_luo_sample_to_round2_channels(sample, config=config)
        round2_channels = round2_sample.projected_physical_channels
        assert round2_channels is not None
        round2_metrics = dict(round2_sample.round2_projection_metrics)
        round1_metrics = _round1_reconstruction_metrics(sample)
        delta_x, delta_y, delta_z = source_pairing_tensors_meV(sample)
        recon_x, recon_y, recon_z = reconstruct_source_tensors_from_channels(round2_channels)
        weak_channel_abs = abs(round2_channels.delta_zx_s)
        core_channel_abs = max(abs(getattr(round2_channels, name)) for name in CORE_PHYSICAL_PAIRING_CHANNELS)

        per_sample.append(
            {
                "sample_id": sample.sample_id,
                "round2_channels": round2_channels.to_dict(),
                "round2_metrics": round2_metrics,
                "round1_metrics": round1_metrics,
                "round2_metadata": dict(round2_sample.round2_projection_metadata),
                "retained_ratio_improvement": float(round2_metrics["retained_ratio_total"] - round1_metrics["retained_ratio_total"]),
                "residual_norm_reduction": float(round1_metrics["residual_norm_total"] - round2_metrics["residual_norm_total"]),
                "omitted_fraction_reduction": float(round1_metrics["omitted_fraction_total"] - round2_metrics["omitted_fraction_total"]),
                "optional_to_core_ratio": float(weak_channel_abs / core_channel_abs) if core_channel_abs > 0.0 else 0.0,
                "round2_source_triplet": {
                    "source_norm_total": float(round2_metrics["source_norm_total"]),
                    "recon_norm_total": float(round2_metrics["recon_norm_total"]),
                    "residual_norm_total": float(round2_metrics["residual_norm_total"]),
                    "delta_x_residual_norm": float(np.linalg.norm(delta_x - recon_x, ord="fro")),
                    "delta_y_residual_norm": float(np.linalg.norm(delta_y - recon_y, ord="fro")),
                    "delta_z_residual_norm": float(np.linalg.norm(delta_z - recon_z, ord="fro")),
                },
            }
        )

    gauge_anchor_names = [item["round2_metadata"]["anchor_channel"] for item in per_sample]
    unique_anchor_names = sorted(set(gauge_anchor_names))
    optional_policy_rows = [item["round2_metadata"].get("optional_channel_policy", {}) for item in per_sample]
    activated_mask = np.asarray([bool(row.get("activated", True)) for row in optional_policy_rows], dtype=np.bool_)
    round2_summary = {
        "num_samples": len(per_sample),
        "metric_definition": {
            "retained_ratio_total": "1 - residual_norm_total / source_norm_total",
            "retained_ratio_x_y_z": "1 - residual_norm_block / source_norm_block",
            "omitted_fraction_total": "residual_norm_total / source_norm_total",
            "residual_norm_total": "sqrt(residual_x^2 + residual_y^2 + residual_z^2) with Frobenius block norms",
        },
        "projection_config": config.to_dict(),
        "channel_groups": {
            "core_channels": list(CORE_PHYSICAL_PAIRING_CHANNELS),
            "optional_channels": list(OPTIONAL_PHYSICAL_PAIRING_CHANNELS),
        },
        "baseline_cluster_summary": baseline_summary,
        "baseline_channels": baseline_channels.to_dict(),
        "channel_magnitude_stats": {
            name: _abs_stats(np.asarray([item["round2_channels"][name] for item in per_sample]))
            for name in ROUND2_CHANNEL_NAMES
        },
        "optional_channel_relative_scale": _real_stats(np.asarray([item["optional_to_core_ratio"] for item in per_sample])),
        "optional_channel_policy_summary": {
            "mode": optional_policy_rows[0].get("mode", "not_recorded") if optional_policy_rows else "not_recorded",
            "activation_fraction": float(np.mean(activated_mask.astype(np.float64))) if activated_mask.size else 0.0,
            "frozen_fraction": float(np.mean((~activated_mask).astype(np.float64))) if activated_mask.size else 0.0,
            "relative_magnitude_if_full_fit": _real_stats(
                np.asarray([float(row.get("relative_magnitude", 0.0)) for row in optional_policy_rows], dtype=np.float64)
            ),
            "residual_reduction_if_activated": _real_stats(
                np.asarray([float(row.get("residual_reduction_if_activated", 0.0)) for row in optional_policy_rows], dtype=np.float64)
            ),
        },
        "gauge_anchor_stats": {
            name: gauge_anchor_names.count(name)
            for name in unique_anchor_names
        },
        "gauge_phase_radians": _real_stats(
            np.asarray([item["round2_metadata"]["gauge_phase_radians"] for item in per_sample], dtype=np.float64)
        ),
        "round2_retained_ratio_total": _real_stats(np.asarray([item["round2_metrics"]["retained_ratio_total"] for item in per_sample])),
        "round2_residual_norm_total": _real_stats(np.asarray([item["round2_metrics"]["residual_norm_total"] for item in per_sample])),
        "round2_omitted_fraction_total": _real_stats(np.asarray([item["round2_metrics"]["omitted_fraction_total"] for item in per_sample])),
    }
    comparison_summary = {
        "num_samples": len(per_sample),
        "metric_definition": round2_summary["metric_definition"],
        "round1_retained_ratio_total": _real_stats(np.asarray([item["round1_metrics"]["retained_ratio_total"] for item in per_sample])),
        "round2_retained_ratio_total": _real_stats(np.asarray([item["round2_metrics"]["retained_ratio_total"] for item in per_sample])),
        "retained_ratio_improvement": _real_stats(np.asarray([item["retained_ratio_improvement"] for item in per_sample])),
        "round1_residual_norm_total": _real_stats(np.asarray([item["round1_metrics"]["residual_norm_total"] for item in per_sample])),
        "round2_residual_norm_total": _real_stats(np.asarray([item["round2_metrics"]["residual_norm_total"] for item in per_sample])),
        "residual_norm_reduction": _real_stats(np.asarray([item["residual_norm_reduction"] for item in per_sample])),
        "round1_omitted_fraction_total": _real_stats(np.asarray([item["round1_metrics"]["omitted_fraction_total"] for item in per_sample])),
        "round2_omitted_fraction_total": _real_stats(np.asarray([item["round2_metrics"]["omitted_fraction_total"] for item in per_sample])),
        "omitted_fraction_reduction": _real_stats(np.asarray([item["omitted_fraction_reduction"] for item in per_sample])),
        "verdict": (
            "Round-2 physical channels improve source reconstruction relative to round 1."
            if np.median(np.asarray([item["retained_ratio_improvement"] for item in per_sample], dtype=np.float64)) > 0.0
            else "Round-2 physical channels do not improve the retained ratio relative to round 1."
        ),
    }
    return per_sample, round2_summary, comparison_summary


def write_round2_projection_outputs(
    output_dir: Path,
    per_sample: list[dict[str, object]],
    round2_summary: dict[str, object],
    comparison_summary: dict[str, object] | None = None,
) -> tuple[Path, Path, Path | None, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "round2_projection_summary.json"
    summary_path.write_text(json.dumps(_json_ready(round2_summary), indent=2), encoding="utf-8")

    baseline_path = output_dir / "round2_baseline_selection.json"
    baseline_path.write_text(json.dumps(_json_ready(round2_summary["baseline_cluster_summary"]), indent=2), encoding="utf-8")

    examples_path = output_dir / "round2_projection_examples.csv"
    ranked = sorted(per_sample, key=lambda item: item["retained_ratio_improvement"], reverse=True)
    candidate_rows = ranked[:3] + sorted(per_sample, key=lambda item: item["retained_ratio_improvement"])[:3]
    seen: set[str] = set()
    fieldnames = [
        "sample_id",
        *ROUND2_CHANNEL_NAMES,
        "gauge_anchor_channel",
        "gauge_phase_radians",
        "round1_retained_ratio_total",
        "round2_retained_ratio_total",
        "retained_ratio_improvement",
        "round1_residual_norm_total",
        "round2_residual_norm_total",
        "residual_norm_reduction",
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
                    **{name: row["round2_channels"][name] for name in ROUND2_CHANNEL_NAMES},
                    "gauge_anchor_channel": row["round2_metadata"]["anchor_channel"],
                    "gauge_phase_radians": row["round2_metadata"]["gauge_phase_radians"],
                    "round1_retained_ratio_total": row["round1_metrics"]["retained_ratio_total"],
                    "round2_retained_ratio_total": row["round2_metrics"]["retained_ratio_total"],
                    "retained_ratio_improvement": row["retained_ratio_improvement"],
                    "round1_residual_norm_total": row["round1_metrics"]["residual_norm_total"],
                    "round2_residual_norm_total": row["round2_metrics"]["residual_norm_total"],
                    "residual_norm_reduction": row["residual_norm_reduction"],
                }
            )

    comparison_path: Path | None = None
    if comparison_summary is not None:
        comparison_path = output_dir / "round1_vs_round2_projection_comparison.json"
        comparison_path.write_text(json.dumps(_json_ready(comparison_summary), indent=2), encoding="utf-8")
    return summary_path, examples_path, comparison_path, baseline_path
