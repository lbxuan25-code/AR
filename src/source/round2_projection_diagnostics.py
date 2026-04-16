"""Diagnostics and comparisons for the round-2 physical channel projection."""

from __future__ import annotations

from pathlib import Path
import csv
import json

import numpy as np

from .luo_loader import load_luo_samples
from .luo_projection import EV_TO_MEV, project_luo_sample_to_pairing
from .round2_projection import (
    ROUND2_CHANNEL_NAMES,
    project_luo_sample_to_round2_channels,
    reconstruct_source_tensors_from_channels,
    source_pairing_tensors_meV,
)


def _round1_reconstruction_metrics(sample) -> dict[str, float]:
    projected = project_luo_sample_to_pairing(sample).projected_pairing_params
    assert projected is not None
    delta_x, delta_y, delta_z = source_pairing_tensors_meV(sample)

    recon_x = np.zeros((4, 4), dtype=np.complex128)
    recon_y = np.zeros((4, 4), dtype=np.complex128)
    recon_z = np.zeros((4, 4), dtype=np.complex128)
    recon_x[0, 0] = recon_x[2, 2] = projected.eta_z_s
    recon_y[0, 0] = recon_y[2, 2] = projected.eta_z_s
    recon_x[1, 1] = recon_x[3, 3] = projected.eta_x_s + projected.eta_x_d
    recon_y[1, 1] = recon_y[3, 3] = projected.eta_x_s - projected.eta_x_d
    recon_z[0, 2] = recon_z[2, 0] = projected.eta_z_perp

    source_norm_x = float(np.linalg.norm(delta_x, ord="fro"))
    source_norm_y = float(np.linalg.norm(delta_y, ord="fro"))
    source_norm_z = float(np.linalg.norm(delta_z, ord="fro"))
    recon_norm_x = float(np.linalg.norm(recon_x, ord="fro"))
    recon_norm_y = float(np.linalg.norm(recon_y, ord="fro"))
    recon_norm_z = float(np.linalg.norm(recon_z, ord="fro"))
    residual_norm_x = float(np.linalg.norm(delta_x - recon_x, ord="fro"))
    residual_norm_y = float(np.linalg.norm(delta_y - recon_y, ord="fro"))
    residual_norm_z = float(np.linalg.norm(delta_z - recon_z, ord="fro"))
    source_norm_total = float(np.sqrt(source_norm_x**2 + source_norm_y**2 + source_norm_z**2))
    residual_norm_total = float(np.sqrt(residual_norm_x**2 + residual_norm_y**2 + residual_norm_z**2))
    return {
        "retained_ratio_total": float(1.0 - residual_norm_total / source_norm_total) if source_norm_total > 0.0 else 1.0,
        "residual_norm_total": residual_norm_total,
        "omitted_fraction_total": float(residual_norm_total / source_norm_total) if source_norm_total > 0.0 else 0.0,
    }


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


def summarize_round2_projection(max_samples: int | None = None) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    samples = load_luo_samples()
    if max_samples is not None:
        samples = samples[: int(max_samples)]

    per_sample: list[dict[str, object]] = []
    for sample in samples:
        round2_sample = project_luo_sample_to_round2_channels(sample)
        round2_channels = round2_sample.projected_physical_channels
        assert round2_channels is not None
        round2_metrics = dict(round2_sample.round2_projection_metrics)
        round1_metrics = _round1_reconstruction_metrics(sample)
        per_sample.append(
            {
                "sample_id": sample.sample_id,
                "round2_channels": round2_channels.to_dict(),
                "round2_metrics": round2_metrics,
                "round1_metrics": round1_metrics,
                "retained_ratio_improvement": float(round2_metrics["retained_ratio_total"] - round1_metrics["retained_ratio_total"]),
                "residual_norm_reduction": float(round1_metrics["residual_norm_total"] - round2_metrics["residual_norm_total"]),
                "omitted_fraction_reduction": float(round1_metrics["omitted_fraction_total"] - round2_metrics["omitted_fraction_total"]),
            }
        )

    round2_summary = {
        "num_samples": len(per_sample),
        "channel_magnitude_stats": {
            name: _abs_stats(np.asarray([item["round2_channels"][name] for item in per_sample]))
            for name in ROUND2_CHANNEL_NAMES
        },
        "round2_retained_ratio_total": _real_stats(np.asarray([item["round2_metrics"]["retained_ratio_total"] for item in per_sample])),
        "round2_residual_norm_total": _real_stats(np.asarray([item["round2_metrics"]["residual_norm_total"] for item in per_sample])),
        "round2_omitted_fraction_total": _real_stats(np.asarray([item["round2_metrics"]["omitted_fraction_total"] for item in per_sample])),
    }
    comparison_summary = {
        "num_samples": len(per_sample),
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
            if np.median(np.asarray([item["retained_ratio_improvement"] for item in per_sample])) > 0.0
            else "Round-2 physical channels do not improve the retained ratio relative to round 1."
        ),
    }
    return per_sample, round2_summary, comparison_summary


def write_round2_projection_outputs(
    output_dir: Path,
    per_sample: list[dict[str, object]],
    round2_summary: dict[str, object],
    comparison_summary: dict[str, object] | None = None,
) -> tuple[Path, Path, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "round2_projection_summary.json"
    summary_path.write_text(json.dumps(round2_summary, indent=2), encoding="utf-8")

    examples_path = output_dir / "round2_projection_examples.csv"
    ranked = sorted(per_sample, key=lambda item: item["retained_ratio_improvement"], reverse=True)
    candidate_rows = ranked[:3] + sorted(per_sample, key=lambda item: item["retained_ratio_improvement"])[:3]
    seen: set[str] = set()
    fieldnames = [
        "sample_id",
        *ROUND2_CHANNEL_NAMES,
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
        comparison_path.write_text(json.dumps(comparison_summary, indent=2), encoding="utf-8")
    return summary_path, examples_path, comparison_path
