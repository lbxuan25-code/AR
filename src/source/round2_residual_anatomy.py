"""Residual-anatomy diagnostics for the current round-2 truth layer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np

from .luo_loader import load_luo_samples
from .round2_projection import (
    DEFAULT_ROUND2_PROJECTION_CONFIG,
    Round2ProjectionConfig,
    project_luo_sample_to_round2_channels,
    reconstruct_source_tensors_from_channels,
    round2_basis_tensors,
    source_pairing_tensors_meV,
)

BLOCK_NAMES: tuple[str, str, str] = ("delta_x", "delta_y", "delta_z")
GROUP_NAMES: tuple[str, str, str, str, str] = ("zz", "xx", "zx", "perp", "other")


@dataclass(slots=True)
class ResidualAnatomyArtifacts:
    summary_path: Path
    examples_csv_path: Path
    docs_path: Path
    aggregate_heatmap_path: Path | None
    representative_heatmap_path: Path | None


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


def _fro_norm(matrix: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(matrix, dtype=np.complex128), ord="fro"))


def _empty_mask() -> np.ndarray:
    return np.zeros((4, 4), dtype=bool)


def round2_group_masks() -> dict[str, dict[str, np.ndarray]]:
    """Return matrix-entry masks grouped by physical channel family."""

    masks = {
        "delta_x": {name: _empty_mask() for name in GROUP_NAMES},
        "delta_y": {name: _empty_mask() for name in GROUP_NAMES},
        "delta_z": {name: _empty_mask() for name in GROUP_NAMES},
    }

    for block_name in ("delta_x", "delta_y"):
        masks[block_name]["zz"][0, 0] = True
        masks[block_name]["zz"][2, 2] = True
        masks[block_name]["xx"][1, 1] = True
        masks[block_name]["xx"][3, 3] = True
        masks[block_name]["zx"][0, 1] = True
        masks[block_name]["zx"][1, 0] = True
        masks[block_name]["zx"][2, 3] = True
        masks[block_name]["zx"][3, 2] = True

    masks["delta_z"]["perp"][0, 2] = True
    masks["delta_z"]["perp"][2, 0] = True
    masks["delta_z"]["perp"][1, 3] = True
    masks["delta_z"]["perp"][3, 1] = True

    for block_name in BLOCK_NAMES:
        supported = np.zeros((4, 4), dtype=bool)
        for group_name in ("zz", "xx", "zx", "perp"):
            supported |= masks[block_name][group_name]
        masks[block_name]["other"] = ~supported
    return masks


def _aggregate_supported_mask(masks: dict[str, dict[str, np.ndarray]], block_name: str) -> np.ndarray:
    supported = np.zeros((4, 4), dtype=bool)
    for group_name in ("zz", "xx", "zx", "perp"):
        supported |= masks[block_name][group_name]
    return supported


def _sample_category_indices(per_sample: list[dict[str, object]]) -> dict[str, int]:
    retained = np.asarray([item["retained_ratio_total"] for item in per_sample], dtype=np.float64)
    ranking = np.argsort(retained)
    return {
        "best": int(ranking[-1]),
        "median": int(ranking[len(ranking) // 2]),
        "worst": int(ranking[0]),
    }


def residual_anatomy_for_sample(
    sample,
    config: Round2ProjectionConfig = DEFAULT_ROUND2_PROJECTION_CONFIG,
) -> dict[str, object]:
    projected = project_luo_sample_to_round2_channels(sample, config=config)
    channels = projected.projected_physical_channels
    assert channels is not None

    source_x, source_y, source_z = source_pairing_tensors_meV(sample)
    gauge_phase = float(projected.round2_projection_metadata["gauge_phase_radians"])
    rotation = np.exp(-1.0j * gauge_phase)
    source_x = source_x * rotation
    source_y = source_y * rotation
    source_z = source_z * rotation
    recon_x, recon_y, recon_z = reconstruct_source_tensors_from_channels(channels)
    residuals = {
        "delta_x": source_x - recon_x,
        "delta_y": source_y - recon_y,
        "delta_z": source_z - recon_z,
    }
    sources = {"delta_x": source_x, "delta_y": source_y, "delta_z": source_z}
    recons = {"delta_x": recon_x, "delta_y": recon_y, "delta_z": recon_z}
    masks = round2_group_masks()

    block_metrics: dict[str, dict[str, float]] = {}
    group_metrics: dict[str, dict[str, float]] = {}
    residual_abs_heatmaps = {
        block_name: np.abs(np.asarray(matrix, dtype=np.complex128))
        for block_name, matrix in residuals.items()
    }
    supported_norm_sq = 0.0
    unsupported_norm_sq = 0.0

    for block_name in BLOCK_NAMES:
        source_matrix = sources[block_name]
        recon_matrix = recons[block_name]
        residual_matrix = residuals[block_name]
        source_norm = _fro_norm(source_matrix)
        recon_norm = _fro_norm(recon_matrix)
        residual_norm = _fro_norm(residual_matrix)
        supported_mask = _aggregate_supported_mask(masks, block_name)
        supported_residual = np.where(supported_mask, residual_matrix, 0.0)
        unsupported_residual = np.where(~supported_mask, residual_matrix, 0.0)
        supported_residual_norm = _fro_norm(supported_residual)
        unsupported_residual_norm = _fro_norm(unsupported_residual)
        supported_norm_sq += supported_residual_norm**2
        unsupported_norm_sq += unsupported_residual_norm**2

        block_metrics[block_name] = {
            "source_norm": source_norm,
            "recon_norm": recon_norm,
            "residual_norm": residual_norm,
            "retained_ratio": float(1.0 - residual_norm / source_norm) if source_norm > 0.0 else 1.0,
            "supported_residual_norm": supported_residual_norm,
            "unsupported_residual_norm": unsupported_residual_norm,
        }

        for group_name in GROUP_NAMES:
            mask = masks[block_name][group_name]
            masked_source = np.where(mask, source_matrix, 0.0)
            masked_recon = np.where(mask, recon_matrix, 0.0)
            masked_residual = np.where(mask, residual_matrix, 0.0)
            group_metrics[f"{block_name}::{group_name}"] = {
                "source_norm": _fro_norm(masked_source),
                "recon_norm": _fro_norm(masked_recon),
                "residual_norm": _fro_norm(masked_residual),
            }

    total_residual_norm_sq = float(sum(block_metrics[name]["residual_norm"] ** 2 for name in BLOCK_NAMES))
    total_residual_norm = float(np.sqrt(total_residual_norm_sq))
    supported_total = float(np.sqrt(supported_norm_sq))
    unsupported_total = float(np.sqrt(unsupported_norm_sq))
    dominant_block = max(BLOCK_NAMES, key=lambda name: block_metrics[name]["residual_norm"])

    hotspot_candidates: list[dict[str, object]] = []
    for block_name in BLOCK_NAMES:
        residual_abs = residual_abs_heatmaps[block_name]
        top_index = np.unravel_index(int(np.argmax(residual_abs)), residual_abs.shape)
        top_value = float(residual_abs[top_index])
        hotspot_group = next(
            group_name
            for group_name in GROUP_NAMES
            if round2_group_masks()[block_name][group_name][top_index]
        )
        hotspot_candidates.append(
            {
                "block": block_name,
                "entry": [int(top_index[0]), int(top_index[1])],
                "group": hotspot_group,
                "abs_residual": top_value,
            }
        )
    dominant_hotspot = max(hotspot_candidates, key=lambda row: row["abs_residual"])

    return {
        "sample_id": sample.sample_id,
        "sample_kind": sample.sample_kind,
        "coordinates": sample.coordinates,
        "retained_ratio_total": float(projected.round2_projection_metrics["retained_ratio_total"]),
        "residual_norm_total": float(projected.round2_projection_metrics["residual_norm_total"]),
        "omitted_fraction_total": float(projected.round2_projection_metrics["omitted_fraction_total"]),
        "block_metrics": block_metrics,
        "group_metrics": group_metrics,
        "supported_residual_norm_total": supported_total,
        "unsupported_residual_norm_total": unsupported_total,
        "supported_residual_fraction": float(supported_norm_sq / total_residual_norm_sq) if total_residual_norm_sq > 0.0 else 0.0,
        "unsupported_residual_fraction": float(unsupported_norm_sq / total_residual_norm_sq) if total_residual_norm_sq > 0.0 else 0.0,
        "dominant_block": dominant_block,
        "dominant_hotspot": dominant_hotspot,
        "residual_abs_heatmaps": residual_abs_heatmaps,
    }


def summarize_round2_residual_anatomy(
    max_samples: int | None = None,
    config: Round2ProjectionConfig = DEFAULT_ROUND2_PROJECTION_CONFIG,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    samples = load_luo_samples()
    if max_samples is not None:
        samples = samples[: int(max_samples)]

    per_sample = [residual_anatomy_for_sample(sample, config=config) for sample in samples]
    category_indices = _sample_category_indices(per_sample)
    group_by_family = {
        group_name: [item["group_metrics"][f"{block_name}::{group_name}"]["residual_norm"] for item in per_sample for block_name in BLOCK_NAMES]
        for group_name in GROUP_NAMES
    }
    hotspot_tables: dict[str, list[dict[str, object]]] = {}
    mean_abs_heatmaps: dict[str, np.ndarray] = {}
    for block_name in BLOCK_NAMES:
        stacked = np.stack([item["residual_abs_heatmaps"][block_name] for item in per_sample], axis=0)
        mean_abs_heatmaps[block_name] = np.mean(stacked, axis=0)
        hotspot_rows: list[dict[str, object]] = []
        for i in range(4):
            for j in range(4):
                values = stacked[:, i, j]
                hotspot_rows.append(
                    {
                        "entry": [i, j],
                        "mean_abs": float(np.mean(values)),
                        "median_abs": float(np.median(values)),
                        "p95_abs": float(np.percentile(values, 95.0)),
                        "group": next(
                            group_name
                            for group_name in GROUP_NAMES
                            if round2_group_masks()[block_name][group_name][i, j]
                        ),
                    }
                )
        hotspot_tables[block_name] = sorted(hotspot_rows, key=lambda row: row["mean_abs"], reverse=True)[:8]

    block_summary = {
        block_name: {
            "residual_norm": _real_stats(np.asarray([item["block_metrics"][block_name]["residual_norm"] for item in per_sample], dtype=np.float64)),
            "retained_ratio": _real_stats(np.asarray([item["block_metrics"][block_name]["retained_ratio"] for item in per_sample], dtype=np.float64)),
            "supported_residual_norm": _real_stats(np.asarray([item["block_metrics"][block_name]["supported_residual_norm"] for item in per_sample], dtype=np.float64)),
            "unsupported_residual_norm": _real_stats(np.asarray([item["block_metrics"][block_name]["unsupported_residual_norm"] for item in per_sample], dtype=np.float64)),
        }
        for block_name in BLOCK_NAMES
    }
    group_summary = {
        group_name: _real_stats(np.asarray(group_by_family[group_name], dtype=np.float64))
        for group_name in GROUP_NAMES
    }
    supported_fraction = np.asarray([item["supported_residual_fraction"] for item in per_sample], dtype=np.float64)
    unsupported_fraction = np.asarray([item["unsupported_residual_fraction"] for item in per_sample], dtype=np.float64)

    if float(np.median(unsupported_fraction)) > 0.6:
        dominant_cause = "missing_channel_structure"
        dominant_note = (
            "Most residual weight sits on entries outside the current round-2 supported masks, "
            "so the mismatch is more likely dominated by missing source-channel structure than by the current fit weights."
        )
    elif float(np.median(supported_fraction)) > 0.6:
        dominant_cause = "imperfect_projection_weighting"
        dominant_note = (
            "Most residual weight stays on entries already covered by the current round-2 masks, "
            "so the mismatch is more likely due to imperfect weighting or regularization inside the existing channel set."
        )
    else:
        dominant_cause = "mixed"
        dominant_note = (
            "Residual weight is split between supported and unsupported entries, so both missing channel structure "
            "and weighting choices remain plausible contributors."
        )

    representative_samples = {
        label: {
            "sample_id": per_sample[index]["sample_id"],
            "retained_ratio_total": float(per_sample[index]["retained_ratio_total"]),
            "residual_norm_total": float(per_sample[index]["residual_norm_total"]),
            "dominant_block": per_sample[index]["dominant_block"],
            "dominant_hotspot": per_sample[index]["dominant_hotspot"],
        }
        for label, index in category_indices.items()
    }

    summary = {
        "num_samples": len(per_sample),
        "config": config.to_dict(),
        "block_residual_summary": block_summary,
        "channel_group_residual_summary": group_summary,
        "aggregate_entry_hotspots": hotspot_tables,
        "supported_vs_unsupported_residual_fraction": {
            "supported": _real_stats(supported_fraction),
            "unsupported": _real_stats(unsupported_fraction),
        },
        "representative_samples": representative_samples,
        "diagnosis": {
            "dominant_cause": dominant_cause,
            "note": dominant_note,
            "dominant_block_by_median_residual": max(
                BLOCK_NAMES,
                key=lambda name: block_summary[name]["residual_norm"]["median"],
            ),
        },
    }
    return per_sample, summary


def write_round2_residual_anatomy_outputs(
    output_dir: Path,
    docs_path: Path,
    per_sample: list[dict[str, object]],
    summary: dict[str, object],
    make_plots: bool = True,
) -> ResidualAnatomyArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    docs_path.parent.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "round2_residual_anatomy_summary.json"
    summary_path.write_text(json.dumps(_json_ready(summary), indent=2), encoding="utf-8")

    examples_path = output_dir / "round2_residual_examples.csv"
    categories = _sample_category_indices(per_sample)
    with examples_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "category",
                "sample_id",
                "retained_ratio_total",
                "residual_norm_total",
                "omitted_fraction_total",
                "dominant_block",
                "dominant_hotspot_block",
                "dominant_hotspot_entry",
                "dominant_hotspot_group",
                "delta_x_residual_norm",
                "delta_y_residual_norm",
                "delta_z_residual_norm",
                "supported_residual_fraction",
                "unsupported_residual_fraction",
            ],
        )
        writer.writeheader()
        for category, index in categories.items():
            item = per_sample[index]
            writer.writerow(
                {
                    "category": category,
                    "sample_id": item["sample_id"],
                    "retained_ratio_total": float(item["retained_ratio_total"]),
                    "residual_norm_total": float(item["residual_norm_total"]),
                    "omitted_fraction_total": float(item["omitted_fraction_total"]),
                    "dominant_block": item["dominant_block"],
                    "dominant_hotspot_block": item["dominant_hotspot"]["block"],
                    "dominant_hotspot_entry": tuple(item["dominant_hotspot"]["entry"]),
                    "dominant_hotspot_group": item["dominant_hotspot"]["group"],
                    "delta_x_residual_norm": float(item["block_metrics"]["delta_x"]["residual_norm"]),
                    "delta_y_residual_norm": float(item["block_metrics"]["delta_y"]["residual_norm"]),
                    "delta_z_residual_norm": float(item["block_metrics"]["delta_z"]["residual_norm"]),
                    "supported_residual_fraction": float(item["supported_residual_fraction"]),
                    "unsupported_residual_fraction": float(item["unsupported_residual_fraction"]),
                }
            )

    docs_lines = [
        "# Round-2 Residual Anatomy",
        "",
        "## Scope",
        "",
        "This note only audits where the current round-2 truth-layer reconstruction still misses the Luo source.",
        "It does not change BTK, surrogate, inverse, or normal-state logic.",
        "",
        "## Main Finding",
        "",
        f"- Dominant cause: `{summary['diagnosis']['dominant_cause']}`",
        f"- Diagnostic note: {summary['diagnosis']['note']}",
        f"- Median-dominant residual block: `{summary['diagnosis']['dominant_block_by_median_residual']}`",
        "",
        "## Block Breakdown",
        "",
        *[
            (
                f"- `{block_name}`: median residual norm = "
                f"{stats['residual_norm']['median']:.6g}, median retained ratio = {stats['retained_ratio']['median']:.4f}"
            )
            for block_name, stats in summary["block_residual_summary"].items()
        ],
        "",
        "## Residual Hotspots",
        "",
        *[
            (
                f"- `{block_name}` top hotspot `{rows[0]['entry']}` in group `{rows[0]['group']}` "
                f"with mean abs residual {rows[0]['mean_abs']:.6g} meV"
            )
            for block_name, rows in summary["aggregate_entry_hotspots"].items()
        ],
        "",
        "## Representative Samples",
        "",
        *[
            (
                f"- `{category}`: `{payload['sample_id']}`, retained ratio = {payload['retained_ratio_total']:.4f}, "
                f"dominant block = `{payload['dominant_block']}`, hotspot = {payload['dominant_hotspot']['block']}[{payload['dominant_hotspot']['entry'][0]},{payload['dominant_hotspot']['entry'][1]}]"
            )
            for category, payload in summary["representative_samples"].items()
        ],
    ]
    docs_path.write_text("\n".join(docs_lines) + "\n", encoding="utf-8")

    aggregate_heatmap_path: Path | None = None
    representative_heatmap_path: Path | None = None
    if make_plots:
        aggregate_heatmap_path = output_dir / "round2_residual_anatomy_heatmaps.png"
        representative_heatmap_path = output_dir / "round2_residual_representatives.png"

        figure, axes = plt.subplots(1, 3, figsize=(11.0, 3.8), constrained_layout=True)
        for axis, block_name in zip(axes, BLOCK_NAMES, strict=True):
            heatmap = np.mean(
                np.stack([item["residual_abs_heatmaps"][block_name] for item in per_sample], axis=0),
                axis=0,
            )
            image = axis.imshow(heatmap, cmap="magma")
            axis.set_title(block_name)
            axis.set_xticks(range(4))
            axis.set_yticks(range(4))
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.savefig(aggregate_heatmap_path, dpi=160)
        plt.close(figure)

        representative = _sample_category_indices(per_sample)
        figure, axes = plt.subplots(3, 3, figsize=(11.0, 10.0), constrained_layout=True)
        for row, (category, index) in enumerate(representative.items()):
            item = per_sample[index]
            for col, block_name in enumerate(BLOCK_NAMES):
                axis = axes[row, col]
                image = axis.imshow(item["residual_abs_heatmaps"][block_name], cmap="viridis")
                axis.set_title(f"{category} | {block_name}")
                axis.set_xticks(range(4))
                axis.set_yticks(range(4))
                figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.savefig(representative_heatmap_path, dpi=160)
        plt.close(figure)

    return ResidualAnatomyArtifacts(
        summary_path=summary_path,
        examples_csv_path=examples_path,
        docs_path=docs_path,
        aggregate_heatmap_path=aggregate_heatmap_path,
        representative_heatmap_path=representative_heatmap_path,
    )


def run_round2_residual_anatomy_audit(
    output_dir: Path,
    docs_path: Path,
    max_samples: int | None = None,
    config: Round2ProjectionConfig = DEFAULT_ROUND2_PROJECTION_CONFIG,
    make_plots: bool = True,
) -> tuple[list[dict[str, object]], dict[str, object], ResidualAnatomyArtifacts]:
    per_sample, summary = summarize_round2_residual_anatomy(max_samples=max_samples, config=config)
    artifacts = write_round2_residual_anatomy_outputs(
        output_dir=output_dir,
        docs_path=docs_path,
        per_sample=per_sample,
        summary=summary,
        make_plots=make_plots,
    )
    return per_sample, summary, artifacts
