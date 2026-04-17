"""Projection consistency diagnostics for Luo RMFT -> local PairingParams.

This module intentionally does not change the projection rules. It only checks:

1. source semantics and unit evidence,
2. algebraic consistency of the currently implemented retained channels,
3. source-level reconstruction residuals for the retained observables,
4. omitted-channel magnitudes and retained ratios,
5. final round-1 verdict language.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json
import re

import matplotlib.pyplot as plt
import numpy as np

from core.conventions import BASIS_ORDER
from core.presets import BASE_MU_DIAG

from .luo_loader import PAIRING_COMPONENT_NAMES, ensure_luo_repo, load_luo_samples
from .luo_projection import EV_TO_MEV, project_luo_sample_to_pairing
from .projection_metrics import build_projection_metric_bundle
from .schema import LuoSample


EVIDENCE_STRONG = "strong_evidence_support"
EVIDENCE_WEAK = "weak_evidence_support"
EVIDENCE_ASSUMPTION = "current_engineering_assumption"

RETAINED_CHANNEL_NAMES = ("eta_z_s", "eta_z_perp", "eta_x_s", "eta_x_d")
RECON_RESIDUAL_NAMES = (
    "R_x_xx_A",
    "R_x_xx_B",
    "R_y_xx_A",
    "R_y_xx_B",
    "R_x_zz_A",
    "R_x_zz_B",
    "R_y_zz_A",
    "R_y_zz_B",
    "R_z_perp_02",
    "R_z_perp_20",
)
ZX_SCAN_ENTRY_NAMES = (
    "delta_x_sym_01",
    "delta_x_sym_23",
    "delta_y_sym_01",
    "delta_y_sym_23",
    "delta_z_sym_01",
    "delta_z_sym_23",
    "delta_z_sym_03",
    "delta_z_sym_12",
)


@dataclass(slots=True)
class ProjectionDiagnosticsArtifacts:
    summary_path: Path
    examples_csv_path: Path
    docs_path: Path
    hist_path: Path | None
    scatter_path: Path | None


def _symmetrized(matrix: np.ndarray, i: int, j: int) -> complex:
    return complex(0.5 * (matrix[i, j] + matrix[j, i]))


def _abs_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.abs(np.asarray(values, dtype=np.complex128).reshape(-1))
    if arr.size == 0:
        return {"max_abs": 0.0, "mean_abs": 0.0, "median_abs": 0.0, "p95_abs": 0.0}
    return {
        "max_abs": float(np.max(arr)),
        "mean_abs": float(np.mean(arr)),
        "median_abs": float(np.median(arr)),
        "p95_abs": float(np.percentile(arr, 95.0)),
    }


def _real_stats(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "median": 0.0,
            "p05": 0.0,
            "p95": 0.0,
        }
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p05": float(np.percentile(arr, 5.0)),
        "p95": float(np.percentile(arr, 95.0)),
    }


def _evidence_record(level: str, supported: bool, detail: str, metric: dict[str, float] | None = None) -> dict[str, object]:
    record: dict[str, object] = {
        "level": level,
        "supported": bool(supported),
        "detail": detail,
    }
    if metric is not None:
        record["metric"] = metric
    return record


def _source_repo_text(repo_dir: Path, relative_path: str) -> str:
    return (repo_dir / relative_path).read_text(encoding="utf-8")


def collect_source_semantics_checks(repo_dir: Path, samples: list[LuoSample]) -> dict[str, object]:
    h4band_text = _source_repo_text(repo_dir, "H4band.py")
    fig1_text = _source_repo_text(repo_dir, "plot_Fig1.py")
    mu2 = np.asarray(np.load(repo_dir / "Mu2.npy"), dtype=np.float64)
    mu0_meV = np.asarray(mu2[0] * EV_TO_MEV, dtype=np.float64)
    base_mu = np.asarray(BASE_MU_DIAG, dtype=np.float64)

    pairing_mapping_ok = tuple(PAIRING_COMPONENT_NAMES) == ("chi_x", "chi_y", "chi_z", "delta_x", "delta_y", "delta_z")
    basis_comment_ok = "# dz2 dx2 dz2 dx2" in h4band_text
    basis_local_ok = tuple(BASIS_ORDER) == ("Az", "Ax", "Bz", "Bx")
    figure_patterns = [
        r"Dltd\s*=\s*Pms\[:,3\]\s*-\s*Pms\[:,4\]",
        r"Dlts\s*=\s*Pms\[:,3\]\s*\+\s*Pms\[:,4\]",
        r"Dltsp\s*=\s*Pms\[:,5\]",
        r"Dltd1\s*=\s*amp\(Dltd\[:,1,1\]\)",
        r"Dlts1\s*=\s*amp\(Dlts\[:,1,1\]\)",
        r"Dltd2\s*=\s*amp\(Dltd\[:,0,0\]\)",
        r"Dlts2\s*=\s*amp\(Dlts\[:,0,0\]\)",
        r"Dltsp\s*=\s*amp\(Dltsp\[:,0,2\]\)",
    ]
    figure_alignment_ok = all(re.search(pattern, fig1_text) is not None for pattern in figure_patterns)

    mu_abs_diff = np.abs(mu0_meV - base_mu)
    mu_rel_diff = mu_abs_diff / np.maximum(np.abs(base_mu), 1.0e-12)
    mu_supported = bool(np.max(mu_rel_diff) < 5.0e-6)

    raw_delta_values = []
    for sample in samples[: min(len(samples), 2048)]:
        raw_delta_values.extend(
            [
                abs(np.asarray(sample.source_pairing_observables["delta_x"], dtype=np.complex128)[1, 1]),
                abs(np.asarray(sample.source_pairing_observables["delta_y"], dtype=np.complex128)[1, 1]),
                abs(np.asarray(sample.source_pairing_observables["delta_z"], dtype=np.complex128)[0, 2]),
            ]
        )
    raw_delta_values_array = np.asarray(raw_delta_values, dtype=np.float64)
    converted_delta_values_array = raw_delta_values_array * EV_TO_MEV
    delta_unit_supported = bool(mu_supported and np.median(converted_delta_values_array) > 1.0)

    return {
        "pairing_component_semantics": _evidence_record(
            EVIDENCE_STRONG if pairing_mapping_ok else EVIDENCE_ASSUMPTION,
            pairing_mapping_ok,
            "Loader names the six RMFT tensor slices as (chi_x, chi_y, chi_z, delta_x, delta_y, delta_z).",
        ),
        "orbital_basis_alignment": _evidence_record(
            EVIDENCE_STRONG if (basis_comment_ok and basis_local_ok) else EVIDENCE_ASSUMPTION,
            basis_comment_ok and basis_local_ok,
            "H4band.py documents the source order as (dz2, dx2, dz2, dx2), which aligns with local (Az, Ax, Bz, Bx).",
        ),
        "figure_level_channel_usage": _evidence_record(
            EVIDENCE_STRONG if figure_alignment_ok else EVIDENCE_ASSUMPTION,
            figure_alignment_ok,
            "plot_Fig1.py uses Pms[3], Pms[4], Pms[5] as delta_x, delta_y, delta_z and reads [1,1], [0,0], [0,2], matching the retained observables used by the local projection.",
        ),
        "mu_unit_sanity": _evidence_record(
            EVIDENCE_STRONG if mu_supported else EVIDENCE_ASSUMPTION,
            mu_supported,
            "Mu2.npy[0] * 1000 nearly matches local BASE_MU_DIAG, strongly supporting the eV -> meV conversion for chemical potentials.",
            metric={
                "max_abs_diff_meV": float(np.max(mu_abs_diff)),
                "max_rel_diff": float(np.max(mu_rel_diff)),
            },
        ),
        "delta_unit_sanity": _evidence_record(
            EVIDENCE_WEAK if delta_unit_supported else EVIDENCE_ASSUMPTION,
            delta_unit_supported,
            "The same source npz payload family stores Mu fields in eV-scale numbers, and converting delta entries by 1000 moves them into the meV scale used by the local pairing baseline. This is supportive but not a direct proof from source-side comments.",
            metric={
                "median_abs_raw_delta_eV": float(np.median(raw_delta_values_array)),
                "median_abs_converted_delta_meV": float(np.median(converted_delta_values_array)),
            },
        ),
    }


def _expected_retained_channels_meV(sample: LuoSample) -> dict[str, complex]:
    delta_x = np.asarray(sample.source_pairing_observables["delta_x"], dtype=np.complex128) * EV_TO_MEV
    delta_y = np.asarray(sample.source_pairing_observables["delta_y"], dtype=np.complex128) * EV_TO_MEV
    delta_z = np.asarray(sample.source_pairing_observables["delta_z"], dtype=np.complex128) * EV_TO_MEV
    return {
        "eta_z_s": complex(0.5 * (delta_x[0, 0] + delta_y[0, 0])),
        "eta_z_perp": _symmetrized(delta_z, 0, 2),
        "eta_x_s": complex(0.5 * (delta_x[1, 1] + delta_y[1, 1])),
        "eta_x_d": complex(0.5 * (delta_x[1, 1] - delta_y[1, 1])),
    }


def _projected_channel_dict(sample: LuoSample) -> dict[str, complex]:
    projected = project_luo_sample_to_pairing(sample).projected_pairing_params
    assert projected is not None
    return {
        "eta_z_s": projected.eta_z_s,
        "eta_z_perp": projected.eta_z_perp,
        "eta_x_s": projected.eta_x_s,
        "eta_x_d": projected.eta_x_d,
        "eta_zx_d": projected.eta_zx_d,
        "eta_x_perp": projected.eta_x_perp,
    }


def _source_level_reconstruction(projected: dict[str, complex]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta_x_recon = np.zeros((4, 4), dtype=np.complex128)
    delta_y_recon = np.zeros((4, 4), dtype=np.complex128)
    delta_z_recon = np.zeros((4, 4), dtype=np.complex128)

    eta_z_s = projected["eta_z_s"]
    eta_z_perp = projected["eta_z_perp"]
    eta_x_s = projected["eta_x_s"]
    eta_x_d = projected["eta_x_d"]

    delta_x_recon[0, 0] = eta_z_s
    delta_x_recon[2, 2] = eta_z_s
    delta_x_recon[1, 1] = eta_x_s + eta_x_d
    delta_x_recon[3, 3] = eta_x_s + eta_x_d

    delta_y_recon[0, 0] = eta_z_s
    delta_y_recon[2, 2] = eta_z_s
    delta_y_recon[1, 1] = eta_x_s - eta_x_d
    delta_y_recon[3, 3] = eta_x_s - eta_x_d

    delta_z_recon[0, 2] = eta_z_perp
    delta_z_recon[2, 0] = eta_z_perp
    return delta_x_recon, delta_y_recon, delta_z_recon


def _diagnose_single_sample(sample: LuoSample) -> dict[str, object]:
    delta_x = np.asarray(sample.source_pairing_observables["delta_x"], dtype=np.complex128) * EV_TO_MEV
    delta_y = np.asarray(sample.source_pairing_observables["delta_y"], dtype=np.complex128) * EV_TO_MEV
    delta_z = np.asarray(sample.source_pairing_observables["delta_z"], dtype=np.complex128) * EV_TO_MEV

    expected = _expected_retained_channels_meV(sample)
    projected = _projected_channel_dict(sample)
    delta_x_recon, delta_y_recon, delta_z_recon = _source_level_reconstruction(projected)

    retained_residuals = {
        name: projected[name] - expected[name]
        for name in RETAINED_CHANNEL_NAMES
    }
    recon_residuals = {
        "R_x_xx_A": delta_x[1, 1] - (projected["eta_x_s"] + projected["eta_x_d"]),
        "R_x_xx_B": delta_x[3, 3] - (projected["eta_x_s"] + projected["eta_x_d"]),
        "R_y_xx_A": delta_y[1, 1] - (projected["eta_x_s"] - projected["eta_x_d"]),
        "R_y_xx_B": delta_y[3, 3] - (projected["eta_x_s"] - projected["eta_x_d"]),
        "R_x_zz_A": delta_x[0, 0] - projected["eta_z_s"],
        "R_x_zz_B": delta_x[2, 2] - projected["eta_z_s"],
        "R_y_zz_A": delta_y[0, 0] - projected["eta_z_s"],
        "R_y_zz_B": delta_y[2, 2] - projected["eta_z_s"],
        "R_z_perp_02": delta_z[0, 2] - projected["eta_z_perp"],
        "R_z_perp_20": delta_z[2, 0] - projected["eta_z_perp"],
    }
    omitted_metrics = {
        "z_sector_d_like_omitted": complex(0.5 * (delta_x[0, 0] - delta_y[0, 0])),
        "x_perp_candidate": _symmetrized(delta_z, 1, 3),
        "zx_d_candidate_aggregate": complex(
            0.25
            * (
                (_symmetrized(delta_x, 0, 1) + _symmetrized(delta_x, 2, 3))
                - (_symmetrized(delta_y, 0, 1) + _symmetrized(delta_y, 2, 3))
            )
        ),
    }
    zx_scan = {
        "delta_x_sym_01": _symmetrized(delta_x, 0, 1),
        "delta_x_sym_23": _symmetrized(delta_x, 2, 3),
        "delta_y_sym_01": _symmetrized(delta_y, 0, 1),
        "delta_y_sym_23": _symmetrized(delta_y, 2, 3),
        "delta_z_sym_01": _symmetrized(delta_z, 0, 1),
        "delta_z_sym_23": _symmetrized(delta_z, 2, 3),
        "delta_z_sym_03": _symmetrized(delta_z, 0, 3),
        "delta_z_sym_12": _symmetrized(delta_z, 1, 2),
    }

    metrics = build_projection_metric_bundle(delta_x, delta_y, delta_z, delta_x_recon, delta_y_recon, delta_z_recon)

    figure_level = {
        "two_eta_x_s_vs_dx_plus_dy_11": 2.0 * projected["eta_x_s"] - (delta_x[1, 1] + delta_y[1, 1]),
        "two_eta_x_d_vs_dx_minus_dy_11": 2.0 * projected["eta_x_d"] - (delta_x[1, 1] - delta_y[1, 1]),
        "two_eta_z_s_vs_dx_plus_dy_00": 2.0 * projected["eta_z_s"] - (delta_x[0, 0] + delta_y[0, 0]),
        "eta_z_perp_vs_dz_02": projected["eta_z_perp"] - delta_z[0, 2],
    }

    return {
        "sample_id": sample.sample_id,
        "expected": expected,
        "projected": projected,
        "retained_channel_residuals": retained_residuals,
        "reconstruction_residuals": recon_residuals,
        "omitted_metrics": omitted_metrics,
        "zx_scan": zx_scan,
        "norms": {
            **metrics,
        },
        "figure_level_checks": figure_level,
    }


def summarize_projection_diagnostics(sample_diagnostics: list[dict[str, object]]) -> dict[str, object]:
    retained_channel_stats = {
        name: _abs_stats(np.asarray([item["retained_channel_residuals"][name] for item in sample_diagnostics]))
        for name in RETAINED_CHANNEL_NAMES
    }
    reconstruction_residual_stats = {
        name: _abs_stats(np.asarray([item["reconstruction_residuals"][name] for item in sample_diagnostics]))
        for name in RECON_RESIDUAL_NAMES
    }
    omitted_channel_stats = {
        "z_sector_d_like_omitted": _abs_stats(np.asarray([item["omitted_metrics"]["z_sector_d_like_omitted"] for item in sample_diagnostics])),
        "x_perp_candidate": _abs_stats(np.asarray([item["omitted_metrics"]["x_perp_candidate"] for item in sample_diagnostics])),
        "zx_d_candidate_aggregate": _abs_stats(np.asarray([item["omitted_metrics"]["zx_d_candidate_aggregate"] for item in sample_diagnostics])),
    }
    zx_scan_stats = {
        name: _abs_stats(np.asarray([item["zx_scan"][name] for item in sample_diagnostics]))
        for name in ZX_SCAN_ENTRY_NAMES
    }
    zx_reasonable_source_analogues = {
        name: zx_scan_stats[name]
        for name in ("delta_x_sym_01", "delta_x_sym_23", "delta_y_sym_01", "delta_y_sym_23")
    }
    ratio_stats = {
        "retained_ratio_x": _real_stats(np.asarray([item["norms"]["retained_ratio_x"] for item in sample_diagnostics])),
        "retained_ratio_y": _real_stats(np.asarray([item["norms"]["retained_ratio_y"] for item in sample_diagnostics])),
        "retained_ratio_z": _real_stats(np.asarray([item["norms"]["retained_ratio_z"] for item in sample_diagnostics])),
        "retained_ratio_total": _real_stats(np.asarray([item["norms"]["retained_ratio_total"] for item in sample_diagnostics])),
        "omitted_fraction_total": _real_stats(np.asarray([item["norms"]["omitted_fraction_total"] for item in sample_diagnostics])),
    }
    norm_stats = {
        "source_norm_total": _real_stats(np.asarray([item["norms"]["source_norm_total"] for item in sample_diagnostics])),
        "recon_norm_total": _real_stats(np.asarray([item["norms"]["recon_norm_total"] for item in sample_diagnostics])),
        "residual_norm_total": _real_stats(np.asarray([item["norms"]["residual_norm_total"] for item in sample_diagnostics])),
    }
    figure_level_stats = {
        name: _abs_stats(np.asarray([item["figure_level_checks"][name] for item in sample_diagnostics]))
        for name in (
            "two_eta_x_s_vs_dx_plus_dy_11",
            "two_eta_x_d_vs_dx_minus_dy_11",
            "two_eta_z_s_vs_dx_plus_dy_00",
            "eta_z_perp_vs_dz_02",
        )
    }

    implementation_correct = max(stats["max_abs"] for stats in retained_channel_stats.values()) < 1.0e-10
    figure_consistent = max(stats["max_abs"] for stats in figure_level_stats.values()) < 1.0e-10
    full_equivalence = (
        ratio_stats["retained_ratio_total"]["p05"] > 0.98
        and omitted_channel_stats["z_sector_d_like_omitted"]["p95_abs"] < 1.0e-3
        and omitted_channel_stats["x_perp_candidate"]["p95_abs"] < 1.0e-3
        and omitted_channel_stats["zx_d_candidate_aggregate"]["p95_abs"] < 1.0e-3
    )
    final_verdict = {
        "projection_implementation_correct": (
            "Yes. The current luo_projection.py implementation matches its design formulas, "
            "and retained-channel residuals are at machine-precision scale."
            if implementation_correct
            else "No. At least one retained-channel residual is too large, which indicates an implementation bug."
        ),
        "projection_figure_level_consistent": (
            "Yes. The retained combinations align with the Luo figure-level s/d/perp observables used in plot_Fig1.py."
            if figure_consistent
            else "No. The retained combinations do not numerically line up with the Luo figure-level channel definitions."
        ),
        "projection_full_rmft_equivalent": (
            "No. Current round-1 projection is a restricted approximation to the RMFT source, not a full equivalence to the complete RMFT pairing tensor."
            if not full_equivalence
            else "Yes. The retained ratio is very high and omitted channels are negligible across the inspected samples."
        ),
    }

    return {
        "retained_channel_residual_stats": retained_channel_stats,
        "reconstruction_residual_stats": reconstruction_residual_stats,
        "omitted_channel_stats": omitted_channel_stats,
        "candidate_hidden_channel_scan": zx_scan_stats,
        "zx_d_reasonable_source_analogues": zx_reasonable_source_analogues,
        "retained_ratio_stats": ratio_stats,
        "norm_stats": norm_stats,
        "figure_level_consistency_stats": figure_level_stats,
        "final_textual_verdict": final_verdict,
    }


def _choose_example_indices(sample_diagnostics: list[dict[str, object]], limit: int = 12) -> list[int]:
    retained_ratio = np.asarray([item["norms"]["retained_ratio_total"] for item in sample_diagnostics], dtype=np.float64)
    omitted_z = np.asarray([abs(item["omitted_metrics"]["z_sector_d_like_omitted"]) for item in sample_diagnostics], dtype=np.float64)
    x_perp = np.asarray([abs(item["omitted_metrics"]["x_perp_candidate"]) for item in sample_diagnostics], dtype=np.float64)
    zx_d = np.asarray([abs(item["omitted_metrics"]["zx_d_candidate_aggregate"]) for item in sample_diagnostics], dtype=np.float64)
    median_index = int(np.argsort(retained_ratio)[len(retained_ratio) // 2])

    indices = [
        int(np.argmax(retained_ratio)),
        int(np.argmin(retained_ratio)),
        median_index,
        int(np.argmax(omitted_z)),
        int(np.argmax(x_perp)),
        int(np.argmax(zx_d)),
    ]
    unique: list[int] = []
    for index in indices:
        if index not in unique:
            unique.append(index)
    return unique[:limit]


def write_examples_csv(sample_diagnostics: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "eta_z_s_projected_re",
        "eta_z_s_expected_re",
        "eta_z_s_residual_abs",
        "eta_z_perp_projected_abs",
        "eta_z_perp_expected_abs",
        "eta_z_perp_residual_abs",
        "eta_x_s_projected_re",
        "eta_x_s_expected_re",
        "eta_x_s_residual_abs",
        "eta_x_d_projected_re",
        "eta_x_d_expected_re",
        "eta_x_d_residual_abs",
        "z_sector_d_like_omitted_abs",
        "x_perp_candidate_abs",
        "zx_d_candidate_aggregate_abs",
        "retained_ratio_total",
        "omitted_fraction_total",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index in _choose_example_indices(sample_diagnostics):
            item = sample_diagnostics[index]
            writer.writerow(
                {
                    "sample_id": item["sample_id"],
                    "eta_z_s_projected_re": float(np.real(item["projected"]["eta_z_s"])),
                    "eta_z_s_expected_re": float(np.real(item["expected"]["eta_z_s"])),
                    "eta_z_s_residual_abs": float(abs(item["retained_channel_residuals"]["eta_z_s"])),
                    "eta_z_perp_projected_abs": float(abs(item["projected"]["eta_z_perp"])),
                    "eta_z_perp_expected_abs": float(abs(item["expected"]["eta_z_perp"])),
                    "eta_z_perp_residual_abs": float(abs(item["retained_channel_residuals"]["eta_z_perp"])),
                    "eta_x_s_projected_re": float(np.real(item["projected"]["eta_x_s"])),
                    "eta_x_s_expected_re": float(np.real(item["expected"]["eta_x_s"])),
                    "eta_x_s_residual_abs": float(abs(item["retained_channel_residuals"]["eta_x_s"])),
                    "eta_x_d_projected_re": float(np.real(item["projected"]["eta_x_d"])),
                    "eta_x_d_expected_re": float(np.real(item["expected"]["eta_x_d"])),
                    "eta_x_d_residual_abs": float(abs(item["retained_channel_residuals"]["eta_x_d"])),
                    "z_sector_d_like_omitted_abs": float(abs(item["omitted_metrics"]["z_sector_d_like_omitted"])),
                    "x_perp_candidate_abs": float(abs(item["omitted_metrics"]["x_perp_candidate"])),
                    "zx_d_candidate_aggregate_abs": float(abs(item["omitted_metrics"]["zx_d_candidate_aggregate"])),
                    "retained_ratio_total": float(item["norms"]["retained_ratio_total"]),
                    "omitted_fraction_total": float(item["norms"]["omitted_fraction_total"]),
                }
            )


def _plot_histogram(sample_diagnostics: list[dict[str, object]], output_path: Path) -> None:
    values = np.asarray([item["norms"]["retained_ratio_total"] for item in sample_diagnostics], dtype=np.float64)
    figure, axis = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    axis.hist(values, bins=32, color="tab:blue", alpha=0.8)
    axis.set_xlabel("Retained ratio (total Frobenius norm)")
    axis.set_ylabel("Count")
    axis.set_title("Projection Retained-Ratio Distribution")
    axis.grid(alpha=0.2)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_scatter(sample_diagnostics: list[dict[str, object]], output_path: Path) -> None:
    ratio = np.asarray([item["norms"]["retained_ratio_total"] for item in sample_diagnostics], dtype=np.float64)
    omitted = np.asarray([item["norms"]["omitted_fraction_total"] for item in sample_diagnostics], dtype=np.float64)
    color = np.asarray([abs(item["omitted_metrics"]["z_sector_d_like_omitted"]) for item in sample_diagnostics], dtype=np.float64)

    figure, axis = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)
    scatter = axis.scatter(ratio, omitted, c=color, s=12.0, cmap="viridis", linewidths=0.0)
    axis.set_xlabel("Retained ratio (total)")
    axis.set_ylabel("Omitted fraction (total residual / source norm)")
    axis.set_title("Projection Residual Scatter")
    axis.grid(alpha=0.2)
    colorbar = figure.colorbar(scatter, ax=axis)
    colorbar.set_label(r"|0.5*(delta_x[0,0]-delta_y[0,0])| (meV)")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def write_markdown_report(
    docs_path: Path,
    semantics_checks: dict[str, object],
    summary: dict[str, object],
) -> None:
    verdict = summary["final_textual_verdict"]
    ratio = summary["retained_ratio_stats"]["retained_ratio_total"]
    omitted = summary["omitted_channel_stats"]
    retained = summary["retained_channel_residual_stats"]
    reconstruction = summary["reconstruction_residual_stats"]

    candidate_rank = sorted(
        summary["candidate_hidden_channel_scan"].items(),
        key=lambda item: item[1]["mean_abs"],
        reverse=True,
    )
    reasonable_zx_rank = sorted(
        summary["zx_d_reasonable_source_analogues"].items(),
        key=lambda item: item[1]["mean_abs"],
        reverse=True,
    )
    candidate_lines = [
        f"- `{name}`: mean abs = {stats['mean_abs']:.6g} meV, p95 abs = {stats['p95_abs']:.6g} meV"
        for name, stats in candidate_rank[:6]
    ]
    reasonable_zx_lines = [
        f"- `{name}`: mean abs = {stats['mean_abs']:.6g} meV, p95 abs = {stats['p95_abs']:.6g} meV"
        for name, stats in reasonable_zx_rank
    ]

    docs_text = "\n".join(
        [
            "# Projection Consistency Round 1",
            "",
            "## 1. 背景",
            "",
            "本报告只检查当前 `src/source/luo_projection.py` 的 round-1 投影实现是否自洽，",
            "不修改 physics core，也不修改 projection 公式。",
            "",
            "## 2. 当前 Projection 公式",
            "",
            "- `eta_z_s = 0.5 * (delta_x[0,0] + delta_y[0,0]) * 1000`",
            "- `eta_z_perp = 0.5 * (delta_z[0,2] + delta_z[2,0]) * 1000`",
            "- `eta_x_s = 0.5 * (delta_x[1,1] + delta_y[1,1]) * 1000`",
            "- `eta_x_d = 0.5 * (delta_x[1,1] - delta_y[1,1]) * 1000`",
            "- `eta_zx_d = 0`, `eta_x_perp = 0`",
            "",
            "## 3. Source 语义与单位检查",
            "",
            *[
                f"- `{name}`: `{record['level']}`. {record['detail']}"
                for name, record in semantics_checks.items()
            ],
            "",
            "结论口径：",
            "- `Pms/pms` 六分量语义、轨道索引对应、以及 `Mu2.npy` 的 eV -> meV 量纲转换都有直接代码证据。",
            "- `delta_*` 的 eV -> meV 对应目前属于弱证据支持，不应表述成已经被 source 注释严格证明。",
            "",
            "## 4. 保留通道的一致性检查",
            "",
            *[
                f"- `{name}`: max abs residual = {stats['max_abs']:.3e}, mean abs residual = {stats['mean_abs']:.3e}, p95 abs residual = {stats['p95_abs']:.3e}"
                for name, stats in retained.items()
            ],
            "",
            "这部分是严格代数检查。若 residual 不接近 machine precision，应视为实现 bug。",
            "",
            "## 5. Reconstruction Residual",
            "",
            *[
                f"- `{name}`: max abs residual = {stats['max_abs']:.6g} meV, mean abs residual = {stats['mean_abs']:.6g} meV, p95 abs residual = {stats['p95_abs']:.6g} meV"
                for name, stats in reconstruction.items()
            ],
            "",
            "这部分说明：即使 projection 实现正确，本地参数化在 source-level retained observables 之外仍可能留下未建模残差。",
            "",
            "## 6. 被忽略信息的规模",
            "",
            f"- `z-sector d-like omitted = 0.5*(delta_x[0,0]-delta_y[0,0])`: p95 abs = {omitted['z_sector_d_like_omitted']['p95_abs']:.6g} meV",
            f"- `x_perp candidate` from symmetrized `delta_z[1,3]`: p95 abs = {omitted['x_perp_candidate']['p95_abs']:.6g} meV",
            f"- `zx_d candidate aggregate`: p95 abs = {omitted['zx_d_candidate_aggregate']['p95_abs']:.6g} meV",
            f"- total retained ratio (`1 - residual/source`): p05 = {ratio['p05']:.4f}, median = {ratio['median']:.4f}, p95 = {ratio['p95']:.4f}",
            "",
            "更合理的 `zx_d` source analogue 候选条目：",
            *reasonable_zx_lines,
            "",
            "说明：这些条目来自 `delta_x/delta_y` 中 z-x mixed 的同层位置 `[0,1]`, `[2,3]`，",
            "在结构上比 `delta_z` 的 interlayer mixed entries 更接近本地 `eta_zx_d` 的角色。",
            "",
            "候选 `zx_d` source analogue 扫描结果：",
            *candidate_lines,
            "",
            "这里需要区分：",
            "- 上述 omitted-channel 数值是严格统计结果。",
            "- `zx_d` 的 source analogue 只是在 source tensor 上的候选 entry 诊断，不是最终严格物理定义。",
            "- 全扫描里数值更大的 `delta_z_sym_03` / `delta_z_sym_12` 更像 interlayer mixed structure，不应直接等同于本地 `eta_zx_d`。",
            "",
            "## 7. 最终判断",
            "",
            f"- “projection 实现正确”: {verdict['projection_implementation_correct']}",
            f"- “projection 与 Luo 图口径一致”: {verdict['projection_figure_level_consistent']}",
            f"- “projection 与原 RMFT 全量信息完全一致”: {verdict['projection_full_rmft_equivalent']}",
            "",
            "必须明确：当前 round-1 projection 是对 RMFT source 的受限近似映射，不应表述成与 full RMFT tensor 完全等价。",
            "",
            "## 8. 下一步建议",
            "",
            "- 如果后续发现 omitted channels 或 retained-ratio 统计持续偏大，应重新审视 `eta_zx_d`、`eta_x_perp` 和 z-sector anisotropy 的投影定义。",
            "- 但本次任务只做诊断，不修改 `luo_projection.py`。",
        ]
    )
    docs_path.write_text(docs_text, encoding="utf-8")


def run_projection_consistency_check(
    output_dir: Path,
    docs_path: Path,
    max_samples: int | None = None,
    make_plots: bool = True,
) -> tuple[dict[str, object], ProjectionDiagnosticsArtifacts]:
    output_dir.mkdir(parents=True, exist_ok=True)
    docs_path.parent.mkdir(parents=True, exist_ok=True)

    repo_dir = ensure_luo_repo()
    samples = load_luo_samples(repo_dir)
    if max_samples is not None:
        samples = samples[: int(max_samples)]

    semantics_checks = collect_source_semantics_checks(repo_dir, samples)
    sample_diagnostics = [_diagnose_single_sample(sample) for sample in samples]
    summary = summarize_projection_diagnostics(sample_diagnostics)
    summary["num_samples"] = int(len(sample_diagnostics))
    summary["source_semantics_checks"] = semantics_checks

    summary_path = output_dir / "projection_consistency_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    examples_csv_path = output_dir / "projection_consistency_examples.csv"
    write_examples_csv(sample_diagnostics, examples_csv_path)

    hist_path: Path | None = None
    scatter_path: Path | None = None
    if make_plots:
        hist_path = output_dir / "projection_consistency_hist.png"
        scatter_path = output_dir / "projection_residual_scatter.png"
        _plot_histogram(sample_diagnostics, hist_path)
        _plot_scatter(sample_diagnostics, scatter_path)

    write_markdown_report(docs_path, semantics_checks, summary)

    artifacts = ProjectionDiagnosticsArtifacts(
        summary_path=summary_path,
        examples_csv_path=examples_csv_path,
        docs_path=docs_path,
        hist_path=hist_path,
        scatter_path=scatter_path,
    )
    return summary, artifacts
