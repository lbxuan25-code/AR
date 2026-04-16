from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.config import (
    FORMAL_ALLOW_CROSS_BAND_FALLBACK,
    FORMAL_MAX_REFLECTION_MISMATCH,
    FORMAL_MIN_CHANNEL_WEIGHT,
    FORMAL_MISMATCH_PENALTY_SCALE,
    FORMAL_REFLECTED_BRANCH_MODE,
)
from core.pipeline import SpectroscopyPipeline
from core.presets import base_model_params
from core.simulation_model import SimulationModel


def _complex_pair(value: complex) -> list[float]:
    return [float(np.real(value)), float(np.imag(value))]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the round-1 baseline physics forward workflow.")
    parser.add_argument("--nk", type=int, default=81)
    parser.add_argument("--interface-angle", type=float, default=0.0)
    parser.add_argument("--barrier-z", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--bias-max", type=float, default=40.0)
    parser.add_argument("--num-bias", type=int, default=601)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "core" / "baseline_forward",
    )
    return parser.parse_args()


def _plot_spectrum(bias: np.ndarray, conductance: np.ndarray, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    axis.plot(bias, conductance, linewidth=1.6, color="tab:blue")
    axis.set_xlabel("Bias (meV)")
    axis.set_ylabel("Normalized conductance")
    axis.set_title("Baseline multichannel BTK spectrum")
    axis.grid(alpha=0.2)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_gap_on_fs(gap_data: list, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.2, 5.6), constrained_layout=True)
    scatter = None
    vmax = 0.0
    signed_gaps: list[np.ndarray] = []
    for contour in gap_data:
        gap = np.asarray(contour.projected_gaps, dtype=np.complex128)
        signed = np.sign(np.real(gap)) * np.abs(gap)
        signed_gaps.append(signed)
        vmax = max(vmax, float(np.max(np.abs(signed))) if signed.size else 0.0)

    vmax = max(vmax, 1.0e-8)
    for contour, signed in zip(gap_data, signed_gaps, strict=True):
        axis.plot(contour.k_points[:, 0], contour.k_points[:, 1], linewidth=0.5, alpha=0.25, color="black")
        scatter = axis.scatter(
            contour.k_points[:, 0],
            contour.k_points[:, 1],
            c=signed,
            s=10.0,
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
            linewidths=0.0,
        )
    axis.set_xlim(-np.pi, np.pi)
    axis.set_ylim(-np.pi, np.pi)
    axis.set_aspect("equal")
    axis.set_xlabel(r"$k_x$")
    axis.set_ylabel(r"$k_y$")
    axis.set_title("Projected gap on Fermi surface")
    axis.grid(alpha=0.15)
    if scatter is not None:
        figure.colorbar(scatter, ax=axis, label=r"sign(Re $\Delta$)$|\Delta|$")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    params = base_model_params()
    model = SimulationModel(params=params, name="round1_baseline_model")
    pipeline = SpectroscopyPipeline(model=model)
    bias = np.linspace(-float(args.bias_max), float(args.bias_max), int(args.num_bias), dtype=np.float64)

    gap_data = pipeline.gap_on_fermi_surface(nk=int(args.nk))
    result = pipeline.compute_multichannel_btk_conductance(
        interface_angle=float(args.interface_angle),
        bias=bias,
        barrier_z=float(args.barrier_z),
        broadening_gamma=float(args.gamma),
        temperature=float(args.temperature),
        nk=int(args.nk),
        reflected_branch_mode=FORMAL_REFLECTED_BRANCH_MODE,
        allow_cross_band_fallback=FORMAL_ALLOW_CROSS_BAND_FALLBACK,
        max_reflection_mismatch=FORMAL_MAX_REFLECTION_MISMATCH,
        strict_reflection_match=False,
        mismatch_penalty_scale=FORMAL_MISMATCH_PENALTY_SCALE,
        min_channel_weight=FORMAL_MIN_CHANNEL_WEIGHT,
    )

    spectrum_path = output_dir / "baseline_spectrum.png"
    gap_path = output_dir / "baseline_gap_on_fs.png"
    summary_path = output_dir / "baseline_summary.json"

    _plot_spectrum(np.asarray(result.bias), np.asarray(result.conductance), spectrum_path)
    _plot_gap_on_fs(gap_data, gap_path)

    zero_index = int(np.argmin(np.abs(result.bias)))
    summary = {
        "baseline_source": {
            "repository": "local migration from LNO327_AR_Phenomenology",
            "normal_state_preset": "base_normal_state_params",
            "pairing_preset": "base_pairing_params",
            "consistency_note": "migrated round-1 baseline is intended to match the old repository baseline values",
        },
        "transport": {
            "interface_angle": float(args.interface_angle),
            "barrier_z": float(args.barrier_z),
            "gamma": float(args.gamma),
            "temperature_kelvin": float(args.temperature),
        },
        "bias_grid": {
            "bias_max_meV": float(args.bias_max),
            "num_bias": int(args.num_bias),
        },
        "pairing_params": {
            "eta_z_s": _complex_pair(params.pairing.eta_z_s),
            "eta_z_perp": _complex_pair(params.pairing.eta_z_perp),
            "eta_x_s": _complex_pair(params.pairing.eta_x_s),
            "eta_x_d": _complex_pair(params.pairing.eta_x_d),
            "eta_zx_d": _complex_pair(params.pairing.eta_zx_d),
            "eta_x_perp": _complex_pair(params.pairing.eta_x_perp),
        },
        "forward_summary": {
            "num_contours": len(gap_data),
            "num_channels": int(result.num_channels),
            "num_input_channels": int(result.num_input_channels),
            "mean_normal_transparency": float(result.mean_normal_transparency),
            "mean_mismatch_penalty": float(result.mean_mismatch_penalty),
            "zero_bias_conductance": float(result.conductance[zero_index]),
            "max_conductance": float(np.max(result.conductance)),
            "min_conductance": float(np.min(result.conductance)),
        },
        "outputs": {
            "spectrum_figure": str(spectrum_path),
            "gap_on_fs_figure": str(gap_path),
            "summary_json": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary["forward_summary"], indent=2))


if __name__ == "__main__":
    main()
