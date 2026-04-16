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

from core.parameters import ModelParams
from core.pipeline import SpectroscopyPipeline
from core.presets import base_normal_state_params, base_physical_pairing_channels
from core.simulation_model import SimulationModel
from source.luo_loader import load_luo_samples
from source.round2_projection import project_luo_sample_to_round2_channels


def _serialize_complex_dict(values: dict[str, complex]) -> dict[str, list[float]]:
    return {
        key: [float(np.real(value)), float(np.imag(value))]
        for key, value in values.items()
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the forward workflow with the round-2 physical pairing channels.")
    parser.add_argument("--nk", type=int, default=61)
    parser.add_argument("--interface-angle", type=float, default=0.0)
    parser.add_argument("--barrier-z", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--bias-max", type=float, default=40.0)
    parser.add_argument("--num-bias", type=int, default=601)
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Optional Luo sample index. If omitted, use the compatibility baseline physical channels.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "core" / "round2_baseline_forward",
    )
    return parser.parse_args()


def _plot_spectrum(bias: np.ndarray, conductance: np.ndarray, output_path: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    axis.plot(bias, conductance, linewidth=1.6)
    axis.set_xlabel("Bias (meV)")
    axis.set_ylabel("Normalized conductance")
    axis.set_title(title)
    axis.grid(alpha=0.2)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pairing = base_physical_pairing_channels()
    source_label = "compatibility_baseline_physical_channels"
    if args.sample_index is not None:
        sample = load_luo_samples()[int(args.sample_index)]
        projected = project_luo_sample_to_round2_channels(sample)
        assert projected.projected_physical_channels is not None
        pairing = projected.projected_physical_channels
        source_label = f"luo_sample::{sample.sample_id}"

    params = ModelParams(normal_state=base_normal_state_params(), pairing=pairing)
    pipeline = SpectroscopyPipeline(model=SimulationModel(params=params, name="round2_forward_model"))
    bias = np.linspace(-float(args.bias_max), float(args.bias_max), int(args.num_bias), dtype=np.float64)
    result = pipeline.compute_multichannel_btk_conductance(
        interface_angle=float(args.interface_angle),
        bias=bias,
        barrier_z=float(args.barrier_z),
        broadening_gamma=float(args.gamma),
        temperature=float(args.temperature),
        nk=int(args.nk),
    )

    spectrum_path = output_dir / "round2_spectrum.png"
    summary_path = output_dir / "round2_forward_summary.json"
    _plot_spectrum(np.asarray(result.bias), np.asarray(result.conductance), spectrum_path, "Round-2 Forward Spectrum")
    summary = {
        "pairing_source": source_label,
        "pairing_channels": _serialize_complex_dict(pairing.to_dict()),
        "transport": {
            "interface_angle": float(args.interface_angle),
            "barrier_z": float(args.barrier_z),
            "gamma": float(args.gamma),
            "temperature_kelvin": float(args.temperature),
        },
        "forward_summary": {
            "num_channels": int(result.num_channels),
            "num_input_channels": int(result.num_input_channels),
            "num_contours": int(result.num_contours),
            "mean_normal_transparency": float(result.mean_normal_transparency),
            "mean_mismatch_penalty": float(result.mean_mismatch_penalty),
        },
        "outputs": {
            "spectrum": str(spectrum_path),
            "summary": str(summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote spectrum: {spectrum_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
