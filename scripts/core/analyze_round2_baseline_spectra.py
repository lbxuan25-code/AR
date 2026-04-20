from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.round2_baseline_spectral_validation import run_round2_baseline_spectral_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the formal round-2 baseline in AR spectra.")
    parser.add_argument("--nk", type=int, default=41)
    parser.add_argument("--bias-max", type=float, default=40.0)
    parser.add_argument("--num-bias", type=int, default=201)
    parser.add_argument(
        "--representative-selection-max-samples",
        type=int,
        default=None,
        help="Optional limit used only when selecting representative Luo projected samples.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "core" / "round2_baseline_spectral_validation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, artifacts = run_round2_baseline_spectral_validation(
        output_dir=args.output_dir,
        representative_selection_max_samples=args.representative_selection_max_samples,
        nk=args.nk,
        bias_max=args.bias_max,
        num_bias=args.num_bias,
    )
    print(f"Wrote summary: {artifacts.summary_path}")
    print(f"Wrote metrics CSV: {artifacts.metrics_csv_path}")
    if artifacts.model_scan_plot_path is not None:
        print(f"Wrote model comparison plot: {artifacts.model_scan_plot_path}")
    if artifacts.channel_sensitivity_plot_path is not None:
        print(f"Wrote channel sensitivity plot: {artifacts.channel_sensitivity_plot_path}")
    print(summary["verdict"])


if __name__ == "__main__":
    main()
