from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from source.rmft_source_ar_validation import run_rmft_source_vs_round2_ar_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate round-2 AR spectra against RMFT source-reference spectra.")
    parser.add_argument("--nk", type=int, default=41)
    parser.add_argument("--bias-max", type=float, default=40.0)
    parser.add_argument("--num-bias", type=int, default=201)
    parser.add_argument(
        "--max-selection-samples",
        type=int,
        default=None,
        help="Optional sample limit used only for choosing best/median/worst representatives.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "core" / "rmft_source_vs_round2_ar_validation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, _, artifacts = run_rmft_source_vs_round2_ar_validation(
        output_dir=args.output_dir,
        max_selection_samples=args.max_selection_samples,
        nk=args.nk,
        bias_max=args.bias_max,
        num_bias=args.num_bias,
    )
    print(f"Wrote summary: {artifacts.summary_path}")
    print(f"Wrote metrics CSV: {artifacts.metrics_csv_path}")
    for path in artifacts.plot_paths:
        print(f"Wrote representative plot: {path}")
    print(summary["conclusion"]["verdict"])


if __name__ == "__main__":
    main()
