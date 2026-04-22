from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.directional_spread_validation import run_directional_spread_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate narrow directional-spread forward spectra.")
    parser.add_argument("--nk", type=int, default=31)
    parser.add_argument("--bias-max", type=float, default=40.0)
    parser.add_argument("--num-bias", type=int, default=161)
    parser.add_argument("--num-spread-samples", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "core" / "directional_spread_validation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, artifacts = run_directional_spread_validation(
        output_dir=args.output_dir,
        nk=args.nk,
        bias_max=args.bias_max,
        num_bias=args.num_bias,
        num_spread_samples=args.num_spread_samples,
    )
    print(f"Wrote summary: {artifacts.summary_path}")
    print(f"Wrote metrics CSV: {artifacts.metrics_csv_path}")
    if artifacts.plot_path is not None:
        print(f"Wrote validation plot: {artifacts.plot_path}")
    print(summary["final_verdict"])


if __name__ == "__main__":
    main()
