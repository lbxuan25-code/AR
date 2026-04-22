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

from core.directional_modes_validation import run_directional_modes_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate canonical in-plane directional forward modes.")
    parser.add_argument("--nk", type=int, default=41)
    parser.add_argument("--bias-max", type=float, default=40.0)
    parser.add_argument("--num-bias", type=int, default=201)
    parser.add_argument("--barrier-z", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "core" / "directional_modes_validation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, artifacts = run_directional_modes_validation(
        output_dir=args.output_dir,
        nk=args.nk,
        bias_max=args.bias_max,
        num_bias=args.num_bias,
        barrier_z=args.barrier_z,
        gamma=args.gamma,
        temperature=args.temperature,
    )
    print(f"Wrote summary: {artifacts.summary_path}")
    print(f"Wrote metrics CSV: {artifacts.metrics_csv_path}")
    if artifacts.comparison_plot_path is not None:
        print(f"Wrote comparison plot: {artifacts.comparison_plot_path}")
    print(summary["verdict"])


if __name__ == "__main__":
    main()
