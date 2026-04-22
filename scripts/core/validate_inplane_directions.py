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

from core.inplane_direction_scan import run_inplane_direction_scan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generic 2D in-plane raw interface angles.")
    parser.add_argument("--num-angles", type=int, default=33)
    parser.add_argument("--nk", type=int, default=41)
    parser.add_argument("--bias-max", type=float, default=40.0)
    parser.add_argument("--num-bias", type=int, default=201)
    parser.add_argument("--barrier-z", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "core" / "inplane_direction_scan",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, artifacts = run_inplane_direction_scan(
        output_dir=args.output_dir,
        num_angles=args.num_angles,
        nk=args.nk,
        bias_max=args.bias_max,
        num_bias=args.num_bias,
        barrier_z=args.barrier_z,
        gamma=args.gamma,
        temperature=args.temperature,
    )
    print(f"Wrote summary: {artifacts.summary_path}")
    print(f"Wrote metrics CSV: {artifacts.metrics_csv_path}")
    if artifacts.plot_path is not None:
        print(f"Wrote scan plot: {artifacts.plot_path}")
    print(summary["final_verdict"])


if __name__ == "__main__":
    main()
