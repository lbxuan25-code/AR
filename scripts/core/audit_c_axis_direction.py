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

from core.c_axis_direction_audit import run_c_axis_direction_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit whether true c-axis forward transport is supported.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "core" / "c_axis_direction_audit",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, artifacts = run_c_axis_direction_audit(output_dir=args.output_dir)
    print(f"Wrote summary: {artifacts.summary_path}")
    print(f"Wrote capability matrix: {artifacts.capability_matrix_path}")
    print(summary["final_verdict"])


if __name__ == "__main__":
    main()
