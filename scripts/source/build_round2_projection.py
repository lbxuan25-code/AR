from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from source.round2_projection_diagnostics import summarize_round2_projection, write_round2_projection_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build round-2 physical-channel projections for Luo source samples.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "source",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    per_sample, round2_summary, _ = summarize_round2_projection(max_samples=args.max_samples)
    summary_path, examples_path, _ = write_round2_projection_outputs(args.output_dir, per_sample, round2_summary)
    print(f"Wrote summary: {summary_path}")
    print(f"Wrote examples: {examples_path}")


if __name__ == "__main__":
    main()
