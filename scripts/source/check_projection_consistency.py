from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from source.projection_diagnostics import run_projection_consistency_check


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Luo projection consistency diagnostics without changing projection logic.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "source",
    )
    parser.add_argument(
        "--docs-path",
        type=Path,
        default=PROJECT_ROOT / "docs" / "projection_consistency_round1.md",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for smoke runs.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip optional histogram/scatter outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary, artifacts = run_projection_consistency_check(
        output_dir=args.output_dir,
        docs_path=args.docs_path,
        max_samples=args.max_samples,
        make_plots=not args.no_plots,
    )
    print(f"Processed {summary['num_samples']} samples")
    print(f"Wrote summary: {artifacts.summary_path}")
    print(f"Wrote examples CSV: {artifacts.examples_csv_path}")
    print(f"Wrote markdown report: {artifacts.docs_path}")
    if artifacts.hist_path is not None:
        print(f"Wrote histogram: {artifacts.hist_path}")
    if artifacts.scatter_path is not None:
        print(f"Wrote scatter: {artifacts.scatter_path}")


if __name__ == "__main__":
    main()
