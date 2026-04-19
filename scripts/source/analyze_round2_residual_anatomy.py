from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from source.round2_residual_anatomy import run_round2_residual_anatomy_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the residual anatomy of the current round-2 truth-layer projection.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "source",
    )
    parser.add_argument(
        "--docs-path",
        type=Path,
        default=PROJECT_ROOT / "docs" / "round2_residual_anatomy.md",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, summary, artifacts = run_round2_residual_anatomy_audit(
        output_dir=args.output_dir,
        docs_path=args.docs_path,
        max_samples=args.max_samples,
        make_plots=not args.no_plots,
    )
    print(f"Processed {summary['num_samples']} samples")
    print(f"Wrote summary: {artifacts.summary_path}")
    print(f"Wrote examples CSV: {artifacts.examples_csv_path}")
    print(f"Wrote docs note: {artifacts.docs_path}")
    if artifacts.aggregate_heatmap_path is not None:
        print(f"Wrote aggregate heatmaps: {artifacts.aggregate_heatmap_path}")
    if artifacts.representative_heatmap_path is not None:
        print(f"Wrote representative heatmaps: {artifacts.representative_heatmap_path}")


if __name__ == "__main__":
    main()
