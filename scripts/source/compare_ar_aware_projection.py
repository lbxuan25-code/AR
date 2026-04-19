from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from source.ar_aware_projection_diagnostics import (
    summarize_ar_aware_projection_comparison,
    write_ar_aware_projection_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare default and AR-aware round-2 projection.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "source",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--representative-spectrum-nk", type=int, default=41)
    parser.add_argument("--representative-num-bias", type=int, default=201)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    per_sample, summary, representative_samples = summarize_ar_aware_projection_comparison(
        max_samples=args.max_samples,
        representative_spectrum_nk=args.representative_spectrum_nk,
        representative_num_bias=args.representative_num_bias,
    )
    artifacts = write_ar_aware_projection_outputs(
        output_dir=args.output_dir,
        per_sample=per_sample,
        summary=summary,
        representative_samples=representative_samples,
        make_plots=not args.no_plots,
    )
    print(f"Processed {summary['num_samples']} samples")
    print(f"Wrote summary: {artifacts.summary_path}")
    print(f"Wrote examples CSV: {artifacts.examples_csv_path}")
    if artifacts.spectra_plot_path is not None:
        print(f"Wrote spectra plot: {artifacts.spectra_plot_path}")


if __name__ == "__main__":
    main()
