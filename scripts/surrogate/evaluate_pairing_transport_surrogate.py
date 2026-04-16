from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from surrogate.evaluate import evaluate_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the pairing+transport surrogate.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "surrogate" / "eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = evaluate_checkpoint(args.dataset, args.checkpoint, args.output_dir)
    print(f"Wrote evaluation report: {report_path}")


if __name__ == "__main__":
    main()
