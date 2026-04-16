from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from surrogate.inverse import run_inverse_demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the round-1 surrogate-assisted inverse demo.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "inverse")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = run_inverse_demo(args.dataset, args.checkpoint, args.output_dir, top_k=int(args.top_k))
    print(f"Wrote inverse report: {report_path}")


if __name__ == "__main__":
    main()
