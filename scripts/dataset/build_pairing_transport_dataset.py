from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data.dataset_builder import DatasetBuildConfig, build_pairing_transport_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the round-1 pairing+transport dataset.")
    parser.add_argument("--scale", choices=("smoke", "mini", "full"), default="smoke")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--nk", type=int, default=41)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "dataset")
    parser.add_argument("--num-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_counts = {"smoke": 600, "mini": 2400, "full": 10000}
    config = DatasetBuildConfig(
        scale=args.scale,
        num_samples=int(args.num_samples or default_counts[args.scale]),
        seed=int(args.seed),
        nk=int(args.nk),
    )
    dataset_path, manifest_path = build_pairing_transport_dataset(args.output_dir, config)
    print(f"Wrote dataset: {dataset_path}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
