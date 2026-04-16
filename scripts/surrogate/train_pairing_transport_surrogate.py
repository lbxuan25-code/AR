from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from surrogate.config import TrainConfig
from surrogate.train import train_surrogate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the pairing+transport surrogate.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "surrogate" / "train")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(seed=int(args.seed), max_epochs=int(args.max_epochs), patience=int(args.patience))
    checkpoint_path, log_path = train_surrogate(args.dataset, args.output_dir, config)
    print(f"Wrote checkpoint: {checkpoint_path}")
    print(f"Wrote log: {log_path}")


if __name__ == "__main__":
    main()
