"""Authoritative formal round-2 baseline record loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .conventions import PHYSICAL_PAIRING_CHANNELS
from .parameters import PhysicalPairingChannels

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUTHORITATIVE_ROUND2_BASELINE_RECORD = PROJECT_ROOT / "outputs" / "source" / "round2_baseline_selection.json"


def _complex_from_record(value: object, channel_name: str) -> complex:
    if not isinstance(value, dict) or "re" not in value or "im" not in value:
        raise ValueError(f"Malformed complex baseline value for {channel_name!r}: {value!r}.")
    return complex(float(value["re"]), float(value["im"]))


def load_authoritative_round2_baseline_record(
    record_path: Path = AUTHORITATIVE_ROUND2_BASELINE_RECORD,
) -> dict[str, Any]:
    """Load the single authoritative formal round-2 baseline artifact."""

    path = Path(record_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing authoritative round-2 baseline record: {path}. "
            "Regenerate it with `PYTHONPATH=src python scripts/source/build_round2_projection.py`."
        )
    record = json.loads(path.read_text(encoding="utf-8"))
    if "pairing_channels" not in record:
        raise ValueError(f"Baseline record {path} does not contain `pairing_channels`.")
    missing = [name for name in PHYSICAL_PAIRING_CHANNELS if name not in record["pairing_channels"]]
    if missing:
        raise ValueError(f"Baseline record {path} is missing channels: {missing}.")
    return record


def formal_round2_baseline_channels(
    record_path: Path = AUTHORITATIVE_ROUND2_BASELINE_RECORD,
) -> PhysicalPairingChannels:
    """Return formal round-2 baseline channels from the authoritative record."""

    record = load_authoritative_round2_baseline_record(record_path)
    values = {
        name: _complex_from_record(record["pairing_channels"][name], name)
        for name in PHYSICAL_PAIRING_CHANNELS
    }
    return PhysicalPairingChannels(**values)
