from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.formal_baseline import AUTHORITATIVE_ROUND2_BASELINE_RECORD, formal_round2_baseline_channels
from core.presets import base_physical_pairing_channels


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _complex_from_json(value: object) -> complex:
    if isinstance(value, dict):
        return complex(float(value["re"]), float(value["im"]))
    if isinstance(value, list):
        return complex(float(value[0]), float(value[1]))
    raise TypeError(f"Unsupported complex JSON payload: {value!r}")


def _channels_from_json(payload: dict[str, object]) -> dict[str, complex]:
    return {name: _complex_from_json(value) for name, value in payload.items()}


def _assert_channels_match(left: dict[str, complex], right: dict[str, complex]) -> None:
    assert set(left) == set(right)
    for name in left:
        assert np.isclose(left[name], right[name], rtol=0.0, atol=1.0e-12), name


def test_formal_baseline_uses_authoritative_record() -> None:
    record = json.loads(AUTHORITATIVE_ROUND2_BASELINE_RECORD.read_text(encoding="utf-8"))
    record_channels = _channels_from_json(record["pairing_channels"])
    preset_channels = base_physical_pairing_channels().to_dict()
    loader_channels = formal_round2_baseline_channels().to_dict()

    _assert_channels_match(record_channels, preset_channels)
    _assert_channels_match(record_channels, loader_channels)
    assert record["authoritative_record_path"] == "outputs/source/round2_baseline_selection.json"
    assert preset_channels["delta_zx_s"] == 0.0 + 0.0j


def test_formal_baseline_outputs_do_not_drift() -> None:
    preset_channels = base_physical_pairing_channels().to_dict()
    source_summary = json.loads((PROJECT_ROOT / "outputs/source/round2_projection_summary.json").read_text(encoding="utf-8"))
    task_d_summary = json.loads(
        (PROJECT_ROOT / "outputs/core/round2_baseline_spectral_validation/round2_baseline_spectral_validation_summary.json").read_text(
            encoding="utf-8"
        )
    )

    source_baseline_channels = _channels_from_json(source_summary["baseline_channels"])
    source_record_channels = _channels_from_json(source_summary["baseline_cluster_summary"]["pairing_channels"])
    task_d_channels = _channels_from_json(task_d_summary["pairing_states"]["formal_round2_baseline"]["channels"])

    _assert_channels_match(preset_channels, source_baseline_channels)
    _assert_channels_match(preset_channels, source_record_channels)
    _assert_channels_match(preset_channels, task_d_channels)
    assert task_d_channels["delta_zx_s"] == 0.0 + 0.0j
