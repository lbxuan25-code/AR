from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FORWARD_EXAMPLES = PROJECT_ROOT / "outputs" / "core" / "forward_interface"


def _load_example(name: str) -> dict[str, object]:
    return json.loads((FORWARD_EXAMPLES / name).read_text(encoding="utf-8"))


def _assert_clean_metadata(payload: dict[str, object]) -> None:
    metadata = payload["metadata"]
    assert metadata["git_dirty"] is False
    baseline_record = metadata["formal_baseline_record"]
    assert baseline_record == "outputs/source/round2_baseline_selection.json"
    assert not str(baseline_record).startswith("/")


def test_directional_contract_index_exists_and_matches_supported_modes() -> None:
    doc_path = PROJECT_ROOT / "docs" / "directional_capability_contract_task_q.md"
    index_path = FORWARD_EXAMPLES / "directional_capability_index.json"
    assert doc_path.exists()
    assert index_path.exists()

    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["contract_id"] == "directional_capability_task_q_v1"
    assert set(index["supported_named_modes"]) == {"inplane_100", "inplane_110"}
    assert index["supported_named_modes"]["inplane_100"]["interface_angle"] == 0.0
    assert index["supported_named_modes"]["inplane_110"]["interface_angle"] == pytest.approx(math.pi / 4.0)
    assert index["generic_raw_inplane_angles"]["status"] == "diagnostic_caution_only"
    assert index["generic_raw_inplane_angles"]["example_payload"] is None
    assert index["c_axis"]["status"] == "unsupported"
    assert index["c_axis"]["example_payload"] is None


def test_forward_examples_record_direction_provenance() -> None:
    raw_fit = _load_example("fit_layer_example_spectrum.json")
    raw_source = _load_example("source_round2_example_spectrum.json")
    named = _load_example("fit_layer_inplane_110_example_spectrum.json")

    for payload in (raw_fit, raw_source, named):
        _assert_clean_metadata(payload)
        assert "direction_mode" in payload["request"]["transport"]
        assert "interface_angle" in payload["request"]["transport"]
        assert "direction_support_tier" in payload["transport_summary"]

    assert raw_fit["request"]["transport"]["direction_mode"] is None
    assert raw_fit["transport_summary"]["direction_support_tier"] == "raw_2d_inplane_angle"
    assert raw_source["request"]["transport"]["direction_mode"] is None
    assert raw_source["transport_summary"]["direction_support_tier"] == "raw_2d_inplane_angle"

    assert named["request"]["transport"]["direction_mode"] == "inplane_110"
    assert named["request"]["transport"]["interface_angle"] == pytest.approx(math.pi / 4.0)
    assert named["transport_summary"]["direction_mode"] == "inplane_110"
    assert named["transport_summary"]["direction_support_tier"] == "A"
    assert named["transport_summary"]["direction_crystal_label"] == "110"


def test_spread_example_records_spread_contract() -> None:
    payload = _load_example("fit_layer_inplane_110_spread_example_spectrum.json")
    _assert_clean_metadata(payload)

    assert payload["request_kind"] == "fit_layer_directional_spread"
    assert payload["request"]["transport"]["direction_mode"] == "inplane_110"
    assert payload["request"]["transport"]["interface_angle"] == pytest.approx(math.pi / 4.0)

    for container in (payload["request"], payload["metadata"], payload["transport_summary"]):
        spread = container["directional_spread"]
        assert spread["direction_mode"] == "inplane_110"
        assert spread["support_tier"] == "A"
        assert spread["averaging_rule"] == "uniform_symmetric"
        assert spread["half_width"] == pytest.approx(math.pi / 64.0)
        assert spread["max_half_width"] == pytest.approx(math.pi / 32.0)

    samples = payload["transport_summary"]["directional_spread_samples"]
    assert len(samples) == 5
    assert sum(sample["weight"] for sample in samples) == pytest.approx(1.0)
    assert samples[2]["relative_angle"] == pytest.approx(0.0)
