from __future__ import annotations

from core.c_axis_direction_audit import run_c_axis_direction_audit
from forward import get_directional_mode

import pytest


def test_c_axis_direction_audit_smoke(tmp_path) -> None:
    summary, artifacts = run_c_axis_direction_audit(output_dir=tmp_path)

    assert artifacts.summary_path.exists()
    assert artifacts.capability_matrix_path.exists()
    assert summary["decision"] == "c_axis_unsupported"
    assert summary["is_c_axis_forward_mode_available"] is False
    assert summary["public_interface_policy"]["direction_mode_c_axis"] == "forbidden"
    assert any(row["current_status"] == "missing_kz" for row in summary["blocking_gaps"])
    assert "must not be emulated" in summary["final_verdict"] or "must not be emulated" in summary["final_verdict"].replace("2D", "2D")


def test_c_axis_direction_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="c-axis transport is not supported"):
        get_directional_mode("c_axis")
