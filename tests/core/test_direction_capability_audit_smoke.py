from __future__ import annotations

import math

from core.direction_capability_audit import run_direction_capability_audit


def test_direction_capability_audit_smoke(tmp_path) -> None:
    summary, artifacts = run_direction_capability_audit(
        output_dir=tmp_path,
        nk=21,
        bias_max=30.0,
        num_bias=81,
    )

    assert artifacts.summary_path.exists()
    assert artifacts.metrics_csv_path.exists()
    assert artifacts.representative_plot_path is not None and artifacts.representative_plot_path.exists()
    assert summary["semantic_contract"]["dimensionality"] == "strictly 2D in-plane in the current implementation"
    assert "Not supported" in summary["support_status"]["c-axis"]
    assert len(summary["angle_metrics"]) == 5
    assert {row["interface_angle_rad"] for row in summary["angle_metrics"]} >= {0.0, math.pi / 4.0}
    assert summary["direction_tiers"]["Tier C"]["directions"][0]["label"] == "c-axis"
    assert all(row["matched_reflected_channel_count"] > 0 for row in summary["angle_metrics"])
