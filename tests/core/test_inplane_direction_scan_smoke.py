from __future__ import annotations

import math

from core.inplane_direction_scan import run_inplane_direction_scan


def test_inplane_direction_scan_smoke(tmp_path) -> None:
    summary, artifacts = run_inplane_direction_scan(
        output_dir=tmp_path,
        num_angles=9,
        nk=21,
        bias_max=30.0,
        num_bias=81,
    )

    assert artifacts.summary_path.exists()
    assert artifacts.metrics_csv_path.exists()
    assert artifacts.plot_path is not None and artifacts.plot_path.exists()
    assert summary["scan_config"]["angle_domain"] == "[0, pi/2]"
    assert len(summary["angle_metrics"]) == 9
    assert "support_thresholds" in summary
    assert "generic_inplane_support_decision" in summary
    assert summary["angle_metrics"][0]["interface_angle_rad"] == 0.0
    assert summary["angle_metrics"][-1]["interface_angle_rad"] == math.pi / 2.0
    assert all("neighbor_max_abs_conductance_step" in row for row in summary["angle_metrics"])
    assert all("matched_fraction_tolerance_span" in row for row in summary["angle_metrics"])
