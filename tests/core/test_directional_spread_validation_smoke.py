from __future__ import annotations

import math

from core.directional_spread_validation import run_directional_spread_validation


def test_directional_spread_validation_smoke(tmp_path) -> None:
    summary, artifacts = run_directional_spread_validation(
        output_dir=tmp_path,
        direction_modes=("inplane_110",),
        half_widths=(0.0, math.pi / 128.0, math.pi / 64.0),
        barriers=(0.5,),
        pairing_states={"formal_baseline": {}},
        num_spread_samples=3,
        nk=21,
        bias_max=30.0,
        num_bias=81,
    )

    assert artifacts.summary_path.exists()
    assert artifacts.metrics_csv_path.exists()
    assert artifacts.plot_path is not None and artifacts.plot_path.exists()
    assert summary["spread_definition"]["averaging_rule"] == "uniform_symmetric arithmetic average of normalized spectra"
    assert summary["num_cases"] == 3
    assert summary["spectra_smooth_under_width_variation"] is True
    assert all("width_step_max_abs" in row for row in summary["metrics"])
