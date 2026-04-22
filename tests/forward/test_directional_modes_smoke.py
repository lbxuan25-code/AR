from __future__ import annotations

import math

import numpy as np
import pytest

from core.directional_modes_validation import run_directional_modes_validation
from forward import (
    BiasGrid,
    FitLayerSpectrumRequest,
    TransportControls,
    generate_spectrum_from_fit_layer,
    interface_angle_for_direction_mode,
    list_directional_modes,
    transport_with_direction_mode,
)


def test_named_directional_modes_are_callable() -> None:
    mode_names = {mode.name for mode in list_directional_modes()}
    assert mode_names == {"inplane_100", "inplane_110"}
    assert interface_angle_for_direction_mode("inplane_100") == 0.0
    assert interface_angle_for_direction_mode("inplane_110") == pytest.approx(math.pi / 4.0)

    request = FitLayerSpectrumRequest(
        transport=transport_with_direction_mode("inplane_110", nk=11),
        bias_grid=BiasGrid(bias_min_mev=-20.0, bias_max_mev=20.0, num_bias=41),
    )
    payload = generate_spectrum_from_fit_layer(request).to_dict()

    assert payload["request"]["transport"]["direction_mode"] == "inplane_110"
    assert payload["request"]["transport"]["interface_angle"] == pytest.approx(math.pi / 4.0)
    assert payload["transport_summary"]["direction_mode"] == "inplane_110"
    assert payload["transport_summary"]["direction_crystal_label"] == "110"


def test_direction_mode_rejects_inconsistent_raw_angle() -> None:
    request = FitLayerSpectrumRequest(
        transport=TransportControls(direction_mode="inplane_110", interface_angle=0.0, nk=11),
        bias_grid=BiasGrid(bias_min_mev=-20.0, bias_max_mev=20.0, num_bias=41),
    )
    with pytest.raises(ValueError, match="direction_mode and interface_angle disagree"):
        generate_spectrum_from_fit_layer(request)


def test_directional_modes_validation_smoke(tmp_path) -> None:
    summary, artifacts = run_directional_modes_validation(
        output_dir=tmp_path,
        nk=21,
        bias_max=30.0,
        num_bias=81,
    )

    assert artifacts.summary_path.exists()
    assert artifacts.metrics_csv_path.exists()
    assert artifacts.comparison_plot_path is not None and artifacts.comparison_plot_path.exists()
    assert {row["direction_mode"] for row in summary["comparisons"]} == {"inplane_100", "inplane_110"}
    assert summary["max_abs_conductance_diff_across_modes"] <= 1.0e-14
    assert all(np.isclose(row["raw_interface_angle"], row["named_interface_angle"]) for row in summary["comparisons"])
