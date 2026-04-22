import numpy as np
import pytest

from forward import (
    BiasGrid,
    FitLayerSpectrumRequest,
    SourceRound2SpectrumRequest,
    TransportControls,
    generate_spectrum_from_fit_layer,
    generate_spectrum_from_source_round2,
)
from forward.schema import FORWARD_INTERFACE_VERSION, FORWARD_OUTPUT_SCHEMA_VERSION


def _small_transport() -> TransportControls:
    return TransportControls(interface_angle=0.0, barrier_z=0.5, gamma=1.0, temperature_kelvin=3.0, nk=11)


def _small_bias_grid() -> BiasGrid:
    return BiasGrid(bias_min_mev=-20.0, bias_max_mev=20.0, num_bias=41)


def test_fit_layer_forward_interface_smoke() -> None:
    request = FitLayerSpectrumRequest(
        pairing_controls={"delta_zz_s": 0.25, "delta_perp_x": -0.1},
        transport=_small_transport(),
        bias_grid=_small_bias_grid(),
        request_label="fit_layer_smoke",
    )

    result = generate_spectrum_from_fit_layer(request)
    payload = result.to_dict()

    assert payload["schema_version"] == FORWARD_OUTPUT_SCHEMA_VERSION
    assert payload["metadata"]["forward_interface_version"] == FORWARD_INTERFACE_VERSION
    assert payload["metadata"]["pairing_source"] == "task_h_fit_layer_controls"
    assert payload["metadata"]["pairing_convention_id"] == "round2_physical_channels_task_h_fit_layer_v1"
    assert payload["metadata"]["formal_baseline_record"] == "outputs/source/round2_baseline_selection.json"
    assert payload["pairing_channels"]["delta_zx_s"] == {"re": 0.0, "im": 0.0}
    assert payload["request"]["transport"]["direction_mode"] is None
    assert payload["transport_summary"]["direction_mode"] is None
    assert payload["transport_summary"]["direction_support_tier"] == "raw_2d_inplane_angle"
    assert len(payload["bias_mev"]) == 41
    assert len(payload["conductance"]) == 41
    assert np.all(np.isfinite(payload["conductance"]))
    assert payload["transport_summary"]["num_channels"] > 0


def test_fit_layer_weak_channel_requires_explicit_branch() -> None:
    request = FitLayerSpectrumRequest(
        pairing_controls={"delta_zx_s": 1.0},
        transport=_small_transport(),
        bias_grid=_small_bias_grid(),
    )

    with pytest.raises(ValueError, match="delta_zx_s is fixed to zero"):
        generate_spectrum_from_fit_layer(request)


def test_source_round2_forward_interface_smoke() -> None:
    request = SourceRound2SpectrumRequest(
        source_sample_index=0,
        transport=_small_transport(),
        bias_grid=_small_bias_grid(),
        request_label="source_round2_smoke",
    )

    result = generate_spectrum_from_source_round2(request)
    payload = result.to_dict()

    assert payload["schema_version"] == FORWARD_OUTPUT_SCHEMA_VERSION
    assert payload["request_kind"] == "source_round2"
    assert payload["metadata"]["pairing_source"] == "luo_source_default_round2_projection"
    assert payload["metadata"]["source_sample_id"]
    assert payload["metadata"]["round2_projection_metrics"]["retained_ratio_total"] > 0.0
    assert payload["request"]["transport"]["direction_mode"] is None
    assert payload["transport_summary"]["direction_mode"] is None
    assert len(payload["bias_mev"]) == 41
    assert len(payload["conductance"]) == 41
    assert np.all(np.isfinite(payload["conductance"]))
