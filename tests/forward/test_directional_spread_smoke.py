from __future__ import annotations

import math

import numpy as np
import pytest

from forward import (
    BiasGrid,
    DirectionalSpread,
    FitLayerSpectrumRequest,
    generate_spread_spectrum_from_fit_layer,
    transport_with_direction_mode,
)


def test_directional_spread_forward_helper_smoke() -> None:
    request = FitLayerSpectrumRequest(
        transport=transport_with_direction_mode("inplane_110", nk=11),
        bias_grid=BiasGrid(bias_min_mev=-20.0, bias_max_mev=20.0, num_bias=41),
    )
    spread = DirectionalSpread(direction_mode="inplane_110", half_width=math.pi / 128.0, num_samples=3)

    result = generate_spread_spectrum_from_fit_layer(request, spread)
    payload = result.to_dict()

    assert payload["request_kind"] == "fit_layer_directional_spread"
    assert payload["request"]["directional_spread"]["direction_mode"] == "inplane_110"
    assert payload["transport_summary"]["directional_spread"]["num_samples"] == 3
    assert len(payload["transport_summary"]["directional_spread_samples"]) == 3
    assert len(payload["conductance"]) == 41
    assert np.all(np.isfinite(payload["conductance"]))


def test_directional_spread_rejects_too_wide_width() -> None:
    request = FitLayerSpectrumRequest(
        transport=transport_with_direction_mode("inplane_110", nk=11),
        bias_grid=BiasGrid(bias_min_mev=-20.0, bias_max_mev=20.0, num_bias=41),
    )
    spread = DirectionalSpread(direction_mode="inplane_110", half_width=math.pi / 8.0, num_samples=5)

    with pytest.raises(ValueError, match="outside the current narrow-spread contract"):
        generate_spread_spectrum_from_fit_layer(request, spread)
