from __future__ import annotations

import numpy as np

from core.pipeline import SpectroscopyPipeline
from core.presets import base_model_params
from core.simulation_model import SimulationModel


def test_baseline_forward_smoke() -> None:
    pipeline = SpectroscopyPipeline(model=SimulationModel(params=base_model_params(), name="test_baseline"))
    bias = np.linspace(-40.0, 40.0, 121, dtype=np.float64)
    result = pipeline.compute_multichannel_btk_conductance(
        interface_angle=0.0,
        bias=bias,
        barrier_z=0.5,
        broadening_gamma=1.0,
        temperature=3.0,
        nk=31,
    )
    assert result.conductance.shape == bias.shape
    assert np.all(np.isfinite(result.conductance))
    assert result.num_channels > 0

