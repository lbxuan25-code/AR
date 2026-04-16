from __future__ import annotations

import numpy as np

from core.parameters import ModelParams
from core.pipeline import SpectroscopyPipeline
from core.presets import base_normal_state_params
from core.simulation_model import SimulationModel
from source.luo_loader import load_luo_samples
from source.round2_projection import project_luo_sample_to_round2_channels


def test_round2_forward_smoke() -> None:
    sample = load_luo_samples()[0]
    projected = project_luo_sample_to_round2_channels(sample)
    assert projected.projected_physical_channels is not None
    pipeline = SpectroscopyPipeline(
        model=SimulationModel(
            params=ModelParams(
                normal_state=base_normal_state_params(),
                pairing=projected.projected_physical_channels,
            ),
            name="round2_smoke_model",
        )
    )
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
