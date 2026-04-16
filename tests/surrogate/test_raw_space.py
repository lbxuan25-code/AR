from __future__ import annotations

import numpy as np

from core.parameters import PairingParams
from surrogate.raw_space import gauge_fixed_vector_to_pairing_params, pairing_params_to_gauge_fixed_vector


def test_raw_space_round_trip() -> None:
    params = PairingParams(
        eta_z_s=1.0 + 2.0j,
        eta_z_perp=3.0 - 4.0j,
        eta_x_s=-1.5 + 0.3j,
        eta_x_d=0.2 - 0.7j,
        eta_zx_d=0.0 + 0.1j,
        eta_x_perp=-0.4 + 0.6j,
    )
    vector = pairing_params_to_gauge_fixed_vector(params)
    restored = gauge_fixed_vector_to_pairing_params(vector)
    original = pairing_params_to_gauge_fixed_vector(params)
    recovered = pairing_params_to_gauge_fixed_vector(restored)
    assert vector.shape == (12,)
    assert np.allclose(original, recovered)

