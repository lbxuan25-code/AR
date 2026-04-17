from __future__ import annotations

import numpy as np

from core.conventions import CORE_PHYSICAL_PAIRING_CHANNELS, OPTIONAL_PHYSICAL_PAIRING_CHANNELS
from core.pairing import delta_matrix, physical_channels_from_pairing
from core.parameters import PairingParams, PhysicalPairingChannels
from core.presets import base_physical_pairing_channels, compatibility_physical_pairing_channels


def test_round2_physical_channels_preserve_legacy_pairing_matrix() -> None:
    legacy = PairingParams(
        eta_z_s=0.4 + 0.1j,
        eta_z_perp=-0.2 + 0.05j,
        eta_x_s=0.3 - 0.07j,
        eta_x_d=-0.11 + 0.03j,
        eta_zx_d=0.09 - 0.02j,
        eta_x_perp=0.08 + 0.04j,
    )
    channels = physical_channels_from_pairing(legacy)
    assert np.allclose(delta_matrix(0.2, -0.4, legacy), delta_matrix(0.2, -0.4, channels))


def test_round2_pairing_matrix_uses_new_channels() -> None:
    channels = PhysicalPairingChannels(
        delta_zz_s=1.0 + 0.0j,
        delta_zz_d=2.0 + 0.0j,
        delta_xx_s=3.0 + 0.0j,
        delta_xx_d=4.0 + 0.0j,
        delta_zx_s=5.0 + 0.0j,
        delta_zx_d=6.0 + 0.0j,
        delta_perp_z=7.0 + 0.0j,
        delta_perp_x=8.0 + 0.0j,
    )
    matrix = delta_matrix(0.0, 0.0, channels)
    assert matrix.shape == (4, 4)
    assert np.isclose(matrix[0, 0], 2.0)
    assert np.isclose(matrix[1, 1], 6.0)
    assert np.isclose(matrix[0, 1], 10.0)
    assert np.isclose(matrix[0, 2], 7.0)
    assert np.isclose(matrix[1, 3], 8.0)


def test_formal_round2_channel_groups_and_baseline() -> None:
    assert OPTIONAL_PHYSICAL_PAIRING_CHANNELS == ("delta_zx_s",)
    assert "delta_zx_s" not in CORE_PHYSICAL_PAIRING_CHANNELS

    formal = base_physical_pairing_channels()
    compatibility = compatibility_physical_pairing_channels()
    assert abs(formal.delta_perp_x) > 1.0
    assert abs(formal.delta_zx_d) > 1.0
    assert not np.isclose(formal.delta_perp_z, compatibility.delta_perp_z)
