"""Pairing-matrix builders for round-1 and round-2 order-parameter layers."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .form_factors import gamma_d, gamma_s
from .parameters import PairingLike, PairingParams, PhysicalPairingChannels


def physical_channels_from_pairing(params: PairingLike) -> PhysicalPairingChannels:
    """Return round-2 physical channels from either pairing API.

    The legacy round-1 container is interpreted as a restricted subset of the
    round-2 channel language:

    - `eta_z_s -> delta_zz_s`
    - no z-sector d term
    - `eta_x_s -> delta_xx_s`
    - `eta_x_d -> delta_xx_d`
    - no mixed s term
    - `eta_zx_d -> delta_zx_d`
    - `eta_z_perp -> delta_perp_z`
    - `eta_x_perp -> delta_perp_x`
    """

    if isinstance(params, PhysicalPairingChannels):
        return params
    if not isinstance(params, PairingParams):
        raise TypeError(f"Unsupported pairing container type: {type(params)!r}.")
    return PhysicalPairingChannels(
        delta_zz_s=params.eta_z_s,
        delta_zz_d=0.0 + 0.0j,
        delta_xx_s=params.eta_x_s,
        delta_xx_d=params.eta_x_d,
        delta_zx_s=0.0 + 0.0j,
        delta_zx_d=params.eta_zx_d,
        delta_perp_z=params.eta_z_perp,
        delta_perp_x=params.eta_x_perp,
    )


def pairing_terms(kx: float, ky: float, params: PairingLike) -> dict[str, complex]:
    """Return the scalar pairing combinations entering ``Delta(k)``."""

    gamma_s_value = float(np.asarray(gamma_s(kx, ky)))
    gamma_d_value = float(np.asarray(gamma_d(kx, ky)))
    channels = physical_channels_from_pairing(params)
    return {
        "z_diag": channels.delta_zz_s * gamma_s_value + channels.delta_zz_d * gamma_d_value,
        "zx": channels.delta_zx_s * gamma_s_value + channels.delta_zx_d * gamma_d_value,
        "x_diag": channels.delta_xx_s * gamma_s_value + channels.delta_xx_d * gamma_d_value,
        "z_perp": channels.delta_perp_z,
        "x_perp": channels.delta_perp_x,
    }


def delta_matrix(kx: float, ky: float, params: PairingLike) -> NDArray[np.complex128]:
    """Construct the 4x4 pairing matrix in the basis ``(Az, Ax, Bz, Bx)``."""

    terms = pairing_terms(kx, ky, params)
    return np.array(
        [
            [terms["z_diag"], terms["zx"], terms["z_perp"], 0.0],
            [terms["zx"], terms["x_diag"], 0.0, terms["x_perp"]],
            [terms["z_perp"], 0.0, terms["z_diag"], terms["zx"]],
            [0.0, terms["x_perp"], terms["zx"], terms["x_diag"]],
        ],
        dtype=np.complex128,
    )


def round1_pairing_from_physical_channels(channels: PhysicalPairingChannels) -> PairingParams:
    """Return the nearest legacy container when no extra round-2 channels are present.

    This helper is intentionally lossy if the round-2 state uses channels that do
    not exist in the round-1 ansatz. In that case it raises instead of silently
    discarding physics.
    """

    if abs(channels.delta_zz_d) > 1.0e-12 or abs(channels.delta_zx_s) > 1.0e-12:
        raise ValueError("Round-2 channels include components that cannot be represented in legacy PairingParams.")
    return PairingParams(
        eta_z_s=channels.delta_zz_s,
        eta_z_perp=channels.delta_perp_z,
        eta_x_s=channels.delta_xx_s,
        eta_x_d=channels.delta_xx_d,
        eta_zx_d=channels.delta_zx_d,
        eta_x_perp=channels.delta_perp_x,
    )
