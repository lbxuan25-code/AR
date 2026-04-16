"""Unified multi-channel pairing matrix used by the phenomenology framework."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .form_factors import gamma_d, gamma_s
from .parameters import PairingParams


def pairing_terms(kx: float, ky: float, params: PairingParams) -> dict[str, complex]:
    """Return the scalar pairing combinations entering ``Delta(k)``."""

    gamma_s_value = float(np.asarray(gamma_s(kx, ky)))
    gamma_d_value = float(np.asarray(gamma_d(kx, ky)))
    return {
        "z_diag": params.eta_z_s * gamma_s_value,
        "zx_d": params.eta_zx_d * gamma_d_value,
        "x_diag": params.eta_x_s * gamma_s_value + params.eta_x_d * gamma_d_value,
        "z_perp": params.eta_z_perp,
        "x_perp": params.eta_x_perp,
    }


def delta_matrix(kx: float, ky: float, params: PairingParams) -> NDArray[np.complex128]:
    """Construct the 4x4 pairing matrix in the basis ``(Az, Ax, Bz, Bx)``."""

    terms = pairing_terms(kx, ky, params)
    return np.array(
        [
            [terms["z_diag"], terms["zx_d"], terms["z_perp"], 0.0],
            [terms["zx_d"], terms["x_diag"], 0.0, terms["x_perp"]],
            [terms["z_perp"], 0.0, terms["z_diag"], terms["zx_d"]],
            [0.0, terms["x_perp"], terms["zx_d"], terms["x_diag"]],
        ],
        dtype=np.complex128,
    )
