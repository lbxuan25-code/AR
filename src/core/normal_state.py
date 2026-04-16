"""Formal normal-state Hamiltonian for the four-orbital bilayer model."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .parameters import NormalStateParams


def normal_state_terms(kx: float, ky: float, params: NormalStateParams) -> dict[str, float]:
    """Return the scalar building blocks used in ``h0(k)``."""

    cos_kx = float(np.cos(kx))
    cos_ky = float(np.cos(ky))
    cos_sum = cos_kx + cos_ky
    cos_prod = cos_kx * cos_ky
    cos_diff = cos_kx - cos_ky
    mu_az, mu_ax, mu_bz, mu_bx = params.mu_diag
    return {
        "xi_z": params.e1 + 2.0 * params.tx1 * cos_sum + 4.0 * params.txy1 * cos_prod,
        "xi_x": params.e2 + 2.0 * params.tx2 * cos_sum + 4.0 * params.txy2 * cos_prod,
        "v_intra": 2.0 * params.vx * cos_diff,
        "v_cross": 2.0 * params.vxz * cos_diff,
        "v1": params.v1,
        "v2": params.v2,
        "mu_az": float(mu_az),
        "mu_ax": float(mu_ax),
        "mu_bz": float(mu_bz),
        "mu_bx": float(mu_bx),
    }


def h0_matrix(kx: float, ky: float, params: NormalStateParams) -> NDArray[np.complex128]:
    """Construct the formal 4x4 normal-state matrix in the basis ``(Az, Ax, Bz, Bx)``."""

    terms = normal_state_terms(kx, ky, params)
    return np.array(
        [
            [
                terms["xi_z"] - terms["mu_az"],
                terms["v_intra"],
                terms["v1"],
                terms["v_cross"],
            ],
            [
                terms["v_intra"],
                terms["xi_x"] - terms["mu_ax"],
                terms["v_cross"],
                terms["v2"],
            ],
            [
                terms["v1"],
                terms["v_cross"],
                terms["xi_z"] - terms["mu_bz"],
                terms["v_intra"],
            ],
            [
                terms["v_cross"],
                terms["v2"],
                terms["v_intra"],
                terms["xi_x"] - terms["mu_bx"],
            ],
        ],
        dtype=np.complex128,
    )
