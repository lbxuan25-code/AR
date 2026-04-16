"""BdG Hamiltonian assembly for the LNO327 scaffold."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .normal_state import h0_matrix
from .pairing import delta_matrix
from .parameters import ModelParams


def bdg_matrix(kx: float, ky: float, params: ModelParams) -> NDArray[np.complex128]:
    """Construct the 8x8 BdG Hamiltonian H_BdG(k)."""

    h0_k = h0_matrix(kx, ky, params.normal_state)
    delta_k = delta_matrix(kx, ky, params.pairing)
    h0_minus_k = h0_matrix(-kx, -ky, params.normal_state)

    matrix = np.block(
        [
            [h0_k, delta_k],
            [delta_k.conj().T, -h0_minus_k.T],
        ]
    ).astype(np.complex128)
    return matrix
