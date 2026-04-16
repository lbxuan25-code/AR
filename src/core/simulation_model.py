"""Lightweight model wrapper for the unified phenomenology framework."""

from __future__ import annotations

from dataclasses import dataclass

from numpy.typing import NDArray

from .bdg import bdg_matrix
from .conventions import BASIS_ORDER, PAIRING_CHANNELS, PHYSICAL_PAIRING_CHANNELS
from .normal_state import h0_matrix
from .pairing import delta_matrix
from .parameters import ModelParams
from .presets import base_model_params


@dataclass(frozen=True, slots=True)
class SimulationModel:
    """Bundle the current normal-state and pairing definitions."""

    params: ModelParams
    name: str = "base_model"
    basis: tuple[str, str, str, str] = BASIS_ORDER
    pairing_channels: tuple[str, ...] = PHYSICAL_PAIRING_CHANNELS
    legacy_pairing_channels: tuple[str, ...] = PAIRING_CHANNELS

    def build_normal_state(self, kx: float, ky: float) -> NDArray[np.complex128]:
        """Construct the normal-state Hamiltonian ``h0(k)``."""

        return h0_matrix(kx, ky, self.params.normal_state)

    def build_delta(self, kx: float, ky: float) -> NDArray[np.complex128]:
        """Construct the pairing matrix ``Delta(k)``."""

        return delta_matrix(kx, ky, self.params.pairing)

    def build_bdg(self, kx: float, ky: float) -> NDArray[np.complex128]:
        """Construct the BdG Hamiltonian ``H_BdG(k)``."""

        return bdg_matrix(kx, ky, self.params)


def base_simulation_model() -> SimulationModel:
    """Return the repository-local baseline simulation model."""

    return SimulationModel(params=base_model_params(), name="base_model")
