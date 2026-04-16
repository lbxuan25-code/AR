"""Projection helpers for normal-state bands and band-basis pairing."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from .fermi_surface import zx_orbital_weights


class ProjectionCompatibleModel(Protocol):
    """Minimal model interface required by the band-projection helpers."""

    def build_normal_state(self, kx: float, ky: float) -> NDArray[np.complex128]:
        """Return the normal-state Hamiltonian at ``(kx, ky)``."""

    def build_delta(self, kx: float, ky: float) -> NDArray[np.complex128]:
        """Return the pairing matrix at ``(kx, ky)``."""


def normal_state_eigensystem(
    kx: float,
    ky: float,
    model: ProjectionCompatibleModel,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Return normal-state eigenvalues and eigenvectors at a single momentum."""

    eigenvalues, eigenvectors = np.linalg.eigh(model.build_normal_state(kx, ky))
    return np.asarray(eigenvalues, dtype=np.float64), np.asarray(eigenvectors, dtype=np.complex128)


def normal_state_eigensystem_path(
    k_points: NDArray[np.float64],
    model: ProjectionCompatibleModel,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Return normal-state eigenvalues and eigenvectors along a k-space path."""

    num_points = len(k_points)
    eigenvalues = np.empty((num_points, 4), dtype=np.float64)
    eigenvectors = np.empty((num_points, 4, 4), dtype=np.complex128)

    for index, (kx, ky) in enumerate(k_points):
        eigvals, eigvecs = normal_state_eigensystem(float(kx), float(ky), model)
        eigenvalues[index] = eigvals
        eigenvectors[index] = eigvecs

    return eigenvalues, eigenvectors


def orbital_sector_weights(
    eigenvectors: NDArray[np.complex128],
) -> dict[str, NDArray[np.float64]]:
    """Return z-like and x-like orbital-sector weights for each band."""

    z_weight, x_weight = zx_orbital_weights(eigenvectors)
    return {"z_like": z_weight, "x_like": x_weight}


def nearest_fermi_band_indices(
    eigenvalues: NDArray[np.float64],
    energy: float = 0.0,
) -> NDArray[np.intp]:
    """Return the band index closest to the target energy at each momentum point."""

    return np.argmin(np.abs(eigenvalues - energy), axis=-1)


def pairing_in_band_basis(
    kx: float,
    ky: float,
    model: ProjectionCompatibleModel,
) -> NDArray[np.complex128]:
    """Project ``Delta(k)`` into the normal-state band basis.

    For eigenvector matrices returned by ``np.linalg.eigh``, the columns are the
    normal-state eigenvectors in the orbital basis. The band-basis pairing matrix
    is therefore ``U(k)^dagger Delta(k) U(-k)^*``.
    """

    _, eigenvectors_k = normal_state_eigensystem(kx, ky, model)
    _, eigenvectors_minus_k = normal_state_eigensystem(-kx, -ky, model)
    delta_k = model.build_delta(kx, ky)
    return eigenvectors_k.conj().T @ delta_k @ eigenvectors_minus_k.conj()


def projected_gap_along_path(
    k_points: NDArray[np.float64],
    band_indices: NDArray[np.intp],
    model: ProjectionCompatibleModel,
) -> NDArray[np.complex128]:
    """Project the pairing matrix onto selected normal-state bands along a path."""

    projected_gaps = np.empty(len(k_points), dtype=np.complex128)
    for index, (kx, ky) in enumerate(k_points):
        pairing_band_basis = pairing_in_band_basis(float(kx), float(ky), model)
        projected_gaps[index] = pairing_band_basis[band_indices[index], band_indices[index]]
    return projected_gaps
