"""Minimal band-structure utilities for the scaffold project."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray

from .bdg import bdg_matrix
from .config import DEFAULT_BAND_PATH_POINTS, DEFAULT_K_GRID_SIZE
from .lattice import default_k_path, square_k_grid
from .normal_state import h0_matrix
from .parameters import ModelParams

Sector = Literal["normal", "bdg"]


def eigenvalues_at_k(
    kx: float,
    ky: float,
    params: ModelParams,
    sector: Sector = "normal",
) -> NDArray[np.float64]:
    """Return sorted eigenvalues at a single momentum point."""

    if sector == "normal":
        matrix = h0_matrix(kx, ky, params.normal_state)
    else:
        matrix = bdg_matrix(kx, ky, params)

    eigenvalues = np.linalg.eigvalsh(matrix)
    return np.asarray(eigenvalues, dtype=np.float64)


def eigenvalues_on_kgrid(
    params: ModelParams,
    nk: int = DEFAULT_K_GRID_SIZE,
    sector: Sector = "normal",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return eigenvalues on a square momentum grid."""

    kx_grid, ky_grid = square_k_grid(nk=nk)
    bands = 4 if sector == "normal" else 8
    eigenvalues = np.empty((nk, nk, bands), dtype=float)

    for i in range(nk):
        for j in range(nk):
            eigenvalues[i, j, :] = eigenvalues_at_k(kx_grid[i, j], ky_grid[i, j], params, sector=sector)

    return kx_grid, ky_grid, eigenvalues


def normal_state_eigensystem_on_kgrid(
    params: ModelParams,
    nk: int = DEFAULT_K_GRID_SIZE,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.complex128]]:
    """Return normal-state eigenvalues and eigenvectors on a square momentum grid."""

    kx_grid, ky_grid = square_k_grid(nk=nk)
    eigenvalues = np.empty((nk, nk, 4), dtype=float)
    eigenvectors = np.empty((nk, nk, 4, 4), dtype=np.complex128)

    for i in range(nk):
        for j in range(nk):
            matrix = h0_matrix(kx_grid[i, j], ky_grid[i, j], params.normal_state)
            eigvals, eigvecs = np.linalg.eigh(matrix)
            eigenvalues[i, j, :] = np.asarray(eigvals, dtype=np.float64)
            eigenvectors[i, j, :, :] = np.asarray(eigvecs, dtype=np.complex128)

    return kx_grid, ky_grid, eigenvalues, eigenvectors


def band_structure_path(
    params: ModelParams,
    points_per_segment: int = DEFAULT_BAND_PATH_POINTS,
    sector: Sector = "normal",
) -> tuple[NDArray[np.float64], NDArray[np.float64], list[int], list[str]]:
    """Return eigenvalues along the default high-symmetry path."""

    k_path, tick_positions, tick_labels = default_k_path(points_per_segment=points_per_segment)
    distances = np.zeros(len(k_path), dtype=float)
    if len(k_path) > 1:
        step_lengths = np.linalg.norm(np.diff(k_path, axis=0), axis=1)
        distances[1:] = np.cumsum(step_lengths)

    band_values = np.array(
        [eigenvalues_at_k(kx, ky, params=params, sector=sector) for kx, ky in k_path],
        dtype=float,
    )
    return distances, band_values, tick_positions, tick_labels
