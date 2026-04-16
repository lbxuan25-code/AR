"""Minimal Fermi-surface helpers for later phenomenology work."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray

from .config import DEFAULT_FERMI_SURFACE_ATOL


def locate_fermi_surface(
    eigenvalues: NDArray[np.float64],
    energy: float = 0.0,
    atol: float = DEFAULT_FERMI_SURFACE_ATOL,
) -> NDArray[np.bool_]:
    """Return a mask of k-points with at least one band near the target energy."""

    return np.any(np.isclose(eigenvalues, energy, atol=atol), axis=-1)


def find_fermi_crossing_bands(
    eigenvalues: NDArray[np.float64],
    energy: float = 0.0,
    atol: float = DEFAULT_FERMI_SURFACE_ATOL,
) -> list[int]:
    """Return band indices whose energy range intersects the target energy."""

    crossing_bands: list[int] = []
    for band_index in range(eigenvalues.shape[-1]):
        band_values = eigenvalues[..., band_index]
        if np.min(band_values) <= energy + atol and np.max(band_values) >= energy - atol:
            crossing_bands.append(band_index)

    return crossing_bands


def zx_orbital_weights(
    eigenvectors: NDArray[np.complex128],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return z-like and x-like orbital weights from normal-state eigenvectors."""

    z_weight = np.sum(np.abs(eigenvectors[..., (0, 2), :]) ** 2, axis=-2)
    x_weight = np.sum(np.abs(eigenvectors[..., (1, 3), :]) ** 2, axis=-2)
    return np.asarray(z_weight, dtype=np.float64), np.asarray(x_weight, dtype=np.float64)


def extract_fermi_contours(
    kx_grid: NDArray[np.float64],
    ky_grid: NDArray[np.float64],
    eigenvalues: NDArray[np.float64],
    energy: float = 0.0,
    band_indices: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Extract E=energy contour paths for the selected bands."""

    selected_bands = range(eigenvalues.shape[-1]) if band_indices is None else band_indices
    contours: list[dict[str, Any]] = []

    figure = plt.figure(figsize=(1.0, 1.0))
    axis = figure.add_subplot(111)
    axis.set_axis_off()
    try:
        for band_index in selected_bands:
            contour_set = axis.contour(
                kx_grid,
                ky_grid,
                eigenvalues[..., band_index],
                levels=[energy],
            )
            for vertices_raw in contour_set.allsegs[0]:
                vertices = np.asarray(vertices_raw, dtype=float)
                if len(vertices) < 2:
                    continue
                contours.append({"band": int(band_index), "k_points": vertices})
            axis.cla()
    finally:
        plt.close(figure)

    return contours


def extract_band_crossings(
    k_axis: ArrayLike,
    band_energies: NDArray[np.float64],
    energy: float = 0.0,
) -> list[dict[str, Any]]:
    """Estimate one-dimensional band crossings relative to a target energy."""

    k_values = np.asarray(k_axis, dtype=float)
    shifted = np.asarray(band_energies, dtype=float) - energy
    crossings: list[dict[str, Any]] = []

    for band_index in range(shifted.shape[1]):
        band = shifted[:, band_index]
        for i in range(len(k_values) - 1):
            left = band[i]
            right = band[i + 1]
            if left == 0.0:
                crossings.append({"band": band_index, "k": float(k_values[i])})
                continue
            if left * right < 0.0:
                weight = abs(left) / (abs(left) + abs(right))
                k_cross = (1.0 - weight) * k_values[i] + weight * k_values[i + 1]
                crossings.append({"band": band_index, "k": float(k_cross)})

    return crossings
