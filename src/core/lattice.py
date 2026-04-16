"""Simple lattice helpers for square-lattice momentum grids and paths."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def square_k_grid(nk: int, endpoint: bool = False) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a square Brillouin-zone mesh spanning [-pi, pi) in each direction."""

    k_values = np.linspace(-np.pi, np.pi, nk, endpoint=endpoint, dtype=float)
    return np.meshgrid(k_values, k_values, indexing="ij")


def default_k_path(
    points_per_segment: int = 64,
) -> tuple[NDArray[np.float64], list[int], list[str]]:
    """Return a default Gamma-X-M-Gamma path for band-structure plots."""

    labels = [r"$\Gamma$", "X", "M", r"$\Gamma$"]
    vertices = np.array(
        [
            [0.0, 0.0],
            [np.pi, 0.0],
            [np.pi, np.pi],
            [0.0, 0.0],
        ],
        dtype=float,
    )

    segments: list[NDArray[np.float64]] = []
    tick_positions = [0]
    total = 0
    for start, stop in zip(vertices[:-1], vertices[1:]):
        segment = np.linspace(start, stop, points_per_segment, endpoint=False, dtype=float)
        segments.append(segment)
        total += len(segment)
        tick_positions.append(total)

    path = np.vstack([*segments, vertices[-1:]])
    tick_positions[-1] = len(path) - 1
    return path, tick_positions, labels
