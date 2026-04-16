"""Momentum-space form factors used by the pairing expansion."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def gamma_s(kx: ArrayLike, ky: ArrayLike):
    """Return the extended-s form factor cos(kx) + cos(ky)."""

    return np.cos(kx) + np.cos(ky)


def gamma_d(kx: ArrayLike, ky: ArrayLike):
    """Return the d-wave form factor cos(kx) - cos(ky)."""

    return np.cos(kx) - np.cos(ky)
