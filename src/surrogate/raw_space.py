"""Gauge-fixed raw parameter space for pairing parameters."""

from __future__ import annotations

import numpy as np

from core.parameters import PairingParams

PAIRING_CHANNELS = (
    "eta_z_s",
    "eta_z_perp",
    "eta_x_s",
    "eta_x_d",
    "eta_zx_d",
    "eta_x_perp",
)
DEFAULT_GAUGE_INDEX = 1
GAUGE_EPS = 1.0e-8


def _pairing_array(params: PairingParams) -> np.ndarray:
    return np.asarray([getattr(params, name) for name in PAIRING_CHANNELS], dtype=np.complex128)


def _choose_gauge_index(values: np.ndarray) -> int:
    if abs(values[DEFAULT_GAUGE_INDEX]) > GAUGE_EPS:
        return DEFAULT_GAUGE_INDEX
    magnitudes = np.abs(values)
    if np.max(magnitudes) <= GAUGE_EPS:
        return DEFAULT_GAUGE_INDEX
    return int(np.argmax(magnitudes))


def pairing_params_to_gauge_fixed_vector(params: PairingParams) -> np.ndarray:
    values = _pairing_array(params)
    gauge_index = _choose_gauge_index(values)
    reference = values[gauge_index]
    phase = 1.0 + 0.0j if abs(reference) <= GAUGE_EPS else np.exp(-1j * np.angle(reference))
    fixed = np.asarray(values * phase, dtype=np.complex128)
    if np.real(fixed[gauge_index]) < 0.0:
        fixed *= -1.0
    fixed[gauge_index] = complex(float(np.real(fixed[gauge_index])), 0.0)

    pieces = [float(gauge_index), float(np.real(fixed[gauge_index]))]
    for index, value in enumerate(fixed):
        if index == gauge_index:
            continue
        pieces.extend([float(np.real(value)), float(np.imag(value))])
    return np.asarray(pieces, dtype=np.float64)


def gauge_fixed_vector_to_pairing_params(vector: np.ndarray) -> PairingParams:
    values = np.asarray(vector, dtype=np.float64)
    if values.shape != (12,):
        raise ValueError(f"Gauge-fixed vector must have shape (12,), got {values.shape}.")

    gauge_index = int(round(float(values[0])))
    if gauge_index < 0 or gauge_index >= len(PAIRING_CHANNELS):
        raise ValueError(f"Gauge index out of range: {gauge_index}.")

    complex_values = np.zeros((len(PAIRING_CHANNELS),), dtype=np.complex128)
    complex_values[gauge_index] = complex(float(values[1]), 0.0)
    cursor = 2
    for index in range(len(PAIRING_CHANNELS)):
        if index == gauge_index:
            continue
        complex_values[index] = complex(float(values[cursor]), float(values[cursor + 1]))
        cursor += 2

    return PairingParams(
        eta_z_s=complex_values[0],
        eta_z_perp=complex_values[1],
        eta_x_s=complex_values[2],
        eta_x_d=complex_values[3],
        eta_zx_d=complex_values[4],
        eta_x_perp=complex_values[5],
    )

