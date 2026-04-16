"""Minimal phase-sensitive BTK-like conductance kernel.

This module intentionally stays lightweight. It consumes the interface-resolved
``Delta_plus`` / ``Delta_minus`` diagnostics and returns a normalized,
phase-sensitive conductance curve under the following approximations:

- two-dimensional interface geometry
- specular reflection
- band-projected superconducting gaps
- no explicit coupled multiband scattering at the interface
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..interface_gap import InterfaceGapDiagnosticsResult, InterfaceResolvedContour

# BTK spectra in this repository are plotted in meV, so thermal broadening
# must use Boltzmann's constant in meV/K to stay on the same energy scale.
K_B_MEV_PER_K: float = 8.617333262145e-2


def trapezoid_integral(
    values: NDArray[np.float64],
    x: NDArray[np.float64] | None = None,
    axis: int = -1,
) -> NDArray[np.float64]:
    """Integrate with NumPy's trapezoid rule using a version-compatible fallback."""

    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return np.asarray(trapezoid(values, x=x, axis=axis))

    trapz = getattr(np, "trapz", None)
    if trapz is None:
        raise AttributeError("NumPy provides neither trapezoid nor trapz.")
    return np.asarray(trapz(values, x=x, axis=axis))


@dataclass(frozen=True, slots=True)
class BTKChannelData:
    """Flattened interface channels for the minimal BTK-like kernel."""

    delta_plus: NDArray[np.complex128]
    delta_minus: NDArray[np.complex128]
    v_n_in: NDArray[np.float64]
    v_n_out: NDArray[np.float64]
    weights: NDArray[np.float64]
    phase_difference: NDArray[np.float64]
    interface_angle: float
    num_channels: int


@dataclass(frozen=True, slots=True)
class MinimalBTKConductanceResult:
    """Normalized conductance curve from the minimal phase-sensitive kernel.

    ``temperature`` is interpreted in Kelvin and internally converted through
    ``k_B T`` into meV, matching the units used by ``bias`` and the gaps.
    """

    bias: NDArray[np.float64]
    conductance: NDArray[np.float64]
    conductance_unbroadened: NDArray[np.float64]
    interface_angle: float
    barrier_z: float
    broadening_gamma: float
    temperature: float
    num_channels: int
    mean_normal_transparency: float
    strict_reflection_match: bool
    max_reflection_mismatch: float | None
    approximation: str = (
        "2D specular-reflection, band-projected, phase-sensitive BTK-like kernel; "
        "minimal architecture test rather than a full generalized-KT solver."
    )


def _wrapped_step_lengths(k_points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Estimate local arc-length weights along one contour."""

    points = np.asarray(k_points, dtype=np.float64)
    num_points = len(points)
    if num_points == 0:
        return np.empty((0,), dtype=np.float64)
    if num_points == 1:
        return np.ones((1,), dtype=np.float64)

    deltas = points[1:] - points[:-1]
    deltas = (deltas + np.pi) % (2.0 * np.pi) - np.pi
    step_lengths = np.linalg.norm(deltas, axis=1)

    weights = np.empty((num_points,), dtype=np.float64)
    weights[0] = max(0.5 * step_lengths[0], 1.0e-12)
    weights[-1] = max(0.5 * step_lengths[-1], 1.0e-12)
    if num_points > 2:
        weights[1:-1] = 0.5 * (step_lengths[:-1] + step_lengths[1:])
    return np.asarray(weights, dtype=np.float64)


def contour_transport_weights(contour: InterfaceResolvedContour) -> NDArray[np.float64]:
    """Return raw flux-like quadrature weights for one interface contour."""

    arc_weights = _wrapped_step_lengths(contour.k_in)
    flux_weights = np.abs(np.asarray(contour.v_n_in, dtype=np.float64))
    weights = np.asarray(arc_weights * flux_weights, dtype=np.float64)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        return np.ones((len(contour.k_in),), dtype=np.float64)
    return weights


def prepare_btk_channels(diagnostics: InterfaceGapDiagnosticsResult) -> BTKChannelData:
    """Flatten interface diagnostics into weighted BTK channels."""

    delta_plus_parts: list[NDArray[np.complex128]] = []
    delta_minus_parts: list[NDArray[np.complex128]] = []
    v_n_in_parts: list[NDArray[np.float64]] = []
    v_n_out_parts: list[NDArray[np.float64]] = []
    weight_parts: list[NDArray[np.float64]] = []
    phase_parts: list[NDArray[np.float64]] = []

    for contour in diagnostics.contours:
        if len(contour.k_in) == 0:
            continue
        contour_weights = contour_transport_weights(contour)
        delta_plus_parts.append(np.asarray(contour.delta_plus, dtype=np.complex128))
        delta_minus_parts.append(np.asarray(contour.delta_minus, dtype=np.complex128))
        v_n_in_parts.append(np.asarray(contour.v_n_in, dtype=np.float64))
        v_n_out_parts.append(np.asarray(contour.v_n_out, dtype=np.float64))
        weight_parts.append(contour_weights)
        phase_parts.append(np.asarray(contour.phase_difference, dtype=np.float64))

    if not delta_plus_parts:
        raise ValueError("No matched interface states are available for the BTK-like kernel.")

    weights = np.concatenate(weight_parts).astype(np.float64, copy=False)
    weights /= np.sum(weights)

    return BTKChannelData(
        delta_plus=np.concatenate(delta_plus_parts).astype(np.complex128, copy=False),
        delta_minus=np.concatenate(delta_minus_parts).astype(np.complex128, copy=False),
        v_n_in=np.concatenate(v_n_in_parts).astype(np.float64, copy=False),
        v_n_out=np.concatenate(v_n_out_parts).astype(np.float64, copy=False),
        weights=weights,
        phase_difference=np.concatenate(phase_parts).astype(np.float64, copy=False),
        interface_angle=float(diagnostics.interface_angle),
        num_channels=int(weights.size),
    )


def normal_state_transparency(
    v_n_in: NDArray[np.float64],
    v_n_out: NDArray[np.float64],
    barrier_z: float,
) -> NDArray[np.float64]:
    """Return the old-repository BTK transparency ``sigma_N = 1 / (1 + Z^2)``.

    The current baseline comparison uses the same scalar barrier definition as
    the previous repository, so the transparency is independent of channel
    velocity mismatch.
    """

    velocity_in = np.asarray(v_n_in, dtype=np.float64)
    del v_n_out
    sigma_n = 1.0 / (1.0 + float(barrier_z) ** 2)
    return np.asarray(
        np.full_like(velocity_in, fill_value=np.clip(sigma_n, 1.0e-12, 1.0), dtype=np.float64)
    )


def phase_sensitive_btk_kernel(
    bias: NDArray[np.float64],
    delta_plus: NDArray[np.complex128],
    delta_minus: NDArray[np.complex128],
    sigma_n: NDArray[np.float64],
    broadening_gamma: float,
) -> NDArray[np.float64]:
    """Return the channel-resolved phase-sensitive BTK-like kernel."""

    energies = np.abs(np.asarray(bias, dtype=np.float64))[:, None]
    delta_p = np.asarray(delta_plus, dtype=np.complex128)[None, :]
    delta_m = np.asarray(delta_minus, dtype=np.complex128)[None, :]
    sigma = np.asarray(sigma_n, dtype=np.float64)[None, :]
    energy_complex = energies - 1j * float(broadening_gamma)

    omega_plus = np.sqrt(energy_complex**2 - np.abs(delta_p) ** 2)
    omega_minus = np.sqrt(energy_complex**2 - np.abs(delta_m) ** 2)

    gamma_plus = np.zeros_like(omega_plus, dtype=np.complex128)
    gamma_minus = np.zeros_like(omega_minus, dtype=np.complex128)

    mask_plus = np.abs(delta_plus) > 1.0e-12
    mask_minus = np.abs(delta_minus) > 1.0e-12
    if np.any(mask_plus):
        gamma_plus[:, mask_plus] = (energy_complex - omega_plus[:, mask_plus]) / np.abs(delta_plus[mask_plus])
    if np.any(mask_minus):
        gamma_minus[:, mask_minus] = (energy_complex - omega_minus[:, mask_minus]) / np.abs(delta_minus[mask_minus])

    phase_plus = np.ones_like(delta_p, dtype=np.complex128)
    phase_minus = np.ones_like(delta_m, dtype=np.complex128)
    if np.any(mask_plus):
        phase_plus[0, mask_plus] = delta_plus[mask_plus] / np.abs(delta_plus[mask_plus])
    if np.any(mask_minus):
        phase_minus[0, mask_minus] = delta_minus[mask_minus] / np.abs(delta_minus[mask_minus])

    phase_ratio = phase_minus / phase_plus
    numerator = 1.0 + sigma * np.abs(gamma_plus) ** 2 + (sigma - 1.0) * np.abs(gamma_plus * gamma_minus) ** 2
    denominator = np.abs(1.0 + (sigma - 1.0) * gamma_plus * gamma_minus * phase_ratio) ** 2
    kernel = np.asarray(np.real(numerator / denominator), dtype=np.float64)
    return np.asarray(np.clip(kernel, 0.0, None), dtype=np.float64)


def apply_thermal_broadening(
    bias: NDArray[np.float64],
    conductance: NDArray[np.float64],
    temperature: float,
) -> NDArray[np.float64]:
    """Apply a simple Fermi-derivative thermal broadening using Kelvin input.

    The BTK pipeline uses meV for bias, gaps, and Dynes broadening, so the
    thermal energy scale is computed as ``k_B T`` in meV.
    """

    energies = np.asarray(bias, dtype=np.float64)
    curve = np.asarray(conductance, dtype=np.float64)
    thermal_energy = K_B_MEV_PER_K * float(temperature)
    if thermal_energy <= 0.0:
        return curve

    if energies.size < 2:
        return curve

    spacing = float(np.median(np.diff(energies)))
    if not np.isfinite(spacing) or spacing <= 0.0:
        return curve

    # Finite-window convolution produces artificial edge suppression if the
    # kernel is truncated at the plotted bias range. Extend the spectrum with
    # constant edge values before integrating so the BTK curve does not show a
    # spurious cliff near +/- bias_max.
    pad_extent = max(8.0 * thermal_energy, 10.0 * spacing)
    pad_count = max(int(np.ceil(pad_extent / spacing)), 1)
    left_pad = energies[0] - spacing * np.arange(pad_count, 0, -1, dtype=np.float64)
    right_pad = energies[-1] + spacing * np.arange(1, pad_count + 1, dtype=np.float64)
    padded_energies = np.concatenate([left_pad, energies, right_pad]).astype(np.float64, copy=False)
    padded_curve = np.concatenate(
        [
            np.full((pad_count,), curve[0], dtype=np.float64),
            curve,
            np.full((pad_count,), curve[-1], dtype=np.float64),
        ]
    ).astype(np.float64, copy=False)

    delta_e = energies[:, None] - padded_energies[None, :]
    argument = np.clip(delta_e / (2.0 * thermal_energy), -50.0, 50.0)
    kernel = 0.25 / thermal_energy / np.cosh(argument) ** 2
    broadened = trapezoid_integral(kernel * padded_curve[None, :], x=padded_energies, axis=1)
    return np.asarray(np.clip(broadened, 0.0, None), dtype=np.float64)


def extended_broadening_bias_grid(
    bias: NDArray[np.float64],
    temperature: float,
    broadening_gamma: float,
) -> NDArray[np.float64]:
    """Return an internal bias grid that is wider than the displayed window.

    The goal is to decouple thermal broadening from the user-facing ``bias``
    extent. The returned grid preserves the requested bias points exactly, but
    pads both sides so that thermal convolution samples a broader unbroadened
    spectrum before interpolation back to the requested display window.
    """

    energies = np.asarray(bias, dtype=np.float64)
    if energies.ndim != 1 or energies.size < 2:
        return energies

    spacing = float(np.median(np.diff(energies)))
    if not np.isfinite(spacing) or spacing <= 0.0:
        return energies

    half_span = 0.5 * float(energies[-1] - energies[0])
    thermal_energy = K_B_MEV_PER_K * float(temperature)
    extra_extent = max(
        3.0 * half_span,
        12.0 * thermal_energy,
        50.0 * float(abs(broadening_gamma)),
        32.0 * spacing,
    )
    pad_count = max(int(np.ceil(extra_extent / spacing)), 1)
    left_pad = energies[0] - spacing * np.arange(pad_count, 0, -1, dtype=np.float64)
    right_pad = energies[-1] + spacing * np.arange(1, pad_count + 1, dtype=np.float64)
    return np.asarray(np.concatenate([left_pad, energies, right_pad]), dtype=np.float64)


def compute_minimal_btk_conductance(
    diagnostics: InterfaceGapDiagnosticsResult,
    bias: NDArray[np.float64],
    barrier_z: float,
    broadening_gamma: float,
    temperature: float = 0.0,
) -> MinimalBTKConductanceResult:
    """Compute a normalized conductance curve from interface diagnostics."""

    energies = np.asarray(bias, dtype=np.float64)
    channels = prepare_btk_channels(diagnostics)
    sigma_n = normal_state_transparency(channels.v_n_in, channels.v_n_out, barrier_z)
    kernel = phase_sensitive_btk_kernel(
        energies,
        channels.delta_plus,
        channels.delta_minus,
        sigma_n,
        broadening_gamma,
    )
    conductance_unbroadened = np.asarray(kernel @ channels.weights, dtype=np.float64)
    extended_energies = extended_broadening_bias_grid(energies, temperature, broadening_gamma)
    if extended_energies.shape == energies.shape and np.allclose(extended_energies, energies):
        conductance = apply_thermal_broadening(energies, conductance_unbroadened, temperature)
    else:
        extended_kernel = phase_sensitive_btk_kernel(
            extended_energies,
            channels.delta_plus,
            channels.delta_minus,
            sigma_n,
            broadening_gamma,
        )
        conductance_unbroadened_extended = np.asarray(extended_kernel @ channels.weights, dtype=np.float64)
        conductance_extended = apply_thermal_broadening(
            extended_energies,
            conductance_unbroadened_extended,
            temperature,
        )
        conductance = np.asarray(
            np.interp(energies, extended_energies, conductance_extended),
            dtype=np.float64,
        )

    edge_count = max(len(energies) // 12, 1)
    reference = float(
        np.mean(
            np.concatenate(
                [conductance[:edge_count], conductance[-edge_count:]],
                axis=0,
            )
        )
    )
    if np.isfinite(reference) and reference > 1.0e-12:
        conductance = np.asarray(conductance / reference, dtype=np.float64)
        conductance_unbroadened = np.asarray(conductance_unbroadened / reference, dtype=np.float64)

    return MinimalBTKConductanceResult(
        bias=energies,
        conductance=conductance,
        conductance_unbroadened=conductance_unbroadened,
        interface_angle=float(diagnostics.interface_angle),
        barrier_z=float(barrier_z),
        broadening_gamma=float(broadening_gamma),
        temperature=float(temperature),
        num_channels=int(channels.num_channels),
        mean_normal_transparency=float(np.mean(sigma_n)),
        strict_reflection_match=diagnostics.strict_reflection_match,
        max_reflection_mismatch=diagnostics.max_reflection_mismatch,
    )
