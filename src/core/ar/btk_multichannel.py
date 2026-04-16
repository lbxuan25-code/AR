"""Enhanced multichannel phase-sensitive BTK / generalized-KT style solver.

This solver keeps the current repository lightweight while moving beyond the
baseline ``btk_minimal`` kernel. It still uses the band-projected interface
diagnostics layer, but organizes transport more explicitly as:

- matched interface points -> scattering channels
- channels grouped by Fermi-surface contour
- contour-level flux integration
- global normal-state transparency normalization

The current approximation remains:

- two-dimensional interface geometry
- specular reflection
- band-projected superconducting gaps
- phase-sensitive BTK / generalized-KT style scattering kernel
- no explicit full coupled multiband interface-scattering matrix
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .btk_minimal import (
    apply_thermal_broadening,
    contour_transport_weights,
    extended_broadening_bias_grid,
    normal_state_transparency,
    phase_sensitive_btk_kernel,
)
from ..interface_gap import InterfaceGapDiagnosticsResult, InterfaceResolvedContour


@dataclass(frozen=True, slots=True)
class ContourTransportSummary:
    """Compact transport summary for one reflected Fermi-surface contour."""

    source_band_index: int
    num_input_channels: int
    num_used_channels: int
    num_same_band_channels: int
    normal_weight: float
    mean_transparency: float
    same_band_fraction: float


@dataclass(frozen=True, slots=True)
class MultichannelBTKConductanceResult:
    """Conductance result from the enhanced multichannel BTK-style solver.

    ``temperature`` is interpreted in Kelvin and internally converted through
    ``k_B T`` in meV by the shared thermal-broadening helper.
    """

    bias: NDArray[np.float64]
    conductance: NDArray[np.float64]
    conductance_unbroadened: NDArray[np.float64]
    interface_angle: float
    barrier_z: float
    broadening_gamma: float
    temperature: float
    num_input_channels: int
    num_channels: int
    num_filtered_channels: int
    num_same_band_channels: int
    num_contours: int
    mean_normal_transparency: float
    mean_mismatch_penalty: float
    strict_reflection_match: bool
    max_reflection_mismatch: float | None
    mismatch_penalty_scale: float | None
    contour_summaries: tuple[ContourTransportSummary, ...]
    approximation: str = (
        "2D specular-reflection, band-projected, multichannel phase-sensitive "
        "BTK/generalized-KT style solver; still not a full exact coupled "
        "multiband interface-scattering theory."
    )


def mismatch_quality_weights(
    reflection_mismatch: NDArray[np.float64],
    mismatch_penalty_scale: float | None,
) -> NDArray[np.float64]:
    """Return smooth quality weights for reflected-state mismatch diagnostics."""

    mismatch = np.asarray(reflection_mismatch, dtype=np.float64)
    if mismatch_penalty_scale is None:
        return np.ones_like(mismatch, dtype=np.float64)
    scale = float(mismatch_penalty_scale)
    if scale <= 0.0:
        raise ValueError(f"mismatch_penalty_scale must be positive or None, got {mismatch_penalty_scale}.")
    return np.asarray(np.exp(-np.square(mismatch / scale)), dtype=np.float64)


def _channel_keep_mask(
    raw_channel_weight: NDArray[np.float64],
    min_channel_weight: float,
) -> NDArray[np.bool_]:
    """Return a stable keep mask for numerically relevant channels."""

    weights = np.asarray(raw_channel_weight, dtype=np.float64)
    if weights.size == 0:
        return np.empty((0,), dtype=np.bool_)
    threshold = float(min_channel_weight) * float(np.max(weights))
    return np.asarray(weights > max(threshold, 1.0e-14), dtype=np.bool_)


def _contour_conductance(
    contour: InterfaceResolvedContour,
    bias: NDArray[np.float64],
    barrier_z: float,
    broadening_gamma: float,
    mismatch_penalty_scale: float | None,
    min_channel_weight: float,
) -> tuple[NDArray[np.float64], float, ContourTransportSummary, int, float]:
    """Compute one contour contribution and its transport summary."""

    arc_flux_weight = contour_transport_weights(contour)
    sigma_n = normal_state_transparency(contour.v_n_in, contour.v_n_out, barrier_z)
    quality = mismatch_quality_weights(contour.reflection_mismatch, mismatch_penalty_scale)
    raw_channel_weight = np.asarray(arc_flux_weight * sigma_n * quality, dtype=np.float64)
    keep_mask = _channel_keep_mask(raw_channel_weight, min_channel_weight)
    num_filtered = int(np.count_nonzero(~keep_mask))

    if not np.any(keep_mask):
        summary = ContourTransportSummary(
            source_band_index=int(contour.source_band_index),
            num_input_channels=len(contour.k_in),
            num_used_channels=0,
            num_same_band_channels=0,
            normal_weight=0.0,
            mean_transparency=0.0,
            same_band_fraction=0.0,
        )
        return np.zeros_like(np.asarray(bias, dtype=np.float64)), 0.0, summary, num_filtered, 0.0

    kernel = phase_sensitive_btk_kernel(
        bias=np.asarray(bias, dtype=np.float64),
        delta_plus=np.asarray(contour.delta_plus[keep_mask], dtype=np.complex128),
        delta_minus=np.asarray(contour.delta_minus[keep_mask], dtype=np.complex128),
        sigma_n=np.asarray(sigma_n[keep_mask], dtype=np.float64),
        broadening_gamma=broadening_gamma,
    )
    used_weights = np.asarray(raw_channel_weight[keep_mask], dtype=np.float64)
    contour_superconducting = np.asarray(kernel @ used_weights, dtype=np.float64)
    contour_normal = float(np.sum(used_weights))

    summary = ContourTransportSummary(
        source_band_index=int(contour.source_band_index),
        num_input_channels=len(contour.k_in),
        num_used_channels=int(np.count_nonzero(keep_mask)),
        num_same_band_channels=int(np.count_nonzero(contour.matched_same_band[keep_mask])),
        normal_weight=contour_normal,
        mean_transparency=float(np.mean(sigma_n[keep_mask])),
        same_band_fraction=float(np.mean(np.asarray(contour.matched_same_band[keep_mask], dtype=np.float64))),
    )
    mean_penalty = float(np.mean(quality[keep_mask]))
    return contour_superconducting, contour_normal, summary, num_filtered, mean_penalty


def compute_multichannel_btk_conductance(
    diagnostics: InterfaceGapDiagnosticsResult,
    bias: NDArray[np.float64],
    barrier_z: float,
    broadening_gamma: float,
    temperature: float = 0.0,
    mismatch_penalty_scale: float | None = 0.12,
    min_channel_weight: float = 1.0e-4,
) -> MultichannelBTKConductanceResult:
    """Compute an enhanced multichannel BTK/generalized-KT style conductance.

    The solver keeps the reflected-branch phase information from
    ``Delta_minus(k_out_target)`` and organizes transport hierarchically:

    1. each matched interface point is treated as one scattering channel
    2. channels on the same contour are integrated with arc-length and
       interface-normal flux weights
    3. normal-state transparency and optional mismatch quality factors weight
       each channel before contour aggregation
    4. all contour contributions are normalized by the total normal-state
       transport weight

    ``temperature`` is interpreted in Kelvin.
    """

    energies = np.asarray(bias, dtype=np.float64)
    if energies.ndim != 1:
        raise ValueError("bias must be a one-dimensional array.")

    total_superconducting = np.zeros_like(energies, dtype=np.float64)
    total_normal = 0.0
    contour_summaries: list[ContourTransportSummary] = []
    num_input_channels = 0
    num_used_channels = 0
    num_filtered_channels = 0
    num_same_band_channels = 0
    penalty_values: list[float] = []
    kept_contours: list[InterfaceResolvedContour] = []

    for contour in diagnostics.contours:
        num_input_channels += len(contour.k_in)
        contour_superconducting, contour_normal, summary, filtered_here, mean_penalty = _contour_conductance(
            contour=contour,
            bias=energies,
            barrier_z=barrier_z,
            broadening_gamma=broadening_gamma,
            mismatch_penalty_scale=mismatch_penalty_scale,
            min_channel_weight=min_channel_weight,
        )
        num_filtered_channels += filtered_here
        if summary.num_used_channels == 0:
            continue

        total_superconducting += contour_superconducting
        total_normal += contour_normal
        contour_summaries.append(summary)
        kept_contours.append(contour)
        num_used_channels += summary.num_used_channels
        num_same_band_channels += summary.num_same_band_channels
        penalty_values.append(mean_penalty)

    if total_normal <= 0.0 or num_used_channels == 0:
        raise ValueError(
            "No transport channels survived the multichannel BTK solver. "
            "Relax strict reflection matching or lower the mismatch suppression."
        )

    conductance_unbroadened = np.asarray(total_superconducting / total_normal, dtype=np.float64)
    extended_energies = extended_broadening_bias_grid(energies, temperature, broadening_gamma)
    if extended_energies.shape == energies.shape and np.allclose(extended_energies, energies):
        conductance = apply_thermal_broadening(energies, conductance_unbroadened, temperature)
    else:
        total_superconducting_extended = np.zeros_like(extended_energies, dtype=np.float64)
        for contour in kept_contours:
            contour_superconducting_extended, _, summary_extended, _, _ = _contour_conductance(
                contour=contour,
                bias=extended_energies,
                barrier_z=barrier_z,
                broadening_gamma=broadening_gamma,
                mismatch_penalty_scale=mismatch_penalty_scale,
                min_channel_weight=min_channel_weight,
            )
            if summary_extended.num_used_channels == 0:
                continue
            total_superconducting_extended += contour_superconducting_extended
        conductance_unbroadened_extended = np.asarray(total_superconducting_extended / total_normal, dtype=np.float64)
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

    return MultichannelBTKConductanceResult(
        bias=energies,
        conductance=np.asarray(np.clip(conductance, 0.0, None), dtype=np.float64),
        conductance_unbroadened=np.asarray(np.clip(conductance_unbroadened, 0.0, None), dtype=np.float64),
        interface_angle=float(diagnostics.interface_angle),
        barrier_z=float(barrier_z),
        broadening_gamma=float(broadening_gamma),
        temperature=float(temperature),
        num_input_channels=int(num_input_channels),
        num_channels=int(num_used_channels),
        num_filtered_channels=int(num_filtered_channels),
        num_same_band_channels=int(num_same_band_channels),
        num_contours=len(contour_summaries),
        mean_normal_transparency=float(
            np.average(
                np.asarray([item.mean_transparency for item in contour_summaries], dtype=np.float64),
                weights=np.asarray([item.num_used_channels for item in contour_summaries], dtype=np.float64),
            )
        )
        if contour_summaries
        else 0.0,
        mean_mismatch_penalty=float(np.mean(np.asarray(penalty_values, dtype=np.float64))) if penalty_values else 0.0,
        strict_reflection_match=bool(diagnostics.strict_reflection_match),
        max_reflection_mismatch=diagnostics.max_reflection_mismatch,
        mismatch_penalty_scale=mismatch_penalty_scale,
        contour_summaries=tuple(contour_summaries),
    )
