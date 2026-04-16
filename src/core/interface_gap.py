"""Interface-resolved gap diagnostics for the minimal BTK scaffold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
import warnings

import numpy as np
from numpy.typing import NDArray

from .interface_geometry import (
    InterfaceReflectionContour,
    build_interface_segment_catalog,
    interface_normal,
    interface_tangent,
    match_reflected_states_on_contour,
    reflected_gap_from_model,
)
from .projection import projected_gap_along_path
from .pipeline import SpectroscopyPipeline

ReflectedBranchMode = Literal["matched_reflected_band", "strict_incident_band"]


@dataclass(frozen=True, slots=True)
class PhaseDiagnostics:
    """Phase-sensitive diagnostics derived from ``Delta_plus`` and ``Delta_minus``."""

    abs_delta_plus: NDArray[np.float64]
    abs_delta_minus: NDArray[np.float64]
    phase_plus: NDArray[np.float64]
    phase_minus: NDArray[np.float64]
    phase_difference: NDArray[np.float64]
    sign_reversal_indicator: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class InterfaceResolvedContour:
    """Interface-resolved band-projected gap data for one source contour."""

    interface_angle: float
    source_band_index: int
    k_in: NDArray[np.float64]
    k_out_target: NDArray[np.float64]
    k_out_interp: NDArray[np.float64]
    band_in: NDArray[np.intp]
    band_out: NDArray[np.intp]
    band_out_projection: NDArray[np.intp]
    delta_plus: NDArray[np.complex128]
    delta_minus: NDArray[np.complex128]
    abs_delta_plus: NDArray[np.float64]
    abs_delta_minus: NDArray[np.float64]
    phase_plus: NDArray[np.float64]
    phase_minus: NDArray[np.float64]
    phase_difference: NDArray[np.float64]
    sign_reversal_indicator: NDArray[np.float64]
    z_like_weight: NDArray[np.float64]
    x_like_weight: NDArray[np.float64]
    z_like_weight_out_interp: NDArray[np.float64]
    x_like_weight_out_interp: NDArray[np.float64]
    v_n_in: NDArray[np.float64]
    v_n_out: NDArray[np.float64]
    k_parallel: NDArray[np.float64]
    incident_angle: NDArray[np.float64]
    matched_contour_id: NDArray[np.intp]
    matched_segment_index: NDArray[np.intp]
    reflection_mismatch: NDArray[np.float64]
    matched_same_band: NDArray[np.bool_]
    used_cross_band_fallback: NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class InterfaceGapDiagnosticsResult:
    """Structured interface-resolved gap package for transport-kernel tests."""

    interface_angle: float
    normal: NDArray[np.float64]
    tangent: NDArray[np.float64]
    contours: list[InterfaceResolvedContour]
    reflected_branch_mode: ReflectedBranchMode
    allow_cross_band_fallback: bool
    strict_reflection_match: bool
    max_reflection_mismatch: float | None
    approximation: str = (
        "2D specular-reflection, same-band contour-local interpolation, "
        "band-projected approximation; "
        "minimal interface diagnostic layer for architecture tests."
    )


def compute_delta_plus_minus(
    matched: InterfaceReflectionContour,
    pipeline: SpectroscopyPipeline,
    reflected_branch_mode: ReflectedBranchMode = "strict_incident_band",
) -> tuple[NDArray[np.complex128], NDArray[np.complex128], NDArray[np.intp]]:
    """Compute ``Delta_plus`` and ``Delta_minus`` for matched interface states.

    The default physical path is ``strict_incident_band``:

    - use the interpolated reflected point ``k_out_interp``
    - keep the incident band as the reflected projection band

    ``matched_reflected_band`` is retained only for diagnostics / experimental
    comparisons and should not be treated as the main reflected-branch path.
    """

    delta_plus = projected_gap_along_path(matched.k_in, matched.band_in, pipeline.model)
    if reflected_branch_mode == "strict_incident_band":
        projection_band = np.asarray(matched.band_in, dtype=np.intp)
    else:
        projection_band = np.asarray(matched.band_ref, dtype=np.intp)
    delta_minus = reflected_gap_from_model(matched.k_out_interp, projection_band, pipeline.model)
    return (
        np.asarray(delta_plus, dtype=np.complex128),
        np.asarray(delta_minus, dtype=np.complex128),
        np.asarray(projection_band, dtype=np.intp),
    )


def _subset_matched_contour(
    matched: InterfaceReflectionContour,
    keep_mask: NDArray[np.bool_],
) -> InterfaceReflectionContour:
    """Return a filtered matched contour while preserving field semantics."""

    keep = np.asarray(keep_mask, dtype=np.bool_)
    return InterfaceReflectionContour(
        interface_angle=matched.interface_angle,
        source_band_index=matched.source_band_index,
        num_incident_candidates=matched.num_incident_candidates,
        k_in=np.asarray(matched.k_in[keep], dtype=np.float64),
        k_out_target=np.asarray(matched.k_out_target[keep], dtype=np.float64),
        k_out_interp=np.asarray(matched.k_out_interp[keep], dtype=np.float64),
        band_in=np.asarray(matched.band_in[keep], dtype=np.intp),
        band_ref=np.asarray(matched.band_ref[keep], dtype=np.intp),
        projected_gap_in=np.asarray(matched.projected_gap_in[keep], dtype=np.complex128),
        projected_gap_out_interp=np.asarray(matched.projected_gap_out_interp[keep], dtype=np.complex128),
        z_like_weight=np.asarray(matched.z_like_weight[keep], dtype=np.float64),
        x_like_weight=np.asarray(matched.x_like_weight[keep], dtype=np.float64),
        z_like_weight_out_interp=np.asarray(matched.z_like_weight_out_interp[keep], dtype=np.float64),
        x_like_weight_out_interp=np.asarray(matched.x_like_weight_out_interp[keep], dtype=np.float64),
        v_in=np.asarray(matched.v_in[keep], dtype=np.float64),
        v_out=np.asarray(matched.v_out[keep], dtype=np.float64),
        v_n_in=np.asarray(matched.v_n_in[keep], dtype=np.float64),
        v_n_out=np.asarray(matched.v_n_out[keep], dtype=np.float64),
        k_parallel=np.asarray(matched.k_parallel[keep], dtype=np.float64),
        incident_angle=np.asarray(matched.incident_angle[keep], dtype=np.float64),
        matched_contour_id=np.asarray(matched.matched_contour_id[keep], dtype=np.intp),
        matched_segment_index=np.asarray(matched.matched_segment_index[keep], dtype=np.intp),
        reflection_mismatch=np.asarray(matched.reflection_mismatch[keep], dtype=np.float64),
        matched_same_band=np.asarray(matched.matched_same_band[keep], dtype=np.bool_),
        used_cross_band_fallback=np.asarray(matched.used_cross_band_fallback[keep], dtype=np.bool_),
    )


def compute_phase_diagnostics(
    delta_plus: NDArray[np.complex128],
    delta_minus: NDArray[np.complex128],
    eps: float = 1.0e-12,
) -> PhaseDiagnostics:
    """Compute magnitudes, phases, and a phase-sensitive sign-reversal indicator."""

    delta_p = np.asarray(delta_plus, dtype=np.complex128)
    delta_m = np.asarray(delta_minus, dtype=np.complex128)
    abs_delta_plus = np.asarray(np.abs(delta_p), dtype=np.float64)
    abs_delta_minus = np.asarray(np.abs(delta_m), dtype=np.float64)
    phase_plus = np.asarray(np.angle(delta_p), dtype=np.float64)
    phase_minus = np.asarray(np.angle(delta_m), dtype=np.float64)

    phase_ratio = np.ones_like(delta_p, dtype=np.complex128)
    nonzero_mask = (abs_delta_plus > eps) & (abs_delta_minus > eps)
    phase_ratio[nonzero_mask] = delta_m[nonzero_mask] / delta_p[nonzero_mask]
    phase_difference = np.asarray(np.angle(phase_ratio), dtype=np.float64)
    sign_reversal_indicator = np.asarray(0.5 * (1.0 - np.cos(phase_difference)), dtype=np.float64)

    return PhaseDiagnostics(
        abs_delta_plus=abs_delta_plus,
        abs_delta_minus=abs_delta_minus,
        phase_plus=phase_plus,
        phase_minus=phase_minus,
        phase_difference=phase_difference,
        sign_reversal_indicator=sign_reversal_indicator,
    )


def project_interface_gaps(
    pipeline: SpectroscopyPipeline,
    interface_angle: float,
    nk: int = 201,
    energy: float = 0.0,
    dk: float = 1.0e-4,
    normal_velocity_tol: float = 1.0e-4,
    k_parallel_tol: float = 5.0e-2,
    match_distance_tol: float = 1.5e-1,
    reflected_branch_mode: ReflectedBranchMode = "strict_incident_band",
    allow_cross_band_fallback: bool = False,
    strict_reflection_match: bool = False,
    max_reflection_mismatch: float | None = None,
) -> InterfaceGapDiagnosticsResult:
    """Build interface-resolved ``Delta_plus`` / ``Delta_minus`` data."""

    gap_data = pipeline.gap_on_fermi_surface(nk=nk, energy=energy)
    segment_catalog = build_interface_segment_catalog(gap_data)
    contours: list[InterfaceResolvedContour] = []
    loose_match_points = 0

    for contour in gap_data:
        matched = match_reflected_states_on_contour(
            contour,
            segment_catalog,
            pipeline.model,
            angle=interface_angle,
            dk=dk,
            normal_velocity_tol=normal_velocity_tol,
            k_parallel_tol=k_parallel_tol,
            match_distance_tol=match_distance_tol,
            allow_cross_band_fallback=allow_cross_band_fallback,
        )
        if len(matched.k_in) == 0:
            continue

        if max_reflection_mismatch is not None:
            good_match_mask = matched.reflection_mismatch <= max_reflection_mismatch
            loose_match_points += int(np.count_nonzero(~good_match_mask))
            if strict_reflection_match:
                matched = _subset_matched_contour(matched, good_match_mask)
                if len(matched.k_in) == 0:
                    continue

        delta_plus, delta_minus, band_out_projection = compute_delta_plus_minus(
            matched,
            pipeline,
            reflected_branch_mode=reflected_branch_mode,
        )
        diagnostics = compute_phase_diagnostics(delta_plus, delta_minus)
        contours.append(
            InterfaceResolvedContour(
                interface_angle=interface_angle,
                source_band_index=matched.source_band_index,
                k_in=np.asarray(matched.k_in, dtype=np.float64),
                k_out_target=np.asarray(matched.k_out_target, dtype=np.float64),
                k_out_interp=np.asarray(matched.k_out_interp, dtype=np.float64),
                band_in=np.asarray(matched.band_in, dtype=np.intp),
                band_out=np.asarray(matched.band_ref, dtype=np.intp),
                band_out_projection=np.asarray(band_out_projection, dtype=np.intp),
                delta_plus=delta_plus,
                delta_minus=delta_minus,
                abs_delta_plus=diagnostics.abs_delta_plus,
                abs_delta_minus=diagnostics.abs_delta_minus,
                phase_plus=diagnostics.phase_plus,
                phase_minus=diagnostics.phase_minus,
                phase_difference=diagnostics.phase_difference,
                sign_reversal_indicator=diagnostics.sign_reversal_indicator,
                z_like_weight=np.asarray(matched.z_like_weight, dtype=np.float64),
                x_like_weight=np.asarray(matched.x_like_weight, dtype=np.float64),
                z_like_weight_out_interp=np.asarray(matched.z_like_weight_out_interp, dtype=np.float64),
                x_like_weight_out_interp=np.asarray(matched.x_like_weight_out_interp, dtype=np.float64),
                v_n_in=np.asarray(matched.v_n_in, dtype=np.float64),
                v_n_out=np.asarray(matched.v_n_out, dtype=np.float64),
                k_parallel=np.asarray(matched.k_parallel, dtype=np.float64),
                incident_angle=np.asarray(matched.incident_angle, dtype=np.float64),
                matched_contour_id=np.asarray(matched.matched_contour_id, dtype=np.intp),
                matched_segment_index=np.asarray(matched.matched_segment_index, dtype=np.intp),
                reflection_mismatch=np.asarray(matched.reflection_mismatch, dtype=np.float64),
                matched_same_band=np.asarray(matched.matched_same_band, dtype=np.bool_),
                used_cross_band_fallback=np.asarray(matched.used_cross_band_fallback, dtype=np.bool_),
            )
        )

    if loose_match_points > 0 and max_reflection_mismatch is not None and not strict_reflection_match:
        warnings.warn(
            f"Kept {loose_match_points} reflected points with mismatch above "
            f"max_reflection_mismatch={max_reflection_mismatch}; their band identity "
            "is retained for diagnostics, but reflected-gap evaluation still occurs "
            "at the interpolated reflected momentum on the main physical path.",
            stacklevel=2,
        )

    return InterfaceGapDiagnosticsResult(
        interface_angle=float(interface_angle),
        normal=interface_normal(interface_angle),
        tangent=interface_tangent(interface_angle),
        contours=contours,
        reflected_branch_mode=reflected_branch_mode,
        allow_cross_band_fallback=bool(allow_cross_band_fallback),
        strict_reflection_match=bool(strict_reflection_match),
        max_reflection_mismatch=max_reflection_mismatch,
    )
