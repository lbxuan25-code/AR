"""Interface-geometry helpers for reflected band-projected diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from .projection import normal_state_eigensystem_path, orbital_sector_weights, projected_gap_along_path
from .simulation_model import SimulationModel

if TYPE_CHECKING:
    from .pipeline import GapOnFermiSurface


@dataclass(frozen=True, slots=True)
class ContourSegment:
    """One closed contour segment used for local reflected-state interpolation."""

    contour_id: int
    source_band_index: int
    segment_index: int
    endpoint_band_indices: tuple[int, int]
    start_point: NDArray[np.float64]
    end_point: NDArray[np.float64]
    displacement: NDArray[np.float64]
    length: float
    tangent: NDArray[np.float64]
    midpoint: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class InterfaceSegmentCatalog:
    """Pre-built contour segments for reflected-state matching."""

    segments: tuple[ContourSegment, ...]


@dataclass(frozen=True, slots=True)
class InterfaceReflectionContour:
    """Incident and reflected Fermi-surface states for one source contour."""

    interface_angle: float
    source_band_index: int
    num_incident_candidates: int
    k_in: NDArray[np.float64]
    k_out_target: NDArray[np.float64]
    k_out_interp: NDArray[np.float64]
    band_in: NDArray[np.intp]
    band_ref: NDArray[np.intp]
    projected_gap_in: NDArray[np.complex128]
    projected_gap_out_interp: NDArray[np.complex128]
    z_like_weight: NDArray[np.float64]
    x_like_weight: NDArray[np.float64]
    z_like_weight_out_interp: NDArray[np.float64]
    x_like_weight_out_interp: NDArray[np.float64]
    v_in: NDArray[np.float64]
    v_out: NDArray[np.float64]
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
class _SegmentMatch:
    """Internal reflected-segment match result."""

    contour_id: int
    segment_index: int
    k_out_interp: NDArray[np.float64]
    band_ref: int
    projected_gap_out_interp: complex
    z_like_weight_out_interp: float
    x_like_weight_out_interp: float
    v_out: NDArray[np.float64]
    v_n_out: float
    reflection_mismatch: float


def wrap_to_brillouin_zone(k_points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Wrap momenta to the square Brillouin zone ``[-pi, pi)``."""

    points = np.asarray(k_points, dtype=np.float64)
    return np.asarray((points + np.pi) % (2.0 * np.pi) - np.pi, dtype=np.float64)


def interface_normal(angle: float) -> NDArray[np.float64]:
    """Return the unit interface normal for a given polar angle."""

    return np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float64)


def interface_tangent(angle: float) -> NDArray[np.float64]:
    """Return the unit tangent vector associated with the interface normal."""

    return np.asarray([-np.sin(angle), np.cos(angle)], dtype=np.float64)


def decompose_k(
    k_points: NDArray[np.float64],
    angle: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return momentum components normal and parallel to the interface."""

    points = np.asarray(k_points, dtype=np.float64)
    normal = interface_normal(angle)
    tangent = interface_tangent(angle)
    return (
        np.asarray(points @ normal, dtype=np.float64),
        np.asarray(points @ tangent, dtype=np.float64),
    )


def reflect_k_across_interface(
    k_points: NDArray[np.float64],
    angle: float,
) -> NDArray[np.float64]:
    """Reflect momenta specularly across the interface plane."""

    points = np.asarray(k_points, dtype=np.float64)
    normal = interface_normal(angle)
    k_normal, _ = decompose_k(points, angle)
    reflected = points - 2.0 * k_normal[:, None] * normal[None, :]
    return wrap_to_brillouin_zone(reflected)


def _periodic_delta(
    left: NDArray[np.float64],
    right: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the shortest Brillouin-zone displacement from ``right`` to ``left``."""

    return wrap_to_brillouin_zone(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))


def build_contour_segments(
    contour: GapOnFermiSurface,
    contour_id: int,
) -> tuple[ContourSegment, ...]:
    """Build a closed piecewise-linear segment list for one contour."""

    points = np.asarray(contour.k_points, dtype=np.float64)
    band_indices = np.asarray(contour.band_indices, dtype=np.intp)
    num_points = len(points)
    if num_points < 2:
        return tuple()

    segments: list[ContourSegment] = []
    for segment_index in range(num_points):
        next_index = (segment_index + 1) % num_points
        start_point = np.asarray(points[segment_index], dtype=np.float64)
        end_point = np.asarray(points[next_index], dtype=np.float64)
        displacement = np.asarray(_periodic_delta(end_point, start_point), dtype=np.float64)
        length = float(np.linalg.norm(displacement))
        if length <= 1.0e-12:
            continue
        tangent = np.asarray(displacement / length, dtype=np.float64)
        midpoint = wrap_to_brillouin_zone(start_point[None, :] + 0.5 * displacement[None, :])[0]
        segments.append(
            ContourSegment(
                contour_id=int(contour_id),
                source_band_index=int(contour.band_index),
                segment_index=int(segment_index),
                endpoint_band_indices=(int(band_indices[segment_index]), int(band_indices[next_index])),
                start_point=start_point,
                end_point=end_point,
                displacement=displacement,
                length=length,
                tangent=tangent,
                midpoint=np.asarray(midpoint, dtype=np.float64),
            )
        )
    return tuple(segments)


def build_interface_segment_catalog(
    gap_data: list[GapOnFermiSurface],
) -> InterfaceSegmentCatalog:
    """Build a global segment catalog for local contour interpolation."""

    segments: list[ContourSegment] = []
    for contour_id, contour in enumerate(gap_data):
        segments.extend(build_contour_segments(contour, contour_id=contour_id))
    return InterfaceSegmentCatalog(segments=tuple(segments))


def _band_energy(
    kx: float,
    ky: float,
    band_index: int,
    model: SimulationModel,
) -> float:
    """Return a single normal-state band energy at one momentum."""

    eigenvalues = np.linalg.eigvalsh(model.build_normal_state(kx, ky))
    return float(np.asarray(eigenvalues[band_index], dtype=np.float64))


def estimate_group_velocity(
    kx: float,
    ky: float,
    band_index: int,
    model: SimulationModel,
    dk: float = 1.0e-4,
) -> NDArray[np.float64]:
    """Estimate the band velocity from finite differences of the normal-state eigenvalue."""

    kx_plus = wrap_to_brillouin_zone(np.array([[kx + dk, ky]], dtype=np.float64))[0]
    kx_minus = wrap_to_brillouin_zone(np.array([[kx - dk, ky]], dtype=np.float64))[0]
    ky_plus = wrap_to_brillouin_zone(np.array([[kx, ky + dk]], dtype=np.float64))[0]
    ky_minus = wrap_to_brillouin_zone(np.array([[kx, ky - dk]], dtype=np.float64))[0]

    velocity_x = (_band_energy(kx_plus[0], kx_plus[1], band_index, model) - _band_energy(kx_minus[0], kx_minus[1], band_index, model)) / (2.0 * dk)
    velocity_y = (_band_energy(ky_plus[0], ky_plus[1], band_index, model) - _band_energy(ky_minus[0], ky_minus[1], band_index, model)) / (2.0 * dk)

    return np.asarray([velocity_x, velocity_y], dtype=np.float64)


def estimate_group_velocities(
    k_points: NDArray[np.float64],
    band_indices: NDArray[np.intp],
    model: SimulationModel,
    dk: float = 1.0e-4,
) -> NDArray[np.float64]:
    """Estimate group velocities for a batch of Fermi-surface points."""

    points = np.asarray(k_points, dtype=np.float64)
    indices = np.asarray(band_indices, dtype=np.intp)
    velocities = np.empty((len(points), 2), dtype=np.float64)
    for index, ((kx, ky), band_index) in enumerate(zip(points, indices, strict=True)):
        velocities[index] = estimate_group_velocity(float(kx), float(ky), int(band_index), model, dk=dk)
    return velocities


def _empty_reflection_contour(
    interface_angle: float,
    source_band_index: int,
) -> InterfaceReflectionContour:
    """Return an empty reflection contour container."""

    empty_k = np.empty((0, 2), dtype=np.float64)
    empty_i = np.empty((0,), dtype=np.intp)
    empty_c = np.empty((0,), dtype=np.complex128)
    empty_f = np.empty((0,), dtype=np.float64)
    empty_b = np.empty((0,), dtype=np.bool_)
    return InterfaceReflectionContour(
        interface_angle=interface_angle,
        source_band_index=source_band_index,
        num_incident_candidates=0,
        k_in=empty_k.copy(),
        k_out_target=empty_k.copy(),
        k_out_interp=empty_k.copy(),
        band_in=empty_i.copy(),
        band_ref=empty_i.copy(),
        projected_gap_in=empty_c.copy(),
        projected_gap_out_interp=empty_c.copy(),
        z_like_weight=empty_f.copy(),
        x_like_weight=empty_f.copy(),
        z_like_weight_out_interp=empty_f.copy(),
        x_like_weight_out_interp=empty_f.copy(),
        v_in=empty_k.copy(),
        v_out=empty_k.copy(),
        v_n_in=empty_f.copy(),
        v_n_out=empty_f.copy(),
        k_parallel=empty_f.copy(),
        incident_angle=empty_f.copy(),
        matched_contour_id=empty_i.copy(),
        matched_segment_index=empty_i.copy(),
        reflection_mismatch=empty_f.copy(),
        matched_same_band=empty_b.copy(),
        used_cross_band_fallback=empty_b.copy(),
    )


def _state_properties_at_point(
    k_point: NDArray[np.float64],
    band_index: int,
    model: SimulationModel,
    dk: float,
) -> tuple[complex, float, float, NDArray[np.float64]]:
    """Return projected gap, orbital weights, and velocity at one point."""

    point = np.asarray(k_point, dtype=np.float64).reshape(1, 2)
    band_indices = np.asarray([band_index], dtype=np.intp)
    projected_gap = complex(reflected_gap_from_model(point, band_indices, model)[0])
    _, eigenvectors = normal_state_eigensystem_path(point, model)
    weights = orbital_sector_weights(eigenvectors)
    velocity = estimate_group_velocity(float(point[0, 0]), float(point[0, 1]), int(band_index), model, dk=dk)
    return (
        projected_gap,
        float(weights["z_like"][0, band_index]),
        float(weights["x_like"][0, band_index]),
        np.asarray(velocity, dtype=np.float64),
    )


def _segment_accepts_band(
    segment: ContourSegment,
    band_index: int,
) -> bool:
    """Return whether the segment is a viable same-band candidate."""

    return int(band_index) == int(segment.source_band_index) or int(band_index) in segment.endpoint_band_indices


def _project_target_to_segment(
    k_out_target: NDArray[np.float64],
    segment: ContourSegment,
) -> tuple[NDArray[np.float64], float, NDArray[np.float64]]:
    """Project a reflected target momentum onto one contour segment."""

    start = np.asarray(segment.start_point, dtype=np.float64)
    displacement = np.asarray(segment.displacement, dtype=np.float64)
    offset = np.asarray(_periodic_delta(k_out_target, start), dtype=np.float64)
    denominator = float(np.dot(displacement, displacement))
    if denominator <= 1.0e-20:
        projected_offset = np.zeros((2,), dtype=np.float64)
    else:
        t = float(np.clip(np.dot(offset, displacement) / denominator, 0.0, 1.0))
        projected_offset = np.asarray(t * displacement, dtype=np.float64)
    residual = np.asarray(offset - projected_offset, dtype=np.float64)
    projected_point = wrap_to_brillouin_zone(start[None, :] + projected_offset[None, :])[0]
    return np.asarray(projected_point, dtype=np.float64), float(np.linalg.norm(residual)), residual


def _find_best_segment_match(
    k_out_target: NDArray[np.float64],
    candidate_segments: tuple[ContourSegment, ...],
    projection_band: int,
    model: SimulationModel,
    angle: float,
    dk: float,
    normal_velocity_tol: float,
    k_parallel_tol: float,
    match_distance_tol: float,
) -> _SegmentMatch | None:
    """Return the best reflected-state match among candidate contour segments."""

    normal = interface_normal(angle)
    tangent = interface_tangent(angle)
    best_match: _SegmentMatch | None = None

    for segment in candidate_segments:
        k_out_interp, mismatch, residual = _project_target_to_segment(k_out_target, segment)
        if mismatch > match_distance_tol:
            continue
        parallel_diff = float(abs(np.dot(residual, tangent)))
        if parallel_diff > k_parallel_tol:
            continue

        projected_gap, z_weight, x_weight, velocity = _state_properties_at_point(k_out_interp, projection_band, model, dk=dk)
        v_n_out = float(np.dot(velocity, normal))
        if v_n_out <= normal_velocity_tol:
            continue

        candidate = _SegmentMatch(
            contour_id=int(segment.contour_id),
            segment_index=int(segment.segment_index),
            k_out_interp=np.asarray(k_out_interp, dtype=np.float64),
            band_ref=int(projection_band),
            projected_gap_out_interp=projected_gap,
            z_like_weight_out_interp=z_weight,
            x_like_weight_out_interp=x_weight,
            v_out=np.asarray(velocity, dtype=np.float64),
            v_n_out=v_n_out,
            reflection_mismatch=float(mismatch),
        )
        if best_match is None or candidate.reflection_mismatch < best_match.reflection_mismatch:
            best_match = candidate

    return best_match


def match_reflected_states_on_contour(
    contour: GapOnFermiSurface,
    segment_catalog: InterfaceSegmentCatalog,
    model: SimulationModel,
    angle: float,
    dk: float = 1.0e-4,
    normal_velocity_tol: float = 1.0e-4,
    k_parallel_tol: float = 5.0e-2,
    match_distance_tol: float = 1.5e-1,
    allow_cross_band_fallback: bool = False,
) -> InterfaceReflectionContour:
    """Match reflected states by same-band contour interpolation.

    The default physical path is:

    1. reflect the incident momentum geometrically to obtain ``k_out_target``
    2. search only same-band contour segments
    3. locally interpolate the reflected representative ``k_out_interp`` on the
       best same-band segment
    4. recompute reflected quantities directly at ``k_out_interp``

    Cross-band fallback is available only as an explicit diagnostics mode.
    """

    incident_velocities = estimate_group_velocities(contour.k_points, contour.band_indices, model, dk=dk)
    normal = interface_normal(angle)
    tangent = interface_tangent(angle)
    v_n_all = np.asarray(incident_velocities @ normal, dtype=np.float64)
    incident_mask = v_n_all < -normal_velocity_tol
    if not np.any(incident_mask):
        return _empty_reflection_contour(angle, contour.band_index)

    k_in = np.asarray(contour.k_points[incident_mask], dtype=np.float64)
    num_incident_candidates = int(len(k_in))
    band_in = np.asarray(contour.band_indices[incident_mask], dtype=np.intp)
    projected_gap_in = np.asarray(contour.projected_gaps[incident_mask], dtype=np.complex128)
    z_like_weight = np.asarray(contour.z_like_weight[incident_mask], dtype=np.float64)
    x_like_weight = np.asarray(contour.x_like_weight[incident_mask], dtype=np.float64)
    v_in = np.asarray(incident_velocities[incident_mask], dtype=np.float64)
    v_n_in = np.asarray(v_n_all[incident_mask], dtype=np.float64)
    v_parallel_in = np.asarray(v_in @ tangent, dtype=np.float64)
    k_parallel = np.asarray(k_in @ tangent, dtype=np.float64)
    incident_angle = np.asarray(np.arctan2(np.abs(v_parallel_in), np.abs(v_n_in)), dtype=np.float64)
    k_out_target = reflect_k_across_interface(k_in, angle)

    keep_indices: list[int] = []
    k_out_interp_list: list[NDArray[np.float64]] = []
    band_ref_list: list[int] = []
    projected_gap_out_interp_list: list[complex] = []
    z_out_list: list[float] = []
    x_out_list: list[float] = []
    v_out_list: list[NDArray[np.float64]] = []
    v_n_out_list: list[float] = []
    contour_id_list: list[int] = []
    segment_index_list: list[int] = []
    mismatch_list: list[float] = []
    matched_same_band_list: list[bool] = []
    used_cross_band_fallback_list: list[bool] = []

    all_segments = segment_catalog.segments

    for point_index in range(len(k_in)):
        same_band_segments = tuple(segment for segment in all_segments if _segment_accepts_band(segment, int(band_in[point_index])))
        best_match = _find_best_segment_match(
            k_out_target=k_out_target[point_index],
            candidate_segments=same_band_segments,
            projection_band=int(band_in[point_index]),
            model=model,
            angle=angle,
            dk=dk,
            normal_velocity_tol=normal_velocity_tol,
            k_parallel_tol=k_parallel_tol,
            match_distance_tol=match_distance_tol,
        )

        matched_same_band = True
        used_cross_band_fallback = False
        if best_match is None and allow_cross_band_fallback:
            fallback_match: _SegmentMatch | None = None
            for segment in all_segments:
                candidate_band = int(segment.source_band_index)
                candidate = _find_best_segment_match(
                    k_out_target=k_out_target[point_index],
                    candidate_segments=(segment,),
                    projection_band=candidate_band,
                    model=model,
                    angle=angle,
                    dk=dk,
                    normal_velocity_tol=normal_velocity_tol,
                    k_parallel_tol=k_parallel_tol,
                    match_distance_tol=match_distance_tol,
                )
                if candidate is None:
                    continue
                if fallback_match is None or candidate.reflection_mismatch < fallback_match.reflection_mismatch:
                    fallback_match = candidate
            best_match = fallback_match
            matched_same_band = best_match is not None and int(best_match.band_ref) == int(band_in[point_index])
            used_cross_band_fallback = best_match is not None and not matched_same_band

        if best_match is None:
            continue

        keep_indices.append(point_index)
        k_out_interp_list.append(best_match.k_out_interp)
        band_ref_list.append(int(best_match.band_ref))
        projected_gap_out_interp_list.append(complex(best_match.projected_gap_out_interp))
        z_out_list.append(float(best_match.z_like_weight_out_interp))
        x_out_list.append(float(best_match.x_like_weight_out_interp))
        v_out_list.append(np.asarray(best_match.v_out, dtype=np.float64))
        v_n_out_list.append(float(best_match.v_n_out))
        contour_id_list.append(int(best_match.contour_id))
        segment_index_list.append(int(best_match.segment_index))
        mismatch_list.append(float(best_match.reflection_mismatch))
        matched_same_band_list.append(bool(matched_same_band))
        used_cross_band_fallback_list.append(bool(used_cross_band_fallback))

    if not keep_indices:
        return _empty_reflection_contour(angle, contour.band_index)

    keep = np.asarray(keep_indices, dtype=np.intp)
    return InterfaceReflectionContour(
        interface_angle=angle,
        source_band_index=contour.band_index,
        num_incident_candidates=num_incident_candidates,
        k_in=np.asarray(k_in[keep], dtype=np.float64),
        k_out_target=np.asarray(k_out_target[keep], dtype=np.float64),
        k_out_interp=np.asarray(np.stack(k_out_interp_list, axis=0), dtype=np.float64),
        band_in=np.asarray(band_in[keep], dtype=np.intp),
        band_ref=np.asarray(band_ref_list, dtype=np.intp),
        projected_gap_in=np.asarray(projected_gap_in[keep], dtype=np.complex128),
        projected_gap_out_interp=np.asarray(projected_gap_out_interp_list, dtype=np.complex128),
        z_like_weight=np.asarray(z_like_weight[keep], dtype=np.float64),
        x_like_weight=np.asarray(x_like_weight[keep], dtype=np.float64),
        z_like_weight_out_interp=np.asarray(z_out_list, dtype=np.float64),
        x_like_weight_out_interp=np.asarray(x_out_list, dtype=np.float64),
        v_in=np.asarray(v_in[keep], dtype=np.float64),
        v_out=np.asarray(np.stack(v_out_list, axis=0), dtype=np.float64),
        v_n_in=np.asarray(v_n_in[keep], dtype=np.float64),
        v_n_out=np.asarray(v_n_out_list, dtype=np.float64),
        k_parallel=np.asarray(k_parallel[keep], dtype=np.float64),
        incident_angle=np.asarray(incident_angle[keep], dtype=np.float64),
        matched_contour_id=np.asarray(contour_id_list, dtype=np.intp),
        matched_segment_index=np.asarray(segment_index_list, dtype=np.intp),
        reflection_mismatch=np.asarray(mismatch_list, dtype=np.float64),
        matched_same_band=np.asarray(matched_same_band_list, dtype=np.bool_),
        used_cross_band_fallback=np.asarray(used_cross_band_fallback_list, dtype=np.bool_),
    )


def reflected_gap_from_model(
    k_out_target: NDArray[np.float64],
    band_ref: NDArray[np.intp],
    model: SimulationModel,
) -> NDArray[np.complex128]:
    """Reproject the gap on reflected states at the supplied reflected momentum."""

    return projected_gap_along_path(k_out_target, np.asarray(band_ref, dtype=np.intp), model)
