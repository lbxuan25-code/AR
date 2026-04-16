"""Lightweight spectroscopy workflow on top of the core model modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .bands import normal_state_eigensystem_on_kgrid
from .fermi_surface import extract_fermi_contours, find_fermi_crossing_bands
from .projection import (
    nearest_fermi_band_indices,
    normal_state_eigensystem_path,
    orbital_sector_weights,
    projected_gap_along_path,
)
from .simulation_model import SimulationModel
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interface_gap import ReflectedBranchMode


@dataclass(frozen=True, slots=True)
class FermiSurfaceContour:
    """A single Fermi-surface contour tied to one band index."""

    band_index: int
    k_points: NDArray[np.float64]
    energy_values: NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class GapOnFermiSurface:
    """Projected gap and orbital-character data along one contour."""

    band_index: int
    band_indices: NDArray[np.intp]
    k_points: NDArray[np.float64]
    projected_gaps: NDArray[np.complex128]
    z_like_weight: NDArray[np.float64]
    x_like_weight: NDArray[np.float64]


@dataclass(slots=True)
class SpectroscopyPipeline:
    """Minimal workflow layer for Fermi-surface, gap, and interface tasks."""

    model: SimulationModel

    def extract_fermi_surface(
        self,
        nk: int = 201,
        energy: float = 0.0,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], list[FermiSurfaceContour]]:
        """Compute the normal-state grid and extract contours near the target energy."""

        kx_grid, ky_grid, eigenvalues, _ = normal_state_eigensystem_on_kgrid(self.model.params, nk=nk)
        crossing_bands = find_fermi_crossing_bands(eigenvalues, energy=energy)
        contour_data = extract_fermi_contours(
            kx_grid,
            ky_grid,
            eigenvalues,
            energy=energy,
            band_indices=crossing_bands,
        )
        contours = [
            FermiSurfaceContour(
                band_index=item["band"],
                k_points=np.asarray(item["k_points"], dtype=np.float64),
                energy_values=np.asarray(
                    normal_state_eigensystem_path(np.asarray(item["k_points"], dtype=np.float64), self.model)[0][
                        :, item["band"]
                    ],
                    dtype=np.float64,
                ),
            )
            for item in contour_data
        ]
        return kx_grid, ky_grid, eigenvalues, contours

    def gap_on_fermi_surface(
        self,
        nk: int = 201,
        energy: float = 0.0,
    ) -> list[GapOnFermiSurface]:
        """Project the pairing matrix onto the Fermi-surface contours."""

        _, _, _, contours = self.extract_fermi_surface(nk=nk, energy=energy)
        gap_data: list[GapOnFermiSurface] = []

        for contour in contours:
            eigenvalues_path, eigenvectors_path = normal_state_eigensystem_path(contour.k_points, self.model)
            band_indices = np.full(len(contour.k_points), contour.band_index, dtype=np.intp)
            nearest_band = nearest_fermi_band_indices(eigenvalues_path, energy=energy)
            use_nearest = np.abs(eigenvalues_path[np.arange(len(contour.k_points)), contour.band_index] - energy) > 1.0e-2
            band_indices[use_nearest] = nearest_band[use_nearest]

            weights = orbital_sector_weights(eigenvectors_path)
            projected_gaps = projected_gap_along_path(contour.k_points, band_indices, self.model)
            gap_data.append(
                GapOnFermiSurface(
                    band_index=contour.band_index,
                    band_indices=np.asarray(band_indices, dtype=np.intp),
                    k_points=contour.k_points,
                    projected_gaps=projected_gaps,
                    z_like_weight=weights["z_like"][np.arange(len(contour.k_points)), band_indices],
                    x_like_weight=weights["x_like"][np.arange(len(contour.k_points)), band_indices],
                )
            )

        return gap_data

    def interface_gap_diagnostics(
        self,
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
    ):
        """Return the interface-resolved gap package for the minimal BTK scaffold."""

        from .interface_gap import project_interface_gaps

        return project_interface_gaps(
            self,
            interface_angle=interface_angle,
            nk=nk,
            energy=energy,
            dk=dk,
            normal_velocity_tol=normal_velocity_tol,
            k_parallel_tol=k_parallel_tol,
            match_distance_tol=match_distance_tol,
            reflected_branch_mode=reflected_branch_mode,
            allow_cross_band_fallback=allow_cross_band_fallback,
            strict_reflection_match=strict_reflection_match,
            max_reflection_mismatch=max_reflection_mismatch,
        )

    def compute_multichannel_btk_conductance(
        self,
        interface_angle: float,
        bias: NDArray[np.float64],
        barrier_z: float,
        broadening_gamma: float,
        temperature: float = 0.0,
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
        mismatch_penalty_scale: float | None = 0.12,
        min_channel_weight: float = 1.0e-4,
    ):
        """Return the enhanced multichannel BTK/generalized-KT style conductance."""

        from .ar.btk_multichannel import compute_multichannel_btk_conductance

        diagnostics = self.interface_gap_diagnostics(
            interface_angle=interface_angle,
            nk=nk,
            energy=energy,
            dk=dk,
            normal_velocity_tol=normal_velocity_tol,
            k_parallel_tol=k_parallel_tol,
            match_distance_tol=match_distance_tol,
            reflected_branch_mode=reflected_branch_mode,
            allow_cross_band_fallback=allow_cross_band_fallback,
            strict_reflection_match=strict_reflection_match,
            max_reflection_mismatch=max_reflection_mismatch,
        )
        return compute_multichannel_btk_conductance(
            diagnostics=diagnostics,
            bias=np.asarray(bias, dtype=np.float64),
            barrier_z=barrier_z,
            broadening_gamma=broadening_gamma,
            temperature=temperature,
            mismatch_penalty_scale=mismatch_penalty_scale,
            min_channel_weight=min_channel_weight,
        )

    def transport_kernel_hook(self, *args: Any, **kwargs: Any) -> None:
        """Placeholder for future AR / BTK transport integrations."""

        raise NotImplementedError("TODO: attach transport kernels to the spectroscopy pipeline.")


ForwardPipeline = SpectroscopyPipeline
