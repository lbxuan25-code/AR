"""Projection from Luo RMFT observables to local PairingParams."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from core.parameters import PairingParams

from .schema import LuoSample, ProjectionMode, ProjectionRecord

EV_TO_MEV = 1000.0


def _symmetrized(matrix: np.ndarray, i: int, j: int) -> complex:
    return complex(0.5 * (matrix[i, j] + matrix[j, i]))


def project_luo_sample_to_pairing(sample: LuoSample) -> LuoSample:
    delta_x = np.asarray(sample.source_pairing_observables["delta_x"], dtype=np.complex128) * EV_TO_MEV
    delta_y = np.asarray(sample.source_pairing_observables["delta_y"], dtype=np.complex128) * EV_TO_MEV
    delta_z = np.asarray(sample.source_pairing_observables["delta_z"], dtype=np.complex128) * EV_TO_MEV

    eta_z_s = complex(0.5 * (delta_x[0, 0] + delta_y[0, 0]))
    eta_z_perp = _symmetrized(delta_z, 0, 2)
    eta_x_s = complex(0.5 * (delta_x[1, 1] + delta_y[1, 1]))
    eta_x_d = complex(0.5 * (delta_x[1, 1] - delta_y[1, 1]))

    projected = PairingParams(
        eta_z_s=eta_z_s,
        eta_z_perp=eta_z_perp,
        eta_x_s=eta_x_s,
        eta_x_d=eta_x_d,
        eta_zx_d=0.0 + 0.0j,
        eta_x_perp=0.0 + 0.0j,
    )
    provenance = {
        "eta_z_s": ProjectionRecord(
            field_name="eta_z_s",
            mode=ProjectionMode.APPROXIMATE,
            source_expression="0.5 * (delta_x[0,0] + delta_y[0,0]) * 1000",
            note="Luo source resolves z-like diagonal x/y bond pairing; mapped to local z-like s component in meV.",
        ),
        "eta_z_perp": ProjectionRecord(
            field_name="eta_z_perp",
            mode=ProjectionMode.DIRECT,
            source_expression="0.5 * (delta_z[0,2] + delta_z[2,0]) * 1000",
            note="Directly uses the interlayer z-like pairing matrix element and converts eV to meV.",
        ),
        "eta_x_s": ProjectionRecord(
            field_name="eta_x_s",
            mode=ProjectionMode.APPROXIMATE,
            source_expression="0.5 * (delta_x[1,1] + delta_y[1,1]) * 1000",
            note="Maps Luo x-like diagonal x/y bond pairing to the local x-like s component in meV.",
        ),
        "eta_x_d": ProjectionRecord(
            field_name="eta_x_d",
            mode=ProjectionMode.APPROXIMATE,
            source_expression="0.5 * (delta_x[1,1] - delta_y[1,1]) * 1000",
            note="Maps Luo x-like bond anisotropy to the local x-like d component in meV.",
        ),
        "eta_zx_d": ProjectionRecord(
            field_name="eta_zx_d",
            mode=ProjectionMode.ZEROED,
            source_expression="0",
            note="No direct RMFT field for the local z-x off-diagonal d channel was identified in round 1.",
        ),
        "eta_x_perp": ProjectionRecord(
            field_name="eta_x_perp",
            mode=ProjectionMode.ZEROED,
            source_expression="0",
            note="No direct RMFT field for the local x-like interlayer channel was identified in round 1.",
        ),
    }
    projected_sample = replace(sample, projected_pairing_params=projected, projection_provenance=provenance)
    return projected_sample


def project_luo_samples(samples: list[LuoSample]) -> list[LuoSample]:
    return [project_luo_sample_to_pairing(sample) for sample in samples]

