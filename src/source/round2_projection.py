"""Round-2 source-native projection into physical pairing channels."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from core.parameters import PhysicalPairingChannels

from .luo_projection import EV_TO_MEV
from .schema import LuoSample, ProjectionMode, ProjectionRecord

ROUND2_CHANNEL_NAMES: tuple[str, ...] = (
    "delta_zz_s",
    "delta_zz_d",
    "delta_xx_s",
    "delta_xx_d",
    "delta_zx_s",
    "delta_zx_d",
    "delta_perp_z",
    "delta_perp_x",
)


def source_pairing_tensors_meV(sample: LuoSample) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return source pairing tensors in meV with exactly one unit conversion."""

    delta_x = np.asarray(sample.source_pairing_observables["delta_x"], dtype=np.complex128) * EV_TO_MEV
    delta_y = np.asarray(sample.source_pairing_observables["delta_y"], dtype=np.complex128) * EV_TO_MEV
    delta_z = np.asarray(sample.source_pairing_observables["delta_z"], dtype=np.complex128) * EV_TO_MEV
    return delta_x, delta_y, delta_z


def round2_basis_tensors() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return the source-native basis tensors for the round-2 channel layer."""

    def zeros() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.zeros((4, 4), dtype=np.complex128),
            np.zeros((4, 4), dtype=np.complex128),
            np.zeros((4, 4), dtype=np.complex128),
        )

    basis: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    dx, dy, dz = zeros()
    dx[0, 0] = dx[2, 2] = 1.0
    dy[0, 0] = dy[2, 2] = 1.0
    basis["delta_zz_s"] = (dx, dy, dz)

    dx, dy, dz = zeros()
    dx[0, 0] = dx[2, 2] = 1.0
    dy[0, 0] = dy[2, 2] = -1.0
    basis["delta_zz_d"] = (dx, dy, dz)

    dx, dy, dz = zeros()
    dx[1, 1] = dx[3, 3] = 1.0
    dy[1, 1] = dy[3, 3] = 1.0
    basis["delta_xx_s"] = (dx, dy, dz)

    dx, dy, dz = zeros()
    dx[1, 1] = dx[3, 3] = 1.0
    dy[1, 1] = dy[3, 3] = -1.0
    basis["delta_xx_d"] = (dx, dy, dz)

    dx, dy, dz = zeros()
    for matrix in (dx, dy):
        matrix[0, 1] = matrix[1, 0] = 1.0
        matrix[2, 3] = matrix[3, 2] = 1.0
    basis["delta_zx_s"] = (dx, dy, dz)

    dx, dy, dz = zeros()
    dx[0, 1] = dx[1, 0] = 1.0
    dx[2, 3] = dx[3, 2] = 1.0
    dy[0, 1] = dy[1, 0] = -1.0
    dy[2, 3] = dy[3, 2] = -1.0
    basis["delta_zx_d"] = (dx, dy, dz)

    dx, dy, dz = zeros()
    dz[0, 2] = dz[2, 0] = 1.0
    basis["delta_perp_z"] = (dx, dy, dz)

    dx, dy, dz = zeros()
    dz[1, 3] = dz[3, 1] = 1.0
    basis["delta_perp_x"] = (dx, dy, dz)
    return basis


def flatten_source_tensors(delta_x: np.ndarray, delta_y: np.ndarray, delta_z: np.ndarray) -> np.ndarray:
    """Return one complex vector for joint source reconstruction."""

    return np.concatenate(
        [
            np.asarray(delta_x, dtype=np.complex128).reshape(-1),
            np.asarray(delta_y, dtype=np.complex128).reshape(-1),
            np.asarray(delta_z, dtype=np.complex128).reshape(-1),
        ]
    )


def reconstruct_source_tensors_from_channels(
    channels: PhysicalPairingChannels,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct source-native x/y/z bond tensors from the round-2 channels."""

    basis = round2_basis_tensors()
    recon_x = np.zeros((4, 4), dtype=np.complex128)
    recon_y = np.zeros((4, 4), dtype=np.complex128)
    recon_z = np.zeros((4, 4), dtype=np.complex128)
    for name in ROUND2_CHANNEL_NAMES:
        coeff = getattr(channels, name)
        basis_x, basis_y, basis_z = basis[name]
        recon_x = recon_x + coeff * basis_x
        recon_y = recon_y + coeff * basis_y
        recon_z = recon_z + coeff * basis_z
    return recon_x, recon_y, recon_z


def fit_round2_channels(sample: LuoSample) -> PhysicalPairingChannels:
    """Solve the constrained complex least-squares fit on the full source tensor."""

    delta_x, delta_y, delta_z = source_pairing_tensors_meV(sample)
    target = flatten_source_tensors(delta_x, delta_y, delta_z)
    basis = round2_basis_tensors()
    design = np.column_stack(
        [
            flatten_source_tensors(*basis[name])
            for name in ROUND2_CHANNEL_NAMES
        ]
    )
    solution, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
    return PhysicalPairingChannels(**{name: complex(solution[index]) for index, name in enumerate(ROUND2_CHANNEL_NAMES)})


def round2_projection_metrics(
    sample: LuoSample,
    channels: PhysicalPairingChannels,
) -> dict[str, float]:
    """Return reconstruction metrics for the round-2 physical channel fit."""

    delta_x, delta_y, delta_z = source_pairing_tensors_meV(sample)
    recon_x, recon_y, recon_z = reconstruct_source_tensors_from_channels(channels)

    source_norm_x = float(np.linalg.norm(delta_x, ord="fro"))
    source_norm_y = float(np.linalg.norm(delta_y, ord="fro"))
    source_norm_z = float(np.linalg.norm(delta_z, ord="fro"))
    recon_norm_x = float(np.linalg.norm(recon_x, ord="fro"))
    recon_norm_y = float(np.linalg.norm(recon_y, ord="fro"))
    recon_norm_z = float(np.linalg.norm(recon_z, ord="fro"))
    residual_norm_x = float(np.linalg.norm(delta_x - recon_x, ord="fro"))
    residual_norm_y = float(np.linalg.norm(delta_y - recon_y, ord="fro"))
    residual_norm_z = float(np.linalg.norm(delta_z - recon_z, ord="fro"))

    source_norm_total = float(np.sqrt(source_norm_x**2 + source_norm_y**2 + source_norm_z**2))
    recon_norm_total = float(np.sqrt(recon_norm_x**2 + recon_norm_y**2 + recon_norm_z**2))
    residual_norm_total = float(np.sqrt(residual_norm_x**2 + residual_norm_y**2 + residual_norm_z**2))
    return {
        "source_norm_x": source_norm_x,
        "source_norm_y": source_norm_y,
        "source_norm_z": source_norm_z,
        "recon_norm_x": recon_norm_x,
        "recon_norm_y": recon_norm_y,
        "recon_norm_z": recon_norm_z,
        "residual_norm_x": residual_norm_x,
        "residual_norm_y": residual_norm_y,
        "residual_norm_z": residual_norm_z,
        "source_norm_total": source_norm_total,
        "recon_norm_total": recon_norm_total,
        "residual_norm_total": residual_norm_total,
        "retained_ratio_x": float(1.0 - residual_norm_x / source_norm_x) if source_norm_x > 0.0 else 1.0,
        "retained_ratio_y": float(1.0 - residual_norm_y / source_norm_y) if source_norm_y > 0.0 else 1.0,
        "retained_ratio_z": float(1.0 - residual_norm_z / source_norm_z) if source_norm_z > 0.0 else 1.0,
        "retained_ratio_total": float(1.0 - residual_norm_total / source_norm_total) if source_norm_total > 0.0 else 1.0,
        "omitted_fraction_total": float(residual_norm_total / source_norm_total) if source_norm_total > 0.0 else 0.0,
    }


def _round2_projection_provenance() -> dict[str, ProjectionRecord]:
    return {
        "delta_zz_s": ProjectionRecord(
            field_name="delta_zz_s",
            mode=ProjectionMode.DIRECT,
            source_expression="least-squares coefficient of the zz bond-even basis across delta_x/delta_y source tensors",
            note="Equivalent to the symmetric z-sector x/y bond component when the source tensor lies in the round-2 channel subspace.",
        ),
        "delta_zz_d": ProjectionRecord(
            field_name="delta_zz_d",
            mode=ProjectionMode.DIRECT,
            source_expression="least-squares coefficient of the zz bond-odd basis across delta_x/delta_y source tensors",
            note="Captures z-sector d-like anisotropy that was omitted in round 1.",
        ),
        "delta_xx_s": ProjectionRecord(
            field_name="delta_xx_s",
            mode=ProjectionMode.DIRECT,
            source_expression="least-squares coefficient of the xx bond-even basis across delta_x/delta_y source tensors",
            note="x-like diagonal bond-even pairing channel.",
        ),
        "delta_xx_d": ProjectionRecord(
            field_name="delta_xx_d",
            mode=ProjectionMode.DIRECT,
            source_expression="least-squares coefficient of the xx bond-odd basis across delta_x/delta_y source tensors",
            note="x-like diagonal bond-odd pairing channel.",
        ),
        "delta_zx_s": ProjectionRecord(
            field_name="delta_zx_s",
            mode=ProjectionMode.APPROXIMATE,
            source_expression="constrained least-squares coefficient of the same-layer z-x mixed bond-even basis",
            note="Optional round-2 mixed channel introduced because the source tensor shows non-negligible bond-even z-x structure.",
        ),
        "delta_zx_d": ProjectionRecord(
            field_name="delta_zx_d",
            mode=ProjectionMode.APPROXIMATE,
            source_expression="constrained least-squares coefficient of the same-layer z-x mixed bond-odd basis",
            note="Primary round-2 mixed channel motivated by the round-1 omitted diagnostics.",
        ),
        "delta_perp_z": ProjectionRecord(
            field_name="delta_perp_z",
            mode=ProjectionMode.DIRECT,
            source_expression="least-squares coefficient of the interlayer z-like basis in delta_z",
            note="Interlayer z-like pairing channel.",
        ),
        "delta_perp_x": ProjectionRecord(
            field_name="delta_perp_x",
            mode=ProjectionMode.DIRECT,
            source_expression="least-squares coefficient of the interlayer x-like basis in delta_z",
            note="Interlayer x-like pairing channel omitted in round 1.",
        ),
    }


def project_luo_sample_to_round2_channels(sample: LuoSample) -> LuoSample:
    """Project one Luo sample into the round-2 physical channel layer."""

    channels = fit_round2_channels(sample)
    metrics = round2_projection_metrics(sample, channels)
    return replace(
        sample,
        projected_physical_channels=channels,
        round2_projection_provenance=_round2_projection_provenance(),
        round2_projection_metrics=metrics,
    )


def project_luo_samples_to_round2_channels(samples: list[LuoSample]) -> list[LuoSample]:
    """Project a batch of Luo samples into the round-2 physical channel layer."""

    return [project_luo_sample_to_round2_channels(sample) for sample in samples]
