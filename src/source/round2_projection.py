"""Round-2 source-native projection into physical pairing channels."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np

from core.conventions import CORE_PHYSICAL_PAIRING_CHANNELS, OPTIONAL_PHYSICAL_PAIRING_CHANNELS, PHYSICAL_PAIRING_CHANNELS
from core.parameters import PhysicalPairingChannels

from .luo_projection import EV_TO_MEV
from .projection_metrics import build_projection_metric_bundle
from .schema import LuoSample, ProjectionMode, ProjectionRecord

ROUND2_CHANNEL_NAMES: tuple[str, ...] = PHYSICAL_PAIRING_CHANNELS
ROUND2_CORE_CHANNEL_NAMES: tuple[str, ...] = CORE_PHYSICAL_PAIRING_CHANNELS
ROUND2_OPTIONAL_CHANNEL_NAMES: tuple[str, ...] = OPTIONAL_PHYSICAL_PAIRING_CHANNELS


@dataclass(frozen=True, slots=True)
class Round2ProjectionConfig:
    """Stage-3 projection controls for the round-2 truth layer."""

    weight_x: float = 1.0
    weight_y: float = 1.0
    weight_z: float = 1.15
    gauge_anchor_priority: tuple[str, ...] = ROUND2_CORE_CHANNEL_NAMES + ROUND2_OPTIONAL_CHANNEL_NAMES
    gauge_min_anchor_abs: float = 1.0e-10
    channel_regularization: dict[str, float] = field(
        default_factory=lambda: {
            "delta_zz_s": 1.0e-4,
            "delta_zz_d": 2.0e-4,
            "delta_xx_s": 1.0e-4,
            "delta_xx_d": 1.0e-4,
            "delta_zx_s": 5.0e-3,
            "delta_zx_d": 7.5e-4,
            "delta_perp_z": 1.0e-4,
            "delta_perp_x": 7.5e-4,
        }
    )

    def block_weights(self) -> tuple[float, float, float]:
        return (float(self.weight_x), float(self.weight_y), float(self.weight_z))

    def regularization_vector(self) -> np.ndarray:
        return np.asarray(
            [float(self.channel_regularization.get(name, 0.0)) for name in ROUND2_CHANNEL_NAMES],
            dtype=np.float64,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "block_weights": {
                "delta_x": float(self.weight_x),
                "delta_y": float(self.weight_y),
                "delta_z": float(self.weight_z),
            },
            "channel_regularization": {
                name: float(self.channel_regularization.get(name, 0.0))
                for name in ROUND2_CHANNEL_NAMES
            },
            "core_channels": list(ROUND2_CORE_CHANNEL_NAMES),
            "optional_channels": list(ROUND2_OPTIONAL_CHANNEL_NAMES),
            "gauge_anchor_priority": list(self.gauge_anchor_priority),
            "gauge_min_anchor_abs": float(self.gauge_min_anchor_abs),
        }


DEFAULT_ROUND2_PROJECTION_CONFIG = Round2ProjectionConfig()


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


@lru_cache(maxsize=1)
def _round2_design_matrix() -> np.ndarray:
    basis = round2_basis_tensors()
    return np.column_stack([flatten_source_tensors(*basis[name]) for name in ROUND2_CHANNEL_NAMES])


def _weighted_overlap_coefficient(
    target_x: np.ndarray,
    target_y: np.ndarray,
    target_z: np.ndarray,
    basis_triplet: tuple[np.ndarray, np.ndarray, np.ndarray],
    block_weights: tuple[float, float, float],
) -> complex:
    numerator = 0.0 + 0.0j
    denominator = 0.0
    for weight, target, basis_matrix in zip(
        block_weights,
        (target_x, target_y, target_z),
        basis_triplet,
        strict=True,
    ):
        numerator += float(weight) * np.vdot(basis_matrix, target)
        denominator += float(weight) * float(np.vdot(basis_matrix, basis_matrix).real)
    if denominator <= 0.0:
        return 0.0 + 0.0j
    return complex(numerator / denominator)


def gauge_fix_source_tensors(
    delta_x: np.ndarray,
    delta_y: np.ndarray,
    delta_z: np.ndarray,
    config: Round2ProjectionConfig = DEFAULT_ROUND2_PROJECTION_CONFIG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    """Apply a global phase convention before fitting.

    The source tensors are rotated so that the strongest prioritized anchor
    channel becomes real and non-negative. This keeps baseline aggregation and
    channel statistics gauge-consistent across samples.
    """

    basis = round2_basis_tensors()
    block_weights = config.block_weights()
    anchor_name = config.gauge_anchor_priority[0]
    anchor_coeff = 0.0 + 0.0j
    for candidate in config.gauge_anchor_priority:
        coeff = _weighted_overlap_coefficient(delta_x, delta_y, delta_z, basis[candidate], block_weights)
        if abs(coeff) >= config.gauge_min_anchor_abs:
            anchor_name = candidate
            anchor_coeff = coeff
            break
    phase = float(np.angle(anchor_coeff)) if abs(anchor_coeff) > 0.0 else 0.0
    rotation = np.exp(-1.0j * phase)
    return (
        np.asarray(delta_x, dtype=np.complex128) * rotation,
        np.asarray(delta_y, dtype=np.complex128) * rotation,
        np.asarray(delta_z, dtype=np.complex128) * rotation,
        {
            "anchor_channel": anchor_name,
            "anchor_magnitude_meV": float(abs(anchor_coeff)),
            "gauge_phase_radians": phase,
            "gauge_rotation_re": float(np.real(rotation)),
            "gauge_rotation_im": float(np.imag(rotation)),
        },
    )


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


def fit_round2_channels_with_metadata(
    sample: LuoSample,
    config: Round2ProjectionConfig = DEFAULT_ROUND2_PROJECTION_CONFIG,
) -> tuple[PhysicalPairingChannels, dict[str, object]]:
    """Solve the Stage-3 weighted and regularized complex fit.

    The fit keeps the round-2 channel basis but upgrades the old unconstrained
    least-squares step by adding:

    - block weights across ``delta_x / delta_y / delta_z``,
    - per-channel ridge regularization,
    - a global gauge convention before fitting.
    """

    delta_x, delta_y, delta_z = source_pairing_tensors_meV(sample)
    gauge_x, gauge_y, gauge_z, gauge_metadata = gauge_fix_source_tensors(delta_x, delta_y, delta_z, config=config)
    target = flatten_source_tensors(gauge_x, gauge_y, gauge_z)
    design = _round2_design_matrix()

    weights = np.concatenate(
        [
            np.full(delta_x.size, np.sqrt(config.weight_x), dtype=np.float64),
            np.full(delta_y.size, np.sqrt(config.weight_y), dtype=np.float64),
            np.full(delta_z.size, np.sqrt(config.weight_z), dtype=np.float64),
        ]
    ).astype(np.complex128)
    weighted_target = weights * target
    weighted_design = weights[:, None] * design

    regularization = config.regularization_vector()
    augmented_design = np.vstack([weighted_design, np.diag(np.sqrt(regularization)).astype(np.complex128)])
    augmented_target = np.concatenate(
        [weighted_target, np.zeros(len(ROUND2_CHANNEL_NAMES), dtype=np.complex128)]
    )
    solution, _, _, _ = np.linalg.lstsq(augmented_design, augmented_target, rcond=None)
    channels = PhysicalPairingChannels(
        **{name: complex(solution[index]) for index, name in enumerate(ROUND2_CHANNEL_NAMES)}
    )
    metadata = {
        **gauge_metadata,
        "config": config.to_dict(),
        "fit_mode": "weighted_ridge_with_global_gauge_fix",
    }
    return channels, metadata


def fit_round2_channels(
    sample: LuoSample,
    config: Round2ProjectionConfig = DEFAULT_ROUND2_PROJECTION_CONFIG,
) -> PhysicalPairingChannels:
    """Return only the fitted channels from the Stage-3 projection."""

    channels, _ = fit_round2_channels_with_metadata(sample, config=config)
    return channels


def round2_projection_metrics(
    sample: LuoSample,
    channels: PhysicalPairingChannels,
    gauge_phase_radians: float = 0.0,
) -> dict[str, float]:
    """Return reconstruction metrics for the round-2 physical channel fit."""

    delta_x, delta_y, delta_z = source_pairing_tensors_meV(sample)
    rotation = np.exp(-1.0j * float(gauge_phase_radians))
    gauge_x = delta_x * rotation
    gauge_y = delta_y * rotation
    gauge_z = delta_z * rotation
    recon_x, recon_y, recon_z = reconstruct_source_tensors_from_channels(channels)
    return build_projection_metric_bundle(gauge_x, gauge_y, gauge_z, recon_x, recon_y, recon_z)


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


def project_luo_sample_to_round2_channels(
    sample: LuoSample,
    config: Round2ProjectionConfig = DEFAULT_ROUND2_PROJECTION_CONFIG,
) -> LuoSample:
    """Project one Luo sample into the round-2 physical channel layer."""

    channels, metadata = fit_round2_channels_with_metadata(sample, config=config)
    metrics = round2_projection_metrics(
        sample,
        channels,
        gauge_phase_radians=float(metadata["gauge_phase_radians"]),
    )
    return replace(
        sample,
        projected_physical_channels=channels,
        round2_projection_provenance=_round2_projection_provenance(),
        round2_projection_metrics=metrics,
        round2_projection_metadata=metadata,
    )


def project_luo_samples_to_round2_channels(
    samples: list[LuoSample],
    config: Round2ProjectionConfig = DEFAULT_ROUND2_PROJECTION_CONFIG,
) -> list[LuoSample]:
    """Project a batch of Luo samples into the round-2 physical channel layer."""

    return [project_luo_sample_to_round2_channels(sample, config=config) for sample in samples]
