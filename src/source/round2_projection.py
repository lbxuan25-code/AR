"""Round-2 source-native projection into physical pairing channels."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import lru_cache

import numpy as np

from core.parameters import ModelParams
from core.pipeline import SpectroscopyPipeline
from core.conventions import CORE_PHYSICAL_PAIRING_CHANNELS, OPTIONAL_PHYSICAL_PAIRING_CHANNELS, PHYSICAL_PAIRING_CHANNELS
from core.parameters import PhysicalPairingChannels
from core.presets import base_normal_state_params
from core.simulation_model import SimulationModel

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
    source_entry_weight_mode: str = "uniform"
    gauge_anchor_priority: tuple[str, ...] = ROUND2_CORE_CHANNEL_NAMES + ROUND2_OPTIONAL_CHANNEL_NAMES
    gauge_min_anchor_abs: float = 1.0e-10
    freeze_optional_weak_channel_by_default: bool = True
    optional_weak_channel_name: str = "delta_zx_s"
    optional_channel_activation_min_relative_magnitude: float = 8.0e-2
    optional_channel_activation_min_residual_reduction: float = 1.0e-2
    ar_interface_angles: tuple[float, ...] = (0.0, 0.7853981633974483)
    ar_reference_nk: int = 61
    ar_supported_entry_weight_floor: float = 0.75
    ar_unsupported_entry_weight: float = 0.15
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
            "source_entry_weight_mode": self.source_entry_weight_mode,
            "freeze_optional_weak_channel_by_default": bool(self.freeze_optional_weak_channel_by_default),
            "optional_weak_channel_name": self.optional_weak_channel_name,
            "optional_channel_activation_min_relative_magnitude": float(self.optional_channel_activation_min_relative_magnitude),
            "optional_channel_activation_min_residual_reduction": float(self.optional_channel_activation_min_residual_reduction),
            "ar_interface_angles": [float(value) for value in self.ar_interface_angles],
            "ar_reference_nk": int(self.ar_reference_nk),
            "ar_supported_entry_weight_floor": float(self.ar_supported_entry_weight_floor),
            "ar_unsupported_entry_weight": float(self.ar_unsupported_entry_weight),
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


def _unit_channel_state(channel_name: str) -> PhysicalPairingChannels:
    values = {name: 0.0 + 0.0j for name in ROUND2_CHANNEL_NAMES}
    values[channel_name] = 1.0 + 0.0j
    return PhysicalPairingChannels(**values)


@lru_cache(maxsize=8)
def _ar_channel_relevance_scores(
    interface_angles: tuple[float, ...],
    nk: int,
) -> tuple[float, ...]:
    """Return channel relevance scores from interface-gap diagnostics.

    The scores are sample-independent and are anchored to the current baseline
    normal state. They rank how strongly each round-2 channel contributes to the
    projected gaps that later feed the AR / BTK workflow.
    """

    normal_state = base_normal_state_params()
    scores: list[float] = []
    for channel_name in ROUND2_CHANNEL_NAMES:
        model = SimulationModel(
            params=ModelParams(normal_state=normal_state, pairing=_unit_channel_state(channel_name)),
            name=f"ar_relevance::{channel_name}",
        )
        pipeline = SpectroscopyPipeline(model=model)
        angle_scores: list[float] = []
        for interface_angle in interface_angles:
            diagnostics = pipeline.interface_gap_diagnostics(interface_angle=float(interface_angle), nk=int(nk))
            contour_values: list[float] = []
            for contour in diagnostics.contours:
                if len(contour.abs_delta_plus) == 0:
                    continue
                contour_values.append(float(np.mean(contour.abs_delta_plus)))
                contour_values.append(float(np.mean(contour.abs_delta_minus)))
            if contour_values:
                angle_scores.append(float(np.mean(contour_values)))
        scores.append(float(np.mean(angle_scores)) if angle_scores else 0.0)

    arr = np.asarray(scores, dtype=np.float64)
    positive = arr[arr > 0.0]
    scale = float(np.mean(positive)) if positive.size > 0 else 1.0
    normalized = np.where(arr > 0.0, arr / scale, 1.0)
    return tuple(float(value) for value in normalized)


def source_entry_weight_vector(
    config: Round2ProjectionConfig = DEFAULT_ROUND2_PROJECTION_CONFIG,
) -> np.ndarray:
    """Return row weights on source-tensor entries for the current config."""

    if config.source_entry_weight_mode == "uniform":
        return np.ones(48, dtype=np.float64)
    if config.source_entry_weight_mode != "ar_aware":
        raise ValueError(
            f"Unsupported source_entry_weight_mode {config.source_entry_weight_mode!r}. "
            "Expected 'uniform' or 'ar_aware'."
        )

    basis = round2_basis_tensors()
    channel_scores = _ar_channel_relevance_scores(tuple(config.ar_interface_angles), int(config.ar_reference_nk))
    block_vectors: list[np.ndarray] = []
    for block_index in range(3):
        support_mask = np.zeros((4, 4), dtype=bool)
        raw_weights = np.zeros((4, 4), dtype=np.float64)
        for channel_name, channel_score in zip(ROUND2_CHANNEL_NAMES, channel_scores, strict=True):
            basis_matrix = np.asarray(basis[channel_name][block_index], dtype=np.complex128)
            raw_weights += float(channel_score) * np.abs(basis_matrix)
            support_mask |= np.abs(basis_matrix) > 0.0
        block_weights = np.full((4, 4), float(config.ar_unsupported_entry_weight), dtype=np.float64)
        if np.any(support_mask):
            support_values = raw_weights[support_mask]
            scale = float(np.mean(support_values)) if np.mean(support_values) > 0.0 else 1.0
            normalized = raw_weights / scale
            normalized[support_mask] = np.maximum(
                normalized[support_mask],
                float(config.ar_supported_entry_weight_floor),
            )
            block_weights[support_mask] = normalized[support_mask]
        block_vectors.append(block_weights.reshape(-1))
    return np.concatenate(block_vectors)


def _fit_channels_from_weighted_design(
    weighted_design: np.ndarray,
    weighted_target: np.ndarray,
    config: Round2ProjectionConfig,
    active_channel_names: tuple[str, ...] = ROUND2_CHANNEL_NAMES,
) -> PhysicalPairingChannels:
    """Solve the weighted ridge fit for the selected active channels."""

    active_indices = [ROUND2_CHANNEL_NAMES.index(name) for name in active_channel_names]
    reduced_design = weighted_design[:, active_indices]
    regularization = config.regularization_vector()[active_indices]
    augmented_design = np.vstack([reduced_design, np.diag(np.sqrt(regularization)).astype(np.complex128)])
    augmented_target = np.concatenate(
        [weighted_target, np.zeros(len(active_channel_names), dtype=np.complex128)]
    )
    solution, _, _, _ = np.linalg.lstsq(augmented_design, augmented_target, rcond=None)
    values = {name: 0.0 + 0.0j for name in ROUND2_CHANNEL_NAMES}
    for index, name in enumerate(active_channel_names):
        values[name] = complex(solution[index])
    return PhysicalPairingChannels(**values)


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

    block_weights = np.concatenate(
        [
            np.full(delta_x.size, np.sqrt(config.weight_x), dtype=np.float64),
            np.full(delta_y.size, np.sqrt(config.weight_y), dtype=np.float64),
            np.full(delta_z.size, np.sqrt(config.weight_z), dtype=np.float64),
        ]
    )
    entry_weights = source_entry_weight_vector(config=config)
    weights = np.sqrt(block_weights * entry_weights).astype(np.complex128)
    weighted_target = weights * target
    weighted_design = weights[:, None] * design

    full_channels = _fit_channels_from_weighted_design(
        weighted_design=weighted_design,
        weighted_target=weighted_target,
        config=config,
        active_channel_names=ROUND2_CHANNEL_NAMES,
    )
    channels = full_channels
    optional_policy_metadata = {
        "mode": "not_applied",
        "optional_channel_name": config.optional_weak_channel_name,
        "activated": True,
    }
    if config.freeze_optional_weak_channel_by_default:
        active_without_optional = tuple(
            name for name in ROUND2_CHANNEL_NAMES if name != config.optional_weak_channel_name
        )
        frozen_channels = _fit_channels_from_weighted_design(
            weighted_design=weighted_design,
            weighted_target=weighted_target,
            config=config,
            active_channel_names=active_without_optional,
        )
        full_metrics = build_projection_metric_bundle(gauge_x, gauge_y, gauge_z, *reconstruct_source_tensors_from_channels(full_channels))
        frozen_metrics = build_projection_metric_bundle(gauge_x, gauge_y, gauge_z, *reconstruct_source_tensors_from_channels(frozen_channels))
        optional_abs = abs(getattr(full_channels, config.optional_weak_channel_name))
        core_abs = max(abs(getattr(full_channels, name)) for name in ROUND2_CORE_CHANNEL_NAMES)
        relative_magnitude = float(optional_abs / core_abs) if core_abs > 0.0 else 0.0
        residual_reduction = float(frozen_metrics["residual_norm_total"] - full_metrics["residual_norm_total"])
        clear_need = (
            relative_magnitude >= float(config.optional_channel_activation_min_relative_magnitude)
            and residual_reduction >= float(config.optional_channel_activation_min_residual_reduction)
        )
        channels = full_channels if clear_need else frozen_channels
        optional_policy_metadata = {
            "mode": "freeze_by_default_soft_gate",
            "optional_channel_name": config.optional_weak_channel_name,
            "activated": bool(clear_need),
            "relative_magnitude": relative_magnitude,
            "residual_reduction_if_activated": residual_reduction,
            "activation_min_relative_magnitude": float(config.optional_channel_activation_min_relative_magnitude),
            "activation_min_residual_reduction": float(config.optional_channel_activation_min_residual_reduction),
            "selected_fit": "full_channels" if clear_need else "frozen_optional_channel",
        }
    metadata = {
        **gauge_metadata,
        "config": config.to_dict(),
        "source_entry_weight_stats": {
            "min": float(np.min(entry_weights)),
            "max": float(np.max(entry_weights)),
            "mean": float(np.mean(entry_weights)),
        },
        "ar_channel_relevance_scores": {
            name: score
            for name, score in zip(
                ROUND2_CHANNEL_NAMES,
                _ar_channel_relevance_scores(tuple(config.ar_interface_angles), int(config.ar_reference_nk)),
                strict=True,
            )
        }
        if config.source_entry_weight_mode == "ar_aware"
        else {},
        "optional_channel_policy": optional_policy_metadata,
        "fit_mode": (
            "weighted_ridge_with_global_gauge_fix_and_ar_entry_weights"
            if config.source_entry_weight_mode == "ar_aware"
            else "weighted_ridge_with_global_gauge_fix"
        ),
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
