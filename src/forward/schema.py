"""Canonical schemas for the stable forward AR interface."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np

FORWARD_INTERFACE_VERSION = "ar_forward_v1"
FORWARD_INPUT_SCHEMA_VERSION = "ar_forward_input_v1"
FORWARD_OUTPUT_SCHEMA_VERSION = "ar_forward_output_v1"
ROUND2_PAIRING_CONVENTION_ID = "round2_physical_channels_task_h_fit_layer_v1"

FIT_LAYER_FREE_CHANNELS: tuple[str, ...] = (
    "delta_zz_s",
    "delta_xx_s",
    "delta_zx_d",
    "delta_perp_z",
    "delta_perp_x",
)
FIT_LAYER_REGULARIZED_CHANNELS: tuple[str, ...] = (
    "delta_zz_d",
    "delta_xx_d",
)
FIT_LAYER_WEAK_CHANNELS: tuple[str, ...] = ("delta_zx_s",)
FIT_LAYER_CONTROL_CHANNELS: tuple[str, ...] = FIT_LAYER_FREE_CHANNELS + FIT_LAYER_REGULARIZED_CHANNELS + FIT_LAYER_WEAK_CHANNELS

PairingControlMode = Literal["delta_from_baseline_meV", "absolute_meV"]


@dataclass(frozen=True, slots=True)
class BiasGrid:
    """Bias grid in meV for AR spectra."""

    bias_min_mev: float = -40.0
    bias_max_mev: float = 40.0
    num_bias: int = 201

    def values(self) -> np.ndarray:
        if self.num_bias < 2:
            raise ValueError("num_bias must be at least 2.")
        if self.bias_max_mev <= self.bias_min_mev:
            raise ValueError("bias_max_mev must be larger than bias_min_mev.")
        return np.linspace(float(self.bias_min_mev), float(self.bias_max_mev), int(self.num_bias), dtype=np.float64)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TransportControls:
    """Transport controls consumed by the multichannel BTK forward path."""

    interface_angle: float = 0.0
    barrier_z: float = 0.5
    gamma: float = 1.0
    temperature_kelvin: float = 3.0
    nk: int = 41

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FitLayerSpectrumRequest:
    """Canonical fit-layer request for external AR spectrum generation.

    ``pairing_controls`` are real meV controls in the Task-H fit layer. In the
    default mode they are deviations from the authoritative formal round-2
    baseline. ``absolute_meV`` sets the selected channel real amplitudes
    directly while preserving unspecified baseline channels.
    """

    pairing_controls: dict[str, float] = field(default_factory=dict)
    pairing_control_mode: PairingControlMode = "delta_from_baseline_meV"
    allow_weak_delta_zx_s: bool = False
    transport: TransportControls = field(default_factory=TransportControls)
    bias_grid: BiasGrid = field(default_factory=BiasGrid)
    request_label: str = "fit_layer"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": FORWARD_INPUT_SCHEMA_VERSION,
            "request_kind": "fit_layer",
            "request_label": self.request_label,
            "pairing_control_mode": self.pairing_control_mode,
            "pairing_controls": dict(self.pairing_controls),
            "allow_weak_delta_zx_s": bool(self.allow_weak_delta_zx_s),
            "transport": self.transport.to_dict(),
            "bias_grid": self.bias_grid.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class SourceRound2SpectrumRequest:
    """Canonical source-linked request using default round-2 projection."""

    source_sample_id: str | None = None
    source_sample_index: int | None = None
    transport: TransportControls = field(default_factory=TransportControls)
    bias_grid: BiasGrid = field(default_factory=BiasGrid)
    request_label: str = "source_round2"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": FORWARD_INPUT_SCHEMA_VERSION,
            "request_kind": "source_round2",
            "request_label": self.request_label,
            "source_sample_id": self.source_sample_id,
            "source_sample_index": self.source_sample_index,
            "transport": self.transport.to_dict(),
            "bias_grid": self.bias_grid.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ForwardSpectrumResult:
    """Canonical forward-output payload for generated AR spectra."""

    schema_version: str
    request_kind: str
    request: dict[str, object]
    metadata: dict[str, object]
    pairing_channels: dict[str, dict[str, float]]
    bias_mev: list[float]
    conductance: list[float]
    conductance_unbroadened: list[float]
    transport_summary: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
