"""Stable forward-facing interface for external training repositories."""

from .directions import (
    CANONICAL_DIRECTIONAL_MODES,
    DirectionalMode,
    DirectionalSpread,
    directional_spread_samples,
    get_directional_mode,
    interface_angle_for_direction_mode,
    list_directional_modes,
    replace_direction_mode,
    spread_transport_samples,
    transport_with_direction_mode,
    validate_directional_spread,
)
from .engine import (
    generate_spectrum_from_fit_layer,
    generate_spectrum_from_source_round2,
    generate_spread_spectrum_from_fit_layer,
    generate_spread_spectrum_from_source_round2,
)
from .schema import (
    BiasGrid,
    FitLayerSpectrumRequest,
    ForwardSpectrumResult,
    SourceRound2SpectrumRequest,
    TransportControls,
)

__all__ = [
    "BiasGrid",
    "CANONICAL_DIRECTIONAL_MODES",
    "DirectionalMode",
    "DirectionalSpread",
    "FitLayerSpectrumRequest",
    "ForwardSpectrumResult",
    "SourceRound2SpectrumRequest",
    "TransportControls",
    "directional_spread_samples",
    "get_directional_mode",
    "generate_spectrum_from_fit_layer",
    "generate_spectrum_from_source_round2",
    "generate_spread_spectrum_from_fit_layer",
    "generate_spread_spectrum_from_source_round2",
    "interface_angle_for_direction_mode",
    "list_directional_modes",
    "replace_direction_mode",
    "spread_transport_samples",
    "transport_with_direction_mode",
    "validate_directional_spread",
]
