"""Stable forward-facing interface for external training repositories."""

from .engine import generate_spectrum_from_fit_layer, generate_spectrum_from_source_round2
from .schema import (
    BiasGrid,
    FitLayerSpectrumRequest,
    ForwardSpectrumResult,
    SourceRound2SpectrumRequest,
    TransportControls,
)

__all__ = [
    "BiasGrid",
    "FitLayerSpectrumRequest",
    "ForwardSpectrumResult",
    "SourceRound2SpectrumRequest",
    "TransportControls",
    "generate_spectrum_from_fit_layer",
    "generate_spectrum_from_source_round2",
]
