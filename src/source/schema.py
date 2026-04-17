"""Internal source schema for Luo RMFT samples."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from core.parameters import PairingParams, PhysicalPairingChannels


class ProjectionMode(StrEnum):
    DIRECT = "direct_read"
    APPROXIMATE = "approximate_inference"
    ZEROED = "conventionally_zeroed"
    UNKNOWN = "undetermined"


@dataclass(slots=True)
class ProjectionRecord:
    field_name: str
    mode: ProjectionMode
    source_expression: str
    note: str


@dataclass(slots=True)
class LuoSample:
    sample_id: str
    source_name: str
    source_file: Path
    sample_kind: str
    coordinates: dict[str, Any]
    source_metadata: dict[str, Any]
    source_pairing_observables: dict[str, NDArray[np.complex128] | NDArray[np.float64]]
    source_chemical_potential: NDArray[np.float64] | None
    projected_pairing_params: PairingParams | None = None
    projection_provenance: dict[str, ProjectionRecord] = field(default_factory=dict)
    projected_physical_channels: PhysicalPairingChannels | None = None
    round2_projection_provenance: dict[str, ProjectionRecord] = field(default_factory=dict)
    round2_projection_metrics: dict[str, Any] = field(default_factory=dict)
    round2_projection_metadata: dict[str, Any] = field(default_factory=dict)
