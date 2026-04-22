"""Canonical directional modes for the stable forward interface.

Task M only promotes validated in-plane high-symmetry raw angles into named
helpers. Generic in-plane certification and c-axis support remain separate
tasks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Literal

import numpy as np

from .schema import TransportControls

DirectionalModeName = Literal["inplane_100", "inplane_110"]
DirectionalSpreadRule = Literal["uniform_symmetric"]
UNSUPPORTED_C_AXIS_ALIASES: tuple[str, ...] = ("c_axis", "c-axis", "caxis", "axis_c")
MAX_DIRECTIONAL_SPREAD_HALF_WIDTH = float(math.pi / 32.0)


@dataclass(frozen=True, slots=True)
class DirectionalMode:
    """Stable public description of one named directional forward mode."""

    name: DirectionalModeName
    crystal_label: str
    interface_angle: float
    support_tier: str
    dimensionality: str
    description: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DirectionalSpread:
    """Narrow angular spread around one supported named in-plane mode.

    ``half_width`` is in radians. The current primitive is intentionally narrow
    and low-dimensional; it is not an experiment-side mixture model.
    """

    direction_mode: str
    half_width: float = 0.0
    num_samples: int = 1
    averaging_rule: DirectionalSpreadRule = "uniform_symmetric"

    def to_dict(self) -> dict[str, object]:
        return {
            "direction_mode": self.direction_mode,
            "half_width": float(self.half_width),
            "num_samples": int(self.num_samples),
            "averaging_rule": self.averaging_rule,
        }


CANONICAL_DIRECTIONAL_MODES: dict[str, DirectionalMode] = {
    "inplane_100": DirectionalMode(
        name="inplane_100",
        crystal_label="100",
        interface_angle=0.0,
        support_tier="A",
        dimensionality="2D in-plane",
        description="In-plane [100] high-symmetry interface normal; raw interface_angle = 0.",
    ),
    "inplane_110": DirectionalMode(
        name="inplane_110",
        crystal_label="110",
        interface_angle=float(math.pi / 4.0),
        support_tier="A",
        dimensionality="2D in-plane",
        description="In-plane [110] high-symmetry interface normal; raw interface_angle = pi/4.",
    ),
}


def list_directional_modes() -> tuple[DirectionalMode, ...]:
    """Return the currently supported named directional modes."""

    return tuple(CANONICAL_DIRECTIONAL_MODES[name] for name in sorted(CANONICAL_DIRECTIONAL_MODES))


def get_directional_mode(name: str) -> DirectionalMode:
    """Return a named directional mode or raise a clear error."""

    if str(name).lower() in UNSUPPORTED_C_AXIS_ALIASES:
        raise ValueError(
            "c-axis transport is not supported by the current forward model. "
            "Do not emulate c-axis by a 2D in-plane interface_angle."
        )
    try:
        return CANONICAL_DIRECTIONAL_MODES[str(name)]
    except KeyError as exc:
        supported = ", ".join(sorted(CANONICAL_DIRECTIONAL_MODES))
        raise ValueError(f"Unsupported directional mode {name!r}. Supported modes: {supported}.") from exc


def interface_angle_for_direction_mode(name: str) -> float:
    """Return the raw 2D ``interface_angle`` for a named directional mode."""

    return float(get_directional_mode(name).interface_angle)


def transport_with_direction_mode(
    direction_mode: str,
    *,
    barrier_z: float = 0.5,
    gamma: float = 1.0,
    temperature_kelvin: float = 3.0,
    nk: int = 41,
) -> TransportControls:
    """Build transport controls for a named in-plane high-symmetry mode."""

    mode = get_directional_mode(direction_mode)
    return TransportControls(
        direction_mode=mode.name,
        interface_angle=float(mode.interface_angle),
        barrier_z=float(barrier_z),
        gamma=float(gamma),
        temperature_kelvin=float(temperature_kelvin),
        nk=int(nk),
    )


def replace_direction_mode(transport: TransportControls, direction_mode: str) -> TransportControls:
    """Return ``transport`` with its angle set by a named directional mode."""

    mode = get_directional_mode(direction_mode)
    return replace(transport, direction_mode=mode.name, interface_angle=float(mode.interface_angle))


def validate_directional_spread(spread: DirectionalSpread) -> None:
    """Validate the narrow directional-spread primitive."""

    get_directional_mode(spread.direction_mode)
    if spread.averaging_rule != "uniform_symmetric":
        raise ValueError(f"Unsupported directional spread averaging rule: {spread.averaging_rule!r}.")
    if spread.half_width < 0.0:
        raise ValueError("Directional spread half_width must be non-negative.")
    if spread.half_width > MAX_DIRECTIONAL_SPREAD_HALF_WIDTH:
        raise ValueError(
            "Directional spread half_width is outside the current narrow-spread contract: "
            f"got {spread.half_width}, max {MAX_DIRECTIONAL_SPREAD_HALF_WIDTH}."
        )
    if spread.num_samples < 1:
        raise ValueError("Directional spread num_samples must be at least 1.")
    if spread.half_width > 0.0 and spread.num_samples < 3:
        raise ValueError("Nonzero directional spread requires at least 3 samples.")
    if spread.num_samples % 2 != 1:
        raise ValueError("Directional spread num_samples must be odd for symmetric sampling.")


def directional_spread_samples(spread: DirectionalSpread) -> tuple[dict[str, float], ...]:
    """Return raw angle samples and uniform weights for a directional spread."""

    validate_directional_spread(spread)
    mode = get_directional_mode(spread.direction_mode)
    if spread.half_width == 0.0:
        return (
            {
                "interface_angle": float(mode.interface_angle),
                "relative_angle": 0.0,
                "weight": 1.0,
            },
        )
    relative_angles = np.linspace(
        -float(spread.half_width),
        float(spread.half_width),
        int(spread.num_samples),
        dtype=np.float64,
    )
    weight = 1.0 / float(spread.num_samples)
    return tuple(
        {
            "interface_angle": float(mode.interface_angle + relative_angle),
            "relative_angle": float(relative_angle),
            "weight": float(weight),
        }
        for relative_angle in relative_angles
    )


def spread_transport_samples(
    transport: TransportControls,
    spread: DirectionalSpread,
) -> tuple[tuple[dict[str, float], TransportControls], ...]:
    """Return raw-angle transport controls used by a spread average."""

    samples = directional_spread_samples(spread)
    return tuple(
        (
            sample,
            replace(
                transport,
                direction_mode=None,
                interface_angle=float(sample["interface_angle"]),
            ),
        )
        for sample in samples
    )


def validate_transport_direction(transport: TransportControls, *, atol: float = 1.0e-12) -> None:
    """Validate that an optional mode label and raw angle are consistent."""

    if transport.direction_mode is None:
        return
    mode = get_directional_mode(transport.direction_mode)
    if not np.isclose(float(transport.interface_angle), float(mode.interface_angle), atol=atol, rtol=0.0):
        raise ValueError(
            "TransportControls direction_mode and interface_angle disagree: "
            f"{transport.direction_mode!r} requires interface_angle={mode.interface_angle}, "
            f"got {transport.interface_angle}."
        )


def direction_metadata_for_transport(transport: TransportControls) -> dict[str, object]:
    """Return stable direction provenance for forward outputs."""

    if transport.direction_mode is None:
        return {
            "direction_mode": None,
            "direction_support_tier": "raw_2d_inplane_angle",
            "direction_crystal_label": None,
            "direction_dimensionality": "2D in-plane raw angle",
            "direction_is_named_mode": False,
        }
    mode = get_directional_mode(transport.direction_mode)
    return {
        "direction_mode": mode.name,
        "direction_support_tier": mode.support_tier,
        "direction_crystal_label": mode.crystal_label,
        "direction_dimensionality": mode.dimensionality,
        "direction_is_named_mode": True,
    }
