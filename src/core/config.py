"""Runtime and engineering defaults for the project scaffold."""

from __future__ import annotations

from typing import Final, Literal

DEFAULT_K_GRID_SIZE: int = 51
DEFAULT_BAND_PATH_POINTS: int = 64
DEFAULT_FERMI_SURFACE_ATOL: float = 5.0e-3
DEFAULT_OUTPUT_DIR: str = "outputs"
DEFAULT_EXAMPLE_OUTPUT_SUBDIR: str = "examples"
DEFAULT_SCRIPT_OUTPUT_SUBDIR: str = "scripts"

# Formal AR / BTK workflow defaults: the repository's single official
# transport path is the multichannel solver on top of the stored baseline,
# with all scans interpreted as perturbations around it.
FORMAL_REFLECTED_BRANCH_MODE: Final[Literal["strict_incident_band"]] = "strict_incident_band"
FORMAL_ALLOW_CROSS_BAND_FALLBACK: Final[bool] = False
FORMAL_STRICT_REFLECTION_MATCH: Final[bool] = False
FORMAL_MAX_REFLECTION_MISMATCH: Final[float | None] = 0.2
FORMAL_MISMATCH_PENALTY_SCALE: Final[float | None] = 0.12
FORMAL_MIN_CHANNEL_WEIGHT: Final[float] = 1.0e-4
