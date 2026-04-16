"""Parameter containers for the unified LNO327 phenomenology framework."""

from __future__ import annotations

from dataclasses import asdict, dataclass


NORMAL_STATE_FAMILIES: tuple[str, ...] = ("base",)


@dataclass(slots=True)
class NormalStateParams:
    """Parameters entering the formal four-orbital bilayer Hamiltonian.

    The basis ordering stays fixed as ``(Az, Ax, Bz, Bx)``.
    """

    family: str = "base"

    e1: float = 0.0
    e2: float = 0.0
    tx1: float = 0.0
    tx2: float = 0.0
    txy1: float = 0.0
    txy2: float = 0.0
    vx: float = 0.0
    v1: float = 0.0
    v2: float = 0.0
    vxz: float = 0.0
    mu_diag: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        """Validate the supported family and the diagonal chemical potential."""

        if self.family not in NORMAL_STATE_FAMILIES:
            raise ValueError(
                f"Unsupported normal-state family {self.family!r}. "
                f"Expected one of {NORMAL_STATE_FAMILIES}."
            )
        if len(self.mu_diag) != 4:
            raise ValueError(f"mu_diag must contain 4 entries, got {self.mu_diag!r}.")

    def to_dict(self) -> dict[str, object]:
        """Return a plain dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class PairingParams:
    """Unified multi-channel pairing container for the phenomenology framework.

    The container is part of the single formal baseline workflow and remains
    available for controlled perturbations through the physical-controls layer.
    """

    eta_z_s: complex = 0.0 + 0.0j
    eta_z_perp: complex = 0.0 + 0.0j
    eta_x_s: complex = 0.0 + 0.0j
    eta_x_d: complex = 0.0 + 0.0j
    eta_zx_d: complex = 0.0 + 0.0j
    eta_x_perp: complex = 0.0 + 0.0j

    def to_dict(self) -> dict[str, object]:
        """Return a plain dictionary representation."""

        return asdict(self)


@dataclass(slots=True)
class ModelParams:
    """Combined parameter container for normal-state and pairing sectors."""

    normal_state: NormalStateParams
    pairing: PairingParams

    def to_dict(self) -> dict[str, object]:
        """Return a nested dictionary representation."""

        return asdict(self)
