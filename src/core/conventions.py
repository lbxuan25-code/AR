"""Physics conventions shared across the package."""

from __future__ import annotations

BASIS_ORDER: tuple[str, str, str, str] = ("Az", "Ax", "Bz", "Bx")
PAIRING_CHANNELS: tuple[str, str, str, str, str, str] = (
    "eta_z_s",
    "eta_z_perp",
    "eta_x_s",
    "eta_x_d",
    "eta_zx_d",
    "eta_x_perp",
)
