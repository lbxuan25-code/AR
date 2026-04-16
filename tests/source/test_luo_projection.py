from __future__ import annotations

from core.parameters import PairingParams
from source.luo_loader import load_luo_samples
from source.luo_projection import project_luo_sample_to_pairing


def test_luo_projection_produces_pairing_params() -> None:
    sample = load_luo_samples()[0]
    projected = project_luo_sample_to_pairing(sample)
    assert isinstance(projected.projected_pairing_params, PairingParams)
    assert "eta_z_perp" in projected.projection_provenance

