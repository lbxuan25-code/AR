from __future__ import annotations

from source.luo_loader import load_luo_samples


def test_luo_loader_reads_at_least_one_sample() -> None:
    samples = load_luo_samples()
    assert samples
    assert samples[0].source_pairing_observables["delta_z"].shape == (4, 4)

