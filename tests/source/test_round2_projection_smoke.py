from __future__ import annotations

from source.luo_loader import load_luo_samples
from source.round2_projection import project_luo_sample_to_round2_channels
from source.round2_projection_diagnostics import summarize_round2_projection


def test_round2_projection_smoke() -> None:
    sample = load_luo_samples()[0]
    projected = project_luo_sample_to_round2_channels(sample)
    assert projected.projected_physical_channels is not None
    assert projected.round2_projection_metrics["retained_ratio_total"] > 0.0

    _, _, comparison = summarize_round2_projection(max_samples=8)
    assert comparison["retained_ratio_improvement"]["median"] >= -1.0e-10
