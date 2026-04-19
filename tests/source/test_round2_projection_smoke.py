from __future__ import annotations

from source.luo_loader import load_luo_samples
from source.round2_projection import Round2ProjectionConfig, project_luo_sample_to_round2_channels
from source.round2_projection_diagnostics import summarize_round2_projection


def test_round2_projection_smoke() -> None:
    sample = load_luo_samples()[0]
    projected = project_luo_sample_to_round2_channels(sample)
    assert projected.projected_physical_channels is not None
    assert projected.round2_projection_metadata["anchor_channel"] == "delta_zz_s"
    assert projected.round2_projection_metadata["optional_channel_policy"]["mode"] == "freeze_by_default_soft_gate"
    assert projected.round2_projection_metadata["optional_channel_policy"]["activated"] is False
    assert abs(projected.projected_physical_channels.delta_zx_s) < 1.0e-12
    assert projected.round2_projection_metrics["retained_ratio_total"] > 0.0

    _, round2_summary, comparison = summarize_round2_projection(max_samples=8)
    assert round2_summary["baseline_cluster_summary"]["num_samples"] == 8
    assert round2_summary["optional_channel_relative_scale"]["median"] < 1.0e-3
    assert round2_summary["optional_channel_policy_summary"]["frozen_fraction"] > 0.99
    assert comparison["retained_ratio_improvement"]["median"] >= -1.0e-10

    ar_projected = project_luo_sample_to_round2_channels(
        sample,
        config=Round2ProjectionConfig(source_entry_weight_mode="ar_aware", freeze_optional_weak_channel_by_default=False),
    )
    assert ar_projected.projected_physical_channels is not None
    assert ar_projected.round2_projection_metadata["fit_mode"].endswith("ar_entry_weights")
    assert ar_projected.round2_projection_metadata["source_entry_weight_stats"]["max"] >= 1.0
    assert ar_projected.round2_projection_metadata["optional_channel_policy"]["mode"] == "not_applied"
