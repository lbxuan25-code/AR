from __future__ import annotations

from source.projection_diagnostics import run_projection_consistency_check


def test_projection_consistency_smoke(tmp_path) -> None:
    docs_path = tmp_path / "projection_consistency_round1.md"
    summary, artifacts = run_projection_consistency_check(
        output_dir=tmp_path / "outputs",
        docs_path=docs_path,
        max_samples=16,
        make_plots=False,
    )
    assert summary["num_samples"] == 16
    assert summary["retained_channel_residual_stats"]["eta_z_s"]["max_abs"] < 1.0e-10
    assert summary["retained_channel_residual_stats"]["eta_z_perp"]["max_abs"] < 1.0e-10
    assert artifacts.summary_path.exists()
    assert artifacts.examples_csv_path.exists()
    assert artifacts.docs_path.exists()
