from __future__ import annotations

from source.round2_residual_anatomy import run_round2_residual_anatomy_audit


def test_round2_residual_anatomy_smoke(tmp_path) -> None:
    docs_path = tmp_path / "round2_residual_anatomy.md"
    per_sample, summary, artifacts = run_round2_residual_anatomy_audit(
        output_dir=tmp_path / "outputs",
        docs_path=docs_path,
        max_samples=12,
        make_plots=False,
    )
    assert len(per_sample) == 12
    assert summary["num_samples"] == 12
    assert "delta_x" in summary["block_residual_summary"]
    assert "delta_y" in summary["block_residual_summary"]
    assert "delta_z" in summary["block_residual_summary"]
    assert summary["diagnosis"]["dominant_cause"] in {
        "missing_channel_structure",
        "imperfect_projection_weighting",
        "mixed",
    }
    assert set(summary["representative_samples"]) == {"best", "median", "worst"}
    assert artifacts.summary_path.exists()
    assert artifacts.examples_csv_path.exists()
    assert artifacts.docs_path.exists()
