from __future__ import annotations

from source.ar_aware_projection_diagnostics import (
    summarize_ar_aware_projection_comparison,
    write_ar_aware_projection_outputs,
)


def test_ar_aware_projection_smoke(tmp_path) -> None:
    per_sample, summary, representative_samples = summarize_ar_aware_projection_comparison(
        max_samples=8,
        representative_spectrum_nk=31,
        representative_num_bias=121,
    )
    artifacts = write_ar_aware_projection_outputs(
        output_dir=tmp_path / "outputs",
        per_sample=per_sample,
        summary=summary,
        representative_samples=representative_samples,
        make_plots=False,
    )
    assert summary["num_samples"] == 8
    assert "projected_channel_stability" in summary
    assert "representative_spectral_agreement" in summary
    assert len(representative_samples) == 3
    assert artifacts.summary_path.exists()
    assert artifacts.examples_csv_path.exists()
