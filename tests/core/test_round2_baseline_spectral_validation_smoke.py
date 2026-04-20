from __future__ import annotations

from core.round2_baseline_spectral_validation import run_round2_baseline_spectral_validation


def test_round2_baseline_spectral_validation_smoke(tmp_path) -> None:
    summary, artifacts = run_round2_baseline_spectral_validation(
        output_dir=tmp_path,
        representative_selection_max_samples=48,
        nk=21,
        bias_max=30.0,
        num_bias=81,
        scan_values={
            "interface_angle": (0.0, 0.7853981633974483),
            "barrier_z": (0.5, 1.0),
            "gamma": (1.0, 2.0),
            "temperature": (3.0, 8.0),
        },
    )

    assert artifacts.summary_path.exists()
    assert artifacts.metrics_csv_path.exists()
    assert artifacts.model_scan_plot_path is not None and artifacts.model_scan_plot_path.exists()
    assert artifacts.channel_sensitivity_plot_path is not None and artifacts.channel_sensitivity_plot_path.exists()
    assert "formal_vs_compatibility" in summary
    assert "channel_sensitivity" in summary
    assert set(summary["pairing_states"]["representative_samples"]) == {"best", "median", "worst"}
    assert summary["formal_vs_compatibility"]["all_cases"]["num_cases"] == 8
    assert "verdict" in summary
