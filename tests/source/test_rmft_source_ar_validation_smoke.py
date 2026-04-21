from pathlib import Path

from source.rmft_source_ar_validation import run_rmft_source_vs_round2_ar_validation


def test_rmft_source_vs_round2_ar_validation_smoke(tmp_path: Path) -> None:
    scan_values = {
        "interface_angle": (0.0, 0.25),
        "barrier_z": (0.5, 1.0),
        "gamma": (1.0, 2.0),
        "temperature": (3.0, 8.0),
    }
    summary, scan_cases, artifacts = run_rmft_source_vs_round2_ar_validation(
        output_dir=tmp_path,
        max_selection_samples=16,
        scan_values=scan_values,
        nk=11,
        bias_max=20.0,
        num_bias=41,
    )

    assert artifacts.summary_path.exists()
    assert artifacts.metrics_csv_path.exists()
    assert len(artifacts.plot_paths) == 3
    assert all(path.exists() for path in artifacts.plot_paths)
    assert "RMFT source-reference" in summary["validation_axis"]
    assert set(summary["representative_samples"]) == {"best", "median", "worst"}
    assert len(scan_cases) == 3 * 4 * 2
    assert summary["overall_metrics"]["num_cases"] == len(scan_cases)
    assert summary["conclusion"]["round2_sufficient_for_ar"] in {True, False}
    assert "round-1-vs-round-2" in summary["conclusion"]["verdict"]
