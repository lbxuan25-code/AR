# Initial Directory Plan

This is the proposed starting layout for the future training / inversion
repository.

```text
AR-surrogate-inverse/
  AGENTS.md
  TODO.md
  README.md
  pyproject.toml
  src/
    ar_inverse/
      __init__.py
      forward_client.py
      metadata.py
      datasets/
        __init__.py
        schema.py
        sampling.py
        build.py
      surrogate/
        __init__.py
        models.py
        train.py
        evaluate.py
        calibration.py
      inverse/
        __init__.py
        objectives.py
        search.py
        candidates.py
      experiments/
        __init__.py
        ingest.py
        preprocessing.py
        report.py
  scripts/
    datasets/
      build_dataset.py
    surrogate/
      train_surrogate.py
      evaluate_surrogate.py
    inverse/
      run_inverse_search.py
    experiments/
      fit_experiment.py
  configs/
    datasets/
    surrogate/
    inverse/
    experiments/
  docs/
    dataset_schema.md
    sampling_policy.md
    surrogate_model_card_template.md
    inverse_output_contract.md
    experiment_report_template.md
  tests/
    test_forward_dependency_smoke.py
    test_dataset_schema.py
    test_metadata_completeness.py
    test_inverse_output_contract.py
  outputs/
    datasets/
    checkpoints/
    runs/
    inverse/
    experiments/
```

## Ownership Map

`src/ar_inverse/forward_client.py`

- wraps calls to the external forward repository;
- stores forward-interface metadata;
- must not copy forward physics internals.

`src/ar_inverse/datasets/`

- owns dataset schemas, sampling policies, manifests, and dataset builders;
- calls `forward_client.py` for spectra.

`src/ar_inverse/surrogate/`

- owns model definitions, training loops, evaluation, and calibration;
- reads generated datasets and writes checkpoints / metrics.

`src/ar_inverse/inverse/`

- owns inverse objectives, search algorithms, and candidate-family schemas;
- uses surrogate acceleration where safe;
- rechecks final candidates with direct forward calls.

`src/ar_inverse/experiments/`

- owns experimental ingest, preprocessing metadata, and report generation;
- must keep preprocessing separate from physical inference claims.

`configs/`

- stores run configuration files;
- every config should include dataset / forward-interface version expectations.

`outputs/`

- stores generated artifacts;
- large outputs should be excluded from source control unless explicitly chosen
  as tiny canonical examples.

## Required First Smoke Test

The first test in the new repository should:

1. import the external `forward` package;
2. build a small `FitLayerSpectrumRequest`;
3. generate one spectrum;
4. assert finite conductance values;
5. assert metadata includes `forward_interface_version`,
   `output_schema_version`, `pairing_convention_id`, and `git_commit`.

That test is the guardrail against accidentally forking forward physics into
the training repository.
