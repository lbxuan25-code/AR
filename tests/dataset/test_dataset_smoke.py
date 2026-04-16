from __future__ import annotations

import numpy as np

from data.dataset_builder import DatasetBuildConfig, build_pairing_transport_dataset


def test_dataset_builder_smoke(tmp_path) -> None:
    dataset_path, manifest_path = build_pairing_transport_dataset(
        tmp_path,
        DatasetBuildConfig(scale="smoke", num_samples=24, seed=11, nk=21, num_bias=101),
    )
    payload = np.load(dataset_path)
    assert dataset_path.exists()
    assert manifest_path.exists()
    assert payload["features"].shape[0] == 24
    assert payload["spectra"].shape == (24, 101)

