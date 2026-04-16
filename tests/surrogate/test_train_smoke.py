from __future__ import annotations

from data.dataset_builder import DatasetBuildConfig, build_pairing_transport_dataset
from surrogate.config import TrainConfig
from surrogate.train import train_surrogate


def test_train_smoke(tmp_path) -> None:
    dataset_path, _ = build_pairing_transport_dataset(
        tmp_path / "dataset",
        DatasetBuildConfig(scale="smoke", num_samples=32, seed=13, nk=21, num_bias=101),
    )
    checkpoint_path, log_path = train_surrogate(
        dataset_path,
        tmp_path / "train",
        TrainConfig(seed=13, hidden_dim=64, num_blocks=2, batch_size=16, max_epochs=8, patience=3),
    )
    assert checkpoint_path.exists()
    assert log_path.exists()

