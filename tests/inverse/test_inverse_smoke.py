from __future__ import annotations

from data.dataset_builder import DatasetBuildConfig, build_pairing_transport_dataset
from surrogate.config import TrainConfig
from surrogate.inverse import run_inverse_demo
from surrogate.train import train_surrogate


def test_inverse_smoke(tmp_path) -> None:
    dataset_path, _ = build_pairing_transport_dataset(
        tmp_path / "dataset",
        DatasetBuildConfig(scale="smoke", num_samples=36, seed=17, nk=21, num_bias=101),
    )
    checkpoint_path, _ = train_surrogate(
        dataset_path,
        tmp_path / "train",
        TrainConfig(seed=17, hidden_dim=64, num_blocks=2, batch_size=16, max_epochs=8, patience=3),
    )
    report_path = run_inverse_demo(dataset_path, checkpoint_path, tmp_path / "inverse", top_k=3, nk=21)
    assert report_path.exists()
