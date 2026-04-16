"""Configuration objects for surrogate training and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class TrainConfig:
    seed: int = 1234
    hidden_dim: int = 256
    num_blocks: int = 3
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-5
    batch_size: int = 128
    max_epochs: int = 120
    patience: int = 15
    derivative_weight: float = 0.25
    low_bias_weight: float = 1.5
    peak_region_weight: float = 1.25

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

