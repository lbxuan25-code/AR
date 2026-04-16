"""Split helpers for surrogate datasets."""

from __future__ import annotations

import numpy as np


def make_train_val_test_split(
    num_samples: int,
    seed: int = 1234,
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"Split fractions must sum to 1, got {fractions}.")
    rng = np.random.default_rng(seed)
    order = np.arange(num_samples, dtype=np.int64)
    rng.shuffle(order)
    n_train = int(round(num_samples * fractions[0]))
    n_val = int(round(num_samples * fractions[1]))
    train = np.sort(order[:n_train])
    val = np.sort(order[n_train : n_train + n_val])
    test = np.sort(order[n_train + n_val :])
    return train, val, test

