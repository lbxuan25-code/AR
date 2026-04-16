"""Residual MLP implemented with NumPy/autograd."""

from __future__ import annotations

from dataclasses import dataclass

import autograd.numpy as anp
import numpy as np


@dataclass(slots=True)
class ResidualMLPSpec:
    in_features: int
    out_features: int
    hidden_dim: int
    num_blocks: int


def initialize_residual_mlp(
    in_features: int,
    out_features: int,
    hidden_dim: int = 256,
    num_blocks: int = 3,
    seed: int = 1234,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    params: list[np.ndarray] = []

    def rand(shape: tuple[int, ...], scale: float) -> np.ndarray:
        return rng.normal(loc=0.0, scale=scale, size=shape).astype(np.float64)

    params.append(rand((in_features, hidden_dim), np.sqrt(2.0 / max(in_features, 1))))
    params.append(np.zeros((hidden_dim,), dtype=np.float64))
    for _ in range(num_blocks):
        params.append(rand((hidden_dim, hidden_dim), np.sqrt(2.0 / hidden_dim)))
        params.append(np.zeros((hidden_dim,), dtype=np.float64))
        params.append(rand((hidden_dim, hidden_dim), np.sqrt(2.0 / hidden_dim)))
        params.append(np.zeros((hidden_dim,), dtype=np.float64))
    params.append(rand((hidden_dim, out_features), np.sqrt(2.0 / hidden_dim)))
    params.append(np.zeros((out_features,), dtype=np.float64))
    return params


def _activation(values):
    return anp.tanh(values)


def forward_residual_mlp(params: list[np.ndarray], inputs):
    cursor = 0
    hidden = _activation(anp.dot(inputs, params[cursor]) + params[cursor + 1])
    cursor += 2
    num_blocks = (len(params) - 4) // 4
    for _ in range(num_blocks):
        z1 = anp.dot(hidden, params[cursor]) + params[cursor + 1]
        a1 = _activation(z1)
        z2 = anp.dot(a1, params[cursor + 2]) + params[cursor + 3]
        hidden = _activation(hidden + z2)
        cursor += 4
    outputs = anp.dot(hidden, params[cursor]) + params[cursor + 1]
    return outputs


def spec_from_params(params: list[np.ndarray]) -> ResidualMLPSpec:
    in_features, hidden_dim = params[0].shape
    out_features = params[-2].shape[1]
    num_blocks = (len(params) - 4) // 4
    return ResidualMLPSpec(
        in_features=in_features,
        out_features=out_features,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
    )
