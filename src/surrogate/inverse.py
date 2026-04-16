"""Surrogate-assisted inverse demo utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from core.parameters import ModelParams
from core.pipeline import SpectroscopyPipeline
from core.presets import base_normal_state_params
from core.simulation_model import SimulationModel
from surrogate.evaluate import load_checkpoint, predict_from_checkpoint
from surrogate.raw_space import gauge_fixed_vector_to_pairing_params
from surrogate.train import load_dataset


@dataclass(slots=True)
class InverseCandidate:
    feature: np.ndarray
    surrogate_mse: float
    physics_mse: float | None = None


def run_inverse_demo(
    dataset_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    top_k: int = 5,
    nk: int = 41,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_dataset(dataset_path)
    checkpoint = load_checkpoint(checkpoint_path)

    features = np.asarray(data["features"], dtype=np.float32)
    spectra = np.asarray(data["spectra"], dtype=np.float32)
    bias = np.asarray(data["bias"], dtype=np.float32)
    test_idx = np.asarray(data["test_idx"], dtype=np.int64)
    target_index = int(test_idx[0])
    target = spectra[target_index]

    candidate_prediction = predict_from_checkpoint(checkpoint, features)
    surrogate_errors = np.mean((candidate_prediction - target[None, :]) ** 2, axis=1)
    order = np.argsort(surrogate_errors)[:top_k]
    candidates = [InverseCandidate(feature=features[idx], surrogate_mse=float(surrogate_errors[idx])) for idx in order]

    pipeline = SpectroscopyPipeline(
        model=SimulationModel(
            params=ModelParams(normal_state=base_normal_state_params(), pairing=gauge_fixed_vector_to_pairing_params(features[0][:12])),
            name="inverse_demo_model",
        )
    )

    for candidate in candidates:
        pairing = gauge_fixed_vector_to_pairing_params(candidate.feature[:12])
        barrier_z = float(candidate.feature[12])
        gamma = float(candidate.feature[13])
        pipeline.model = SimulationModel(
            params=ModelParams(normal_state=base_normal_state_params(), pairing=pairing),
            name="inverse_candidate_model",
        )
        result = pipeline.compute_multichannel_btk_conductance(
            interface_angle=0.0,
            bias=bias,
            barrier_z=barrier_z,
            broadening_gamma=gamma,
            temperature=3.0,
            nk=nk,
        )
        candidate.physics_mse = float(np.mean((np.asarray(result.conductance) - target) ** 2))

    candidates.sort(key=lambda item: float(item.physics_mse))
    best = candidates[0]
    best_prediction = predict_from_checkpoint(checkpoint, best.feature[None, :])[0]
    best_pairing = gauge_fixed_vector_to_pairing_params(best.feature[:12])
    pipeline.model = SimulationModel(
        params=ModelParams(normal_state=base_normal_state_params(), pairing=best_pairing),
        name="inverse_best_model",
    )
    best_physics = pipeline.compute_multichannel_btk_conductance(
        interface_angle=0.0,
        bias=bias,
        barrier_z=float(best.feature[12]),
        broadening_gamma=float(best.feature[13]),
        temperature=3.0,
        nk=nk,
    )

    figure, axis = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    axis.plot(bias, target, label="target", linewidth=1.8)
    axis.plot(bias, best_prediction, label="surrogate best", linewidth=1.4)
    axis.plot(bias, best_physics.conductance, label="physics recheck", linewidth=1.4)
    axis.set_xlabel("Bias (meV)")
    axis.set_ylabel("Conductance")
    axis.set_title("Surrogate-assisted inverse demo")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.savefig(output_dir / "top_candidate_comparison.png", dpi=160)
    plt.close(figure)

    residual = np.asarray(best_physics.conductance) - target
    figure, axis = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
    axis.plot(bias, residual, linewidth=1.4, color="tab:red")
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_xlabel("Bias (meV)")
    axis.set_ylabel("Residual")
    axis.set_title("Best Candidate Residual")
    axis.grid(alpha=0.2)
    figure.savefig(output_dir / "top_candidate_residual.png", dpi=160)
    plt.close(figure)

    report = {
        "target_index": target_index,
        "candidate_cluster": [
            {
                "rank": rank + 1,
                "surrogate_mse": float(candidate.surrogate_mse),
                "physics_mse": float(candidate.physics_mse),
                "barrier_z": float(candidate.feature[12]),
                "gamma": float(candidate.feature[13]),
                "gauge_index": int(round(candidate.feature[0])),
            }
            for rank, candidate in enumerate(candidates)
        ],
        "output_contract": "top-K candidate cluster, not a unique true parameter point",
    }
    report_path = output_dir / "inverse_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path

