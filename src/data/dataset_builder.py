"""Physics-labeled dataset builder for pairing+transport surrogate training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.parameters import ModelParams, PairingParams
from core.pipeline import SpectroscopyPipeline
from core.presets import base_normal_state_params
from core.simulation_model import SimulationModel
from data.manifest import DatasetManifest, write_manifest
from data.splits import make_train_val_test_split
from source.luo_loader import load_luo_samples
from source.luo_projection import project_luo_samples
from surrogate.raw_space import gauge_fixed_vector_to_pairing_params, pairing_params_to_gauge_fixed_vector


@dataclass(slots=True)
class DatasetBuildConfig:
    scale: str = "smoke"
    num_samples: int = 600
    seed: int = 1234
    nk: int = 41
    bias_max_meV: float = 40.0
    num_bias: int = 601
    interface_angle: float = 0.0
    temperature_kelvin: float = 3.0
    barrier_z_range: tuple[float, float] = (0.1, 1.5)
    gamma_range: tuple[float, float] = (0.2, 2.0)


def _sample_transport(rng: np.random.Generator, config: DatasetBuildConfig) -> tuple[float, float]:
    barrier_z = float(rng.uniform(*config.barrier_z_range))
    gamma = float(rng.uniform(*config.gamma_range))
    return barrier_z, gamma


def _pairing_from_vector(vector: np.ndarray) -> PairingParams:
    return gauge_fixed_vector_to_pairing_params(np.asarray(vector, dtype=np.float64))


def _perturb_pairing(anchor: PairingParams, rng: np.random.Generator) -> PairingParams:
    vector = pairing_params_to_gauge_fixed_vector(anchor)
    perturbed = np.array(vector, copy=True)
    perturbed[1:] += rng.normal(loc=0.0, scale=0.08, size=perturbed[1:].shape) * np.maximum(np.abs(perturbed[1:]), 5.0)
    perturbed[0] = vector[0]
    if perturbed[1] < 0.0:
        perturbed[1] = abs(perturbed[1])
    return _pairing_from_vector(perturbed)


def _choose_anchor(anchors: list, rng: np.random.Generator):
    return anchors[int(rng.integers(0, len(anchors)))]


def build_pairing_transport_dataset(
    output_dir: Path,
    config: DatasetBuildConfig,
) -> tuple[Path, Path]:
    rng = np.random.default_rng(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_samples = load_luo_samples()
    anchors = [sample for sample in project_luo_samples(raw_samples) if sample.projected_pairing_params is not None]
    bias = np.linspace(-config.bias_max_meV, config.bias_max_meV, config.num_bias, dtype=np.float64)

    features: list[np.ndarray] = []
    spectra: list[np.ndarray] = []
    sample_ids: list[str] = []
    categories: list[str] = []

    model = SimulationModel(
        params=ModelParams(normal_state=base_normal_state_params(), pairing=anchors[0].projected_pairing_params),
        name="dataset_builder_base_model",
    )
    pipeline = SpectroscopyPipeline(model=model)

    category_order = ("A_luo_anchor", "B_local_perturbation", "C_transport_scan")
    target_prob = np.asarray([0.25, 0.4, 0.35], dtype=np.float64)

    attempts = 0
    while len(features) < config.num_samples:
        attempts += 1
        category = category_order[int(rng.choice(len(category_order), p=target_prob))]
        anchor = _choose_anchor(anchors, rng)
        barrier_z, gamma = _sample_transport(rng, config)
        pairing = anchor.projected_pairing_params
        if category == "B_local_perturbation":
            pairing = _perturb_pairing(pairing, rng)
        elif category == "C_transport_scan":
            barrier_grid = np.linspace(*config.barrier_z_range, 12)
            gamma_grid = np.linspace(*config.gamma_range, 12)
            barrier_z = float(barrier_grid[int(rng.integers(0, len(barrier_grid)))])
            gamma = float(gamma_grid[int(rng.integers(0, len(gamma_grid)))])

        pipeline.model = SimulationModel(
            params=ModelParams(normal_state=base_normal_state_params(), pairing=pairing),
            name=f"dataset_model_{len(features)}",
        )
        try:
            result = pipeline.compute_multichannel_btk_conductance(
                interface_angle=config.interface_angle,
                bias=bias,
                barrier_z=barrier_z,
                broadening_gamma=gamma,
                temperature=config.temperature_kelvin,
                nk=config.nk,
            )
        except ValueError:
            if attempts > config.num_samples * 20:
                raise
            continue

        feature = np.concatenate(
            [
                pairing_params_to_gauge_fixed_vector(pairing),
                np.asarray([barrier_z, gamma], dtype=np.float64),
            ]
        )
        features.append(feature)
        spectra.append(np.asarray(result.conductance, dtype=np.float64))
        sample_ids.append(anchor.sample_id)
        categories.append(category)

    features_array = np.asarray(features, dtype=np.float64)
    spectra_array = np.asarray(spectra, dtype=np.float64)
    sample_ids_array = np.asarray(sample_ids, dtype="U128")
    categories_array = np.asarray(categories, dtype="U32")

    train_idx, val_idx, test_idx = make_train_val_test_split(len(features_array), seed=config.seed)
    dataset_path = output_dir / f"pairing_transport_dataset_{config.scale}.npz"
    np.savez_compressed(
        dataset_path,
        features=features_array,
        spectra=spectra_array,
        bias=bias,
        sample_ids=sample_ids_array,
        categories=categories_array,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )

    manifest = DatasetManifest(
        dataset_name="pairing_transport_dataset",
        scale=config.scale,
        num_samples=int(len(features_array)),
        bias_max_meV=float(config.bias_max_meV),
        num_bias=int(config.num_bias),
        nk=int(config.nk),
        interface_angle=float(config.interface_angle),
        temperature_kelvin=float(config.temperature_kelvin),
        seed=int(config.seed),
        dataset_path=str(dataset_path),
        categories={name: int(np.count_nonzero(categories_array == name)) for name in category_order},
    )
    manifest_path = output_dir / f"pairing_transport_dataset_{config.scale}_manifest.json"
    write_manifest(manifest, manifest_path)
    return dataset_path, manifest_path

