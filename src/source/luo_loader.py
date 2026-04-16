"""Loader for Luo's RMFT_Ni327 source repository."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .schema import LuoSample

DEFAULT_LUO_REPO_URL = "https://github.com/ZhihuiLuo/RMFT_Ni327.git"
DEFAULT_LUO_CACHE_DIR = Path("outputs/source/cache/RMFT_Ni327")
PAIRING_COMPONENT_NAMES = ("chi_x", "chi_y", "chi_z", "delta_x", "delta_y", "delta_z")


def ensure_luo_repo(local_dir: Path | None = None, repo_url: str = DEFAULT_LUO_REPO_URL) -> Path:
    repo_dir = Path(local_dir or DEFAULT_LUO_CACHE_DIR)
    if repo_dir.exists():
        return repo_dir
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", repo_url, str(repo_dir)], check=True)
    return repo_dir


def _file_semantics(path: Path) -> str:
    stem = path.stem
    if stem.startswith("pms_"):
        return "single RMFT sample snapshot"
    if "figT" in stem:
        return "temperature sweep RMFT pairing data"
    if "fig1" in stem:
        return "doping sweep RMFT pairing data"
    if "figJ" in stem:
        return "exchange-coupling sweep RMFT pairing data"
    if stem.startswith("pams_J"):
        return "two-parameter px/pz RMFT phase-space scan"
    if path.suffix == ".npy":
        return "auxiliary chemical-potential baseline table"
    if path.suffix == ".py":
        return "source-side plotting or model helper"
    if path.suffix == ".pdf":
        return "paper figure output"
    return "unclassified source artifact"


def list_luo_files(repo_dir: Path | None = None) -> list[Path]:
    root = ensure_luo_repo(repo_dir)
    return sorted(path for path in root.iterdir() if path.is_file())


def inspect_luo_files(repo_dir: Path | None = None) -> list[dict[str, Any]]:
    root = ensure_luo_repo(repo_dir)
    rows: list[dict[str, Any]] = []
    for path in list_luo_files(root):
        row: dict[str, Any] = {
            "path": str(path.relative_to(root)),
            "file_type": path.suffix or "<none>",
            "semantics": _file_semantics(path),
        }
        if path.suffix == ".npz":
            payload = np.load(path, allow_pickle=True)
            row["fields"] = {
                key: {
                    "shape": tuple(int(v) for v in np.asarray(payload[key]).shape),
                    "dtype": str(np.asarray(payload[key]).dtype),
                }
                for key in payload.files
            }
        elif path.suffix == ".npy":
            payload = np.load(path, allow_pickle=True)
            row["fields"] = {"array": {"shape": tuple(int(v) for v in payload.shape), "dtype": str(payload.dtype)}}
        else:
            row["fields"] = {}
        rows.append(row)
    return rows


def _as_python(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _extract_sample_coordinates(data: Any, index: tuple[int, ...]) -> dict[str, Any]:
    coordinates: dict[str, Any] = {}
    if "pr" in data and len(index) == 1:
        coordinates["pr"] = float(data["pr"][index[0]])
    if "Tr" in data and len(index) == 1:
        coordinates["temperature_eV"] = float(data["Tr"][index[0]])
    if "pxr" in data and len(index) == 2:
        coordinates["pxr"] = float(data["pxr"][index[0]])
    if "pzr" in data and len(index) == 2:
        coordinates["pzr"] = float(data["pzr"][index[1]])
    if "Jr" in data and len(index) == 1:
        coordinates["Jr"] = _as_python(data["Jr"][index[0]])
    if "p" in data:
        coordinates["p"] = _as_python(data["p"])
    return coordinates


def _build_sample(
    file_path: Path,
    sample_kind: str,
    index: tuple[int, ...],
    pairing_tensor: np.ndarray,
    chemical_potential: np.ndarray | None,
    density: np.ndarray | None,
    file_metadata: dict[str, Any],
    coordinates: dict[str, Any],
) -> LuoSample:
    pairing_array = np.asarray(pairing_tensor, dtype=np.complex128)
    observables = {
        name: np.asarray(pairing_array[position], dtype=np.complex128)
        for position, name in enumerate(PAIRING_COMPONENT_NAMES)
    }
    if density is not None:
        observables["density"] = np.asarray(density, dtype=np.float64)
    sample_id = f"{file_path.stem}::{'_'.join(str(i) for i in index) if index else '0'}"
    return LuoSample(
        sample_id=sample_id,
        source_name="ZhihuiLuo/RMFT_Ni327",
        source_file=file_path,
        sample_kind=sample_kind,
        coordinates=coordinates,
        source_metadata=file_metadata,
        source_pairing_observables=observables,
        source_chemical_potential=None if chemical_potential is None else np.asarray(chemical_potential, dtype=np.float64),
    )


def load_luo_samples(repo_dir: Path | None = None) -> list[LuoSample]:
    root = ensure_luo_repo(repo_dir)
    samples: list[LuoSample] = []
    for path in sorted(root.glob("*.npz")):
        payload = np.load(path, allow_pickle=True)
        file_metadata = {
            key: _as_python(payload[key])
            for key in payload.files
            if key not in {"Pms", "pms", "Mu", "mu", "N", "n"}
        }

        if "pms" in payload:
            samples.append(
                _build_sample(
                    file_path=path,
                    sample_kind=_file_semantics(path),
                    index=(),
                    pairing_tensor=payload["pms"],
                    chemical_potential=payload["mu"] if "mu" in payload else None,
                    density=payload["n"] if "n" in payload else None,
                    file_metadata=file_metadata,
                    coordinates=_extract_sample_coordinates(payload, ()),
                )
            )
            continue

        if "Pms" not in payload:
            continue

        pairing_grid = np.asarray(payload["Pms"], dtype=np.complex128)
        mu_grid = np.asarray(payload["Mu"], dtype=np.float64) if "Mu" in payload else None
        density_grid = np.asarray(payload["N"], dtype=np.float64) if "N" in payload else None

        if pairing_grid.ndim == 4:
            for i in range(pairing_grid.shape[0]):
                samples.append(
                    _build_sample(
                        file_path=path,
                        sample_kind=_file_semantics(path),
                        index=(i,),
                        pairing_tensor=pairing_grid[i],
                        chemical_potential=None if mu_grid is None else mu_grid[i],
                        density=None if density_grid is None else density_grid[i],
                        file_metadata=file_metadata,
                        coordinates=_extract_sample_coordinates(payload, (i,)),
                    )
                )
        elif pairing_grid.ndim == 5:
            for i in range(pairing_grid.shape[0]):
                for j in range(pairing_grid.shape[1]):
                    samples.append(
                        _build_sample(
                            file_path=path,
                            sample_kind=_file_semantics(path),
                            index=(i, j),
                            pairing_tensor=pairing_grid[i, j],
                            chemical_potential=None if mu_grid is None else mu_grid[i, j],
                            density=None if density_grid is None else density_grid[i, j],
                            file_metadata=file_metadata,
                            coordinates=_extract_sample_coordinates(payload, (i, j)),
                        )
                    )
    return samples

