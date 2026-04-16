"""Dataset manifest types."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class DatasetManifest:
    dataset_name: str
    scale: str
    num_samples: int
    bias_max_meV: float
    num_bias: int
    nk: int
    interface_angle: float
    temperature_kelvin: float
    seed: int
    dataset_path: str
    categories: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def write_manifest(manifest: DatasetManifest, output_path: Path) -> None:
    import json

    output_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")

