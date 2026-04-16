from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from source.luo_loader import DEFAULT_LUO_CACHE_DIR, ensure_luo_repo, inspect_luo_files, load_luo_samples
from source.luo_projection import project_luo_sample_to_pairing


def _markdown_table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return ""
    header = "| " + " | ".join(rows[0].keys()) + " |"
    sep = "| " + " | ".join(["---"] * len(rows[0])) + " |"
    body = ["| " + " | ".join(str(value) for value in row.values()) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def main() -> None:
    repo_dir = ensure_luo_repo(PROJECT_ROOT / DEFAULT_LUO_CACHE_DIR)
    inspection = inspect_luo_files(repo_dir)
    samples = load_luo_samples(repo_dir)
    projected = project_luo_sample_to_pairing(samples[0])

    docs_path = PROJECT_ROOT / "docs" / "luo_source_map_round1.md"
    summary_path = PROJECT_ROOT / "outputs" / "source" / "luo_source_inspection_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    file_rows = []
    for item in inspection:
        fields = ", ".join(sorted(item["fields"].keys())) if item["fields"] else "-"
        file_rows.append(
            {
                "path": item["path"],
                "file_type": item["file_type"],
                "semantics": item["semantics"],
                "fields": fields,
            }
        )

    projection_rows = [
        {
            "field": name,
            "mode": record.mode.value,
            "source_expression": record.source_expression,
            "note": record.note,
        }
        for name, record in projected.projection_provenance.items()
    ]

    docs_text = "\n".join(
        [
            "# Luo Source Map Round 1",
            "",
            "## Source",
            "",
            "- Repository: `https://github.com/ZhihuiLuo/RMFT_Ni327`",
            f"- Local inspected path: `{repo_dir}`",
            "- Round-1 bridge assumption: Luo source energies are stored in eV and are converted to meV before mapping into local `PairingParams`.",
            "",
            "## Files",
            "",
            _markdown_table(file_rows),
            "",
            "## Usable Fields",
            "",
            "- `Pms` / `pms`: RMFT pairing observables with six channel slices interpreted as `(chi_x, chi_y, chi_z, delta_x, delta_y, delta_z)`.",
            "- `Mu` / `mu`: source chemical potentials. `Mu2.npy[0] * 1000` matches the migrated baseline `mu_diag`, so these arrays are treated as eV-scale source fields.",
            "- `N` / `n`: orbital occupations or densities.",
            "- sweep coordinates such as `pr`, `Tr`, `pxr`, `pzr`, `Jr`, and file-level metadata such as `Js`, `alpha`, `JH`, `eps`, `Nm`.",
            "",
            "## Not Directly Usable Fields",
            "",
            "- No direct local analogue was identified for the round-1 `eta_zx_d` channel.",
            "- No direct local analogue was identified for the round-1 `eta_x_perp` channel.",
            "- PDF figures are documentation artifacts rather than machine-readable source data.",
            "",
            "## Projection Assumptions",
            "",
            _markdown_table(projection_rows),
            "",
            "## Example Projected Sample",
            "",
            f"- Sample id: `{projected.sample_id}`",
            f"- Source file: `{projected.source_file.name}`",
            f"- Coordinates: `{projected.coordinates}`",
            f"- Projected pairing params: `{projected.projected_pairing_params}`",
        ]
    )
    docs_path.write_text(docs_text, encoding="utf-8")
    summary_path.write_text(json.dumps(inspection, indent=2), encoding="utf-8")

    print(f"Wrote {docs_path}")
    print(f"Wrote {summary_path}")
    print(f"Loaded {len(samples)} Luo samples")


if __name__ == "__main__":
    main()
