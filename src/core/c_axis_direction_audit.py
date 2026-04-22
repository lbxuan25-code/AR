"""Task-O audit for true c-axis directional support.

The current model is a 2D in-plane forward truth chain. This audit records why
true c-axis injection is formally unsupported rather than silently mapped onto a
raw in-plane ``interface_angle``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import inspect
import json

from core import bands, bdg, interface_geometry, normal_state, pairing, simulation_model


@dataclass(frozen=True, slots=True)
class CAxisDirectionAuditArtifacts:
    """Generated Task-O c-axis audit artifact paths."""

    summary_path: Path
    capability_matrix_path: Path


def _signature_parameters(function: object) -> list[str]:
    return list(inspect.signature(function).parameters)


def _capability_rows() -> list[dict[str, object]]:
    return [
        {
            "component": "normal_state.h0_matrix",
            "required_for_c_axis": "Hamiltonian depends on out-of-plane momentum or a physically equivalent c-axis coordinate.",
            "current_evidence": f"signature={_signature_parameters(normal_state.h0_matrix)}",
            "current_status": "missing_kz",
            "blocking": True,
        },
        {
            "component": "pairing.delta_matrix",
            "required_for_c_axis": "Pairing tensor can be evaluated on c-axis-injected states.",
            "current_evidence": f"signature={_signature_parameters(pairing.delta_matrix)}",
            "current_status": "missing_kz",
            "blocking": True,
        },
        {
            "component": "bdg.bdg_matrix",
            "required_for_c_axis": "BdG Hamiltonian supports the same out-of-plane momentum structure.",
            "current_evidence": f"signature={_signature_parameters(bdg.bdg_matrix)}",
            "current_status": "missing_kz",
            "blocking": True,
        },
        {
            "component": "simulation_model.SimulationModel",
            "required_for_c_axis": "Public model methods accept a c-axis coordinate or true 3D momentum.",
            "current_evidence": (
                f"build_normal_state={_signature_parameters(simulation_model.SimulationModel.build_normal_state)}, "
                f"build_delta={_signature_parameters(simulation_model.SimulationModel.build_delta)}"
            ),
            "current_status": "2d_kx_ky_only",
            "blocking": True,
        },
        {
            "component": "bands.normal_state_eigensystem_on_kgrid",
            "required_for_c_axis": "Fermi-surface extraction supports a 3D surface or c-axis injection manifold.",
            "current_evidence": f"signature={_signature_parameters(bands.normal_state_eigensystem_on_kgrid)}",
            "current_status": "2d_square_grid_only",
            "blocking": True,
        },
        {
            "component": "interface_geometry.interface_normal",
            "required_for_c_axis": "Interface normal can represent out-of-plane injection, e.g. a 3-vector along z.",
            "current_evidence": "returns (cos(angle), sin(angle)) in the kx-ky plane",
            "current_status": "2d_inplane_normal_only",
            "blocking": True,
        },
        {
            "component": "interface_geometry.estimate_group_velocity",
            "required_for_c_axis": "Velocity matching includes an out-of-plane velocity component.",
            "current_evidence": "finite differences only in kx and ky; returns a 2-component velocity",
            "current_status": "missing_vz",
            "blocking": True,
        },
        {
            "component": "forward.directions",
            "required_for_c_axis": "Public direction registry exposes c-axis only after a validated c-axis path exists.",
            "current_evidence": "canonical modes are limited to inplane_100 and inplane_110",
            "current_status": "c_axis_intentionally_absent",
            "blocking": False,
        },
    ]


def _write_capability_matrix(rows: list[dict[str, object]], path: Path) -> Path:
    fieldnames = [
        "component",
        "required_for_c_axis",
        "current_evidence",
        "current_status",
        "blocking",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})
    return path


def run_c_axis_direction_audit(
    *,
    output_dir: Path = Path("outputs/core/c_axis_direction_audit"),
) -> tuple[dict[str, object], CAxisDirectionAuditArtifacts]:
    """Write the Task-O c-axis unsupported audit record."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _capability_rows()
    missing = [row for row in rows if bool(row["blocking"])]
    summary: dict[str, object] = {
        "task": "Task O",
        "decision": "c_axis_unsupported",
        "is_c_axis_forward_mode_available": False,
        "capability_matrix": rows,
        "blocking_gaps": missing,
        "public_interface_policy": {
            "direction_mode_c_axis": "forbidden",
            "raw_interface_angle": "must not be labeled as c-axis",
            "allowed_named_modes": ["inplane_100", "inplane_110"],
        },
        "required_extension_plan": [
            "Define a microscopic or phenomenological out-of-plane momentum coordinate, e.g. kz or an equivalent c-axis injection manifold.",
            "Extend the normal-state Hamiltonian and pairing matrix so they can be evaluated on that coordinate.",
            "Extract or sample the correct c-axis transport states rather than reusing the 2D kx-ky Fermi contours.",
            "Define c-axis interface normal, velocity projection, and specular/reflected-state construction with an out-of-plane velocity component.",
            "Validate the c-axis path against dedicated reflection diagnostics and spectra before exposing it as a public mode.",
        ],
        "final_verdict": (
            "The current forward truth chain cannot represent true c-axis transport. "
            "c-axis is formally unsupported and must not be emulated by any raw 2D in-plane interface_angle."
        ),
    }

    capability_matrix_path = _write_capability_matrix(rows, output_dir / "capability_matrix.csv")
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return summary, CAxisDirectionAuditArtifacts(
        summary_path=summary_path,
        capability_matrix_path=capability_matrix_path,
    )
