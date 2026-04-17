"""Shared projection-metric definitions for source reconstruction diagnostics."""

from __future__ import annotations

import numpy as np


def fro_norm(matrix: np.ndarray) -> float:
    """Return the Frobenius norm as a plain Python float."""

    return float(np.linalg.norm(np.asarray(matrix, dtype=np.complex128), ord="fro"))


def build_projection_metric_bundle(
    source_x: np.ndarray,
    source_y: np.ndarray,
    source_z: np.ndarray,
    recon_x: np.ndarray,
    recon_y: np.ndarray,
    recon_z: np.ndarray,
) -> dict[str, float]:
    """Return the repository-standard reconstruction metrics.

    Stage 3 requires a single retained-ratio / residual / omitted definition
    shared across round-1, round-2, and future diagnostics. The standard is:

    - residual norm: Frobenius norm of ``source - recon``
    - retained ratio: ``1 - residual_norm / source_norm``
    - omitted fraction: ``residual_norm / source_norm``
    """

    source_norm_x = fro_norm(source_x)
    source_norm_y = fro_norm(source_y)
    source_norm_z = fro_norm(source_z)
    recon_norm_x = fro_norm(recon_x)
    recon_norm_y = fro_norm(recon_y)
    recon_norm_z = fro_norm(recon_z)
    residual_norm_x = fro_norm(np.asarray(source_x, dtype=np.complex128) - np.asarray(recon_x, dtype=np.complex128))
    residual_norm_y = fro_norm(np.asarray(source_y, dtype=np.complex128) - np.asarray(recon_y, dtype=np.complex128))
    residual_norm_z = fro_norm(np.asarray(source_z, dtype=np.complex128) - np.asarray(recon_z, dtype=np.complex128))

    source_norm_total = float(np.sqrt(source_norm_x**2 + source_norm_y**2 + source_norm_z**2))
    recon_norm_total = float(np.sqrt(recon_norm_x**2 + recon_norm_y**2 + recon_norm_z**2))
    residual_norm_total = float(np.sqrt(residual_norm_x**2 + residual_norm_y**2 + residual_norm_z**2))

    def retained_ratio(source_norm: float, residual_norm: float) -> float:
        if source_norm <= 0.0:
            return 1.0
        return float(1.0 - residual_norm / source_norm)

    def omitted_fraction(source_norm: float, residual_norm: float) -> float:
        if source_norm <= 0.0:
            return 0.0
        return float(residual_norm / source_norm)

    return {
        "source_norm_x": source_norm_x,
        "source_norm_y": source_norm_y,
        "source_norm_z": source_norm_z,
        "recon_norm_x": recon_norm_x,
        "recon_norm_y": recon_norm_y,
        "recon_norm_z": recon_norm_z,
        "residual_norm_x": residual_norm_x,
        "residual_norm_y": residual_norm_y,
        "residual_norm_z": residual_norm_z,
        "source_norm_total": source_norm_total,
        "recon_norm_total": recon_norm_total,
        "residual_norm_total": residual_norm_total,
        "retained_ratio_x": retained_ratio(source_norm_x, residual_norm_x),
        "retained_ratio_y": retained_ratio(source_norm_y, residual_norm_y),
        "retained_ratio_z": retained_ratio(source_norm_z, residual_norm_z),
        "retained_ratio_total": retained_ratio(source_norm_total, residual_norm_total),
        "omitted_fraction_total": omitted_fraction(source_norm_total, residual_norm_total),
    }
