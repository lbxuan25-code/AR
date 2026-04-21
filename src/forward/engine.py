"""Callable stable forward AR interface built on the existing physics core."""

from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from core.formal_baseline import AUTHORITATIVE_ROUND2_BASELINE_RECORD, load_authoritative_round2_baseline_record
from core.parameters import ModelParams, PhysicalPairingChannels
from core.pipeline import SpectroscopyPipeline
from core.presets import base_normal_state_params, base_physical_pairing_channels
from core.simulation_model import SimulationModel
from source.luo_loader import load_luo_samples
from source.round2_projection import DEFAULT_ROUND2_PROJECTION_CONFIG, project_luo_sample_to_round2_channels

from .schema import (
    FIT_LAYER_CONTROL_CHANNELS,
    FIT_LAYER_FREE_CHANNELS,
    FIT_LAYER_REGULARIZED_CHANNELS,
    FIT_LAYER_WEAK_CHANNELS,
    FORWARD_INTERFACE_VERSION,
    FORWARD_OUTPUT_SCHEMA_VERSION,
    ROUND2_PAIRING_CONVENTION_ID,
    FitLayerSpectrumRequest,
    ForwardSpectrumResult,
    SourceRound2SpectrumRequest,
    TransportControls,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _git_revision() -> dict[str, object]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=PROJECT_ROOT,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"git_commit": commit, "git_dirty": dirty}
    except Exception:
        return {"git_commit": "unknown", "git_dirty": None}


def _complex_payload(value: complex) -> dict[str, float]:
    return {"re": float(np.real(value)), "im": float(np.imag(value))}


def _channels_payload(channels: PhysicalPairingChannels) -> dict[str, dict[str, float]]:
    return {name: _complex_payload(value) for name, value in channels.to_dict().items()}


def _baseline_metadata() -> dict[str, object]:
    record = load_authoritative_round2_baseline_record()
    return {
        "formal_baseline_record": str(AUTHORITATIVE_ROUND2_BASELINE_RECORD),
        "formal_baseline_role": record.get("record_role", "authoritative_formal_round2_baseline"),
        "formal_baseline_selection_rule": record.get("selection_rule"),
        "weak_channel_policy": record.get("weak_channel_policy"),
    }


def forward_metadata(pairing_source: str, extra: dict[str, object] | None = None) -> dict[str, object]:
    """Return canonical metadata for external dataset provenance."""

    metadata: dict[str, object] = {
        "forward_interface_version": FORWARD_INTERFACE_VERSION,
        "output_schema_version": FORWARD_OUTPUT_SCHEMA_VERSION,
        "pairing_convention_id": ROUND2_PAIRING_CONVENTION_ID,
        "pairing_source": pairing_source,
        "normal_state_policy": "repository_fixed_base_normal_state",
        "normal_state_family": base_normal_state_params().family,
        "truth_layer": "round2_physical_pairing_channels",
        "fit_layer_policy": {
            "free_channels": list(FIT_LAYER_FREE_CHANNELS),
            "strongly_regularized_channels": list(FIT_LAYER_REGULARIZED_CHANNELS),
            "weak_optional_channels": list(FIT_LAYER_WEAK_CHANNELS),
            "default_complex_phase_policy": "fixed_real_controls_in_projection_gauge",
        },
        "projection_config": DEFAULT_ROUND2_PROJECTION_CONFIG.to_dict(),
        **_baseline_metadata(),
        **_git_revision(),
    }
    if extra:
        metadata.update(extra)
    return metadata


def _validate_transport(transport: TransportControls) -> None:
    if transport.nk < 5:
        raise ValueError("transport.nk must be at least 5.")
    if transport.gamma <= 0:
        raise ValueError("transport.gamma must be positive.")
    if transport.temperature_kelvin < 0:
        raise ValueError("temperature_kelvin must be non-negative.")


def _fit_layer_channels(request: FitLayerSpectrumRequest) -> PhysicalPairingChannels:
    if request.pairing_control_mode not in {"delta_from_baseline_meV", "absolute_meV"}:
        raise ValueError(f"Unsupported pairing_control_mode: {request.pairing_control_mode!r}.")

    baseline = base_physical_pairing_channels()
    values = baseline.to_dict()
    unknown = sorted(set(request.pairing_controls) - set(FIT_LAYER_CONTROL_CHANNELS))
    if unknown:
        raise ValueError(f"Unknown fit-layer pairing control(s): {unknown}.")

    weak_value = float(request.pairing_controls.get("delta_zx_s", 0.0))
    if abs(weak_value) > 0.0 and not request.allow_weak_delta_zx_s:
        raise ValueError("delta_zx_s is fixed to zero by default; set allow_weak_delta_zx_s=True to open this branch.")

    for channel_name, real_value in request.pairing_controls.items():
        current = complex(values[channel_name])
        if request.pairing_control_mode == "delta_from_baseline_meV":
            values[channel_name] = complex(float(np.real(current)) + float(real_value), float(np.imag(current)))
        else:
            values[channel_name] = complex(float(real_value), float(np.imag(current)))
    if not request.allow_weak_delta_zx_s:
        values["delta_zx_s"] = 0.0 + 0.0j
    return PhysicalPairingChannels(**values)


def _find_luo_sample(source_sample_id: str | None, source_sample_index: int | None):
    samples = load_luo_samples()
    if source_sample_id is None and source_sample_index is None:
        raise ValueError("Provide either source_sample_id or source_sample_index.")
    if source_sample_id is not None:
        for sample in samples:
            if sample.sample_id == source_sample_id:
                return sample
        raise ValueError(f"No Luo sample found for source_sample_id={source_sample_id!r}.")
    assert source_sample_index is not None
    return samples[int(source_sample_index)]


def _compute_spectrum(
    channels: PhysicalPairingChannels,
    transport: TransportControls,
    bias: np.ndarray,
    pairing_source: str,
    request_kind: str,
    request_payload: dict[str, object],
    extra_metadata: dict[str, object] | None = None,
) -> ForwardSpectrumResult:
    _validate_transport(transport)
    params = ModelParams(normal_state=base_normal_state_params(), pairing=channels)
    pipeline = SpectroscopyPipeline(model=SimulationModel(params=params, name=f"stable_forward::{request_kind}"))
    result = pipeline.compute_multichannel_btk_conductance(
        interface_angle=float(transport.interface_angle),
        bias=np.asarray(bias, dtype=np.float64),
        barrier_z=float(transport.barrier_z),
        broadening_gamma=float(transport.gamma),
        temperature=float(transport.temperature_kelvin),
        nk=int(transport.nk),
    )
    return ForwardSpectrumResult(
        schema_version=FORWARD_OUTPUT_SCHEMA_VERSION,
        request_kind=request_kind,
        request=request_payload,
        metadata=forward_metadata(pairing_source=pairing_source, extra=extra_metadata),
        pairing_channels=_channels_payload(channels),
        bias_mev=[float(value) for value in np.asarray(result.bias, dtype=np.float64)],
        conductance=[float(value) for value in np.asarray(result.conductance, dtype=np.float64)],
        conductance_unbroadened=[float(value) for value in np.asarray(result.conductance_unbroadened, dtype=np.float64)],
        transport_summary={
            "interface_angle": float(result.interface_angle),
            "barrier_z": float(result.barrier_z),
            "gamma": float(result.broadening_gamma),
            "temperature_kelvin": float(result.temperature),
            "nk": int(transport.nk),
            "num_input_channels": int(result.num_input_channels),
            "num_channels": int(result.num_channels),
            "num_filtered_channels": int(result.num_filtered_channels),
            "num_same_band_channels": int(result.num_same_band_channels),
            "num_contours": int(result.num_contours),
            "mean_normal_transparency": float(result.mean_normal_transparency),
            "mean_mismatch_penalty": float(result.mean_mismatch_penalty),
            "approximation": result.approximation,
        },
    )


def generate_spectrum_from_fit_layer(request: FitLayerSpectrumRequest) -> ForwardSpectrumResult:
    """Generate an AR spectrum from Task-H fit-layer controls."""

    channels = _fit_layer_channels(request)
    return _compute_spectrum(
        channels=channels,
        transport=request.transport,
        bias=request.bias_grid.values(),
        pairing_source="task_h_fit_layer_controls",
        request_kind="fit_layer",
        request_payload=request.to_dict(),
        extra_metadata={
            "fit_layer_request_label": request.request_label,
            "pairing_control_mode": request.pairing_control_mode,
            "allow_weak_delta_zx_s": bool(request.allow_weak_delta_zx_s),
        },
    )


def generate_spectrum_from_source_round2(request: SourceRound2SpectrumRequest) -> ForwardSpectrumResult:
    """Generate an AR spectrum from a Luo sample projected into round-2 channels."""

    sample = _find_luo_sample(request.source_sample_id, request.source_sample_index)
    projected = project_luo_sample_to_round2_channels(sample)
    assert projected.projected_physical_channels is not None
    request_payload = request.to_dict()
    request_payload["resolved_source_sample_id"] = sample.sample_id
    return _compute_spectrum(
        channels=projected.projected_physical_channels,
        transport=request.transport,
        bias=request.bias_grid.values(),
        pairing_source="luo_source_default_round2_projection",
        request_kind="source_round2",
        request_payload=request_payload,
        extra_metadata={
            "source_sample_id": sample.sample_id,
            "source_sample_kind": sample.sample_kind,
            "source_coordinates": dict(sample.coordinates),
            "round2_projection_metrics": dict(projected.round2_projection_metrics),
            "round2_projection_metadata": dict(projected.round2_projection_metadata),
        },
    )


def fit_layer_request_with_controls(
    controls: dict[str, float],
    transport: TransportControls | None = None,
    **kwargs: Any,
) -> FitLayerSpectrumRequest:
    """Convenience helper for callers that want a small programmatic request."""

    request = FitLayerSpectrumRequest(pairing_controls=dict(controls), **kwargs)
    return request if transport is None else replace(request, transport=transport)
