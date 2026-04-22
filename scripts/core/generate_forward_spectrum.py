from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from forward import (
    BiasGrid,
    DirectionalSpread,
    FitLayerSpectrumRequest,
    SourceRound2SpectrumRequest,
    TransportControls,
    generate_spectrum_from_fit_layer,
    generate_spectrum_from_source_round2,
    generate_spread_spectrum_from_fit_layer,
    generate_spread_spectrum_from_source_round2,
    list_directional_modes,
    replace_direction_mode,
)


def _parse_control(item: str) -> tuple[str, float]:
    if "=" not in item:
        raise argparse.ArgumentTypeError(f"Expected CHANNEL=VALUE, got {item!r}.")
    name, value = item.split("=", 1)
    try:
        return name, float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid control value in {item!r}.") from exc


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    direction_choices = tuple(mode.name for mode in list_directional_modes())
    parser.add_argument(
        "--direction-mode",
        choices=direction_choices,
        default=None,
        help="Optional named in-plane high-symmetry mode. Overrides --interface-angle when provided.",
    )
    parser.add_argument(
        "--spread-half-width",
        type=float,
        default=0.0,
        help="Optional narrow directional-spread half width in radians. Requires --direction-mode when nonzero.",
    )
    parser.add_argument(
        "--spread-num-samples",
        type=int,
        default=5,
        help="Odd number of uniformly weighted samples for a nonzero directional spread.",
    )
    parser.add_argument("--interface-angle", type=float, default=0.0)
    parser.add_argument("--barrier-z", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--nk", type=int, default=41)
    parser.add_argument("--bias-min", type=float, default=-40.0)
    parser.add_argument("--bias-max", type=float, default=40.0)
    parser.add_argument("--num-bias", type=int, default=201)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "core" / "forward_interface" / "forward_spectrum.json",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AR spectra through the stable forward interface.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    fit_parser = subparsers.add_parser("fit-layer", help="Generate from Task-H fit-layer controls.")
    fit_parser.add_argument(
        "--control",
        action="append",
        default=[],
        type=_parse_control,
        help="Real fit-layer control as CHANNEL=VALUE in meV. Repeat as needed.",
    )
    fit_parser.add_argument(
        "--control-mode",
        choices=("delta_from_baseline_meV", "absolute_meV"),
        default="delta_from_baseline_meV",
    )
    fit_parser.add_argument("--allow-weak-delta-zx-s", action="store_true")
    _add_common_args(fit_parser)

    source_parser = subparsers.add_parser("source-round2", help="Generate from a Luo sample projected to round-2.")
    source_group = source_parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--sample-id", type=str)
    source_group.add_argument("--sample-index", type=int)
    _add_common_args(source_parser)
    return parser.parse_args()


def _transport_from_args(args: argparse.Namespace) -> TransportControls:
    transport = TransportControls(
        interface_angle=float(args.interface_angle),
        barrier_z=float(args.barrier_z),
        gamma=float(args.gamma),
        temperature_kelvin=float(args.temperature),
        nk=int(args.nk),
    )
    if args.direction_mode is not None:
        transport = replace_direction_mode(transport, args.direction_mode)
    return transport


def _bias_grid_from_args(args: argparse.Namespace) -> BiasGrid:
    return BiasGrid(
        bias_min_mev=float(args.bias_min),
        bias_max_mev=float(args.bias_max),
        num_bias=int(args.num_bias),
    )


def main() -> None:
    args = parse_args()
    transport = _transport_from_args(args)
    bias_grid = _bias_grid_from_args(args)
    label = args.label or args.mode
    spread = None
    spread_half_width = float(args.spread_half_width)
    if spread_half_width < 0.0:
        raise SystemExit("--spread-half-width must be non-negative.")
    if spread_half_width > 0.0:
        if args.direction_mode is None:
            raise SystemExit("--spread-half-width requires --direction-mode so the spread has a named central mode.")
        spread = DirectionalSpread(
            direction_mode=str(args.direction_mode),
            half_width=spread_half_width,
            num_samples=int(args.spread_num_samples),
        )
    if args.mode == "fit-layer":
        request = FitLayerSpectrumRequest(
            pairing_controls=dict(args.control),
            pairing_control_mode=args.control_mode,
            allow_weak_delta_zx_s=bool(args.allow_weak_delta_zx_s),
            transport=transport,
            bias_grid=bias_grid,
            request_label=label,
        )
        if spread is None:
            result = generate_spectrum_from_fit_layer(request)
        else:
            result = generate_spread_spectrum_from_fit_layer(request, spread)
    else:
        request = SourceRound2SpectrumRequest(
            source_sample_id=args.sample_id,
            source_sample_index=args.sample_index,
            transport=transport,
            bias_grid=bias_grid,
            request_label=label,
        )
        if spread is None:
            result = generate_spectrum_from_source_round2(request)
        else:
            result = generate_spread_spectrum_from_source_round2(request, spread)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    print(f"Wrote forward spectrum JSON: {args.output_json}")
    print(f"Forward interface version: {result.metadata['forward_interface_version']}")
    print(f"Pairing source: {result.metadata['pairing_source']}")


if __name__ == "__main__":
    main()
