"""Small baseline presets for the unified phenomenology framework."""

from __future__ import annotations

from .parameters import ModelParams, NormalStateParams, PairingParams, PhysicalPairingChannels


BASE_MU_DIAG: tuple[float, float, float, float] = (
    1.2045798275923976,
    1.4929398332585742,
    1.2045801819027133,
    1.49294117770948,
)


def base_normal_state_params() -> NormalStateParams:
    """Return the repository-local fixed baseline normal-state preset.

    The Hamiltonian form is the single formal analytic 4x4 bilayer/two-orbital
    ansatz used throughout the repository. The baseline chemical-potential
    diagonal is fixed locally inside the repository.
    """

    return NormalStateParams(
        family="base",
        e1=409.0,
        e2=776.0,
        tx1=-110.0,
        tx2=-483.0,
        txy1=-17.0,
        txy2=69.0,
        vx=239.0,
        v1=-635.0,
        v2=5.0,
        vxz=-34.0,
        mu_diag=BASE_MU_DIAG,
    )


def base_pairing_params() -> PairingParams:
    """Return the repository-local fixed baseline pairing preset.

    The numerical values are stored as the repository's single formal baseline
    pairing preset rather than reconstructed from any external provenance path.
    """

    return PairingParams(
        eta_z_s=0.0 + 0.0j,
        eta_z_perp=-12.63182175722998 + 1.2811826512138103j,
        eta_x_s=-0.26763071583212805 + 0.02714444223421438j,
        eta_x_d=-2.0731608573908547e-05 - 9.873133906390484e-06j,
        eta_zx_d=0.0 + 0.0j,
        eta_x_perp=0.0 + 0.0j,
    )


def compatibility_physical_pairing_channels() -> PhysicalPairingChannels:
    """Return the legacy baseline translated into the round-2 channel basis.

    This is a compatibility baseline rather than a newly refit round-2 source
    anchor. It stays available for explicit round-1 compatibility workflows.
    """

    legacy = base_pairing_params()
    return PhysicalPairingChannels(
        delta_zz_s=legacy.eta_z_s,
        delta_zz_d=0.0 + 0.0j,
        delta_xx_s=legacy.eta_x_s,
        delta_xx_d=legacy.eta_x_d,
        delta_zx_s=0.0 + 0.0j,
        delta_zx_d=legacy.eta_zx_d,
        delta_perp_z=legacy.eta_z_perp,
        delta_perp_x=legacy.eta_x_perp,
    )


def base_physical_pairing_channels() -> PhysicalPairingChannels:
    """Return the formal Stage-3 round-2 truth-layer baseline.

    The values are fixed from the median of the low-temperature charge-balanced
    Luo temperature-sweep cluster selected in the Stage-3 baseline audit
    (`temperature_eV <= 1e-3`, `p≈0`, first 8 samples), then passed through the
    default Task-C weak-channel freeze convention for ``delta_zx_s``.
    """

    return PhysicalPairingChannels(
        delta_zz_s=43.47120957876885 - 5.990118039475453e-16j,
        delta_zz_d=-6.466862302353316e-08 + 1.2953226292173955e-07j,
        delta_xx_s=-1.7820360737854513 + 1.2994633925908852e-12j,
        delta_xx_d=-1.3049740214993564e-07 + 2.5302036891064664e-07j,
        delta_zx_s=0.0 + 0.0j,
        delta_zx_d=-3.5075801360800885 + 4.1074084233644336e-14j,
        delta_perp_z=-63.513372199351885 + 4.1920631483844906e-13j,
        delta_perp_x=-10.177855352139929 - 1.7339087624688947e-14j,
    )


def base_model_params() -> ModelParams:
    """Return the repository-local fixed baseline model parameter set."""

    return ModelParams(
        normal_state=base_normal_state_params(),
        pairing=base_pairing_params(),
    )
