import numpy as np
import pytest

from openlaval.physics import (
    Kstar_max,
    compute_isentropic_pressure_coef,
    compute_turning_angles,
    mach_after_edge,
    mach_from_prandtl,
    mach_to_mstar,
    mass_flow_parameter,
    mstar_to_mach,
    phi_Rstar,
    prandtl_meyer,
    vl_minimum,
    vortex_C,
    vortex_Q,
)

GAMMA = 1.4


def test_prandtl_meyer_monotone():
    assert (
        prandtl_meyer(1.1, GAMMA)
        < prandtl_meyer(2.0, GAMMA)
        < prandtl_meyer(3.0, GAMMA)
    )


@pytest.mark.parametrize("M", [1.1, 1.5, 2.0, 3.0])
def test_prandtl_roundtrip(M):
    nu = prandtl_meyer(M, GAMMA)
    M2 = mach_from_prandtl(nu, GAMMA)
    assert np.isclose(M, M2, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("M", [1.1, 2.0, 3.0])
def test_mstar_roundtrip(M):
    Ms = mach_to_mstar(M, GAMMA)
    M2 = mstar_to_mach(Ms, GAMMA)
    assert np.isclose(M, M2, rtol=1e-6, atol=1e-6)


def test_mach_after_edge_increases_nu():
    M1 = 2.0
    dnu = 5.0
    M2 = mach_after_edge(M1, dnu, GAMMA)
    assert prandtl_meyer(M2, GAMMA) > prandtl_meyer(M1, GAMMA)


def test_phi_Rstar_varies_with_Rs():
    phi_lo = phi_Rstar(0.92, GAMMA)
    phi_mid = phi_Rstar(0.98, GAMMA)
    phi_hi = phi_Rstar(1.04, GAMMA)

    assert np.isfinite(phi_lo)
    assert np.isfinite(phi_mid)
    assert np.isfinite(phi_hi)

    assert not np.isclose(phi_lo, phi_mid)
    assert not np.isclose(phi_mid, phi_hi)


def test_Kstar_max_valid_range():
    Mlstar = 0.65
    Mustar = 0.95
    Kmax = Kstar_max(Mlstar, Mustar, GAMMA)
    assert np.isfinite(Kmax)
    assert 0 < Kmax < 1


def test_vortex_Q_positive():
    Mlstar = 0.65
    Mustar = 0.95
    Q = vortex_Q(Mlstar, Mustar, GAMMA)
    assert np.isfinite(Q)
    assert Q > 0


def test_vortex_C_between_zero_and_one():
    Mlstar = 0.65
    Mustar = 0.95
    C = vortex_C(Mlstar, Mustar, GAMMA)
    assert np.isfinite(C)
    assert 0 <= C < 1


def test_vl_minimum_is_repeatable():
    nu1 = vl_minimum(GAMMA)
    nu2 = vl_minimum(GAMMA)

    assert np.isfinite(nu1)
    assert np.isclose(nu1, nu2, atol=1e-12)


def test_mass_flow_parameter_maximum_at_mach_one():
    q_sub = mass_flow_parameter(0.5, GAMMA)
    q_star = mass_flow_parameter(1.0, GAMMA)
    q_sup = mass_flow_parameter(2.0, GAMMA)

    assert q_star > q_sub
    assert q_star > q_sup


def test_compute_turning_angles():
    beta_in = 60.0
    M_in = 1.8
    M_out = 2.2
    turning = compute_turning_angles(beta_in, M_in, M_out, GAMMA)

    assert "nu_in" in turning
    assert "nu_out" in turning
    assert "delta_nu" in turning
    assert "total_turning" in turning

    assert turning["delta_nu"] > 0
    assert turning["total_turning"] > beta_in


def test_compute_isentropic_pressure_coef_identity():
    cp = compute_isentropic_pressure_coef(
        M=2.0,
        M_ref=2.0,
        gamma=GAMMA,
    )

    assert np.isclose(cp, 0.0, atol=1e-12)


def test_compute_isentropic_pressure_coef_acceleration():
    cp = compute_isentropic_pressure_coef(
        M=2.2,
        M_ref=1.8,
        gamma=GAMMA,
    )

    assert cp < 0.0


def test_prandtl_meyer_reference():
    assert np.isclose(
        prandtl_meyer(2.0, GAMMA),
        26.3797608,
        atol=1e-5,
    )
