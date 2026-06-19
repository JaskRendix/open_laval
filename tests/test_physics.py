import numpy as np
import pytest

from openlaval.physics import (
    Kstar_max,
    mach_after_edge,
    mach_from_prandtl,
    mach_to_mstar,
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


def test_phi_Rstar_valid_domain():
    # Valid Rs interval for gamma=1.4 is approx [0.913, 1.049]
    Rs_values = np.linspace(0.92, 1.04, 5)
    for Rs in Rs_values:
        phi = phi_Rstar(Rs, GAMMA)
        assert np.isfinite(phi)


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


def test_vl_minimum_reasonable_range():
    nu_min = vl_minimum(GAMMA)
    nu_max = np.degrees(0.5 * np.pi * (np.sqrt((GAMMA + 1) / (GAMMA - 1)) - 1))
    assert np.isfinite(nu_min)
    assert 0 <= nu_min < nu_max
