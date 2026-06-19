import numpy as np
import pytest

from openlaval.characteristics import (
    chara_angles,
    chara_xy,
    characteristic_curve,
    rotate,
    solve_Rstar_intersection,
    transition_arc,
)
from openlaval.physics import phi_Rstar

GAMMA = 1.4


def test_chara_angles_basic():
    Rs = 0.95
    v1, v2 = 10.0, 20.0
    phi_const = 0.1

    th1, th2 = chara_angles(Rs, GAMMA, v1, v2, phi_const)

    assert np.isfinite(th1)
    assert np.isfinite(th2)
    assert th1 != th2


@pytest.mark.parametrize("Rs", [0.9, 0.95, 1.0])
def test_chara_angles_monotone_in_v(Rs):
    th1a, th2a = chara_angles(Rs, GAMMA, 5.0, 15.0, 0.0)
    th1b, th2b = chara_angles(Rs, GAMMA, 10.0, 20.0, 0.0)

    assert th1b > th1a
    assert th2b > th2a


def test_chara_xy_radius_preserved():
    Rs = 0.8
    theta = np.pi / 6
    x, y = chara_xy(Rs, theta)
    assert np.isclose(np.hypot(x, y), Rs, atol=1e-12)


def test_chara_xy_symmetry_about_x_axis():
    Rs = 1.0
    x1, y1 = chara_xy(Rs, np.pi / 4)
    x2, y2 = chara_xy(Rs, -np.pi / 4)
    # cos is even, sin is odd
    assert np.isclose(x1, -x2)
    assert np.isclose(y1, y2)


def test_characteristic_curve_shapes():
    x0, y0, x1, y1 = characteristic_curve(5.0, 15.0, GAMMA, 0.0, n=200)
    assert x0.shape == y0.shape == x1.shape == y1.shape == (200,)


def test_characteristic_curve_endpoints_R_equals_1():
    x0, y0, x1, y1 = characteristic_curve(5.0, 15.0, GAMMA, 0.0, n=50)
    assert np.isclose(np.hypot(x0[-1], y0[-1]), 1.0, atol=1e-12)
    assert np.isclose(np.hypot(x1[-1], y1[-1]), 1.0, atol=1e-12)


def test_solve_Rstar_intersection_basic():
    Rstar, x, y = solve_Rstar_intersection(5.0, 15.0, GAMMA, 0.0)
    assert 0 < Rstar < 1
    assert np.isfinite(x)
    assert np.isfinite(y)


def test_solve_Rstar_intersection_consistency():
    # For swapped v1/v2 we still expect a valid intersection, not necessarily same R*
    R1, x1, y1 = solve_Rstar_intersection(5.0, 15.0, GAMMA, 0.0)
    R2, x2, y2 = solve_Rstar_intersection(15.0, 5.0, GAMMA, 0.0)
    assert 0 < R2 < 1
    assert np.isfinite(x2) and np.isfinite(y2)
    # Intersection points should not be wildly different
    assert np.isclose(R1, R2, rtol=0.1)


def test_solve_Rstar_intersection_failure_message():
    # Construct a case where the two characteristic lines never intersect:
    # - v1 and v2 extremely close
    # - phi_const chosen to push both curves to the same side
    v1 = 10.0
    v2 = 10.0001  # nearly identical → no crossing
    phi_const = 5.0  # shifts both curves so x1 - x2 keeps same sign

    with pytest.raises(RuntimeError):
        solve_Rstar_intersection(v1, v2, GAMMA, phi_const)


def test_rotate_identity():
    x, y = rotate(1.0, 2.0, 0.0)
    assert x == 1.0
    assert y == 2.0


def test_rotate_90deg():
    x, y = rotate(1.0, 0.0, 90.0)
    assert np.isclose(x, 0.0, atol=1e-12)
    assert np.isclose(y, 1.0, atol=1e-12)


def test_rotate_full_circle():
    x, y = rotate(1.0, 2.0, 360.0)
    assert np.isclose(x, 1.0)
    assert np.isclose(y, 2.0)


def test_rotate_radius_preserved():
    x0, y0 = 0.3, -0.7
    x1, y1 = rotate(x0, y0, 37.0)
    assert np.isclose(np.hypot(x0, y0), np.hypot(x1, y1))


def test_transition_arc_basic():
    X, Y = transition_arc(
        R0=0.9,
        vmin=5.0,
        vmax=15.0,
        gamma=GAMMA,
        rotate_angle=0.0,
        dv=1.0,
    )
    assert len(X) == len(Y)
    assert len(X) > 0
    assert np.all(np.isfinite(X))
    assert np.all(np.isfinite(Y))


def test_transition_arc_R_bounded():
    X, Y = transition_arc(
        R0=0.9,
        vmin=5.0,
        vmax=15.0,
        gamma=GAMMA,
        rotate_angle=0.0,
        dv=1.0,
    )
    R = np.hypot(X, Y)
    # R* should stay in a reasonable range around R0
    assert np.all(R > 0.5)
    assert np.all(R < 1.5)


def test_transition_arc_rotation_changes_coordinates():
    X0, Y0 = transition_arc(0.9, 5.0, 15.0, GAMMA, 0.0)
    X1, Y1 = transition_arc(0.9, 5.0, 15.0, GAMMA, 45.0)
    assert not np.allclose(X0, X1)
    assert not np.allclose(Y0, Y1)
