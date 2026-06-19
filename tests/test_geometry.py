import numpy as np
import pytest

from openlaval.geometry import (
    circular_arc,
    interpolate_contour,
    leading_edge_adjust,
    lower_concave_transition,
    straight_line_to_match,
    upper_convex_transition,
)


@pytest.mark.parametrize(
    "R, alpha_start, alpha_end",
    [
        (1.0, 0.0, 90.0),  # increasing
        (1.0, 90.0, 0.0),  # decreasing
        (1.0, 45.0, 45.0),  # degenerate single angle
    ],
)
def test_circular_arc_basic(R, alpha_start, alpha_end):
    x, y = circular_arc(R, alpha_start, alpha_end)

    # Always at least one point
    assert x.size >= 1
    assert y.size == x.size

    # All points lie on circle of radius R (within tolerance)
    r = np.sqrt(x**2 + y**2)
    assert np.allclose(r, R, atol=1e-6)

    # No NaNs
    assert np.isfinite(x).all()
    assert np.isfinite(y).all()


def test_circular_arc_direction_symmetry():
    R = 1.0
    x_inc, y_inc = circular_arc(R, 0.0, 90.0)
    x_dec, y_dec = circular_arc(R, 90.0, 0.0)

    # Same set of points, different order
    assert x_inc.size == x_dec.size
    assert np.allclose(np.sort(x_inc), np.sort(x_dec), atol=1e-6)
    assert np.allclose(np.sort(y_inc), np.sort(y_dec), atol=1e-6)


@pytest.mark.parametrize(
    "vmin, vmax",
    [
        (30.0, 40.0),
        (10.0, 30.0),
    ],
)
def test_lower_concave_transition_no_nans(vmin, vmax):
    R0 = 1.0
    gamma = 1.4
    rotate_angle = 60.0

    x, y = lower_concave_transition(R0, vmin, vmax, gamma, rotate_angle)

    assert x.size == y.size
    assert np.isfinite(x).all()
    assert np.isfinite(y).all()

    if x.size > 0:
        # Not all points collapsed
        assert not np.allclose(x, x[0])
        assert not np.allclose(y, y[0])


@pytest.mark.parametrize("direction", ["in", "out"])
def test_upper_convex_transition_no_nans(direction):
    Ru = 1.0
    vu = 40.0
    beta_in = 70.0
    gamma = 1.4

    x, y = upper_convex_transition(Ru, vu, beta_in, gamma, direction=direction)

    assert x.size == y.size
    assert np.isfinite(x).all()
    assert np.isfinite(y).all()

    if x.size > 0:
        assert not np.allclose(x, x[0])
        assert not np.allclose(y, y[0])


def test_straight_line_to_match_geometry():
    x_target = 1.0
    x_ref = 0.0
    y_ref = 0.0
    beta = 45.0

    x, y = straight_line_to_match(x_target, x_ref, y_ref, beta)

    assert x.size == 2
    assert y.size == 2

    m = np.tan(np.radians(beta))
    b = y_ref - m * x_ref
    assert np.allclose(y, m * x + b, atol=1e-6)


def test_interpolate_contour_normal_case():
    x_lower = np.linspace(0.0, 1.0, 5)
    y_lower = x_lower
    x_upper = np.linspace(0.0, 1.0, 5)
    y_upper = 2 * x_upper

    n = 50
    x, yL, yU = interpolate_contour(x_lower, y_lower, x_upper, y_upper, n)

    assert x.size == n
    assert yL.size == n
    assert yU.size == n

    assert np.allclose(yL, x, atol=1e-6)
    assert np.allclose(yU, 2 * x, atol=1e-6)


def test_interpolate_contour_handles_nans_and_duplicates():
    x_lower = np.array([0.0, 0.5, 0.5, 1.0])
    y_lower = np.array([0.0, 0.5, np.nan, 1.0])
    x_upper = np.array([0.0, 1.0])
    y_upper = np.array([1.0, 2.0])

    x, yL, yU = interpolate_contour(x_lower, y_lower, x_upper, y_upper, 20)

    assert np.isfinite(x).all()
    assert np.isfinite(yL).all()
    assert np.isfinite(yU).all()

    dx = np.diff(x)
    assert np.all(dx > 0.0)


def test_interpolate_contour_bails_on_short_curves():
    x_lower = np.array([0.0])
    y_lower = np.array([0.0])
    x_upper = np.array([0.0, 1.0])
    y_upper = np.array([1.0, 2.0])

    x, yL, yU = interpolate_contour(x_lower, y_lower, x_upper, y_upper, 10)

    assert x.size == 10
    assert np.isnan(yL).all()
    assert np.isnan(yU).all()


@pytest.mark.parametrize(
    "beta_in, delta, offset, shift",
    [
        (70.0, 8.0, 0.1, 0.0),
        (70.0, 8.0, 0.3, 0.5),
        (45.0, 5.0, 0.0, 0.0),
    ],
)
def test_leading_edge_adjust_root_convergence(beta_in, delta, offset, shift):
    xu, yu = 0.0, 0.0
    xl, yl = 0.0, 0.0

    x_edge, y_edge = leading_edge_adjust(xu, yu, xl, yl, beta_in, delta, offset, shift)

    assert np.isfinite(x_edge)
    assert np.isfinite(y_edge)

    m1 = np.tan(np.radians(beta_in))
    m2 = np.tan(np.radians(beta_in + delta))

    t = shift * offset

    y1 = m1 * x_edge + yu - m1 * xu - t
    y2 = m2 * x_edge + yl - m2 * xl + shift

    assert np.allclose(y1, y2, atol=1e-6)
    assert np.allclose(y_edge, y2, atol=1e-6)
