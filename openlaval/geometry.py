from __future__ import annotations

import numpy as np
from scipy import optimize
from scipy.interpolate import interp1d

from .characteristics import rotate, solve_Rstar_intersection
from .physics import mach_after_edge, mach_from_prandtl, prandtl_meyer


def circular_arc(R: float, alpha_start: float, alpha_end: float, step: float = 0.1):
    """
    Generate a circular arc of radius R between angles α_start → α_end (deg).

    Handles both increasing and decreasing angle ranges.
    """
    if np.isclose(alpha_start, alpha_end):
        alpha = np.array([alpha_start], dtype=float)
    elif alpha_end > alpha_start:
        alpha = np.arange(alpha_start, alpha_end + step, abs(step))
    else:
        alpha = np.arange(alpha_start, alpha_end - step, -abs(step))

    x = R * np.cos(np.radians(alpha))
    y = R * np.sin(np.radians(alpha))
    return x, y


def lower_concave_transition(
    R0: float,
    vmin: float,
    vmax: float,
    gamma: float,
    rotate_angle: float,
    dv: float = 0.5,
):
    """
    Generate the lower-surface concave transition arc.

    Args:
        R0: initial R*
        vmin: lower Prandtl–Meyer angle (deg)
        vmax: inlet/outlet Prandtl–Meyer angle (deg)
        gamma: specific heat ratio
        rotate_angle: rotation into blade coordinates
        dv: step in Prandtl–Meyer angle

    Returns:
        (x[], y[]) arrays
    """
    X, Y = [], []
    k = np.arange(0, int((vmax - vmin) / dv) + 1)
    v_k = vmax - k * dv

    for v in v_k:
        try:
            Rstar, x0, y0 = solve_Rstar_intersection(-vmin, v, gamma, phi_const=0.0)
        except RuntimeError:
            continue

        fai = np.radians(vmax - vmin - (vmax - v))
        xk = -Rstar * np.sin(fai)
        yk = Rstar * np.cos(fai)

        xr, yr = rotate(xk, yk, rotate_angle)
        X.append(xr)
        Y.append(yr)

    return np.array(X), np.array(Y)


def upper_convex_transition(
    Ru: float, vu: float, beta_in: float, gamma: float, direction: str = "in"
):
    """
    Robust upper-surface convex transition arc.

    Skips unsolvable characteristic intersections instead of crashing.
    """
    X, Y = [], []
    xtmp, ytmp = 0.0, Ru

    # Simple, monotone marching in v
    dv = 1.0  # deg

    if direction == "in":
        sign = +1
    else:
        sign = -1

    v_values = np.arange(vu, 0.0 - 1e-6, -dv)

    for idx, v in enumerate(v_values):
        # Try to find intersection; stop if impossible
        try:
            Rstar, Xa, Ya = solve_Rstar_intersection(-vu, v, gamma, phi_const=0.0)
        except RuntimeError:
            break

        M = mach_from_prandtl(v, gamma)
        if not np.isfinite(M) or M <= 1.0:
            break

        mu = np.arcsin(1.0 / M)

        n_half = idx * dv / 2.0
        a1 = np.tan(mu + np.radians(n_half))
        b1 = Ya - a1 * Xa
        a2 = np.tan(np.radians(n_half))
        b2 = ytmp - a2 * xtmp

        if np.isclose(a1, a2):
            continue  # nearly parallel, skip

        xtmp = (b2 - b1) / (a1 - a2)
        ytmp = a2 * xtmp + b2

        xr, yr = rotate(sign * xtmp, ytmp, beta_in)
        X.append(xr)
        Y.append(yr)

    return np.array(X), np.array(Y)


def straight_line_to_match(x_target: float, x_ref: float, y_ref: float, beta: float):
    """
    Compute straight-line connector between two points at angle β.
    """
    y_target = (
        np.tan(np.radians(beta)) * x_target + y_ref - np.tan(np.radians(beta)) * x_ref
    )
    return np.array([x_target, x_ref]), np.array([y_target, y_ref])


def interpolate_contour(
    x_lower: np.ndarray,
    y_lower: np.ndarray,
    x_upper: np.ndarray,
    y_upper: np.ndarray,
    n: int,
):
    """
    Interpolate lower and upper blade curves to a uniform x-grid.
    Cleans NaNs, duplicates, and non-monotone x.
    """

    def clean_curve(x, y):
        # Remove NaNs
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]

        if x.size < 2:
            return x, y

        # Sort by x
        idx = np.argsort(x)
        x, y = x[idx], y[idx]

        # Remove duplicate x
        dx = np.diff(x)
        good = np.hstack(([True], dx > 1e-9))
        x, y = x[good], y[good]

        return x, y

    x_lower, y_lower = clean_curve(x_lower, y_lower)
    x_upper, y_upper = clean_curve(x_upper, y_upper)

    # If either curve is too short, bail out with NaNs
    if x_lower.size < 2 or x_upper.size < 2:
        x = np.linspace(0.0, 1.0, n)
        return x, np.full_like(x, np.nan), np.full_like(x, np.nan)

    x_min = max(x_lower.min(), x_upper.min())
    x_max = min(x_lower.max(), x_upper.max())

    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        x = np.linspace(0.0, 1.0, n)
        return x, np.full_like(x, np.nan), np.full_like(x, np.nan)

    x = np.linspace(x_min, x_max, n)

    f_lower = interp1d(x_lower, y_lower, fill_value="extrapolate")
    f_upper = interp1d(x_upper, y_upper, fill_value="extrapolate")

    return x, f_lower(x), f_upper(x)


def leading_edge_adjust(
    xu: float,
    yu: float,
    xl: float,
    yl: float,
    beta_in: float,
    delta: float,
    offset: float,
    shift: float,
):
    """
    Adjust leading-edge geometry by modifying inlet angle and offset.

    Args:
        xu, yu: upper-surface inlet point
        xl, yl: lower-surface inlet point
        beta_in: inlet flow angle (deg)
        delta: increment of edge angle (deg)
        offset: offset ratio
        shift: vertical shift of lower surface

    Returns:
        (x_edge, y_edge)
    """
    t = shift * offset

    def f(x):
        f1 = np.tan(np.radians(beta_in)) * x + yu - np.tan(np.radians(beta_in)) * xu - t
        f2 = (
            np.tan(np.radians(beta_in + delta)) * x
            + yl
            - np.tan(np.radians(beta_in + delta)) * xl
            + shift
        )
        return f1 - f2

    sol = optimize.root(f, -1.0)
    x_edge = sol.x[0]
    y_edge = (
        np.tan(np.radians(beta_in + delta)) * x_edge
        + yl
        - np.tan(np.radians(beta_in + delta)) * xl
        + shift
    )

    return x_edge, y_edge
