from __future__ import annotations

import numpy as np
from scipy import optimize

from .physics import phi_Rstar


def chara_angles(
    Rs: float, gamma: float, v1_deg: float, v2_deg: float, phi_const: float
) -> tuple[float, float]:
    """
    Compute θ1 and θ2 for a given R* and Prandtl–Meyer angles v1, v2.
    """
    phi = phi_Rstar(Rs, gamma)
    v1 = np.radians(v1_deg)
    v2 = np.radians(v2_deg)

    theta1 = (phi - phi_const) + v1
    theta2 = -(phi - phi_const) + v2
    return theta1, theta2


def chara_xy(Rs: float, theta: float) -> tuple[float, float]:
    """
    Convert (R*, θ) to Cartesian coordinates.
    """
    return Rs * np.sin(theta), Rs * np.cos(theta)


def characteristic_curve(
    v1_deg: float, v2_deg: float, gamma: float, phi_const: float, n: int = 1000
):
    """
    Generate two characteristic curves for Prandtl–Meyer angles v1 < v2.

    Returns:
        (x0, y0, x1, y1)
    """
    Rmin = np.sqrt((gamma - 1) / (gamma + 1))
    Rs = np.linspace(Rmin, 1.0, n)

    x0, y0, x1, y1 = [], [], [], []

    for R in Rs:
        th1, th2 = chara_angles(R, gamma, v1_deg, v2_deg, phi_const)
        X0, Y0 = chara_xy(R, th1)
        X1, Y1 = chara_xy(R, th2)
        x0.append(X0)
        y0.append(Y0)
        x1.append(X1)
        y1.append(Y1)

    return np.array(x0), np.array(y0), np.array(x1), np.array(y1)


def solve_Rstar_intersection(
    v1_deg: float, v2_deg: float, gamma: float, phi_const: float
) -> tuple[float, float, float]:
    """
    Robustly solve for R* such that the two characteristic lines intersect.

    Returns:
        (R*, x, y)

    Raises:
        RuntimeError if no valid intersection (no sign change) can be found.
    """
    Rmin = np.sqrt((gamma - 1) / (gamma + 1))
    Rlo = Rmin + 1e-6
    Rhi = 1.0 - 1e-6

    def f(R: float) -> float:
        th1, th2 = chara_angles(R, gamma, v1_deg, v2_deg, phi_const)
        x1, _ = chara_xy(R, th1)
        x2, _ = chara_xy(R, th2)
        return x1 - x2  # intersection when x1 == x2

    # Initial bracket
    fa = f(Rlo)
    fb = f(Rhi)

    # If no sign change, try to find one by scanning
    if not (np.isfinite(fa) and np.isfinite(fb)) or fa * fb > 0:
        Rs = np.linspace(Rlo, Rhi, 200)
        fvals = np.array([f(R) for R in Rs])

        # Find any interval with sign change
        idx = np.where(
            np.isfinite(fvals[:-1] * fvals[1:]) & (fvals[:-1] * fvals[1:] < 0)
        )[0]
        if idx.size == 0:
            raise RuntimeError(
                f"solve_Rstar_intersection: no sign change for v1={v1_deg}, v2={v2_deg}"
            )

        i = idx[0]
        Rlo, Rhi = Rs[i], Rs[i + 1]

    Rstar = optimize.brentq(f, Rlo, Rhi, xtol=1e-8, rtol=1e-8, maxiter=100)
    th1, th2 = chara_angles(Rstar, gamma, v1_deg, v2_deg, phi_const)
    x, y = chara_xy(Rstar, th1)
    return Rstar, x, y


def rotate(x: float, y: float, angle_deg: float) -> tuple[float, float]:
    """
    Rotate (x, y) by angle_deg degrees.
    """
    th = np.radians(angle_deg)
    c, s = np.cos(th), np.sin(th)
    xr = c * x - s * y
    yr = s * x + c * y
    return xr, yr


def transition_arc(
    R0: float,
    vmin: float,
    vmax: float,
    gamma: float,
    rotate_angle: float,
    dv: float = 0.5,
):
    """
    Generate a transition arc between Prandtl–Meyer angles vmin → vmax.

    This is a cleaned, vectorized version of the original OpenLaval logic.
    """

    k = np.arange(0, int((vmax - vmin) / dv) + 1)
    v_k = vmax - k * dv
    phi_target = (
        2 * np.radians(vmax)
        - np.pi / 2 * (np.sqrt((gamma + 1) / (gamma - 1)) - 1)
        - 2 * np.radians(k * dv)
    )

    # Solve for R*(k)
    def eq(R, phi_t):
        return 2 * phi_Rstar(R, gamma) - phi_t

    Rk = np.array([optimize.root(eq, R0, args=(phi_t,)).x[0] for phi_t in phi_target])

    # Compute coordinates
    fai_k = np.radians(vmax - vmin - k * dv)
    xk = -Rk * np.sin(fai_k)
    yk = Rk * np.cos(fai_k)

    # Rotate into blade coordinates
    X, Y = [], []
    for x, y in zip(xk, yk):
        xr, yr = rotate(x, y, rotate_angle)
        X.append(xr)
        Y.append(yr)

    return np.array(X), np.array(Y)
