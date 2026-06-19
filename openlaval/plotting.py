from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def _is_headless() -> bool:
    return matplotlib.get_backend().lower() == "agg"


def plot_contour(
    lower_x,
    lower_y,
    upper_x,
    upper_y,
    title: str = "Blade Contour",
    save_path: str | None = None,
    color_lower="k",
    color_upper="k",
):
    """
    Plot full blade contour (raw geometry).
    """
    plt.figure(figsize=(8, 4))
    plt.plot(lower_x, lower_y, color=color_lower, label="Lower surface")
    plt.plot(upper_x, upper_y, color=color_upper, label="Upper surface")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(title)
    plt.xlabel("x*")
    plt.ylabel("y*")
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if not _is_headless():
        plt.show()


def plot_interpolated_contour(
    x,
    lower,
    upper,
    title: str = "Interpolated Blade Contour",
    save_path: str | None = None,
):
    """
    Plot interpolated blade contour.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(x, lower, "k-", label="Lower surface (interp)")
    plt.plot(x, upper, "k-", label="Upper surface (interp)")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(title)
    plt.xlabel("x*")
    plt.ylabel("y*")
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if not _is_headless():
        plt.show()


def plot_characteristics(
    x0, y0, x1, y1, title: str = "Characteristic Lines", save_path: str | None = None
):
    """
    Plot two characteristic lines.
    """
    plt.figure(figsize=(6, 6))
    plt.plot(x0, y0, "r-", label="Compression")
    plt.plot(x1, y1, "b-", label="Expansion")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(title)
    plt.xlabel("x*")
    plt.ylabel("y*")
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if not _is_headless():
        plt.show()


def plot_prandtl_meyer(
    gamma: float,
    mach_points: dict[str, float] | None = None,
    save_path: str | None = None,
):
    """
    Plot Prandtl–Meyer angle vs Mach number.

    mach_points: optional dict of labeled Mach numbers to highlight.
        Example:
            {
                "Inlet": M_in,
                "Outlet": M_out,
                "Upper": M_upper,
                "Lower": M_lower,
            }
    """
    from .physics import prandtl_meyer

    M = np.linspace(1.0, 9.0, 500)
    nu = np.array([prandtl_meyer(m, gamma) for m in M])

    plt.figure(figsize=(6, 5))
    plt.plot(nu, M, "k-", label="ν(M)")

    if mach_points:
        for label, m in mach_points.items():
            plt.plot(prandtl_meyer(m, gamma), m, "o", label=label)

    plt.title(f"Prandtl–Meyer Function (γ = {gamma:.2f})")
    plt.xlabel("Prandtl–Meyer angle ν [deg]")
    plt.ylabel("Mach number")
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if not _is_headless():
        plt.show()


def plot_thickness(x, lower, upper, title="Thickness Distribution", save_path=None):
    thickness = upper - lower

    plt.figure(figsize=(8, 4))
    plt.plot(x, thickness, "k-", label="Thickness = upper - lower")

    plt.title(title)
    plt.xlabel("x*")
    plt.ylabel("Thickness")
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if not _is_headless():
        plt.show()


def plot_camber(x, lower, upper, title="Camber Line", save_path=None):
    camber = 0.5 * (upper + lower)

    plt.figure(figsize=(8, 4))
    plt.plot(x, camber, "k-", label="Camber line")

    plt.title(title)
    plt.xlabel("x*")
    plt.ylabel("Camber")
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if not _is_headless():
        plt.show()


def plot_curvature(x, y, title="Curvature Distribution", save_path=None):
    # Numerical curvature
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)

    curvature = np.abs(dx * ddy - dy * ddx) / (dx * dx + dy * dy) ** 1.5

    plt.figure(figsize=(8, 4))
    plt.plot(x, curvature, "k-", label="Curvature")

    plt.title(title)
    plt.xlabel("x*")
    plt.ylabel("Curvature")
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if not _is_headless():
        plt.show()


def plot_raw_vs_interp(
    raw_x_lower,
    raw_y_lower,
    raw_x_upper,
    raw_y_upper,
    x_interp,
    lower_interp,
    upper_interp,
    title="Raw vs Interpolated",
    save_path=None,
):
    plt.figure(figsize=(8, 4))

    # Raw
    plt.plot(raw_x_lower, raw_y_lower, "r--", label="Lower raw")
    plt.plot(raw_x_upper, raw_y_upper, "b--", label="Upper raw")

    # Interpolated
    plt.plot(x_interp, lower_interp, "k-", label="Lower interp")
    plt.plot(x_interp, upper_interp, "g-", label="Upper interp")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.title(title)
    plt.xlabel("x*")
    plt.ylabel("y*")
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if not _is_headless():
        plt.show()


def plot_asymmetry(x, lower, upper, title="Asymmetry (Upper - Lower)", save_path=None):
    diff = upper - lower

    plt.figure(figsize=(8, 4))
    plt.plot(x, diff, "k-", label="Upper - Lower")

    plt.title(title)
    plt.xlabel("x*")
    plt.ylabel("Asymmetry")
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    if not _is_headless():
        plt.show()
