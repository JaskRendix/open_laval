from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


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

    plt.show()
