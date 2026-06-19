from __future__ import annotations

import numpy as np

from .characteristics import solve_Rstar_intersection
from .config import BladeConfig
from .geometry import (
    circular_arc,
    interpolate_contour,
    leading_edge_adjust,
    lower_concave_transition,
    straight_line_to_match,
    upper_convex_transition,
)
from .physics import mach_after_edge, prandtl_meyer


class Blade:
    """
    High-level blade generator.
    """

    def __init__(self, cfg: BladeConfig):
        self.cfg = cfg

        # Storage for geometry
        self.lower_x = None
        self.lower_y = None
        self.upper_x = None
        self.upper_y = None
        self.x_interp = None
        self.lower_interp = None
        self.upper_interp = None

        # Internal state
        self.shift = 0.0
        self.solidity = None

    def compute_flow_relations(self) -> None:
        c = self.cfg

        # Copy raw config parameters
        self.gamma = c.gamma
        self.vl = c.vl
        self.vu = c.vu
        self.beta_in = c.beta_in

        # Mach after leading-edge expansion
        self.mach_in_eff = mach_after_edge(c.mach_in, c.edge_delta, c.gamma)

        # Prandtl–Meyer angles
        self.nu_in = prandtl_meyer(c.mach_in, c.gamma)
        self.nu_out = prandtl_meyer(c.mach_out, c.gamma)

        # Compute outlet flow angle (beta_out)
        b1 = 1 + (c.gamma - 1) / 2 * (c.mach_out**2)
        b2 = 1 + (c.gamma - 1) / 2 * (c.mach_in**2)
        b3 = (c.gamma + 1) / (2 * (c.gamma - 1))

        beta_out_rad = -np.arccos(
            c.mach_in / c.mach_out * ((b1 / b2) ** b3) * np.cos(np.deg2rad(c.beta_in))
        )
        self.beta_out = np.rad2deg(beta_out_rad)

        # R* values for upper/lower surfaces
        self.Rstar_upper = solve_Rstar_intersection(
            -c.vu, c.vu, c.gamma, phi_const=0.0
        )[0]
        self.Rstar_lower = solve_Rstar_intersection(
            -c.vl, c.vl, c.gamma, phi_const=0.0
        )[0]

    def generate_geometry(self) -> None:
        c = self.cfg

        # 2.1 Circular arcs
        lower_arc_x, lower_arc_y = circular_arc(
            self.Rstar_lower,
            90 + (self.beta_in - (self.nu_in - self.vl)),
            90 + (self.beta_out + (self.nu_out - self.vl)),
        )

        upper_arc_x, upper_arc_y = circular_arc(
            self.Rstar_upper,
            90 + (self.beta_in - (self.vu - self.nu_in)),
            90 + (self.beta_out + (self.vu - self.nu_out)),
        )

        # 2.2 Lower concave transitions
        lc_in_x, lc_in_y = lower_concave_transition(
            self.Rstar_lower, self.vl, self.nu_in, self.gamma, self.beta_in
        )
        lc_out_x, lc_out_y = lower_concave_transition(
            self.Rstar_lower, self.vl, self.nu_out, self.gamma, self.beta_out
        )

        # 2.3 Upper convex transitions
        uc_in_x, uc_in_y = upper_convex_transition(
            self.Rstar_upper, self.vu, self.beta_in, self.gamma, direction="in"
        )
        uc_out_x, uc_out_y = upper_convex_transition(
            self.Rstar_upper, self.vu, self.beta_out, self.gamma, direction="out"
        )

        # Helpers
        def _safe_last(arr: np.ndarray, fallback: float) -> float:
            return arr[-1] if arr.size > 0 else fallback

        # Safe endpoints for leading edge and trailing edge
        lc_in_last_x = _safe_last(lc_in_x, lower_arc_x[0])
        lc_in_last_y = _safe_last(lc_in_y, lower_arc_y[0])
        lc_out_last_x = _safe_last(lc_out_x, lower_arc_x[-1])
        lc_out_last_y = _safe_last(lc_out_y, lower_arc_y[-1])

        uc_in_last_x = _safe_last(uc_in_x, upper_arc_x[0])
        uc_in_last_y = _safe_last(uc_in_y, upper_arc_y[0])
        uc_out_last_x = _safe_last(uc_out_x, upper_arc_x[-1])
        uc_out_last_y = _safe_last(uc_out_y, upper_arc_y[-1])

        # 2.5 Leading-edge adjustment (using safe endpoints)
        x_edge, y_edge = leading_edge_adjust(
            uc_in_last_x,
            uc_in_last_y,
            lc_in_last_x,
            lc_in_last_y,
            self.beta_in,
            c.edge_delta,
            c.edge_offset,
            shift=self.shift,
        )

        # Assemble lower surface (allow empty transitions)
        self.lower_x = np.concatenate(
            [
                lc_in_x[::-1] if lc_in_x.size > 0 else np.array([], dtype=float),
                lower_arc_x,
                lc_out_x if lc_out_x.size > 0 else np.array([], dtype=float),
            ]
        )
        self.lower_y = np.concatenate(
            [
                lc_in_y[::-1] if lc_in_y.size > 0 else np.array([], dtype=float),
                lower_arc_y,
                lc_out_y if lc_out_y.size > 0 else np.array([], dtype=float),
            ]
        )

        # Assemble upper surface safely
        self.upper_x = np.concatenate(
            [
                np.array([x_edge, uc_in_last_x], dtype=float),
                uc_in_x[::-1],
                upper_arc_x,
                uc_out_x,
                np.array([uc_out_last_x, x_edge], dtype=float),
            ]
        )
        self.upper_y = np.concatenate(
            [
                np.array([y_edge, uc_in_last_y], dtype=float),
                uc_in_y[::-1],
                upper_arc_y,
                uc_out_y,
                np.array([uc_out_last_y, y_edge], dtype=float),
            ]
        )

        # Final cleaning: numeric, finite, sorted, no duplicate x
        def _clean_xy(x, y):
            x = np.array(x, dtype=float)
            y = np.array(y, dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            if x.size < 2:
                return x, y
            idx = np.argsort(x)
            x, y = x[idx], y[idx]
            dx = np.diff(x)
            good = np.hstack(([True], dx > 1e-9))
            return x[good], y[good]

        self.lower_x, self.lower_y = _clean_xy(self.lower_x, self.lower_y)
        self.upper_x, self.upper_y = _clean_xy(self.upper_x, self.upper_y)

    def interpolate(self):
        self.x_interp, self.lower_interp, self.upper_interp = interpolate_contour(
            self.lower_x,
            self.lower_y,
            self.upper_x,
            self.upper_y,
            self.cfg.num_points,
        )

    def compute_solidity(self) -> None:
        # Guard against empty geometry
        if self.lower_x is None or self.lower_x.size < 2:
            self.solidity = np.nan
            return

        Cstar = self.lower_x[-1] - self.lower_x[0]

        # Temporary: use chord as pitch proxy if shift is zero
        if self.shift == 0.0:
            Gstar = Cstar if Cstar != 0 else 1.0
        else:
            Gstar = abs(self.shift)

        self.solidity = Cstar / Gstar

    def compute(self):
        """
        Full blade-generation pipeline.
        """
        self.compute_flow_relations()
        self.generate_geometry()
        self.interpolate()
        self.compute_solidity()

        return {
            "x": self.x_interp,
            "lower": self.lower_interp,
            "upper": self.upper_interp,
            "solidity": self.solidity,
        }
