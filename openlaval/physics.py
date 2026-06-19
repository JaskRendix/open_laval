from __future__ import annotations

import numpy as np
from scipy import integrate, optimize


def prandtl_meyer(M: float, gamma: float) -> float:
    """
    Prandtl–Meyer angle ν(M) in degrees.
    """
    a = np.sqrt((gamma + 1) / (gamma - 1))
    term = np.sqrt((gamma - 1) / (gamma + 1) * (M**2 - 1))
    nu = a * np.arctan(term) - np.arctan(np.sqrt(M**2 - 1))
    return np.degrees(nu)


def mach_from_prandtl(nu_deg: float, gamma: float) -> float:
    """
    Invert Prandtl–Meyer function: find Mach number for a given ν.
    """
    f = lambda M: prandtl_meyer(M, gamma) - nu_deg
    sol = optimize.root(f, 2.0)
    return sol.x[0]


def mach_to_mstar(M: float, gamma: float) -> float:
    """
    Convert Mach number to M* (critical Mach parameter).
    """
    return np.sqrt(((gamma + 1) / 2 * M**2) / (1 + (gamma - 1) / 2 * M**2))


def mstar_to_mach(Ms: float, gamma: float) -> float:
    """
    Convert M* back to Mach number.
    """
    return np.sqrt((2 * Ms**2) / ((gamma + 1) - (gamma - 1) * Ms**2))


def phi_Rstar(Rs: float, gamma: float) -> float:
    """
    Characteristic-line angle φ(R*) from NASA TN D‑4421.
    Returns φ in radians.
    """
    a = np.sqrt((gamma + 1) / (gamma - 1))

    arg1 = (gamma - 1) / Rs**2 - gamma
    arg2 = (gamma + 1) * Rs**2 - gamma

    # Numerical safety: keep arcsin arguments in [-1, 1]
    arg1 = np.clip(arg1, -1.0, 1.0)
    arg2 = np.clip(arg2, -1.0, 1.0)

    term1 = a * np.arcsin(arg1)
    term2 = np.arcsin(arg2)
    return 0.5 * (term1 + term2)


def mach_after_edge(M_in: float, delta_nu: float, gamma: float) -> float:
    """
    Compute Mach number after a Prandtl–Meyer expansion of Δν degrees.
    """
    nu_new = prandtl_meyer(M_in, gamma) + delta_nu
    return mach_from_prandtl(nu_new, gamma)


def Kstar_max(Mlstar: float, Mustar: float, gamma: float) -> float:
    """
    Maximum K* for which mass flow is real.
    NASA TN D‑4421 eq. 23–25.
    """

    # Upper bound so that 1 - (K/Ml*)^2 Ms^2 >= 0 for Ms in [Ml*, Mu*]
    K_upper = Mlstar / Mustar * 0.999

    def func(K):
        a1 = (1 - K**2) ** (1 / (gamma - 1))

        a2 = 1 - (K**2) * (Mustar / Mlstar) ** 2
        a3 = a2 ** (1 / (gamma - 1))

        def integrand(Ms):
            base = 1 - (K / Mlstar) ** 2 * Ms**2
            if base <= 0.0:
                return 0.0
            return base ** (1 / (gamma - 1))

        b = integrate.quad(integrand, Mlstar, Mustar)[0]

        return a1 - a3 - b

    return optimize.brentq(func, 1e-6, K_upper)


def vortex_C(Mlstar: float, Mustar: float, gamma: float) -> float:
    """
    Reduction in maximum weight flow due to 2D flow.
    NASA TN D‑4421 eq. 34b.
    """

    # Outside the valid M* range, treat the 2D reduction as negligible.
    # This avoids pathological calls from vl_minimum's root search.
    if Mlstar < 1e-3:
        return 0.0

    Kmax = Kstar_max(Mlstar, Mustar, gamma)

    a = 1 + np.sqrt((gamma + 1) / (gamma - 1)) * ((gamma + 1) / 2) ** (
        1 / (gamma - 1)
    ) * (Mustar / (Mustar - Mlstar))

    def integrand(Ms):
        base = 1 - (Kmax / Mlstar) ** 2 * Ms**2
        if base <= 0.0:
            return 0.0
        return Kmax * base ** (1 / (gamma - 1))

    b = integrate.quad(integrand, Mlstar, Mustar)[0]

    return a * b


def vortex_Q(Mlstar: float, Mustar: float, gamma: float) -> float:
    """
    Vortex-flow parameter Q.
    NASA TN D‑4421 eq. 34a.
    """

    def integrand(Ms):
        base = (gamma + 1) / 2 - (gamma - 1) / 2 * Ms**2
        if base <= 0.0:
            return 0.0
        return base ** (1 / (gamma - 1))

    a = integrate.quad(integrand, Mlstar, Mustar)[0]
    return (Mlstar * Mustar) / (Mustar - Mlstar) * a


def vl_minimum(gamma: float) -> float:
    """
    Minimum lower-surface Prandtl–Meyer angle ν_l,min from separation criterion.
    Returns ν_l,min in degrees.
    """

    def Mi_func(Mi):
        Mi = float(Mi)

        Q = vortex_Q(1 / Mi, 1.0, gamma)
        C = vortex_C(1 / Mi, 1.0, gamma)

        num = 1 - ((gamma - 1) / (gamma + 1)) * Mi**2
        den = Mi**2 - ((gamma - 1) / (gamma + 1))
        base = num / den

        if base <= 0.0:
            base_term = 0.0
        else:
            base_term = base ** (1 / (gamma - 1))

        lhs = Mi ** (2 * gamma / (gamma - 1)) * base_term
        rhs = Q / (1 - C)
        return lhs - rhs

    Mi_lo, Mi_hi = 1.0 + 1e-6, 5.0
    try:
        Mi = optimize.brentq(Mi_func, Mi_lo, Mi_hi)
    except ValueError:
        # No sign change → no physical separation limit in this model
        return 0.0

    a = np.sqrt((gamma + 1) / (gamma - 1))
    b = 1 - ((gamma - 1) / (gamma + 1)) * Mi**2
    c = 1 + 0.5 * ((gamma / (gamma + 1)) * Mi**2) / b

    Mlmin = a * np.sqrt(1 - b * c ** ((gamma - 1) / gamma))

    nu_min_rad = np.pi / 4 * np.sqrt((gamma + 1) / (gamma - 1)) + phi_Rstar(
        1 / Mlmin, gamma
    )
    return np.degrees(nu_min_rad)
