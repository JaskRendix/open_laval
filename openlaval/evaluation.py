from __future__ import annotations

import numpy as np

from .physics import (
    compute_isentropic_pressure_coef,
    compute_turning_angles,
    mass_flow_parameter,
)


def compute_curvature_smoothness(x: np.ndarray, y: np.ndarray) -> float:
    """Computes a smoothness index based on the standard deviation of numerical curvature."""
    if x.size < 3:
        return 0.0
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    # Menger curvature or standard finite difference curvature formula
    denominator = (dx**2 + dy**2) ** 1.5
    # Avoid division by zero
    denominator[denominator == 0.0] = 1e-8
    curvature = np.abs(dx * ddy - dy * ddx) / denominator
    return float(np.std(curvature))


def estimate_losses_and_efficiency(M_in: float, M_out: float, gamma: float) -> dict:
    """Estimates shock losses and isentropic loss coefficients for supersonic blades."""
    # Total pressure ratio approximation for oblique/normal shock component if M > 1
    # Using standard normal shock total pressure recovery relation as a baseline proxy
    if M_in > 1.0:
        # P_t2 / P_t1 for normal shock
        gamma_term = (gamma + 1.0) / (2.0 * gamma * M_in**2 - (gamma - 1.0))
        power = gamma / (gamma - 1.0)
        p_ratio_normal = (
            (gamma + 1.0) * M_in**2 / (2.0 + (gamma - 1.0) * M_in**2)
        ) ** power * gamma_term**power
        shock_loss_coeff = max(0.0, 1.0 - p_ratio_normal)
    else:
        shock_loss_coeff = 0.0

    # Profile/isentropic loss coefficient approximation
    # Proportional to Mach number change and turning severity
    mach_diff = abs(M_out - M_in)
    profile_loss_coeff = 0.02 * (1.0 + 0.5 * mach_diff + shock_loss_coeff)

    # Turning efficiency (ratio of actual kinetic energy gain/retention to ideal)
    turning_efficiency = max(
        0.0, min(1.0, 1.0 - (profile_loss_coeff + shock_loss_coeff))
    )

    return {
        "shock_loss_coefficient": float(shock_loss_coeff),
        "profile_loss_coefficient": float(profile_loss_coeff),
        "turning_efficiency": float(turning_efficiency),
    }


def evaluate_blade_performance(result: dict, cfg) -> dict:
    """
    Evaluate aerodynamic and geometric metrics for a computed blade result,
    utilizing gas dynamics and mass flow physics.
    """
    x = np.array(result.get("x", []))
    lower = np.array(result.get("lower", []))
    upper = np.array(result.get("upper", []))

    # Geometric metrics
    if x.size > 0 and lower.size > 0 and upper.size > 0:
        thickness = upper - lower
        min_thickness = float(np.min(thickness))
        max_thickness = float(result.get("max_thickness", np.max(thickness)))
        chord = float(result.get("chord", x[-1] - x[0]))

        # Curvature smoothness indices for upper and lower surfaces
        upper_smoothness = compute_curvature_smoothness(x, upper)
        lower_smoothness = compute_curvature_smoothness(x, lower)
    else:
        min_thickness = 0.0
        max_thickness = float(result.get("max_thickness", 0.0))
        chord = float(result.get("chord", 0.0))
        upper_smoothness = 0.0
        lower_smoothness = 0.0

    solidity = float(result.get("solidity", 0.0))

    # Aerodynamic evaluations
    turning_data = compute_turning_angles(
        beta_in_deg=cfg.beta_in, M_in=cfg.mach_in, M_out=cfg.mach_out, gamma=cfg.gamma
    )

    m_dot_in = mass_flow_parameter(cfg.mach_in, cfg.gamma)
    m_dot_out = mass_flow_parameter(cfg.mach_out, cfg.gamma)
    cp_est = compute_isentropic_pressure_coef(cfg.mach_out, cfg.mach_in, cfg.gamma)

    loss_data = estimate_losses_and_efficiency(cfg.mach_in, cfg.mach_out, cfg.gamma)

    metrics = {
        "name": cfg.name,
        "chord": chord,
        "min_thickness": min_thickness,
        "max_thickness": max_thickness,
        "max_thickness_location": result.get("max_thickness_location"),
        "max_camber": result.get("max_camber"),
        "solidity": solidity,
        "upper_curvature_smoothness": upper_smoothness,
        "lower_curvature_smoothness": lower_smoothness,
        "mass_flow_param_in": m_dot_in,
        "mass_flow_param_out": m_dot_out,
        "isentropic_cp_est": cp_est,
        **loss_data,
        **turning_data,
    }

    return metrics
