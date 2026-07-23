from __future__ import annotations

import numpy as np
from .physics import (
    prandtl_meyer,
    mass_flow_parameter,
    compute_turning_angles,
    compute_isentropic_pressure_coef,
)


def evaluate_blade_performance(result: dict, cfg) -> dict:
    """
    Evaluate aerodynamic and geometric metrics for a computed blade result,
    utilizing gas dynamics and mass flow physics.
    
    Args:
        result: Dictionary containing x, lower, upper, and geometric features.
        cfg: BladeConfig instance containing design parameters like gamma, mach_in, mach_out, beta_in.
        
    Returns:
        Dictionary of comprehensive geometric and aerodynamic performance metrics.
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
    else:
        min_thickness = 0.0
        max_thickness = float(result.get("max_thickness", 0.0))
        chord = float(result.get("chord", 0.0))
        
    solidity = float(result.get("solidity", 0.0))
    
    # Aerodynamic evaluations using physics module functions
    turning_data = compute_turning_angles(
        beta_in_deg=cfg.beta_in, 
        M_in=cfg.mach_in, 
        M_out=cfg.mach_out, 
        gamma=cfg.gamma
    )
    
    m_dot_in = mass_flow_parameter(cfg.mach_in, cfg.gamma)
    m_dot_out = mass_flow_parameter(cfg.mach_out, cfg.gamma)
    
    # Compressible pressure coefficient estimation relative to inlet reference conditions
    cp_est = compute_isentropic_pressure_coef(cfg.mach_out, cfg.mach_in, cfg.gamma)

    metrics = {
        "name": cfg.name,
        "chord": chord,
        "min_thickness": min_thickness,
        "max_thickness": max_thickness,
        "max_thickness_location": result.get("max_thickness_location"),
        "max_camber": result.get("max_camber"),
        "solidity": solidity,
        "mass_flow_param_in": m_dot_in,
        "mass_flow_param_out": m_dot_out,
        "isentropic_cp_est": cp_est,
        **turning_data,  # unpacks nu_in, nu_out, delta_nu, total_turning
    }
    
    return metrics
