from __future__ import annotations

import numpy as np
import pytest

from openlaval.blade import Blade, BladeConfig
from openlaval.evaluation import evaluate_blade_performance
from openlaval.physics import prandtl_meyer


@pytest.fixture
def naca_l52b06_baseline_config() -> BladeConfig:
    """
    Configuration based on typical supersonic impulse cascade parameters 
    referenced in NACA RM L52B06 (e.g., design inlet Mach ~1.57, turning ~120°).
    """
    gamma = 1.4
    mach_in = 1.57
    nu_in = float(prandtl_meyer(mach_in, gamma))
    
    # Ensure vl and vu are within valid range [0, nu_in] or valid characteristic bounds
    return BladeConfig(
        name="naca_l52b06_baseline",
        gamma=gamma,
        mach_in=mach_in,
        mach_out=mach_in,  # Isentropic impulse reference
        beta_in=60.0,
        vu=nu_in * 0.8,
        vl=nu_in * 0.5,
        edge_delta=5.0,
        edge_offset=0.1,
        num_points=100,
        save_fig=False,
        save_excel=False,
        asymmetric=False,
    )


def test_validation_naca_baseline_metrics(naca_l52b06_baseline_config):
    """
    Assert that running a classic baseline configuration produces physically 
    valid supersonic flow properties, sensible solidity, and matching turning expectations.
    """
    blade = Blade(naca_l52b06_baseline_config)
    result = blade.compute()
    metrics = evaluate_blade_performance(result, naca_l52b06_baseline_config)

    # 1. Geometric integrity checks
    assert metrics["chord"] > 0.0
    assert metrics["min_thickness"] > 0.0
    assert metrics["max_thickness"] >= metrics["min_thickness"]
    assert 0.0 < metrics["solidity"] < 5.0

    # 2. Aerodynamic gas-dynamics checks
    assert metrics["nu_in"] > 0.0
    assert np.isfinite(metrics["mass_flow_param_in"])
    assert metrics["mass_flow_param_in"] > 0.0

    # 3. Turning angle bounds check
    assert metrics["total_turning"] > naca_l52b06_baseline_config.beta_in


@pytest.mark.parametrize(
    "mach_in,beta_in,expected_min_turning",
    [
        (1.5, 45.0, 45.0),
        (2.0, 55.0, 55.0),
        (2.5, 65.0, 65.0),
    ],
)
def test_validation_parametric_trends(mach_in, beta_in, expected_min_turning):
    """
    Regression validation across varying historical design entry conditions 
    to ensure monotonic and physically sound behavior.
    """
    gamma = 1.4
    nu_in = float(prandtl_meyer(mach_in, gamma))

    cfg = BladeConfig(
        name="sweep_regression",
        gamma=gamma,
        mach_in=mach_in,
        mach_out=mach_in,
        beta_in=beta_in,
        vu=nu_in * 0.8,
        vl=nu_in * 0.5,
        edge_delta=5.0,
        edge_offset=0.1,
        num_points=50,
        save_fig=False,
        save_excel=False,
    )

    blade = Blade(cfg)
    result = blade.compute()
    metrics = evaluate_blade_performance(result, cfg)

    assert metrics["total_turning"] >= expected_min_turning
    assert metrics["chord"] > 0.5
    assert np.isfinite(metrics["isentropic_cp_est"])
