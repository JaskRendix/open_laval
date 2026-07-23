from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock

from openlaval.evaluation import evaluate_blade_performance


@pytest.fixture
def sample_blade_result():
    x = np.linspace(0.0, 1.0, 10)
    lower = np.zeros(10)
    upper = np.linspace(0.1, 0.2, 10)
    return {
        "x": x,
        "lower": lower,
        "upper": upper,
        "max_thickness": 0.2,
        "max_thickness_location": 1.0,
        "max_camber": 0.1,
        "solidity": 1.5,
        "chord": 1.0,
    }


@pytest.fixture
def sample_config():
    cfg = MagicMock()
    cfg.name = "test_eval_blade"
    cfg.gamma = 1.4
    cfg.mach_in = 1.5
    cfg.mach_out = 2.0
    cfg.beta_in = 65.0
    return cfg


def test_evaluate_blade_performance_basic(sample_blade_result, sample_config):
    metrics = evaluate_blade_performance(sample_blade_result, sample_config)

    assert isinstance(metrics, dict)
    assert metrics["name"] == "test_eval_blade"
    assert metrics["chord"] == 1.0
    assert metrics["min_thickness"] == pytest.approx(0.1)
    assert metrics["max_thickness"] == 0.2
    assert metrics["solidity"] == 1.5

    # Check aerodynamic metrics added from physics module
    assert "mass_flow_param_in" in metrics
    assert "mass_flow_param_out" in metrics
    assert "isentropic_cp_est" in metrics
    assert "nu_in" in metrics
    assert "nu_out" in metrics
    assert "delta_nu" in metrics
    assert "total_turning" in metrics

    assert metrics["delta_nu"] > 0
    assert metrics["total_turning"] > sample_config.beta_in


def test_evaluate_blade_performance_empty_result(sample_config):
    empty_result = {
        "x": np.array([]),
        "lower": np.array([]),
        "upper": np.array([]),
    }
    metrics = evaluate_blade_performance(empty_result, sample_config)

    assert metrics["chord"] == 0.0
    assert metrics["min_thickness"] == 0.0
    assert metrics["max_thickness"] == 0.0
    assert metrics["solidity"] == 0.0
    assert np.isfinite(metrics["total_turning"])
