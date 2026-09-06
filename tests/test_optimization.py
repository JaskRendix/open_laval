from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from openlaval.optimization import optimize_blade_performance
from openlaval.validation import BladeValidationError


def write_config(tmp_path, **overrides):
    cfg = {
        "config": {
            "name": "testblade",
            "save_fig": False,
            "save_excel": False,
            "num_points": 50,
        },
        "turbine_blade": {
            "gamma": 1.4,
            "mach_in": 2.0,
            "mach_out": 1.5,
            "beta_in": 30.0,
            "vu": 20.0,
            "vl": 12.0,
        },
        "edge": {"delta": 5.0, "offset": 0.1},
    }
    for section, values in overrides.items():
        cfg[section].update(values)

    from openlaval.config import load_config

    toml_text = f"""
[config]
name = "{cfg['config']['name']}"
save_fig = {str(cfg['config']['save_fig']).lower()}
save_excel = {str(cfg['config']['save_excel']).lower()}
num_points = {cfg['config']['num_points']}

[turbine_blade]
gamma = {cfg['turbine_blade']['gamma']}
mach_in = {cfg['turbine_blade']['mach_in']}
mach_out = {cfg['turbine_blade']['mach_out']}
beta_in = {cfg['turbine_blade']['beta_in']}
vu = {cfg['turbine_blade']['vu']}
vl = {cfg['turbine_blade']['vl']}

[edge]
delta = {cfg['edge']['delta']}
offset = {cfg['edge']['offset']}
""".strip()
    path = tmp_path / "example.toml"
    path.write_text(toml_text)
    return load_config(str(path))


def test_optimize_blade_basic(tmp_path):
    cfg = write_config(tmp_path)

    with (
        patch("openlaval.optimization.Blade") as MockBlade,
        patch("openlaval.optimization.evaluate_blade_performance") as mock_eval,
    ):
        mock_blade_inst = MagicMock()
        mock_blade_inst.compute.return_value = {"solidity": 0.75}
        MockBlade.return_value = mock_blade_inst

        mock_eval.return_value = {
            "turning_efficiency": 0.95,
            "shock_loss_coefficient": 0.02,
            "profile_loss_coefficient": 0.03,
        }

        result = optimize_blade_performance(cfg, design_vars=["vu", "vl"], maxiter=3)

    assert "optimized_parameters" in result
    assert "final_metrics" in result
    assert "success" in result
    assert "vu" in result["optimized_parameters"]
    assert "vl" in result["optimized_parameters"]


@pytest.mark.parametrize("method", ["Nelder-Mead", "Powell"])
def test_optimize_blade_methods(tmp_path, method):
    cfg = write_config(tmp_path)

    with (
        patch("openlaval.optimization.Blade") as MockBlade,
        patch("openlaval.optimization.evaluate_blade_performance") as mock_eval,
    ):
        mock_blade_inst = MagicMock()
        mock_blade_inst.compute.return_value = {"solidity": 0.8}
        MockBlade.return_value = mock_blade_inst

        mock_eval.return_value = {
            "turning_efficiency": 0.92,
            "shock_loss_coefficient": 0.025,
            "profile_loss_coefficient": 0.025,
        }

        result = optimize_blade_performance(
            cfg, design_vars=["vu"], method=method, maxiter=2
        )

    assert result["optimized_parameters"] is not None


def test_optimize_blade_handles_validation_error(tmp_path):
    cfg = write_config(tmp_path)

    with (
        patch("openlaval.optimization.Blade") as MockBlade,
        patch("openlaval.optimization.evaluate_blade_performance") as mock_eval,
    ):
        MockBlade.side_effect = BladeValidationError("Invalid geometry encountered")

        result = optimize_blade_performance(cfg, design_vars=["vu", "vl"], maxiter=2)

    # Even if optimization encounters validation errors, it should handle the penalty and return structure
    assert "optimized_parameters" in result
    assert "final_metrics" in result


def test_optimize_blade_single_variable(tmp_path):
    cfg = write_config(tmp_path)

    with (
        patch("openlaval.optimization.Blade") as MockBlade,
        patch("openlaval.optimization.evaluate_blade_performance") as mock_eval,
    ):
        mock_blade_inst = MagicMock()
        mock_blade_inst.compute.return_value = {"solidity": 0.7}
        MockBlade.return_value = mock_blade_inst

        mock_eval.return_value = {
            "turning_efficiency": 0.90,
            "shock_loss_coefficient": 0.03,
            "profile_loss_coefficient": 0.03,
        }

        result = optimize_blade_performance(cfg, design_vars=["vl"], maxiter=2)

    assert list(result["optimized_parameters"].keys()) == ["vl"]
