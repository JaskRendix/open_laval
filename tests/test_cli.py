from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

import tomllib
from typer.testing import CliRunner

from openlaval.cli import app

runner = CliRunner()


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

    # Apply overrides
    for section, values in overrides.items():
        cfg[section].update(values)

    # Manual TOML serialization (simple and robust)
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
    return path


def test_cli_run_basic(tmp_path):
    cfg_path = write_config(tmp_path)

    with patch("openlaval.cli.Blade") as MockBlade:
        mock_blade = MagicMock()
        mock_blade.compute.return_value = {"solidity": 0.75}
        MockBlade.return_value = mock_blade

        result = runner.invoke(app, ["run", str(cfg_path)])

    assert result.exit_code == 0
    assert "Computed blade: testblade" in result.stdout
    assert "Solidity: 0.7500" in result.stdout


def test_cli_plot_calls_plotting(tmp_path):
    cfg_path = write_config(tmp_path)

    with (
        patch("openlaval.cli.Blade") as MockBlade,
        patch("openlaval.cli.plot_interpolated_contour") as mock_plot,
    ):
        mock_blade = MagicMock()
        mock_blade.compute.return_value = {
            "x": [0, 1],
            "lower": [0, -1],
            "upper": [1, 2],
            "solidity": 0.8,
        }
        MockBlade.return_value = mock_blade

        result = runner.invoke(app, ["plot", str(cfg_path)])

    assert result.exit_code == 0
    mock_plot.assert_called_once()


def test_cli_export_creates_files(tmp_path):
    cfg_path = write_config(tmp_path)

    with (
        patch("openlaval.cli.Blade") as MockBlade,
        patch("openlaval.cli.save_all_results") as mock_save,
    ):
        mock_blade = MagicMock()
        mock_blade.lower_x = [0, 1]
        mock_blade.lower_y = [0, -1]
        mock_blade.upper_x = [0, 1]
        mock_blade.upper_y = [1, 2]
        mock_blade.compute.return_value = {
            "x": [0, 1],
            "lower": [0, -1],
            "upper": [1, 2],
            "solidity": 0.8,
        }
        MockBlade.return_value = mock_blade

        result = runner.invoke(
            app, ["export", str(cfg_path), "--outdir", str(tmp_path)]
        )

    assert result.exit_code == 0
    mock_save.assert_called_once()
    assert "Exported results" in result.stdout


def test_cli_export_excel_flag(tmp_path):
    cfg_path = write_config(
        tmp_path,
        config={"save_excel": True},
    )

    with (
        patch("openlaval.cli.Blade") as MockBlade,
        patch("openlaval.cli.save_all_results") as mock_save,
    ):
        mock_blade = MagicMock()
        mock_blade.lower_x = [0, 1]
        mock_blade.lower_y = [0, -1]
        mock_blade.upper_x = [0, 1]
        mock_blade.upper_y = [1, 2]
        mock_blade.compute.return_value = {
            "x": [0, 1],
            "lower": [0, -1],
            "upper": [1, 2],
            "solidity": 0.8,
        }
        MockBlade.return_value = mock_blade

        result = runner.invoke(
            app, ["export", str(cfg_path), "--outdir", str(tmp_path)]
        )

    assert result.exit_code == 0
    mock_save.assert_called_once()
    # Ensure save_excel=True was passed through
    assert mock_save.call_args.kwargs["save_excel"] is True


def test_cli_run_missing_file(tmp_path):
    missing = tmp_path / "does_not_exist.toml"
    result = runner.invoke(app, ["run", str(missing)])

    assert result.exit_code != 0
    assert "No such file" in result.stdout or "Error" in result.stdout


def test_cli_run_invalid_toml(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("this is not valid toml = = =")

    result = runner.invoke(app, ["run", str(bad)])

    assert result.exit_code != 0
    assert "Error" in result.stdout or "invalid" in result.stdout.lower()


def test_cli_run_invalid_config(tmp_path):
    path = tmp_path / "invalid_config.toml"
    path.write_text(
        """
[config]
name = "bad"
save_fig = false
save_excel = false
num_points = 50

[turbine_blade]
gamma = 1.4
mach_in = 2.0
mach_out = 1.5
beta_in = 30.0
vu = 10.0
vl = 50.0    # invalid: vl > vu

[edge]
delta = 5.0
offset = 0.1
"""
    )

    result = runner.invoke(app, ["run", str(path)])

    assert result.exit_code != 0
    assert "Invalid Prandtl–Meyer angles" in result.stdout


def test_cli_plot_invalid_config(tmp_path):
    path = tmp_path / "invalid_plot.toml"
    path.write_text(
        """
[config]
name = "bad"
save_fig = false
save_excel = false
num_points = 50

[turbine_blade]
gamma = 1.4
mach_in = 2.0
mach_out = 1.5
beta_in = 30.0
vu = 10.0
vl = 50.0

[edge]
delta = 5.0
offset = 0.1
"""
    )

    result = runner.invoke(app, ["plot", str(path)])

    assert result.exit_code != 0
    assert "Invalid Prandtl–Meyer" in result.stdout


def test_cli_export_invalid_config(tmp_path):
    path = tmp_path / "invalid_export.toml"
    path.write_text(
        """
[config]
name = "bad"
save_fig = false
save_excel = false
num_points = 50

[turbine_blade]
gamma = 1.4
mach_in = 2.0
mach_out = 1.5
beta_in = 30.0
vu = 10.0
vl = 50.0

[edge]
delta = 5.0
offset = 0.1
"""
    )

    result = runner.invoke(app, ["export", str(path), "--outdir", str(tmp_path)])

    assert result.exit_code != 0
    assert "Invalid Prandtl–Meyer" in result.stdout


def test_cli_plot_thickness(tmp_path):
    cfg_path = write_config(tmp_path)

    with (
        patch("openlaval.cli.Blade") as MockBlade,
        patch("openlaval.cli.plot_thickness") as mock_plot,
    ):
        mock_blade = MagicMock()
        mock_blade.compute.return_value = {
            "x": [0, 1],
            "lower": [0, -1],
            "upper": [1, 2],
            "solidity": 0.8,
        }
        MockBlade.return_value = mock_blade

        result = runner.invoke(app, ["plot-thickness", str(cfg_path)])

    assert result.exit_code == 0
    mock_plot.assert_called_once()


def test_cli_plot_camber(tmp_path):
    cfg_path = write_config(tmp_path)

    with (
        patch("openlaval.cli.Blade") as MockBlade,
        patch("openlaval.cli.plot_camber") as mock_plot,
    ):
        mock_blade = MagicMock()
        mock_blade.compute.return_value = {
            "x": [0, 1],
            "lower": [0, -1],
            "upper": [1, 2],
            "solidity": 0.8,
        }
        MockBlade.return_value = mock_blade

        result = runner.invoke(app, ["plot-camber", str(cfg_path)])

    assert result.exit_code == 0
    mock_plot.assert_called_once()


def test_cli_plot_curvature(tmp_path):
    cfg_path = write_config(tmp_path)

    with (
        patch("openlaval.cli.Blade") as MockBlade,
        patch("openlaval.cli.plot_curvature") as mock_plot,
    ):
        mock_blade = MagicMock()
        mock_blade.lower_x = [0, 1]
        mock_blade.lower_y = [0, -1]
        mock_blade.upper_x = [0, 1]
        mock_blade.upper_y = [1, 2]
        mock_blade.compute.return_value = {
            "x": [0, 1],
            "lower": [0, -1],
            "upper": [1, 2],
            "solidity": 0.8,
        }
        MockBlade.return_value = mock_blade

        result = runner.invoke(app, ["plot-curvature", str(cfg_path)])

    assert result.exit_code == 0
    # curvature is called twice (lower + upper)
    assert mock_plot.call_count == 2


def test_cli_plot_raw_vs_interp(tmp_path):
    cfg_path = write_config(tmp_path)

    with (
        patch("openlaval.cli.Blade") as MockBlade,
        patch("openlaval.cli.plot_raw_vs_interp") as mock_plot,
    ):
        mock_blade = MagicMock()
        mock_blade.lower_x = [0, 1]
        mock_blade.lower_y = [0, -1]
        mock_blade.upper_x = [0, 1]
        mock_blade.upper_y = [1, 2]
        mock_blade.compute.return_value = {
            "x": [0, 1],
            "lower": [0, -1],
            "upper": [1, 2],
            "solidity": 0.8,
        }
        MockBlade.return_value = mock_blade

        result = runner.invoke(app, ["plot-raw-vs-interp", str(cfg_path)])

    assert result.exit_code == 0
    mock_plot.assert_called_once()


def test_cli_plot_asymmetry(tmp_path):
    cfg_path = write_config(tmp_path)

    with (
        patch("openlaval.cli.Blade") as MockBlade,
        patch("openlaval.cli.plot_asymmetry") as mock_plot,
    ):
        mock_blade = MagicMock()
        mock_blade.compute.return_value = {
            "x": [0, 1],
            "lower": [0, -1],
            "upper": [1, 2],
            "solidity": 0.8,
        }
        MockBlade.return_value = mock_blade

        result = runner.invoke(app, ["plot-asymmetry", str(cfg_path)])

    assert result.exit_code == 0
    mock_plot.assert_called_once()


def test_cli_plot_curvature_combined(tmp_path):
    cfg_path = write_config(tmp_path)

    with (
        patch("openlaval.cli.Blade") as MockBlade,
        patch("openlaval.cli.plot_combined_curvature") as mock_plot,
    ):
        mock_blade = MagicMock()
        mock_blade.lower_x = [0, 1]
        mock_blade.lower_y = [0, -1]
        mock_blade.upper_x = [0, 1]
        mock_blade.upper_y = [1, 2]
        mock_blade.compute.return_value = {
            "x": [0, 1],
            "lower": [0, -1],
            "upper": [1, 2],
            "solidity": 0.8,
        }
        MockBlade.return_value = mock_blade

        result = runner.invoke(app, ["plot-curvature-combined", str(cfg_path)])

    assert result.exit_code == 0
    mock_plot.assert_called_once()


def test_cli_batch(tmp_path):
    cfg_path = write_config(tmp_path)

    with patch("openlaval.cli.Blade") as MockBlade:
        mock_blade = MagicMock()
        mock_blade.compute.return_value = {
            "chord": 1.0,
            "max_thickness": 0.2,
            "solidity": 0.75,
        }
        MockBlade.return_value = mock_blade

        pattern = str(tmp_path / "*.toml")
        result = runner.invoke(app, ["batch", pattern, "--output-dir", str(tmp_path / "batch_out")])

    assert result.exit_code == 0
    assert "Batch Summary" in result.stdout


def test_cli_sweep(tmp_path):
    cfg_path = write_config(tmp_path)

    with patch("openlaval.cli.Blade") as MockBlade:
        mock_blade = MagicMock()
        mock_blade.compute.return_value = {
            "chord": 1.0,
            "max_thickness": 0.2,
            "solidity": 0.75,
        }
        MockBlade.return_value = mock_blade

        result = runner.invoke(
            app,
            [
                "sweep",
                str(cfg_path),
                "--param",
                "mach_in",
                "--start",
                "1.5",
                "--end",
                "2.5",
                "--steps",
                "3",
                "--output-dir",
                str(tmp_path / "sweep_out"),
            ],
        )

    assert result.exit_code == 0
    assert "Sweep completed successfully" in result.stdout
    assert (tmp_path / "sweep_out" / "sweep_mach_in_summary.csv").exists()


def test_cli_export_cad(tmp_path):
    cfg_path = write_config(tmp_path)

    with (
        patch("openlaval.cli.Blade") as MockBlade,
        patch("openlaval.cli.save_cfd_dat") as mock_dat,
        patch("openlaval.cli.save_csv_coordinates") as mock_csv,
    ):
        mock_blade = MagicMock()
        mock_blade.compute.return_value = {
            "x": [0, 1],
            "lower": [0, -1],
            "upper": [1, 2],
            "solidity": 0.8,
        }
        MockBlade.return_value = mock_blade

        result = runner.invoke(app, ["export-cad", str(cfg_path), "--outdir", str(tmp_path)])

    assert result.exit_code == 0
    mock_dat.assert_called_once()
    mock_csv.assert_called_once()
    assert "Successfully exported CAD/CFD formats" in result.stdout


def test_cli_validate(tmp_path):
    cfg_path = write_config(tmp_path)

    with (
        patch("openlaval.cli._safe_load_config") as mock_load,
        patch("openlaval.cli._safe_compute_blade") as mock_compute,
    ):
        mock_cfg = MagicMock()
        mock_cfg.name = "test_blade"
        mock_load.return_value = mock_cfg

        mock_blade = MagicMock()
        mock_result = {
            "lower": np.array([0.0, 0.0]),
            "upper": np.array([1.0, 1.0]),
            "max_thickness": 2.0,
        }
        mock_compute.return_value = (mock_blade, mock_result)

        result = runner.invoke(app, ["validate", str(cfg_path)])

    assert result.exit_code == 0
    assert "Validation PASSED" in result.stdout


def test_cli_evaluate(tmp_path):
    cfg_path = write_config(tmp_path)

    with (
        patch("openlaval.cli._safe_load_config") as mock_load,
        patch("openlaval.cli._safe_compute_blade") as mock_compute,
        patch("openlaval.cli.evaluate_blade_performance") as mock_eval,
    ):
        mock_cfg = MagicMock()
        mock_load.return_value = mock_cfg

        mock_blade = MagicMock()
        mock_result = {"x": [0, 1], "lower": [0, -1], "upper": [1, 2]}
        mock_compute.return_value = (mock_blade, mock_result)

        mock_eval.return_value = {
            "name": "testblade",
            "chord": 1.0000,
            "min_thickness": 0.0500,
            "max_thickness": 0.2000,
            "solidity": 0.7500,
            "nu_in": 25.50,
            "nu_out": 45.20,
            "delta_nu": 19.70,
            "total_turning": 35.00,
            "mass_flow_param_in": 1.2345,
            "mass_flow_param_out": 1.1111,
            "isentropic_cp_est": 0.8500,
        }

        result = runner.invoke(app, ["evaluate", str(cfg_path)])

    assert result.exit_code == 0
    mock_eval.assert_called_once()
    assert "Aerodynamic & Geometric Evaluation: testblade" in result.stdout
    assert "Chord:" in result.stdout
    assert "Expansion Delta (Δν):" in result.stdout
