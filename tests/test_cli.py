import json
import tomllib
from pathlib import Path
from unittest.mock import MagicMock, patch

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

    path = tmp_path / "settings.toml"
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
    path.write_text("""
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
vl = 50.0   # invalid: vl > vu

[edge]
delta = 5.0
offset = 0.1
""")

    result = runner.invoke(app, ["run", str(path)])

    assert result.exit_code != 0
    assert "Invalid Prandtl–Meyer angles" in result.stdout


def test_cli_plot_invalid_config(tmp_path):
    path = tmp_path / "invalid_plot.toml"
    path.write_text("""
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
""")

    result = runner.invoke(app, ["plot", str(path)])

    assert result.exit_code != 0
    assert "Invalid Prandtl–Meyer" in result.stdout


def test_cli_export_invalid_config(tmp_path):
    path = tmp_path / "invalid_export.toml"
    path.write_text("""
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
""")

    result = runner.invoke(app, ["export", str(path), "--outdir", str(tmp_path)])

    assert result.exit_code != 0
    assert "Invalid Prandtl–Meyer" in result.stdout
