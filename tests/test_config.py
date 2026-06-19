import numpy as np
import pytest

from openlaval.blade import Blade
from openlaval.config import BladeConfig
from openlaval.physics import prandtl_meyer

GAMMA = 1.4


def make_cfg(
    gamma=GAMMA,
    mach_in=2.0,
    mach_out=1.5,
    vl=None,
    vu=None,
    beta_in=30.0,
    edge_delta=5.0,
    edge_offset=0.02,
    num_points=200,
    name="test_blade",
    save_fig=False,
    save_excel=False,
    asymmetric=False,
    vl_lower=None,
    vl_upper=None,
    vu_lower=None,
    vu_upper=None,
):
    # Compute valid Prandtl–Meyer range
    nu_in = prandtl_meyer(mach_in, gamma)
    nu_out = prandtl_meyer(mach_out, gamma)
    nu_min = min(nu_in, nu_out)
    nu_max = max(nu_in, nu_out)

    # Default symmetric values
    if vl is None:
        vl = nu_min + 0.1 * (nu_max - nu_min)
    if vu is None:
        vu = nu_min + 0.6 * (nu_max - nu_min)

    # Clamp to valid range
    vl = max(nu_min + 1e-6, min(vl, nu_max - 1e-6))
    vu = max(vl + 1e-6, min(vu, nu_max - 1e-6))

    return BladeConfig(
        name=name,
        gamma=gamma,
        mach_in=mach_in,
        mach_out=mach_out,
        vl=vl,
        vu=vu,
        beta_in=beta_in,
        edge_delta=edge_delta,
        edge_offset=edge_offset,
        num_points=num_points,
        save_fig=save_fig,
        save_excel=save_excel,
        asymmetric=asymmetric,
        vl_lower=vl_lower,
        vl_upper=vl_upper,
        vu_lower=vu_lower,
        vu_upper=vu_upper,
    )


def test_compute_flow_relations_basic():
    cfg = make_cfg()
    blade = Blade(cfg)
    blade.compute_flow_relations()

    assert np.isfinite(blade.mach_in_eff)
    assert np.isfinite(blade.nu_in)
    assert np.isfinite(blade.nu_out)
    assert np.isfinite(blade.beta_out)
    assert 0 < blade.Rstar_upper < 1
    assert 0 < blade.Rstar_lower < 1


@pytest.mark.parametrize("mach_in,mach_out", [(2.0, 1.5), (3.0, 2.0), (1.8, 1.2)])
def test_compute_flow_relations_param_mach(mach_in, mach_out):
    cfg = make_cfg(mach_in=mach_in, mach_out=mach_out)
    blade = Blade(cfg)
    blade.compute_flow_relations()
    assert np.isfinite(blade.beta_out)


def test_generate_geometry_basic():
    cfg = make_cfg()
    blade = Blade(cfg)
    blade.compute_flow_relations()
    blade.generate_geometry()

    assert blade.lower_x.size > 0
    assert blade.upper_x.size > 0
    assert np.all(np.isfinite(blade.lower_x))
    assert np.all(np.isfinite(blade.upper_x))
    assert np.all(np.diff(blade.lower_x) > 0)
    assert np.all(np.diff(blade.upper_x) > 0)


def test_generate_geometry_handles_empty_transitions():
    cfg = make_cfg(vl=1.0, vu=1.0)
    blade = Blade(cfg)
    blade.compute_flow_relations()
    blade.generate_geometry()

    assert blade.lower_x.size > 0
    assert blade.upper_x.size > 0


def test_interpolate_basic():
    cfg = make_cfg(num_points=300)
    blade = Blade(cfg)
    blade.compute_flow_relations()
    blade.generate_geometry()
    blade.interpolate()

    assert blade.x_interp.size == cfg.num_points
    assert blade.lower_interp.size == cfg.num_points
    assert blade.upper_interp.size == cfg.num_points
    assert np.all(np.isfinite(blade.lower_interp))
    assert np.all(np.isfinite(blade.upper_interp))


def test_compute_solidity_zero_shift_uses_chord():
    cfg = make_cfg()
    blade = Blade(cfg)
    blade.compute()
    assert blade.solidity > 0


def test_compute_solidity_nonzero_shift():
    cfg = make_cfg()
    blade = Blade(cfg)
    blade.shift = 0.5
    blade.compute()
    assert blade.solidity > 0


def test_compute_solidity_empty_geometry():
    cfg = make_cfg()
    blade = Blade(cfg)
    blade.lower_x = np.array([], dtype=float)
    blade.compute_solidity()
    assert np.isnan(blade.solidity)


def test_blade_full_pipeline_basic():
    cfg = make_cfg()
    blade = Blade(cfg)
    result = blade.compute()

    assert result["x"].size == cfg.num_points
    assert result["lower"].size == cfg.num_points
    assert result["upper"].size == cfg.num_points
    assert np.all(np.isfinite(result["lower"]))
    assert np.all(np.isfinite(result["upper"]))
    assert np.isfinite(result["solidity"])


@pytest.mark.parametrize("vl,vu", [(12.0, 15.0), (12.0, 20.0), (15.0, 25.0)])
def test_blade_full_pipeline_param_angles(vl, vu):
    cfg = make_cfg(vl=vl, vu=vu)
    blade = Blade(cfg)
    result = blade.compute()
    assert np.isfinite(result["solidity"])


def test_asymmetric_flow_relations_rstar_differs():
    cfg = make_cfg(
        asymmetric=True,
        vl_lower=13.0,
        vl_upper=14.0,
        vu_lower=20.0,
        vu_upper=22.0,
    )
    blade = Blade(cfg)
    blade.compute_flow_relations()

    assert blade.Rstar_lower != blade.Rstar_upper
    assert 0 < blade.Rstar_lower < 1
    assert 0 < blade.Rstar_upper < 1


def test_asymmetric_geometry_differs():
    cfg = make_cfg(
        asymmetric=True,
        vl_lower=13.0,
        vl_upper=14.0,
        vu_lower=20.0,
        vu_upper=22.0,
    )
    blade = Blade(cfg)
    blade.compute()
    assert not np.allclose(blade.lower_interp, blade.upper_interp)


def test_asymmetric_pipeline_runs():
    cfg = make_cfg(
        asymmetric=True,
        vl_lower=12.5,
        vl_upper=13.5,
        vu_lower=19.0,
        vu_upper=21.0,
    )
    blade = Blade(cfg)
    result = blade.compute()

    assert result["x"].size == cfg.num_points
    assert np.isfinite(result["solidity"])


def test_asymmetric_fallback_to_symmetric():
    cfg = make_cfg(asymmetric=True)  # no overrides → fallback
    blade = Blade(cfg)
    blade.compute_flow_relations()

    # Should match symmetric behavior
    assert blade.vl_lower == blade.vl_upper == cfg.vl
    assert blade.vu_lower == blade.vu_upper == cfg.vu


def test_cli_asymmetric_override(tmp_path):
    from typer.testing import CliRunner

    from openlaval.cli import app

    runner = CliRunner()

    # Create a minimal symmetric config file
    toml_data = """
[config]
name = "cli_test"
save_fig = false
save_excel = false
num_points = 50

[turbine_blade]
gamma = 1.4
mach_in = 2.0
mach_out = 1.5
beta_in = 30.0
vl = 13.0
vu = 20.0

[edge]
delta = 5.0
offset = 0.1
"""
    path = tmp_path / "blade.toml"
    path.write_text(toml_data)

    # Run CLI with asymmetric overrides
    result = runner.invoke(
        app,
        [
            "run",
            str(path),
            "--asymmetric",
            "--vl-lower",
            "13.5",
            "--vl-upper",
            "14.0",
            "--vu-lower",
            "19.0",
            "--vu-upper",
            "21.0",
        ],
    )

    # CLI should exit cleanly
    assert result.exit_code == 0

    # Output should mention the blade name
    assert "cli_test" in result.stdout

    # Should print solidity
    assert "Solidity:" in result.stdout
