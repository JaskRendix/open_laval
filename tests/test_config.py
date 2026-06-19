import tomllib
from pathlib import Path

import numpy as np
import pytest

from openlaval.config import BladeConfig, load_config
from openlaval.physics import prandtl_meyer


def valid_pm_range(mach_in=2.0, mach_out=1.5, gamma=1.4):
    nu_in = prandtl_meyer(mach_in, gamma)
    nu_out = prandtl_meyer(mach_out, gamma)
    return min(nu_in, nu_out), max(nu_in, nu_out)


def make_valid_config(**overrides):
    mach_in = overrides.get("mach_in", 2.0)
    mach_out = overrides.get("mach_out", 1.5)
    gamma = overrides.get("gamma", 1.4)

    # If gamma <= 1, skip PM computation — let BladeConfig catch the error
    if gamma <= 1.0:
        vl = 10.0
        vu = 20.0
    else:
        nu_in = prandtl_meyer(mach_in, gamma)
        nu_out = prandtl_meyer(mach_out, gamma)
        nu_min = min(nu_in, nu_out)
        nu_max = max(nu_in, nu_out)
        vl = nu_min + 0.1 * (nu_max - nu_min)
        vu = nu_min + 0.6 * (nu_max - nu_min)

    cfg = dict(
        name="test",
        save_fig=False,
        save_excel=False,
        num_points=200,
        gamma=gamma,
        mach_in=mach_in,
        mach_out=mach_out,
        beta_in=30.0,
        vl=vl,
        vu=vu,
        edge_delta=5.0,
        edge_offset=0.1,
    )

    cfg.update(overrides)
    return cfg


def test_bladeconfig_valid_basic():
    cfg = make_valid_config()
    c = BladeConfig(**cfg)
    assert c.name == "test"
    assert c.num_points == 200
    assert c.gamma == 1.4
    assert np.isfinite(c.vl)
    assert np.isfinite(c.vu)


def test_missing_required_fields():
    with pytest.raises(Exception):
        BladeConfig()  # missing everything


@pytest.mark.filterwarnings("ignore:invalid value encountered in sqrt")
@pytest.mark.parametrize("mach_in", [0.5, 1.0])
def test_invalid_mach_in(mach_in):
    cfg = make_valid_config(mach_in=mach_in)
    with pytest.raises(ValueError):
        BladeConfig(**cfg)


@pytest.mark.filterwarnings("ignore:invalid value encountered in sqrt")
@pytest.mark.parametrize("mach_out", [0.5, 1.0])
def test_invalid_mach_out(mach_out):
    cfg = make_valid_config(mach_out=mach_out)
    with pytest.raises(ValueError):
        BladeConfig(**cfg)


@pytest.mark.parametrize("gamma", [0.5, 1.0])
def test_invalid_gamma(gamma):
    cfg = make_valid_config(gamma=gamma)
    with pytest.raises(ValueError):
        BladeConfig(**cfg)


def test_invalid_vl_vu_order():
    cfg = make_valid_config()
    cfg["vl"], cfg["vu"] = cfg["vu"], cfg["vl"]  # swap → invalid
    with pytest.raises(ValueError):
        BladeConfig(**cfg)


def test_vl_below_nu_min():
    cfg = make_valid_config()
    cfg["vl"] = -5.0
    with pytest.raises(ValueError):
        BladeConfig(**cfg)


def test_vu_above_nu_max():
    cfg = make_valid_config()
    cfg["vu"] = 999.0
    with pytest.raises(ValueError):
        BladeConfig(**cfg)


@pytest.mark.parametrize("beta", [-10, 0, 90, 120])
def test_invalid_beta_in(beta):
    cfg = make_valid_config(beta_in=beta)
    with pytest.raises(ValueError):
        BladeConfig(**cfg)


@pytest.mark.parametrize("offset", [-0.1, 0.6, 1.0])
def test_invalid_edge_offset(offset):
    cfg = make_valid_config(edge_offset=offset)
    with pytest.raises(ValueError):
        BladeConfig(**cfg)


def test_valid_edge_offset():
    cfg = make_valid_config(edge_offset=0.3)
    c = BladeConfig(**cfg)
    assert c.edge_offset == 0.3


def test_load_config_valid(tmp_path):
    cfg = make_valid_config()

    toml_data = f"""
[config]
name = "{cfg['name']}"
save_fig = {str(cfg['save_fig']).lower()}
save_excel = {str(cfg['save_excel']).lower()}
num_points = {cfg['num_points']}

[turbine_blade]
gamma = {cfg['gamma']}
mach_in = {cfg['mach_in']}
mach_out = {cfg['mach_out']}
beta_in = {cfg['beta_in']}
vu = {cfg['vu']}
vl = {cfg['vl']}

[edge]
delta = {cfg['edge_delta']}
offset = {cfg['edge_offset']}
"""

    path = tmp_path / "blade.toml"
    path.write_text(toml_data)

    loaded = load_config(path)
    assert loaded.name == cfg["name"]
    assert loaded.gamma == cfg["gamma"]
    assert loaded.vl == cfg["vl"]
    assert loaded.vu == cfg["vu"]


def test_load_config_invalid(tmp_path):
    toml_data = """
[config]
name = "bad"
save_fig = false
save_excel = false
num_points = 200

[turbine_blade]
gamma = 1.4
mach_in = 2.0
mach_out = 1.5
beta_in = 30.0
vu = 5.0
vl = 100.0   # invalid

[edge]
delta = 5.0
offset = 0.1
"""
    path = tmp_path / "bad.toml"
    path.write_text(toml_data)

    with pytest.raises(ValueError):
        load_config(path)
