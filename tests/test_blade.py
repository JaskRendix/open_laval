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
):
    # Compute valid Prandtl–Meyer range
    nu_in = prandtl_meyer(mach_in, gamma)
    nu_out = prandtl_meyer(mach_out, gamma)
    nu_min = min(nu_in, nu_out)
    nu_max = max(nu_in, nu_out)

    # If vl/vu not provided, choose safe defaults inside the valid range
    if vl is None:
        vl = nu_min + 0.1 * (nu_max - nu_min)
    if vu is None:
        vu = nu_min + 0.6 * (nu_max - nu_min)

    # Clamp to valid range (just in case)
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

    assert blade.lower_x is not None
    assert blade.upper_x is not None
    assert blade.lower_x.size > 0
    assert blade.upper_x.size > 0
    assert np.all(np.isfinite(blade.lower_x))
    assert np.all(np.isfinite(blade.lower_y))
    assert np.all(np.isfinite(blade.upper_x))
    assert np.all(np.isfinite(blade.upper_y))
    assert np.all(np.diff(blade.lower_x) > 0)
    assert np.all(np.diff(blade.upper_x) > 0)


def test_generate_geometry_handles_empty_transitions():
    # Use extreme angles to stress transition logic
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

    assert blade.x_interp is not None
    assert blade.lower_interp is not None
    assert blade.upper_interp is not None
    assert blade.x_interp.size == cfg.num_points
    assert blade.lower_interp.size == cfg.num_points
    assert blade.upper_interp.size == cfg.num_points
    assert np.all(np.isfinite(blade.lower_interp))
    assert np.all(np.isfinite(blade.upper_interp))


def test_compute_solidity_zero_shift_uses_chord():
    cfg = make_cfg()
    blade = Blade(cfg)
    blade.compute_flow_relations()
    blade.generate_geometry()
    blade.interpolate()
    blade.compute_solidity()

    assert np.isfinite(blade.solidity)
    assert blade.solidity > 0


def test_compute_solidity_nonzero_shift():
    cfg = make_cfg()
    blade = Blade(cfg)
    blade.shift = 0.5
    blade.compute_flow_relations()
    blade.generate_geometry()
    blade.interpolate()
    blade.compute_solidity()

    assert np.isfinite(blade.solidity)
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

    assert (
        "x" in result
        and "lower" in result
        and "upper" in result
        and "solidity" in result
    )
    assert result["x"].size == cfg.num_points
    assert result["lower"].size == cfg.num_points
    assert result["upper"].size == cfg.num_points
    assert np.all(np.isfinite(result["lower"]))
    assert np.all(np.isfinite(result["upper"]))
    assert np.isfinite(result["solidity"])


@pytest.mark.parametrize(
    "vl,vu",
    [
        (12.0, 15.0),
        (12.0, 20.0),
        (
            15.0,
            25.0,
        ),  # 25 is slightly above ν_max but still acceptable if your validator clamps or warns
    ],
)
def test_blade_full_pipeline_param_angles(vl, vu):
    cfg = make_cfg(vl=vl, vu=vu)
    blade = Blade(cfg)
    result = blade.compute()
    assert np.isfinite(result["solidity"])
