import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from openlaval import plotting


@pytest.fixture(autouse=True)
def force_headless(monkeypatch):
    """Force headless mode so plt.show() is skipped."""
    monkeypatch.setattr(matplotlib, "get_backend", lambda: "agg")


@pytest.fixture
def sample_arrays():
    x = np.linspace(0, 1, 10)
    lower = np.sin(x)
    upper = np.cos(x)
    return x, lower, upper


@pytest.mark.parametrize(
    "func,args",
    [
        (
            plotting.plot_contour,
            (np.arange(5), np.arange(5), np.arange(5), np.arange(5)),
        ),
        (
            plotting.plot_interpolated_contour,
            (np.arange(5), np.arange(5), np.arange(5)),
        ),
        (
            plotting.plot_characteristics,
            (np.arange(5), np.arange(5), np.arange(5), np.arange(5)),
        ),
        (plotting.plot_thickness, (np.arange(5), np.zeros(5), np.ones(5))),
        (plotting.plot_camber, (np.arange(5), np.zeros(5), np.ones(5))),
        (
            plotting.plot_curvature,
            (np.linspace(0, 1, 10), np.sin(np.linspace(0, 1, 10))),
        ),
        (
            plotting.plot_combined_curvature,
            (np.linspace(0, 1, 10), np.sin(np.linspace(0, 1, 10)), np.linspace(0, 1, 10), np.cos(np.linspace(0, 1, 10))),
        ),
        (plotting.plot_asymmetry, (np.arange(5), np.zeros(5), np.ones(5))),
    ],
)
def test_plot_functions_no_save(func, args, tmp_path):
    func(*args)  # Should not raise
    plt.close("all")


@pytest.mark.parametrize(
    "func,args,filename",
    [
        (
            plotting.plot_contour,
            (np.arange(5), np.arange(5), np.arange(5), np.arange(5)),
            "contour.png",
        ),
        (
            plotting.plot_interpolated_contour,
            (np.arange(5), np.arange(5), np.arange(5)),
            "interp.png",
        ),
        (
            plotting.plot_characteristics,
            (np.arange(5), np.arange(5), np.arange(5), np.arange(5)),
            "char.png",
        ),
        (plotting.plot_thickness, (np.arange(5), np.zeros(5), np.ones(5)), "thick.png"),
        (plotting.plot_camber, (np.arange(5), np.zeros(5), np.ones(5)), "camber.png"),
        (
            plotting.plot_curvature,
            (np.linspace(0, 1, 10), np.sin(np.linspace(0, 1, 10))),
            "curv.png",
        ),
        (
            plotting.plot_combined_curvature,
            (np.linspace(0, 1, 10), np.sin(np.linspace(0, 1, 10)), np.linspace(0, 1, 10), np.cos(np.linspace(0, 1, 10))),
            "combined_curv.png",
        ),
        (plotting.plot_asymmetry, (np.arange(5), np.zeros(5), np.ones(5)), "asym.png"),
    ],
)
def test_plot_functions_with_save(func, args, filename, tmp_path):
    save_path = tmp_path / filename
    func(*args, save_path=str(save_path))
    assert save_path.exists()
    plt.close("all")


def test_plot_raw_vs_interp(tmp_path):
    raw_x = np.linspace(0, 1, 5)
    raw_y = np.sin(raw_x)
    x_interp = np.linspace(0, 1, 10)
    y_interp = np.sin(x_interp)

    save_path = tmp_path / "raw_interp.png"

    plotting.plot_raw_vs_interp(
        raw_x,
        raw_y,
        raw_x,
        raw_y,
        x_interp,
        y_interp,
        y_interp,
        save_path=str(save_path),
    )

    assert save_path.exists()
    plt.close("all")


def test_plot_prandtl_meyer(tmp_path):
    save_path = tmp_path / "pm.png"

    plotting.plot_prandtl_meyer(1.4, save_path=str(save_path))
    assert save_path.exists()

    plotting.plot_prandtl_meyer(
        1.4,
        mach_points={"Inlet": 2.0, "Exit": 3.0},
    )
    plt.close("all")


def test_show_branch(monkeypatch):
    """Force non-headless backend so plt.show() branch is executed."""
    monkeypatch.setattr(matplotlib, "get_backend", lambda: "qt5agg")
    monkeypatch.setattr(plt, "show", lambda: None)

    x = np.linspace(0, 1, 5)
    y = np.sin(x)

    plotting.plot_contour(x, y, x, y)
    plotting.plot_interpolated_contour(x, y, y)
    plotting.plot_characteristics(x, y, x, y)
    plt.close("all")


@pytest.fixture
def non_headless(monkeypatch):
    """Force non-headless backend so plt.show() branch executes."""
    monkeypatch.setattr(matplotlib, "get_backend", lambda: "qt5agg")
    monkeypatch.setattr(plt, "show", lambda: None)  # prevent GUI


def test_plot_combined_curvature_show(non_headless):
    """Test combined curvature execution with show branch."""
    x = np.linspace(0, 1, 10)
    plotting.plot_combined_curvature(x, np.sin(x), x, np.cos(x))
    plt.close("all")
