import json

import numpy as np
import pandas as pd

from openlaval.io import (
    ensure_dir,
    save_all_results,
    save_contour_csv,
    save_contour_excel,
    save_metadata,
    save_raw_geometry,
)


def test_ensure_dir_creates_directory(tmp_path):
    target = tmp_path / "newdir"
    out = ensure_dir(target)
    assert out.exists()
    assert out.is_dir()


def test_ensure_dir_existing_directory(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    out = ensure_dir(tmp_path)
    assert out == tmp_path


def test_save_contour_csv(tmp_path):
    x = np.linspace(0, 1, 5)
    lower = x**2
    upper = x**3

    path = tmp_path / "contour.csv"
    save_contour_csv(path, x, lower, upper)

    df = pd.read_csv(path)
    assert list(df.columns) == ["x", "lower_surface", "upper_surface"]
    assert np.allclose(df["x"], x)
    assert np.allclose(df["lower_surface"], lower)
    assert np.allclose(df["upper_surface"], upper)


def test_save_contour_excel(tmp_path):
    x = np.linspace(0, 1, 5)
    lower = x**2
    upper = x**3

    path = tmp_path / "contour.xlsx"
    save_contour_excel(path, x, lower, upper)

    df = pd.read_excel(path)
    assert list(df.columns) == ["x", "lower_surface", "upper_surface"]
    assert np.allclose(df["x"], x)
    assert np.allclose(df["lower_surface"], lower)
    assert np.allclose(df["upper_surface"], upper)


def test_save_raw_geometry(tmp_path):
    lower_x = np.array([0, 1, 2])
    lower_y = np.array([0, -1, -2])
    upper_x = np.array([0, 1, 2])
    upper_y = np.array([1, 2, 3])

    path = tmp_path / "raw.csv"
    save_raw_geometry(path, lower_x, lower_y, upper_x, upper_y)

    df = pd.read_csv(path)
    assert list(df.columns) == ["lower_x", "lower_y", "upper_x", "upper_y"]
    assert np.allclose(df["lower_x"], lower_x)
    assert np.allclose(df["lower_y"], lower_y)
    assert np.allclose(df["upper_x"], upper_x)
    assert np.allclose(df["upper_y"], upper_y)


def test_save_metadata(tmp_path):
    metadata = {"solidity": 1.23, "mach_in": 2.0}
    path = tmp_path / "meta.json"

    save_metadata(path, metadata)

    with open(path) as f:
        loaded = json.load(f)

    assert loaded == metadata


def test_save_all_results_csv_only(tmp_path):
    x = np.linspace(0, 1, 5)
    lower = x**2
    upper = x**3

    lower_x = np.array([0, 1])
    lower_y = np.array([0, -1])
    upper_x = np.array([0, 1])
    upper_y = np.array([1, 2])

    metadata = {"solidity": 0.8}

    save_all_results(
        outdir=tmp_path,
        name="bladeA",
        x_interp=x,
        lower_interp=lower,
        upper_interp=upper,
        lower_x=lower_x,
        lower_y=lower_y,
        upper_x=upper_x,
        upper_y=upper_y,
        metadata=metadata,
        save_excel=False,
    )

    # Check files
    assert (tmp_path / "bladeA_contour.csv").exists()
    assert not (tmp_path / "bladeA_contour.xlsx").exists()
    assert (tmp_path / "bladeA_raw_geometry.csv").exists()
    assert (tmp_path / "bladeA_metadata.json").exists()


def test_save_all_results_with_excel(tmp_path):
    x = np.linspace(0, 1, 5)
    lower = x**2
    upper = x**3

    lower_x = np.array([0, 1])
    lower_y = np.array([0, -1])
    upper_x = np.array([0, 1])
    upper_y = np.array([1, 2])

    metadata = {"solidity": 0.8}

    save_all_results(
        outdir=tmp_path,
        name="bladeB",
        x_interp=x,
        lower_interp=lower,
        upper_interp=upper,
        lower_x=lower_x,
        lower_y=lower_y,
        upper_x=upper_x,
        upper_y=upper_y,
        metadata=metadata,
        save_excel=True,
    )

    assert (tmp_path / "bladeB_contour.csv").exists()
    assert (tmp_path / "bladeB_contour.xlsx").exists()
    assert (tmp_path / "bladeB_raw_geometry.csv").exists()
    assert (tmp_path / "bladeB_metadata.json").exists()


def test_save_contour_csv_empty_arrays(tmp_path):
    path = tmp_path / "empty.csv"
    save_contour_csv(path, [], [], [])
    df = pd.read_csv(path)
    assert df.empty


def test_save_raw_geometry_empty_arrays(tmp_path):
    path = tmp_path / "empty_raw.csv"
    save_raw_geometry(path, [], [], [], [])
    df = pd.read_csv(path)
    assert df.empty


def test_save_all_results_creates_directory(tmp_path):
    outdir = tmp_path / "nested" / "deep"
    x = [0, 1]
    lower = [0, -1]
    upper = [1, 2]

    save_all_results(
        outdir=outdir,
        name="bladeC",
        x_interp=x,
        lower_interp=lower,
        upper_interp=upper,
        lower_x=x,
        lower_y=lower,
        upper_x=x,
        upper_y=upper,
        metadata={"ok": True},
        save_excel=False,
    )

    assert (outdir / "bladeC_contour.csv").exists()
