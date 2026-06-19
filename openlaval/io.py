from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_contour_csv(path: str | Path, x, lower, upper) -> None:
    """
    Save interpolated blade contour to CSV.
    """
    df = pd.DataFrame(
        {
            "x": x,
            "lower_surface": lower,
            "upper_surface": upper,
        }
    )
    df.to_csv(path, index=False)


def save_contour_excel(path: str | Path, x, lower, upper) -> None:
    """
    Save interpolated blade contour to Excel.
    """
    df = pd.DataFrame(
        {
            "x": x,
            "lower_surface": lower,
            "upper_surface": upper,
        }
    )
    df.to_excel(path, index=False)


def save_raw_geometry(path: str | Path, lower_x, lower_y, upper_x, upper_y) -> None:
    """
    Save raw (non-interpolated) blade geometry.
    """
    df = pd.DataFrame(
        {
            "lower_x": lower_x,
            "lower_y": lower_y,
            "upper_x": upper_x,
            "upper_y": upper_y,
        }
    )
    df.to_csv(path, index=False)


def save_metadata(path: str | Path, metadata: dict) -> None:
    """
    Save metadata as JSON.
    """
    with open(path, "w") as f:
        json.dump(metadata, f, indent=4)


def save_all_results(
    outdir: str | Path,
    name: str,
    x_interp,
    lower_interp,
    upper_interp,
    lower_x,
    lower_y,
    upper_x,
    upper_y,
    metadata: dict,
    save_excel: bool = False,
) -> None:
    """
    Save all blade results (CSV, Excel, raw geometry, metadata).
    """
    outdir = ensure_dir(outdir)

    # Interpolated contour
    save_contour_csv(
        outdir / f"{name}_contour.csv", x_interp, lower_interp, upper_interp
    )

    if save_excel:
        save_contour_excel(
            outdir / f"{name}_contour.xlsx", x_interp, lower_interp, upper_interp
        )

    # Raw geometry
    save_raw_geometry(
        outdir / f"{name}_raw_geometry.csv", lower_x, lower_y, upper_x, upper_y
    )

    # Metadata
    save_metadata(outdir / f"{name}_metadata.json", metadata)
