from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_contour_csv(path: str | Path, x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> None:
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


def save_contour_excel(path: str | Path, x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> None:
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


def save_raw_geometry(
    path: str | Path, 
    lower_x: np.ndarray, 
    lower_y: np.ndarray, 
    upper_x: np.ndarray, 
    upper_y: np.ndarray
) -> None:
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


def save_cfd_dat(
    outdir: str | Path, 
    name: str, 
    x_interp: np.ndarray, 
    lower_interp: np.ndarray, 
    upper_interp: np.ndarray
) -> Path:
    """
    Export blade profile in standard .dat format (useful for CFD grids/solvers like SU2/ANSYS).
    Orders coordinates from trailing edge, along upper surface to leading edge, and back along lower surface.
    """
    out_path = ensure_dir(outdir)
    file_path = out_path / f"{name}_profile.dat"

    with open(file_path, "w") as f:
        f.write(f'Title="OpenLaval Blade Profile: {name}"\n')
        f.write("Variables=X, Y\n")
        f.write(f'Zone T="Blade", I={len(x_interp) * 2}\n')
        
        # Write upper surface (from trailing edge to leading edge)
        for x_val, y_val in zip(reversed(x_interp), reversed(upper_interp)):
            f.write(f"{x_val:.6f} {y_val:.6f}\n")
            
        # Write lower surface (from leading edge to trailing edge)
        for x_val, y_val in zip(x_interp, lower_interp):
            f.write(f"{x_val:.6f} {y_val:.6f}\n")

    return file_path


def save_csv_coordinates(
    outdir: str | Path, 
    name: str, 
    x_interp: np.ndarray, 
    lower_interp: np.ndarray, 
    upper_interp: np.ndarray
) -> Path:
    """
    Export separate structured CSV file with comprehensive points, thickness, and camber.
    """
    out_path = ensure_dir(outdir)
    file_path = out_path / f"{name}_coordinates.csv"
    
    thickness = upper_interp - lower_interp
    camber = 0.5 * (upper_interp + lower_interp)
    
    df = pd.DataFrame({
        "x": x_interp,
        "y_lower": lower_interp,
        "y_upper": upper_interp,
        "thickness": thickness,
        "camber": camber,
    })
    df.to_csv(file_path, index=False)
    return file_path


def save_all_results(
    outdir: str | Path,
    name: str,
    x_interp: np.ndarray,
    lower_interp: np.ndarray,
    upper_interp: np.ndarray,
    lower_x: np.ndarray,
    lower_y: np.ndarray,
    upper_x: np.ndarray,
    upper_y: np.ndarray,
    metadata: dict,
    save_excel: bool = False,
) -> None:
    """
    Save all blade results (CSV, Excel, raw geometry, CFD/CAD formats, metadata).
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

    # CFD and CAD Formats (.dat and structured coordinate CSV)
    save_cfd_dat(outdir, name, x_interp, lower_interp, upper_interp)
    save_csv_coordinates(outdir, name, x_interp, lower_interp, upper_interp)

    # Metadata
    save_metadata(outdir / f"{name}_metadata.json", metadata)
