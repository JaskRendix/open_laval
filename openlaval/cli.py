from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import typer

from .blade import Blade, BladeConfig
from .config import load_config
from .io import save_all_results, save_cfd_dat, save_csv_coordinates
from .plotting import (
    plot_asymmetry,
    plot_camber,
    plot_characteristics,
    plot_combined_curvature,
    plot_contour,
    plot_curvature,
    plot_interpolated_contour,
    plot_prandtl_meyer,
    plot_raw_vs_interp,
    plot_thickness,
)

app = typer.Typer(help="OpenLaval — Supersonic Impulse Turbine Blade Generator")


def _safe_load_config(path: str) -> BladeConfig:
    p = Path(path).expanduser().resolve()

    if not p.exists():
        typer.echo(f"Error: configuration file not found: {p}")
        raise typer.Exit(code=1)

    try:
        return load_config(str(p))
    except Exception as e:
        typer.echo(f"Error loading configuration: {e}")
        raise typer.Exit(code=1)


def _safe_compute_blade(cfg):
    try:
        blade = Blade(cfg)
        return blade, blade.compute()
    except Exception as e:
        typer.echo(f"Error computing blade: {e}")
        raise typer.Exit(code=1)


@app.command()
def run(
    settings: str,
    asymmetric: bool = typer.Option(
        None,
        help="Enable asymmetric blade mode (overrides config).",
    ),
    vl_lower: float = typer.Option(
        None,
        help="Lower-surface vl (overrides config).",
    ),
    vl_upper: float = typer.Option(
        None,
        help="Upper-surface vl (overrides config).",
    ),
    vu_lower: float = typer.Option(
        None,
        help="Lower-surface vu (overrides config).",
    ),
    vu_upper: float = typer.Option(
        None,
        help="Upper-surface vu (overrides config).",
    ),
):
    """
    Compute blade geometry from a settings file.
    """
    cfg = _safe_load_config(settings)

    # Override asymmetric mode
    if asymmetric is not None:
        cfg.asymmetric = asymmetric

    # Override asymmetric PM angles
    if vl_lower is not None:
        cfg.vl_lower = vl_lower
    if vl_upper is not None:
        cfg.vl_upper = vl_upper
    if vu_lower is not None:
        cfg.vu_lower = vu_lower
    if vu_upper is not None:
        cfg.vu_upper = vu_upper

    blade, result = _safe_compute_blade(cfg)

    typer.echo(f"Computed blade: {cfg.name}")
    typer.echo(f"Solidity: {result['solidity']:.4f}")


@app.command()
def batch(
    pattern: str = typer.Argument(..., help="Glob pattern for config files, e.g., 'configs/*.toml'"),
    output_dir: str = typer.Option("output_batch", help="Directory to save batch results"),
):
    """
    Run a batch of configuration files and summarize their geometric metrics.
    """
    files = glob.glob(pattern)
    if not files:
        typer.echo(f"No configuration files found matching pattern: {pattern}", err=True)
        raise typer.Exit(code=1)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    typer.echo(f"Found {len(files)} configurations. Running batch...")

    summary_data = []

    for file_path in files:
        try:
            cfg = _safe_load_config(file_path)
            blade, result = _safe_compute_blade(cfg)
            
            summary_data.append({
                "file": file_path,
                "chord": result.get("chord"),
                "max_thickness": result.get("max_thickness"),
                "solidity": result.get("solidity"),
            })
            typer.echo(f"  [SUCCESS] {file_path}")
        except Exception as e:
            typer.echo(f"  [ERROR] {file_path}: {e}", err=True)

    typer.echo("\n--- Batch Summary ---")
    for item in summary_data:
        typer.echo(f"{item['file']} -> Chord: {item['chord']:.3f}, Thick: {item['max_thickness']:.3f}, Solidity: {item['solidity']:.3f}")


@app.command()
def plot(
    settings: str,
    asymmetric: bool = typer.Option(
        None,
        help="Enable asymmetric blade mode (overrides config).",
    ),
    vl_lower: float = typer.Option(None),
    vl_upper: float = typer.Option(None),
    vu_lower: float = typer.Option(None),
    vu_upper: float = typer.Option(None),
):
    """
    Plot blade contour from a settings file.
    """
    cfg = _safe_load_config(settings)

    if asymmetric is not None:
        cfg.asymmetric = asymmetric
    if vl_lower is not None:
        cfg.vl_lower = vl_lower
    if vl_upper is not None:
        cfg.vl_upper = vl_upper
    if vu_lower is not None:
        cfg.vu_lower = vu_lower
    if vu_upper is not None:
        cfg.vu_upper = vu_upper

    blade, result = _safe_compute_blade(cfg)

    lower = np.array(result["lower"])
    upper = np.array(result["upper"])

    print("Lower surface (interp):")
    print(lower[:10])
    print("Upper surface (interp):")
    print(upper[:10])
    print("Difference (upper - lower):")
    print((upper - lower)[:10])

    try:
        plot_interpolated_contour(
            result["x"],
            result["lower"],
            result["upper"],
            title=f"Blade Contour — {cfg.name}",
            save_path=f"result/{cfg.name}_contour.png" if cfg.save_fig else None,
        )
    except Exception as e:
        typer.echo(f"Error plotting blade: {e}")
        raise typer.Exit(code=1)


@app.command("plot-raw")
def plot_raw(
    settings: str,
    asymmetric: bool = typer.Option(None),
    vl_lower: float = typer.Option(None),
    vl_upper: float = typer.Option(None),
    vu_lower: float = typer.Option(None),
    vu_upper: float = typer.Option(None),
):
    """
    Plot raw blade geometry (no interpolation).
    """
    cfg = _safe_load_config(settings)

    if asymmetric is not None:
        cfg.asymmetric = asymmetric
    if vl_lower is not None:
        cfg.vl_lower = vl_lower
    if vl_upper is not None:
        cfg.vl_upper = vl_upper
    if vu_lower is not None:
        cfg.vu_lower = vu_lower
    if vu_upper is not None:
        cfg.vu_upper = vu_upper

    blade, result = _safe_compute_blade(cfg)

    try:
        plot_contour(
            blade.lower_x,
            blade.lower_y,
            blade.upper_x,
            blade.upper_y,
            title=f"Raw Blade Geometry — {cfg.name}",
            save_path=f"result/{cfg.name}_raw.png" if cfg.save_fig else None,
        )
    except Exception as e:
        typer.echo(f"Error plotting raw geometry: {e}")
        raise typer.Exit(code=1)


@app.command("plot-char")
def plot_char(
    settings: str,
    asymmetric: bool = typer.Option(None),
    vl_lower: float = typer.Option(None),
    vl_upper: float = typer.Option(None),
    vu_lower: float = typer.Option(None),
    vu_upper: float = typer.Option(None),
):
    """
    Plot characteristic lines (compression & expansion).
    """
    cfg = _safe_load_config(settings)

    if asymmetric is not None:
        cfg.asymmetric = asymmetric
    if vl_lower is not None:
        cfg.vl_lower = vl_lower
    if vl_upper is not None:
        cfg.vl_upper = vl_upper
    if vu_lower is not None:
        cfg.vu_lower = vu_lower
    if vu_upper is not None:
        cfg.vu_upper = vu_upper

    blade, result = _safe_compute_blade(cfg)

    try:
        plot_characteristics(
            blade.x0,
            blade.y0,
            blade.x1,
            blade.y1,
            title=f"Characteristics — {cfg.name}",
            save_path=f"result/{cfg.name}_char.png" if cfg.save_fig else None,
        )
    except Exception as e:
        typer.echo(f"Error plotting characteristics: {e}")
        raise typer.Exit(code=1)


@app.command("plot-nu")
def plot_nu(
    settings: str,
):
    """
    Plot Prandtl–Meyer function ν(M) with inlet/outlet markers.
    """
    cfg = _safe_load_config(settings)

    mach_points = {
        "Inlet": cfg.mach_in,
        "Outlet": cfg.mach_out,
    }

    try:
        plot_prandtl_meyer(
            gamma=cfg.gamma,
            mach_points=mach_points,
            save_path=f"result/{cfg.name}_nu.png" if cfg.save_fig else None,
        )
    except Exception as e:
        typer.echo(f"Error plotting Prandtl–Meyer: {e}")
        raise typer.Exit(code=1)


@app.command("plot-thickness")
def plot_thickness_cmd(settings: str):
    cfg = _safe_load_config(settings)
    blade, result = _safe_compute_blade(cfg)

    plot_thickness(
        result["x"],
        result["lower"],
        result["upper"],
        title=f"Thickness — {cfg.name}",
        save_path=f"result/{cfg.name}_thickness.png" if cfg.save_fig else None,
    )


@app.command("plot-camber")
def plot_camber_cmd(settings: str):
    cfg = _safe_load_config(settings)
    blade, result = _safe_compute_blade(cfg)

    plot_camber(
        result["x"],
        result["lower"],
        result["upper"],
        title=f"Camber — {cfg.name}",
        save_path=f"result/{cfg.name}_camber.png" if cfg.save_fig else None,
    )


@app.command("plot-curvature")
def plot_curvature_cmd(settings: str):
    cfg = _safe_load_config(settings)
    blade, result = _safe_compute_blade(cfg)

    plot_curvature(
        blade.lower_x,
        blade.lower_y,
        title=f"Curvature Lower — {cfg.name}",
        save_path=f"result/{cfg.name}_curv_lower.png" if cfg.save_fig else None,
    )

    plot_curvature(
        blade.upper_x,
        blade.upper_y,
        title=f"Curvature Upper — {cfg.name}",
        save_path=f"result/{cfg.name}_curv_upper.png" if cfg.save_fig else None,
    )


@app.command("plot-curvature-combined")
def plot_curvature_combined_cmd(settings: str):
    """
    Plot upper and lower surface curvatures in a single comparative figure.
    """
    cfg = _safe_load_config(settings)
    blade, result = _safe_compute_blade(cfg)

    try:
        plot_combined_curvature(
            blade.lower_x,
            blade.lower_y,
            blade.upper_x,
            blade.upper_y,
            title=f"Comparative Curvature — {cfg.name}",
            save_path=f"result/{cfg.name}_curv_combined.png" if cfg.save_fig else None,
        )
        typer.echo(f"Successfully generated combined curvature plot for {cfg.name}")
    except Exception as e:
        typer.echo(f"Error plotting combined curvature: {e}", err=True)
        raise typer.Exit(code=1)


@app.command("plot-raw-vs-interp")
def plot_raw_vs_interp_cmd(settings: str):
    cfg = _safe_load_config(settings)
    blade, result = _safe_compute_blade(cfg)

    plot_raw_vs_interp(
        blade.lower_x,
        blade.lower_y,
        blade.upper_x,
        blade.upper_y,
        result["x"],
        result["lower"],
        result["upper"],
        title=f"Raw vs Interpolated — {cfg.name}",
        save_path=f"result/{cfg.name}_raw_vs_interp.png" if cfg.save_fig else None,
    )


@app.command("plot-asymmetry")
def plot_asymmetry_cmd(settings: str):
    cfg = _safe_load_config(settings)
    blade, result = _safe_compute_blade(cfg)

    plot_asymmetry(
        result["x"],
        result["lower"],
        result["upper"],
        title=f"Asymmetry — {cfg.name}",
        save_path=f"result/{cfg.name}_asymmetry.png" if cfg.save_fig else None,
    )


@app.command("export-cad")
def export_cad_cmd(settings: str, outdir: str = "result"):
    """
    Export specific CFD/CAD coordinate files (.dat and .csv).
    """
    cfg = _safe_load_config(settings)
    blade, result = _safe_compute_blade(cfg)

    try:
        dat_path = save_cfd_dat(outdir, cfg.name, result["x"], result["lower"], result["upper"])
        csv_path = save_csv_coordinates(outdir, cfg.name, result["x"], result["lower"], result["upper"])
        typer.echo(f"Successfully exported CAD/CFD formats:\n  - {dat_path}\n  - {csv_path}")
    except Exception as e:
        typer.echo(f"Error exporting CAD files: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def export(
    settings: str,
    outdir: str = "result",
    asymmetric: bool = typer.Option(
        None,
        help="Enable asymmetric blade mode (overrides config).",
    ),
    vl_lower: float = typer.Option(None),
    vl_upper: float = typer.Option(None),
    vu_lower: float = typer.Option(None),
    vu_upper: float = typer.Option(None),
):
    """
    Export blade geometry and metadata to disk.
    """
    cfg = _safe_load_config(settings)

    if asymmetric is not None:
        cfg.asymmetric = asymmetric
    if vl_lower is not None:
        cfg.vl_lower = vl_lower
    if vl_upper is not None:
        cfg.vl_upper = vl_upper
    if vu_lower is not None:
        cfg.vu_lower = vu_lower
    if vu_upper is not None:
        cfg.vu_upper = vu_upper

    blade, result = _safe_compute_blade(cfg)

    metadata = {
        "name": cfg.name,
        "gamma": cfg.gamma,
        "mach_in": cfg.mach_in,
        "mach_out": cfg.mach_out,
        "solidity": result["solidity"],
        "chord": result.get("chord"),
        "max_thickness": result.get("max_thickness"),
        "max_thickness_location": result.get("max_thickness_location"),
        "max_camber": result.get("max_camber"),
        "asymmetric": cfg.asymmetric,
        "vl_lower": cfg.vl_lower,
        "vl_upper": cfg.vl_upper,
        "vu_lower": cfg.vu_lower,
        "vu_upper": cfg.vu_upper,
    }

    try:
        save_all_results(
            outdir=outdir,
            name=cfg.name,
            x_interp=result["x"],
            lower_interp=result["lower"],
            upper_interp=result["upper"],
            lower_x=blade.lower_x,
            lower_y=blade.lower_y,
            upper_x=blade.upper_x,
            upper_y=blade.upper_y,
            metadata=metadata,
            save_excel=cfg.save_excel,
        )
    except Exception as e:
        typer.echo(f"Error exporting results: {e}")
        raise typer.Exit(code=1)

    typer.echo(f"Exported results to {outdir}")


def main() -> None:
    app()
