import typer

from .blade import Blade
from .config import load_config
from .io import save_all_results
from .plotting import plot_interpolated_contour

app = typer.Typer(help="OpenLaval — Supersonic Impulse Turbine Blade Generator")


def _safe_load_config(path: str):
    try:
        return load_config(path)
    except FileNotFoundError:
        typer.echo(f"Error: configuration file not found: {path}")
        raise typer.Exit(code=1)
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
