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
def run(settings: str) -> None:
    """
    Compute blade geometry from a settings file.
    """
    cfg = _safe_load_config(settings)
    blade, result = _safe_compute_blade(cfg)

    typer.echo(f"Computed blade: {cfg.name}")
    typer.echo(f"Solidity: {result['solidity']:.4f}")


@app.command()
def plot(settings: str) -> None:
    """
    Plot blade contour from a settings file.
    """
    cfg = _safe_load_config(settings)
    blade, result = _safe_compute_blade(cfg)

    try:
        plot_interpolated_contour(
            result["x"],
            result["lower"],
            result["upper"],
            title=f"Blade Contour — {cfg.name}",
        )
    except Exception as e:
        typer.echo(f"Error plotting blade: {e}")
        raise typer.Exit(code=1)


@app.command()
def export(settings: str, outdir: str = "result") -> None:
    """
    Export blade geometry and metadata to disk.
    """
    cfg = _safe_load_config(settings)
    blade, result = _safe_compute_blade(cfg)

    metadata = {
        "name": cfg.name,
        "gamma": cfg.gamma,
        "mach_in": cfg.mach_in,
        "mach_out": cfg.mach_out,
        "solidity": result["solidity"],
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
