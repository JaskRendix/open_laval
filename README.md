# OpenLaval

OpenLaval is a numerical tool for generating **supersonic impulse turbine blade geometry** using the method of characteristics and classical vortex‑flow theory.

It implements the design approach described in several NACA and NASA technical reports and produces upper and lower blade surfaces, interpolated contours, and derived quantities such as solidity.

This repository is a refactored and modular version of the original project published by Interstellar Technologies Inc.:

[https://github.com/istellartech/OpenLaval](https://github.com/istellartech/OpenLaval)

---

## Features

* Supersonic impulse turbine blade design based on Prandtl–Meyer expansion theory
* Method of characteristics for upper and lower surface construction
* Circular arcs, concave/convex transitions, and leading‑edge shaping
* Interpolated blade contour suitable for CAD or CFD preprocessing
* Export of geometry, metadata, and CSV files
* Export of CAD/CFD formats (`.dat` and coordinate CSVs)
* Plotting of raw geometry, interpolated contours, thickness, camber, curvature, combined curvature, raw‑vs‑interpolated, asymmetry, and Prandtl–Meyer diagrams
* Automated batch processing of multiple configuration files
* Parametric design sweeps with CSV summary generation
* Validation of design constraints and configuration parameters
* Configuration through a simple TOML file
* Asymmetric blade support

---

## Installation

OpenLaval requires Python 3.12 or newer.

Install in editable mode:

```
pip install -e .

```

This installs the `openlaval` CLI.

---

## Usage

Prepare a configuration file, for example:

```
example.toml

```

You may also store multiple configurations under a directory such as `configs/`:

```
configs/example.toml
configs/asymmetric.toml
configs/high_mach.toml

```

### Compute geometry

```
openlaval run configs/example.toml

```

### Plot the interpolated blade contour

```
openlaval plot configs/example.toml

```

If `save_fig = true`, the plot is saved to:

```
result/<name>_contour.png

```

### Export geometry, metadata, and CSV files

```
openlaval export configs/example.toml --outdir result/

```

This writes:

* interpolated geometry CSV
* raw geometry CSV
* metadata JSON
* optional Excel file (`save_excel = true`)

---

## Additional plot commands

All plots respect `save_fig = true` and write PNG files to `result/`.

### Raw geometry

```
openlaval plot-raw <config>

```

Plots raw MOC‑generated upper and lower surfaces.

### Thickness distribution

```
openlaval plot-thickness <config>

```

Plots thickness as `(upper − lower)` along the chord.

### Camber line

```
openlaval plot-camber <config>

```

Plots camber as `0.5 * (upper + lower)`.

### Curvature distribution

```
openlaval plot-curvature <config>

```

Plots numerical curvature for lower and upper surfaces.

### Combined curvature view

```
openlaval plot-curvature-combined <config>

```

Plots upper and lower surface curvatures in a dual-subplot layout.

### Raw vs interpolated

```
openlaval plot-raw-vs-interp <config>

```

Overlays raw MOC geometry with the interpolated contour.

### Asymmetry

```
openlaval plot-asymmetry <config>

```

Plots the deviation `(upper − lower)` to highlight asymmetric designs.

### Prandtl–Meyer diagram

```
openlaval plot-nu <config>

```

Plots the Prandtl–Meyer function with inlet and outlet Mach markers.

---

## Advanced CLI operations

### Validate configuration

```
openlaval validate configs/example.toml

```

Checks configuration parameters and design constraints before computation.

### Export CAD and CFD formats

```
openlaval export-cad configs/example.toml --outdir result/

```

Exports `.dat` files and coordinate files for CAD and CFD preprocessing.

### Batch processing

```
openlaval batch "configs/*.toml" --output-dir batch_results/

```

Processes multiple TOML configuration files matching a glob pattern and outputs a consolidated summary.

### Parameter sweep

```
openlaval sweep configs/example.toml --param mach_in --start 1.5 --end 2.5 --steps 5 --output-dir sweep_results/

```

Performs parameter variation sweeps and generates a summary CSV tracking performance metrics.

---

## Input parameters

The configuration file defines:

* specific heat ratio
* inlet and outlet Mach numbers
* inlet flow angle
* symmetric or asymmetric Prandtl–Meyer angles
* leading‑edge parameters
* number of interpolation points
* output options (`save_fig`, `save_excel`)

See `example.toml` for a reference.

---

## References

OpenLaval implements the design methods described in the following reports:

1. NACA RM L52B06 — Application of Supersonic Vortex‑Flow Theory to the Design of Supersonic Impulse Compressor or Turbine‑Blade Sections
2. Design of Turbine Blades Suitable for Supersonic Relative Inlet Velocities and the Investigation of Their Performance in Cascades: Part I — Theory and Design
3. Design of Turbine Blades Suitable for Supersonic Relative Inlet Velocities and the Investigation of Their Performance in Cascades: Part II — Experiments, Results and Discussion
4. NASA TN D‑4421 — Analytical Investigation of Supersonic Turbomachinery Blading: I — Computer Program for Blading Design
5. NASA TN D‑4422 — Analytical Investigation of Supersonic Turbomachinery Blading: II — Analysis of Impulse Turbine‑Blade Sections

---

## Status and future work

The refactored version provides a modular structure suitable for extension.

Possible future additions:

* additional export formats
* validation against published cascade data
* automated blade optimization workflows
