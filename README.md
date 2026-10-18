# OpenLaval — Supersonic Impulse Turbine Blade Generator

OpenLaval is a numerical tool for generating supersonic impulse turbine blade geometry using the method of characteristics and classical vortex‑flow theory.

It implements the design approach described in several NACA and NASA technical reports and produces upper and lower blade surfaces, interpolated contours, and derived quantities such as solidity.

This repository is a refactored and modular version of the original project published by Interstellar Technologies Inc.:  
<https://github.com/istellartech/OpenLaval>

---

## Features

- Supersonic impulse turbine blade design based on Prandtl–Meyer expansion theory  
- Method of characteristics for upper and lower surface construction  
- Circular arcs, concave/convex transitions, and leading‑edge shaping  
- Interpolated blade contour suitable for CAD or CFD preprocessing  
- Aerodynamic performance evaluation (isentropic loss coefficients, shock losses, turning efficiency, curvature smoothness)  
- Automated blade performance optimization workflows using `scipy.optimize`  
- Export of geometry, metadata, and CSV files  
- Export of CAD/CFD formats (`.dat` and coordinate CSVs)  
- Plotting of raw geometry, interpolated contours, thickness, camber, curvature, combined curvature, raw‑vs‑interpolated, asymmetry, and Prandtl–Meyer diagrams  
- Automated batch processing of multiple configuration files  
- Parametric design sweeps with CSV summary generation  
- Validation of design constraints and configuration parameters  
- Configuration through a simple TOML file  
- Asymmetric blade support  

---

## Installation

OpenLaval requires Python 3.12 or newer.

Install in editable mode:

```bash
pip install -e .
```

This installs the `openlaval` CLI.

---

## Usage

Prepare a configuration file, for example:

```
example.toml
```

Or store multiple configurations under a directory:

```
configs/example.toml
configs/asymmetric.toml
configs/high_mach.toml
```

### Compute geometry

```bash
openlaval run configs/example.toml
```

### Plot the interpolated blade contour

```bash
openlaval plot configs/example.toml
```

If `save_fig = true`, the plot is saved to:

```
result/<name>_contour.png
```

### Export geometry, metadata, and CSV files

```bash
openlaval export configs/example.toml --outdir result/
```

This writes:

- interpolated geometry CSV  
- raw geometry CSV  
- metadata JSON  
- optional Excel file (`save_excel = true`)  

---

## Additional plot commands

All plots respect `save_fig = true` and write PNG files to `result/`.

### Raw geometry

```bash
openlaval plot-raw <config>
```

### Thickness distribution

```bash
openlaval plot-thickness <config>
```

### Camber line

```bash
openlaval plot-camber <config>
```

### Curvature distribution

```bash
openlaval plot-curvature <config>
```

### Combined curvature view

```bash
openlaval plot-curvature-combined <config>
```

### Raw vs interpolated

```bash
openlaval plot-raw-vs-interp <config>
```

### Asymmetry

```bash
openlaval plot-asymmetry <config>
```

### Prandtl–Meyer diagram

```bash
openlaval plot-nu <config>
```

---

## Advanced CLI operations

### Validate configuration

```bash
openlaval validate configs/example.toml
```

### Evaluate blade performance

```bash
openlaval evaluate configs/example.toml
```

### Automated optimization

```bash
openlaval optimize configs/example.toml
```

### Export CAD and CFD formats

```bash
openlaval export-cad configs/example.toml --outdir result/
```

### Batch processing

```bash
openlaval batch "configs/*.toml" --output-dir batch_results/
```

### Parameter sweep

```bash
openlaval sweep configs/example.toml --param mach_in --start 1.5 --end 2.5 --steps 5 --output-dir sweep_results/
```

---

## Input parameters

The configuration file defines:

- specific heat ratio  
- inlet and outlet Mach numbers  
- inlet flow angle  
- symmetric or asymmetric Prandtl–Meyer angles  
- leading‑edge parameters  
- number of interpolation points  
- output options (`save_fig`, `save_excel`)  

See `example.toml` for a reference.

---

## References

OpenLaval implements the design methods described in:

1. Boxer, E., Sterrett, J. R., & Wlodarski, J. (1952). *Application of Supersonic Vortex-Flow Theory to the Design of Supersonic Impulse Compressor- or Turbine-Blade Sections* (NACA RM L52B06). National Advisory Committee for Aeronautics.  
   https://ntrs.nasa.gov/citations/19930087012

2. Colclough, C. D. (1966). *Design of Turbine Blades Suitable for Supersonic Relative Inlet Velocities and the Investigation of Their Performance in Cascades: Part I – Theory and Design*. Journal of Mechanical Engineering Science, 8(1).  
   https://doi.org/10.1243/JMES_JOUR_1966_008_015_02

3. Colclough, C. D. (1966). *Design of Turbine Blades Suitable for Supersonic Relative Inlet Velocities and the Investigation of Their Performance in Cascades: Part II – Experiments, Results and Discussion*. Journal of Mechanical Engineering Science, 8(2).  
   https://doi.org/10.1243/JMES_JOUR_1966_008_023_02

4. Goldman, L. J., & Scullin, V. J. (1968). *Analytical Investigation of Supersonic Turbomachinery Blading: I – Computer Program for Blading Design* (NASA TN D-4421). NASA Lewis Research Center.  
   https://ntrs.nasa.gov/citations/19680009151

5. Goldman, L. J. (1968). *Analytical Investigation of Supersonic Turbomachinery Blading: II – Analysis of Impulse Turbine-Blade Sections* (NASA TN D-4422). NASA Lewis Research Center.  
   https://ntrs.nasa.gov/citations/19680010807

---

## Status and future work

The refactored version provides a modular structure suitable for extension.

Possible future additions:

- Additional export formats: STEP, DXF, STL using libraries like `trimesh`, `CadQuery`, or `gmsh`.
