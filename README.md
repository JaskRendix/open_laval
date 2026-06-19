# OpenLaval

OpenLaval is a numerical tool for generating **supersonic impulse turbine blade geometry** using the method of characteristics and classical vortex‑flow theory.  
It implements the design approach described in several NACA and NASA technical reports and produces upper and lower blade surfaces, interpolated contours, and derived quantities such as solidity.

This repository is a refactored and modular version of the original project published by Interstellar Technologies Inc.:  
[https://github.com/istellartech/OpenLaval](https://github.com/istellartech/OpenLaval)

---

## Features

- Supersonic impulse turbine blade design based on Prandtl–Meyer expansion theory  
- Method of characteristics for upper and lower surface construction  
- Circular arcs, concave/convex transitions, and leading‑edge shaping  
- Interpolated blade contour suitable for CAD or CFD preprocessing  
- Export of geometry and metadata  
- Plotting of contours and Prandtl–Meyer diagrams (headless‑safe)  
- Configuration through a simple `example.toml` file  
- Asymmetric blade support  

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

### Plot the blade contour

```
openlaval plot configs/example.toml
```

If `save_fig = true` in the configuration, the plot is saved automatically to:

```
result/<name>_contour.png
```

### Export geometry and metadata

```
openlaval export example.toml --outdir result/
```

All output files are written to the chosen directory.

---

## Input parameters

The configuration file defines:

- specific heat ratio  
- inlet and outlet Mach numbers  
- inlet flow angle  
- upper and lower Prandtl–Meyer angles (symmetric or asymmetric)  
- leading‑edge adjustment parameters  
- number of interpolation points  
- output options (`save_fig`, `save_excel`)  

See `example.toml` for an example.

---

## References

OpenLaval implements the design methods described in the following reports:

1. **NACA RM L52B06** — Application of Supersonic Vortex‑Flow Theory to the Design of Supersonic Impulse Compressor or Turbine‑Blade Sections  
2. **Design of Turbine Blades Suitable for Supersonic Relative Inlet Velocities and the Investigation of Their Performance in Cascades: Part I — Theory and Design**  
3. **Design of Turbine Blades Suitable for Supersonic Relative Inlet Velocities and the Investigation of Their Performance in Cascades: Part II — Experiments, Results and Discussion**  
4. **NASA TN D‑4421** — Analytical Investigation of Supersonic Turbomachinery Blading: I — Computer Program for Blading Design  
5. **NASA TN D‑4422** — Analytical Investigation of Supersonic Turbomachinery Blading: II — Analysis of Impulse Turbine‑Blade Sections  

---

## Status and future work

The refactored version provides a modular structure suitable for extension.

Possible future additions:

- improved evaluation functions  
- additional export formats  
- validation against published cascade data  
