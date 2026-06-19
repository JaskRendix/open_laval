# OpenLaval

OpenLaval is a numerical tool for generating **supersonic impulse turbine blade geometry** using the method of characteristics and classical vortex‑flow theory.  
It implements the design approach described in several NACA and NASA technical reports and produces upper and lower blade surfaces, interpolated contours, and derived quantities such as solidity.

This repository is a refactored and modular version of the original project published by Interstellar Technologies Inc.:  
[https://github.com/istellartech/OpenLaval](https://github.com/istellartech/OpenLaval)

---

## Features

- Supersonic impulse turbine blade design based on Prandtl–Meyer expansion theory  
- Method of characteristics for upper and lower surface construction  
- Circular arcs, concave and convex transition regions, and leading‑edge shaping  
- Interpolated blade contour suitable for CAD or CFD preprocessing  
- Export of geometry and metadata  
- Plotting of contours and Prandtl–Meyer diagrams  
- Configuration through a simple `settings.ini` file  
- asymmetric blade support  

---

## Installation

OpenLaval requires Python 3.12 or newer.

Install the package in editable mode:

```
pip install -e .
```

This installs the `openlaval` CLI.

---

## Usage

Prepare a configuration file:

```
settings.ini
```

Then run one of the CLI commands.

### Compute geometry

```
openlaval run settings.ini
```

### Plot the blade contour

```
openlaval plot settings.ini
```

### Export geometry and metadata

```
openlaval export settings.ini --outdir result/
```

All output files are written to the chosen directory.

---

## Input parameters

The configuration file defines:

- specific heat ratio  
- inlet and outlet Mach numbers  
- inlet flow angle  
- upper and lower Prandtl–Meyer angles  
- leading‑edge adjustment parameters  
- number of interpolation points  
- output options  

See `settings.ini` for an example.

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
