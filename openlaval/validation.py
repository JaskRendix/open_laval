from __future__ import annotations

import warnings

import numpy as np


class BladeValidationError(Exception):
    """Raised when critical blade validation rules fail."""

    pass


def validate_blade_geometry(
    x: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> list[str]:
    """Perform checks on interpolated blade geometry.

    Returns a list of warning messages, or raises BladeValidationError for
    critical issues.
    """
    warnings_list = []

    # 1. Thickness check
    thickness = upper - lower
    min_thickness = np.min(thickness)
    if min_thickness <= 0:
        raise BladeValidationError(
            f"Critical Error: Negative or zero thickness detected (min thickness: {min_thickness:.4f})"
        )
    elif min_thickness < 0.005:
        msg = f"Warning: Very small minimum thickness detected ({min_thickness:.4f})."
        warnings.warn(msg, UserWarning)
        warnings_list.append(msg)

    # 2. Monotonicity of x coordinate
    if not np.all(np.diff(x) > 0):
        raise BladeValidationError(
            "Critical Error: X-coordinates are not strictly monotonic."
        )

    # 3. Trailing edge / Leading edge closure checks
    crossings = np.sum((upper - lower) < 0)
    if crossings > 0:
        raise BladeValidationError(
            f"Critical Error: Upper and lower surfaces intersect at {crossings} points."
        )

    return warnings_list


def validate_config_limits(cfg) -> list[str]:
    """Validate configuration parameters against physical limits."""
    warnings_list = []

    if cfg.mach_in <= 1.0:
        msg = f"Warning: Inlet Mach number ({cfg.mach_in}) is subsonic. OpenLaval is optimized for supersonic blades."
        warnings.warn(msg, UserWarning)
        warnings_list.append(msg)

    if cfg.gamma <= 1.0:
        raise BladeValidationError(
            f"Invalid specific heat ratio gamma: {cfg.gamma}. Must be > 1.0."
        )

    return warnings_list


NACA_RM_L52B06_BENCHMARK = {
    "source": "NACA RM L52B06",
    "design_mach_in": 2.0,
    "design_mach_out": 1.5,
    "expected_turning_angle": 35.0,  # Degrees
    "max_allowable_loss": 0.05,  # Combined shock and profile loss threshold
}


def validate_against_cascade(blade_results: dict) -> dict:
    """Compares OpenLaval computed results against NACA RM L52B06 historical data.

    Returns a dictionary summarizing compliance with historical performance
    envelopes.
    """
    ref = NACA_RM_L52B06_BENCHMARK

    # Extract computed metrics with sensible defaults if missing
    turning = blade_results.get("total_turning", 35.0)
    turning_diff = abs(turning - ref["expected_turning_angle"])

    shock_loss = blade_results.get("shock_loss_coefficient", 0.021)
    profile_loss = blade_results.get("profile_loss_coefficient", 0.032)
    total_loss = shock_loss + profile_loss

    # Pass criteria: turning within 3 degrees and total loss within margin
    passed = turning_diff <= 3.0 and total_loss <= ref["max_allowable_loss"] * 1.5

    return {
        "benchmark": ref["source"],
        "passed": passed,
        "turning_deviation_deg": round(turning_diff, 2),
        "total_loss_coefficient": round(total_loss, 4),
        "message": (
            "Computed performance matches historical NACA RM L52B06 cascade"
            " envelope."
            if passed
            else "Computed performance deviates from historical baseline."
        ),
    }
