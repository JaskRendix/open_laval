from __future__ import annotations

import warnings
import numpy as np


class BladeValidationError(Exception):
    """Raised when critical blade validation rules fail."""
    pass


def validate_blade_geometry(x: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> list[str]:
    """
    Perform checks on interpolated blade geometry.
    Returns a list of warning messages, or raises BladeValidationError for critical issues.
    """
    warnings_list = []
    
    # 1. Thickness check
    thickness = upper - lower
    min_thickness = np.min(thickness)
    if min_thickness <= 0:
        raise BladeValidationError(f"Critical Error: Negative or zero thickness detected (min thickness: {min_thickness:.4f})")
    elif min_thickness < 0.005:
        msg = f"Warning: Very small minimum thickness detected ({min_thickness:.4f})."
        warnings.warn(msg, UserWarning)
        warnings_list.append(msg)

    # 2. Monotonicity of x coordinate
    if not np.all(np.diff(x) > 0):
        raise BladeValidationError("Critical Error: X-coordinates are not strictly monotonic.")

    # 3. Trailing edge / Leading edge closure checks
    crossings = np.sum((upper - lower) < 0)
    if crossings > 0:
        raise BladeValidationError(f"Critical Error: Upper and lower surfaces intersect at {crossings} points.")

    return warnings_list


def validate_config_limits(cfg) -> list[str]:
    """
    Validate configuration parameters against physical limits.
    """
    warnings_list = []
    
    if cfg.mach_in <= 1.0:
        msg = f"Warning: Inlet Mach number ({cfg.mach_in}) is subsonic. OpenLaval is optimized for supersonic blades."
        warnings.warn(msg, UserWarning)
        warnings_list.append(msg)
        
    if cfg.gamma <= 1.0:
        raise BladeValidationError(f"Invalid specific heat ratio gamma: {cfg.gamma}. Must be > 1.0.")
        
    return warnings_list
