from __future__ import annotations

from scipy.optimize import minimize

from .blade import Blade
from .evaluation import evaluate_blade_performance
from .validation import BladeValidationError


def optimize_blade_performance(
    cfg,
    design_vars: list[str] = ["vu", "vl"],
    method: str = "Nelder-Mead",
    maxiter: int = 50,
) -> dict:
    """
    Optimizes blade configuration parameters to maximize turning efficiency.
    """
    initial_x = [getattr(cfg, var) for var in design_vars]

    def objective(x):
        for var, val in zip(design_vars, x):
            setattr(cfg, var, val)
        try:
            blade = Blade(cfg)
            result = blade.compute()
            metrics = evaluate_blade_performance(result, cfg)
            # Minimize negative efficiency (equivalent to maximizing efficiency) + penalty for constraint failure
            efficiency = metrics.get("turning_efficiency", 0.0)
            loss = metrics.get("shock_loss_coefficient", 0.0) + metrics.get(
                "profile_loss_coefficient", 0.0
            )
            return loss - (0.5 * efficiency)
        except (BladeValidationError, Exception):
            # Heavy penalty for invalid geometries during optimization search
            return 10.0

    opt_options = {"maxiter": maxiter}
    if method == "Powell":
        opt_options["xtol"] = 1e-3
    else:
        opt_options["xatol"] = 1e-3

    res = minimize(
        objective, initial_x, method=method, options=opt_options
    )

    optimized_values = {var: val for var, val in zip(design_vars, res.x)}
    for var, val in optimized_values.items():
        setattr(cfg, var, val)

    try:
        final_blade, final_result = Blade(cfg), Blade(cfg).compute()
        final_metrics = evaluate_blade_performance(final_result, cfg)
    except (BladeValidationError, Exception):
        final_metrics = {
            "turning_efficiency": 0.0,
            "shock_loss_coefficient": 0.0,
            "profile_loss_coefficient": 0.0,
        }

    return {
        "success": res.success,
        "message": res.message,
        "optimized_parameters": optimized_values,
        "initial_objective": res.fun,
        "final_metrics": final_metrics,
    }
