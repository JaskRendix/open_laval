import math
import tomllib
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from openlaval.physics import phi_Rstar, prandtl_meyer


class BladeConfig(BaseModel):
    # [config]
    name: str
    save_fig: bool
    save_excel: bool
    num_points: int = Field(gt=0)

    # [turbine_blade]
    gamma: float = Field(gt=1.0)
    mach_in: float = Field(gt=1.0)
    mach_out: float = Field(gt=1.0)
    beta_in: float
    vu: float
    vl: float

    # [edge]
    edge_delta: float = Field(ge=0.0)
    edge_offset: float = Field(ge=0.0, le=0.5)

    @model_validator(mode="after")
    def validate_physics(self):
        nu_in = prandtl_meyer(self.mach_in, self.gamma)
        nu_out = prandtl_meyer(self.mach_out, self.gamma)

        nu_min = min(nu_in, nu_out)
        nu_max = max(nu_in, nu_out)

        # vl, vu must lie between ν_in and ν_out
        if not (nu_min <= self.vl < self.vu <= nu_max):
            raise ValueError(
                f"Invalid Prandtl–Meyer angles:\n"
                f"Expected: ν_min ≤ vl < vu ≤ ν_max\n"
                f"Given: vl={self.vl}, vu={self.vu}, "
                f"ν_in={nu_in:.2f}, ν_out={nu_out:.2f}"
            )

        # beta_in sanity
        if not (0 < self.beta_in < 90):
            raise ValueError("beta_in must be between 0 and 90 degrees")

        # leading-edge offset
        if not (0.0 <= self.edge_offset <= 0.5):
            raise ValueError("edge_offset must be between 0.0 and 0.5")

        return self


def load_config(path: str | Path) -> BladeConfig:
    data = tomllib.loads(Path(path).read_text())

    c = data["config"]
    t = data["turbine_blade"]
    e = data["edge"]

    return BladeConfig(
        name=c["name"],
        save_fig=c["save_fig"],
        save_excel=c["save_excel"],
        num_points=c["num_points"],
        gamma=t["gamma"],
        mach_in=t["mach_in"],
        mach_out=t["mach_out"],
        beta_in=t["beta_in"],
        vu=t["vu"],
        vl=t["vl"],
        edge_delta=e["delta"],
        edge_offset=e["offset"],
    )
