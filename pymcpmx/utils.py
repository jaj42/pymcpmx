import pymc as pm
import numpy as np


def add_omegas(model):
    with model:
        for name, var in model.named_vars.copy().items():
            if name.startswith("sd_"):
                pm.Deterministic(f"omega_{name[3:]}", var**2)


def rate_at(t, infu_time, infu_rate):
    """Return piecewise-constant infusion rate at time t (numpy, for static use)."""
    if t < infu_time[0]:
        return 0.0
    idx = np.searchsorted(infu_time[1:], t, side="right")
    idx = int(np.clip(idx, 0, len(infu_rate) - 1))
    return float(infu_rate[idx])
