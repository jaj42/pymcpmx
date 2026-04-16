import pymc as pm
import numpy as np
import jax.numpy as jnp


def add_omegas(model):
    with model:
        for name, var in model.named_vars.copy().items():
            if name.startswith("sd_"):
                pm.Deterministic(f"omega_{name[3:]}", var**2)


def rate_at_old(t, infu_time, infu_rate):
    """Return piecewise-constant infusion rate at time t (numpy, for static use)."""
    if t < infu_time[0]:
        return 0.0
    idx = np.searchsorted(infu_time[1:], t, side="right")
    idx = int(np.clip(idx, 0, len(infu_rate) - 1))
    return float(infu_rate[idx])


def rate_at(t, infu_time, infu_rate):
    """Return piecewise-constant infusion rate at time t (numpy, for static use)."""
    _infu = jnp.asarray(infu_time)
    idx = jnp.searchsorted(infu_time[1:], t, side="right")
    idx = jnp.clip(idx, 0, len(infu_rate) - 1)
    return jnp.where(t < _infu[0], 0.0, _infu[idx])


def build_rate_func(infu_time, infu_rate):
    def worker(t):
        return rate_at(t, infu_time, infu_rate)

    return worker
