import jax.numpy as jnp
import numpy as np
import pymc as pm
import pytensor.tensor as pt


def add_omegas(model=None):
    if model is None:
        model = pm.Model.get_context()
    with model:
        for name, var in model.named_vars.copy().items():
            if name.startswith("sd_"):
                pm.Deterministic(f"omega_{name[3:]}", var**2)


def add_IIV(variable, sigma, n_subj, model=None):
    if model is None:
        model = pm.Model.get_context()
    with model:
        for name, var in model.named_vars.copy().items():
            if name == f"theta_{variable}":
                pass
                sd = pm.HalfNormal(f"sd_{variable}", sigma=sigma)
                eta = pm.Normal(f"eta_{variable}", mu=0, sigma=1, shape=n_subj)
                return var * pt.exp(eta * sd)
        else:
            raise ValueError(f"theta_{variable} not in model")


def rate_at_numpy(t, infu_time, infu_rate):
    """Return piecewise-constant infusion rate at time t (numpy, for static use)."""
    if t < infu_time[0]:
        return 0.0
    idx = np.searchsorted(infu_time[1:], t, side="right")
    idx = np.clip(idx, 0, len(infu_rate) - 1)
    return infu_rate[idx]


def rate_at(t, infu_time, infu_rate):
    """Return piecewise-constant infusion rate at time t.

    Works with plain numpy scalars/arrays and with JAX traced values,
    so it can be called inside a jax.lax.scan with a traced lag offset.
    """
    infu_time = jnp.asarray(infu_time).ravel()
    infu_rate = jnp.asarray(infu_rate).ravel()
    idx = jnp.searchsorted(infu_time[1:], t, side="right")
    idx = jnp.clip(idx, 0, len(infu_rate) - 1)
    return jnp.where(t < infu_time[0], 0.0, infu_rate[idx])


def build_rate_func(infu_time, infu_rate):
    def worker(t):
        _infu_time = jnp.asarray(infu_time)
        _infu_rate = jnp.asarray(infu_rate)
        idx = jnp.searchsorted(_infu_time[1:], t, side="right")
        idx = jnp.clip(idx, 0, len(infu_rate) - 1)
        return jnp.where(t < _infu_time[0], 0.0, _infu_rate[idx])

    return worker
