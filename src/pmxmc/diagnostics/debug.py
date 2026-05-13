import numpy as np
import pymc as pm
import pytensor
from pymc_extras import find_MAP


def logp_contributions(model=None):
    """Return {var_name: logp} for every free and observed RV at initial_point."""
    if model is None:
        model = pm.Model.get_context()
    ip = model.initial_point()
    out = {}
    for rv in model.free_RVs + model.observed_RVs:
        try:
            val = float(model.compile_logp(vars=[rv], jacobian=False)(ip))
        except Exception:
            val = float("nan")
        out[rv.name] = val
    return out


def print_logp_at_ip(model=None):
    """Print per-variable logp at initial_point, flagging NaN/Inf contributors."""
    contributions = logp_contributions(model)
    finite_total = sum(v for v in contributions.values() if np.isfinite(v))
    width = max(len(k) for k in contributions) + 2
    print(f"{'Variable':<{width}} {'logp':>12}")
    print("-" * (width + 14))
    for name, val in contributions.items():
        flag = "  <-- NaN/Inf" if not np.isfinite(val) else ""
        print(f"{name:<{width}} {val:>12.4f}{flag}")
    print("-" * (width + 14))
    print(f"{'Total (finite only)':<{width}} {finite_total:>12.4f}")


def eval_at_ip(expr, model=None):
    """Evaluate a pytensor expression at model.initial_point().

    Uses pytensor.function directly with on_unused_input="ignore" so that
    the full initial_point dict can be passed regardless of which value_vars
    expr actually depends on.
    """
    if model is None:
        model = pm.Model.get_context()
    ip = model.initial_point()
    f = pytensor.function(
        inputs=model.value_vars,
        outputs=expr,
        on_unused_input="ignore",
        allow_input_downcast=True,
        mode="FAST_COMPILE",
    )
    vv_names = {v.name for v in model.value_vars}
    return f(**{k: v for k, v in ip.items() if k in vv_names})


def ofv_decomposed(point, model=None):
    """Return OFV split into likelihood and prior contributions at a given point.

    Parameters
    ----------
    point : dict
        Point in the model's transformed (value) space, as returned by
        pm.find_MAP() or model.initial_point().
    """
    if model is None:
        model = pm.Model.get_context()
    # find_MAP skips discrete value_vars (e.g. trace_prior_ from prior_from_idata)
    # and may return spurious keys (e.g. "posterior"). Build the evaluation point
    # by starting from initial_point() — which has every value_var — then
    # overwriting with the MAP values for the continuous parameters.
    vv_names = {v.name for v in model.value_vars}
    point = {
        **model.initial_point(),
        **{k: v for k, v in point.items() if k in vv_names},
    }
    log_lik = float(model.compile_logp(vars=model.observed_RVs, jacobian=False)(point))
    log_pri = float(model.compile_logp(vars=model.free_RVs, jacobian=False)(point))
    return {
        "likelihood": -2 * log_lik,
        "prior": -2 * log_pri,
        "joint": -2 * (log_lik + log_pri),
    }


def print_ofv_at_map(model=None):
    """Print OFV decomposed into likelihood and prior at a given point."""
    if model is None:
        model = pm.Model.get_context()
    print("Estimating maximum a posteriori (MAP) OFV")
    with model:
        map_est = find_MAP(gradient_backend="jax")
    ofv = ofv_decomposed(map_est, model)
    print(f"−2·logp (likelihood): {ofv['likelihood']:>12.2f}")
    print(f"−2·logp (prior):      {ofv['prior']:>12.2f}")
    print(f"−2·logp (joint):      {ofv['joint']:>12.2f}")
