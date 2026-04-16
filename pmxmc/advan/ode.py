import jax.numpy as jnp
import diffrax as dfx
from pytensor import wrap_jax
from pmxmc.utils import build_rate_func


@wrap_jax
def ode_advan(meas_time, infu_time, infu_rate, pk_ode, params, y0):
    y0 = jnp.asarray(y0)
    ode_term = dfx.ODETerm(pk_ode)
    solver = dfx.Tsit5()
    pid_controller = dfx.PIDController(rtol=1e-6, atol=1e-8)
    stepsize_controller = dfx.ClipStepSizeController(pid_controller, jump_ts=infu_time)
    max_steps = 100_000
    saveat = dfx.SaveAt(ts=meas_time)
    rate = build_rate_func(infu_time, infu_rate)
    tbeg = min(meas_time[0], infu_time[0])
    tend = meas_time[-1]

    sol = dfx.diffeqsolve(
        terms=ode_term,
        solver=solver,
        t0=tbeg,
        t1=tend,
        y0=y0,
        dt0=None,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
        saveat=saveat,
        args={**params, "rate": rate},
        throw=False,
    )

    return sol.ys[:, 0]
