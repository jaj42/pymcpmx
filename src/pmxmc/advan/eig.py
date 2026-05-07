import jax.numpy as jnp
import numpy as np
from jax import lax
from pytensor import wrap_jax

from pmxmc.utils import rate_at


def eigendecomposition(S, B, f=None, real_eigenvalues=True):
    eigvals, eigvecs = lax.linalg.eig(
        S,
        enable_eigvec_derivs=True,
        compute_right_eigenvectors=True,
        compute_left_eigenvectors=False,
    )
    if real_eigenvalues:
        eigvals = eigvals.real
        eigvecs = eigvecs.real
    lambdas = -eigvals

    Vinv = jnp.linalg.inv(eigvecs)
    alpha_B = Vinv @ B
    coefs = eigvecs @ jnp.diag(alpha_B / lambdas).T

    if f is not None:
        alpha_f = Vinv @ f
        coefs_f = eigvecs @ jnp.diag(alpha_f / lambdas).T
    else:
        coefs_f = jnp.zeros_like(coefs)

    return lambdas, coefs, coefs_f


@wrap_jax
def eig_advan(
    system_matrix,
    input_matrix,
    meas_time,
    infu_time,
    infu_rate,
    y0=None,
    forcing=None,
    real_eigenvalues=True,
):
    lambdas, coefs, coefs_f = eigendecomposition(
        system_matrix, input_matrix, forcing, real_eigenvalues
    )

    tbeg = min(meas_time[0], infu_time[0])
    tend = meas_time[-1]

    _relevant_itimes = infu_time[(infu_time >= tbeg) & (infu_time <= tend)]
    _all_times = np.unique(np.concatenate([_relevant_itimes, meas_time]))
    _dts = np.diff(_all_times)
    _rates = np.array([rate_at(t, infu_time, infu_rate) for t in _all_times[:-1]])

    dts = jnp.array(_dts)
    rates = jnp.array(_rates)

    state_dtype = jnp.float64 if real_eigenvalues else jnp.complex128
    if y0 is None:
        n_cmt = len(lambdas)
        y0 = jnp.zeros((n_cmt, n_cmt), dtype=state_dtype)
    else:
        y0 = jnp.asarray(y0, dtype=state_dtype)

    def step_fn(A, inputs):
        dt, rate = inputs
        decay = jnp.exp(-lambdas * dt)
        A_new = A * decay + (coefs * rate + coefs_f) * (1 - decay)
        return A_new, A_new

    _, all_states = lax.scan(step_fn, y0, (dts, rates))
    all_states_with_init = jnp.concatenate([y0[None, :, :], all_states], axis=0)

    _meas_indices = np.where(np.isin(_all_times, meas_time))[0]
    states_at_meas = all_states_with_init[_meas_indices]  # (n_meas, n_cmt, n_cmt)

    res = jnp.sum(states_at_meas, axis=-1)  # (n_meas, n_cmt)
    if real_eigenvalues:
        return res
    else:
        return res.real
