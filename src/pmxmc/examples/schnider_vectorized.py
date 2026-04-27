import os
from multiprocessing import cpu_count

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={cpu_count() - 2}"
os.environ["JAX_PLATFORMS"] = "cpu"

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import pymc as pm
import pytensor.tensor as pt
import nutpie
from pmxmc.advan.eigh import eigendecomposition
from pmxmc.io import read_nonmem_dataset_padded
from pmxmc.utils import add_omegas
from pymc_extras import inference
from pytensor import wrap_jax

jax.config.update("jax_enable_x64", True)


@wrap_jax
def _threecomp_single_occasion(
    dts, rates, boluses, meas_indices, k10, k12, k21, k13, k31, V1
):
    """Single-occasion 3-compartment PK solver (pure JAX, no batching)."""
    a2 = k12 * jnp.sqrt(k21 / k12)
    a3 = k13 * jnp.sqrt(k31 / k13)
    k123 = k10 + k12 + k13
    S = jnp.array(
        [
            [-k123,   a2,   a3],
            [   a2, -k21,    0],
            [   a3,    0, -k31],
        ]
    )  # fmt: skip
    lambdas, p_coef = eigendecomposition(S, V1, 0)
    state0 = jnp.zeros(3, dtype=jnp.float64)

    def step_fn(A, inputs):
        dt, rate, bolus = inputs
        A = A + bolus * lambdas * p_coef
        decay = jnp.exp(-lambdas * dt)
        A_new = A * decay + p_coef * rate * (1 - decay)
        return A_new, A_new

    _, all_states = jax.lax.scan(step_fn, state0, (dts, rates, boluses))
    all_states = jnp.concatenate([state0[None, :], all_states], axis=0)
    return jnp.sum(all_states[meas_indices], axis=-1)


threecomp_advan = pt.vectorize(
    _threecomp_single_occasion,
    signature="(n),(n),(n),(m),(),(),(),(),(),()->(m)",
)

def build_model(ds) -> pm.Model:
    n_subj = len(set(ds['occid_map'].values()))
    with pm.Model() as model:
        theta_V1 = pm.LogNormal("theta_V1", mu=np.log(4.5), sigma=0.5)
        theta_V2 = pm.LogNormal("theta_V2", mu=np.log(15), sigma=0.7)
        theta_V3 = pm.LogNormal("theta_V3", mu=np.log(250), sigma=1.0)
        theta_CL = pm.LogNormal("theta_CL", mu=np.log(1.5), sigma=0.5)
        theta_Q2 = pm.LogNormal("theta_Q2", mu=np.log(1.5), sigma=0.5)
        theta_Q3 = pm.LogNormal("theta_Q3", mu=np.log(0.8), sigma=0.5)

        sd_V1 = pm.HalfNormal("sd_V1", sigma=0.5)
        sd_V2 = pm.HalfNormal("sd_V2", sigma=0.5)
        # sd_V3 = pm.HalfNormal("sd_V3", sigma=0.5)
        sd_CL = pm.HalfNormal("sd_CL", sigma=0.5)
        sd_Q2 = pm.HalfNormal("sd_Q2", sigma=0.5)
        # sd_Q3 = pm.HalfNormal("sd_Q3", sigma=0.5)

        # sigma_add = pm.HalfNormal("sigma_add", sigma=5)
        sigma_prop = pm.HalfNormal("sigma_prop", sigma=0.5)

        eta_V1 = pm.Normal("eta_V1", mu=0, sigma=1, shape=n_subj)
        eta_V2 = pm.Normal("eta_V2", mu=0, sigma=1, shape=n_subj)
        # eta_V3 = pm.Normal("eta_V3", mu=0, sigma=1, shape=n_subj)
        eta_CL = pm.Normal("eta_CL", mu=0, sigma=1, shape=n_subj)
        eta_Q2 = pm.Normal("eta_Q2", mu=0, sigma=1, shape=n_subj)
        # eta_Q3 = pm.Normal("eta_Q3", mu=0, sigma=1, shape=n_subj)

        V1_i = theta_V1 * pt.exp(sd_V1 * eta_V1)
        V2_i = theta_V2 * pt.exp(sd_V2 * eta_V2)
        # V3_i = theta_V3 * pt.exp(sd_V3 * eta_V3)
        CL_i = theta_CL * pt.exp(sd_CL * eta_CL)
        Q2_i = theta_Q2 * pt.exp(sd_Q2 * eta_Q2)
        # Q3_i = theta_Q3 * pt.exp(sd_Q3 * eta_Q3)

        V1 = V1_i[ds['id']]
        V2 = V2_i[ds['id']]
        # V3 = V3_i[bio_indices]
        V3 = theta_V3
        CL = CL_i[ds['id']]
        Q2 = Q2_i[ds['id']]
        # Q3 = Q3_i[bio_indices]
        Q3 = theta_Q3

        all_Cp = threecomp_advan(
            ds['dt'],
            ds['rate'],
            ds['bolus'],
            ds['meas_idx'],
            CL / V1,  # k10
            Q2 / V1,  # k12
            Q2 / V2,  # k21
            Q3 / V1,  # k13
            Q3 / V3,  # k31
            V1,
        )
        IPRED = pt.flatten(all_Cp)[ds['valid_idx']]
        ERR = IPRED * sigma_prop
        # ERR = pt.sqrt((IPRED * sigma_prop )**2 + sigma_add**2)
        pm.Normal("C_obs", mu=IPRED, sigma=ERR, observed=ds['dv'])

    return model


def main():
    # rate, dv, covar, bio_map, bolus = read_nonmem_dataset("./eleveld.csv")

    from pmxmc import assets
    from importlib import resources
    # with resources.open_text(assets, "eleveld.csv") as fd:
    #     ds= read_nonmem_dataset(fd, sep=" ", dv_col="DV")
    with resources.open_text(assets, "schnider.csv") as fd:
        dataset= read_nonmem_dataset_padded(fd,sep=',',dv_col='CP')
    model = build_model(dataset)
    add_omegas(model)
    with model:
        compiled = nutpie.compile_pymc_model(model, backend="jax", gradient_backend="jax")
        idata = nutpie.sample(compiled)
        idata = inference.fit_laplace(model=model, gradient_backend="jax")
    az.to_netcdf(idata, "idata.nc")
    print(az.summary(idata))


if __name__ == "__main__":
    main()
