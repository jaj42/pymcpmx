import os
from multiprocessing import cpu_count
from importlib import resources

os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={cpu_count() - 2}"
os.environ["JAX_PLATFORMS"] = "cpu"

from pmxmc import assets
import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import nutpie
import pymc as pm
import pytensor.tensor as pt
from pmxmc.advan import ode_advan as advan
from pmxmc.io import read_nonmem_dataset
from pmxmc.utils import add_omegas
from pymc.distributions import transforms
from pymc_extras.utils.prior import prior_from_idata

jax.config.update("jax_enable_x64", True)


def pk_ode(t, y, p):
    A1, A2, A3 = y
    k123 = p["k10"] + p["k12"] + p["k13"]
    ddt_A1 = -k123 * A1 + p["k21"] * A2 + p["k31"] * A3 + p["rate"](t)
    ddt_A2 = p["k12"] * A1 - p["k21"] * A2
    ddt_A3 = p["k13"] * A1 - p["k31"] * A3
    return jnp.array([ddt_A1, ddt_A2, ddt_A3])


def build_model(rates, dv, covar, bio_map, prior_idata=None) -> pm.Model:
    unique_occ_ids = dv.index.get_level_values("ID").unique()
    unique_bio_ids = sorted(bio_map.unique())
    n_subj = len(unique_bio_ids)
    bio_idx_map = {int(bid): i for i, bid in enumerate(unique_bio_ids)}

    DV = dv.to_numpy()

    with pm.Model() as model:
        if prior_idata is not None:
            prior_vars = [
                v
                for v in prior_idata["posterior"].data_vars
                if any(v.startswith(pfx) for pfx in ["theta", "sd", "sigma"])
            ]
            prior_kwargs = {v: transforms.log for v in prior_vars}
            p = prior_from_idata(prior_idata, **prior_kwargs)
            theta_V1, theta_V2, theta_V3 = p["theta_V1"], p["theta_V2"], p["theta_V3"]
            theta_CL, theta_Q2, theta_Q3 = p["theta_CL"], p["theta_Q2"], p["theta_Q3"]
            sd_CL, sd_V1, sd_V2, sd_Q2 = p["sd_CL"], p["sd_V1"], p["sd_V2"], p["sd_Q2"]
            sigma_prop = p["sigma_prop"]
        else:
            theta_V1 = pm.LogNormal("theta_V1", mu=np.log(4.5), sigma=0.5)
            theta_V2 = pm.LogNormal("theta_V2", mu=np.log(15), sigma=0.7)
            theta_V3 = pm.LogNormal("theta_V3", mu=np.log(250), sigma=1.0)
            theta_CL = pm.LogNormal("theta_CL", mu=np.log(1.5), sigma=0.5)
            theta_Q2 = pm.LogNormal("theta_Q2", mu=np.log(1.5), sigma=0.5)
            theta_Q3 = pm.LogNormal("theta_Q3", mu=np.log(0.8), sigma=0.5)

            sd_CL = pm.HalfNormal("sd_CL", sigma=0.5)
            sd_V1 = pm.HalfNormal("sd_V1", sigma=0.5)
            sd_V2 = pm.HalfNormal("sd_V2", sigma=0.5)
            # sd_V3 = pm.HalfNormal("sd_V3", sigma=0.5)
            sd_Q2 = pm.HalfNormal("sd_Q2", sigma=0.5)
            # sd_Q3 = pm.HalfNormal("sd_Q3", sigma=0.5)

            sigma_prop = pm.HalfNormal("sigma_prop", sigma=0.5)

        eta_CL = pm.Normal("eta_CL", mu=0, sigma=1, shape=n_subj)
        eta_V1 = pm.Normal("eta_V1", mu=0, sigma=1, shape=n_subj)
        eta_V2 = pm.Normal("eta_V2", mu=0, sigma=1, shape=n_subj)
        # eta_V3 = pm.Normal("eta_V3", mu=0, sigma=1, shape=n_subj)
        eta_Q2 = pm.Normal("eta_Q2", mu=0, sigma=1, shape=n_subj)
        # eta_Q3 = pm.Normal("eta_Q3", mu=0, sigma=1, shape=n_subj)

        V1_i = theta_V1 * pt.exp(sd_V1 * eta_V1)
        V2_i = theta_V2 * pt.exp(sd_V2 * eta_V2)
        V3 = theta_V3
        CL_i = theta_CL * pt.exp(sd_CL * eta_CL)
        Q2_i = theta_Q2 * pt.exp(sd_Q2 * eta_Q2)
        Q3 = theta_Q3

        C_preds = []
        for occ_id in unique_occ_ids:
            bio_id = int(bio_map.loc[occ_id])
            bio_idx = bio_idx_map[bio_id]

            meas_time = dv.xs(occ_id, level="ID").index.to_numpy().flatten()
            patient_rate = rates.xs(occ_id, level="ID")
            infu_time = patient_rate.index.to_numpy().flatten()
            infu_rate = patient_rate.to_numpy().flatten()

            V1 = V1_i[bio_idx]
            V2 = V2_i[bio_idx]
            # V3 = V3_i[bio_idx]
            CL = CL_i[bio_idx]
            Q2 = Q2_i[bio_idx]
            # Q3 = Q3_i[bio_idx]

            k10 = CL / V1
            k12 = Q2 / V1
            k21 = Q2 / V2
            k13 = Q3 / V1
            k31 = Q3 / V3
            params = {
                "k10": k10,
                "k12": k12, "k21": k21,
                "k13": k13, "k31": k31,
                "V1": V1, "V2": V2, "V3": V3,
            }  # fmt: skip

            Ap = advan(meas_time, infu_time, infu_rate, pk_ode, params, y0=[0, 0, 0])
            C_preds.append(Ap / V1)

        IPRED = pt.concatenate(C_preds)
        ERR = IPRED * sigma_prop
        pm.Normal("C_obs", mu=IPRED, sigma=ERR, observed=DV)

    return model


def main():
    with resources.open_text(assets, "schnider.csv") as fd:
        rate, dv, covar, bio_map, _bolus = read_nonmem_dataset(fd, sep=",", dv_col="CP")
    prior_idata = az.from_netcdf("idata_prior.nc")
    model = build_model(rate, dv, covar, bio_map, prior_idata=prior_idata)
    add_omegas(model)
    with model:
        # idata = inference.fit_dadvi(gradient_backend="jax")
        compiled = nutpie.compile_pymc_model(model, backend="jax")
        idata = nutpie.sample(
            compiled,
            draws=50,
            tune=20,
            chains=2,
        )
    az.to_netcdf(idata, "idata.nc")


if __name__ == "__main__":
    main()
