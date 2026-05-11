import re
from sys import argv

import arviz as az
import arviz_base as azb
import arviz_plots as azp
import arviz_stats as azs
import matplotlib.pyplot as plt
import pymc as pm
from matplotlib.backends.backend_pdf import PdfPages

azb.rcParams["plot.max_subplots"] = 500

patterns = {
    "omega": "^omega.*[^_]$",
    "theta": "^theta.*[^_]$",
    "sigma": "^sigma.*[^_]$",
    "eta": "^eta.*[^_]$",
}


def _available_parameters(idata):
    params = {k: False for k in patterns.keys()}
    for param_type, param_pattern in patterns.items():
        if any(re.search(param_pattern, v) for v in idata.posterior.data_vars):
            params[param_type] = True
    return params


def sample_predictive(trace, model=None):
    if model is None:
        model = pm.Model.get_context()
    with model:
        prior = pm.sample_prior_predictive(draws=1000)
        posterior = pm.sample_posterior_predictive(trace)
    idata = trace.copy()
    idata.extend(prior)
    idata.extend(posterior)
    return idata


def plot_param_type(idata, name, pattern, device):
    az.plot_trace(idata, var_names=[pattern], filter_vars="regex")
    plt.suptitle(f"{name} Trace")
    plt.tight_layout()
    device.savefig()
    plt.close()

    az.plot_posterior(idata, var_names=[pattern], filter_vars="regex")
    plt.suptitle(f"{name} posterior")
    plt.tight_layout()
    device.savefig()
    plt.close()

    # azp.plot_prior_posterior(idata, var_names=[pattern], filter_vars="regex")
    # plt.suptitle(f"{name} prior vs posterior")
    # plt.tight_layout()
    # pdf.savefig()
    # plt.close()

    fig, ax = plt.subplots()
    ax.axis("off")
    ax.table(
        azs.summary(idata, var_names=[pattern], filter_vars="regex"),
        loc="center",
    )
    fig.suptitle(name)
    plt.tight_layout()
    device.savefig(fig)
    plt.close()


def plot_idata(
    idata,
    output="output.pdf",
    model=None,
    prior_predictive=False,
    posterior_predictive=False,
):
    params = _available_parameters(idata)

    with PdfPages(output) as pdf:
        if prior_predictive:
            azp.plot_ppc_dist(idata, group="prior")
            plt.suptitle("Prior Predictive")
            pdf.savefig()
            plt.close()

        if posterior_predictive:
            az.plot_ppc(idata, group="posterior")
            plt.suptitle("Posterior Predictive")
            pdf.savefig()
            plt.close()

        # THETA
        if params["theta"]:
            plot_param_type(idata, "THETA", patterns["theta"], pdf)

        # OMEGA
        if params["omega"]:
            plot_param_type(idata, "OMEGA", patterns["omega"], pdf)

        # SIGMA
        if params["sigma"]:
            plot_param_type(idata, "SIGMA", patterns["sigma"], pdf)
        # az.plot_trace(idata, var_names=["^sigma.*[^_]$"], filter_vars="regex")
        # plt.suptitle("SIGMA Trace")
        # plt.tight_layout()
        # pdf.savefig()
        # plt.close()

        # ETA
        az.plot_trace(
            idata,
            var_names=[patterns["eta"]],
            filter_vars="regex",
        )
        plt.suptitle("ETA Trace")
        plt.tight_layout()
        pdf.savefig()
        plt.close()


def main():
    try:
        infile = argv[1]
    except IndexError:
        infile = "./idata.nc"
    idata = az.from_netcdf(infile)

    plot_idata(idata)


if __name__ == "__main__":
    main()
