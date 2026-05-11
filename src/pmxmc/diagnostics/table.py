import re
from sys import argv

import arviz as az


def print_summary(idata, pattern):
    if any(re.search(pattern, v) for v in idata.posterior.data_vars):
        print(az.summary(idata, var_names=[pattern], filter_vars="regex"))


def print_table(idata):
    print_summary(idata, "^omega.*[^_]$")
    print_summary(idata, "^theta.*[^_]$")
    print_summary(idata, "^sigma.*[^_]$")
    # print(az.summary(idata, var_names=["^omega.*[^_]$"], filter_vars="regex"))
    # print(az.summary(idata, var_names=["^theta.*[^_]$"], filter_vars="regex"))
    # print(az.summary(idata, var_names=["^sigma.*[^_]$"], filter_vars="regex"))


def main():
    try:
        infile = argv[1]
    except IndexError:
        infile = "./idata.nc"
    idata = az.from_netcdf(infile)
    print_table(idata)


if __name__ == "__main__":
    main()
