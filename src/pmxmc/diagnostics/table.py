from sys import argv

import arviz as az


def print_table(idata):
    print(az.summary(idata, var_names=["^omega.*[^_]$"], filter_vars="regex"))
    print(az.summary(idata, var_names=["^theta.*[^_]$"], filter_vars="regex"))
    print(az.summary(idata, var_names=["^sigma.*[^_]$"], filter_vars="regex"))


def main():
    try:
        infile = argv[1]
    except IndexError:
        infile = "./idata.nc"
    idata = az.from_netcdf(infile)
    print_table(idata)


if __name__ == "__main__":
    main()
