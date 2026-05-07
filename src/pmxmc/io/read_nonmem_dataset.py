from functools import partial
from pathlib import Path
from typing import IO

import pandas as pd

pdtonum = partial(pd.to_numeric, errors="coerce")


def extract_rates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=['RATE','AMT'])
    doses = df.query("RATE > 0")
    doses["TINF"] = doses.eval("AMT / RATE")
    doses["TEND"] = doses.eval("TIME + TINF")

    rows = []
    for _, row in doses.iterrows():
        subjectid = row["ID"]
        # Infusion turns on
        rows.append({"ID": subjectid, "TIME": row["TIME"], "RATE": row["RATE"]})
        # Infusion turns off
        rows.append({"ID": subjectid, "TIME": row["TEND"], "RATE": 0.0})

    result = pd.DataFrame(rows)

    # If an end-time coincides exactly with the next start-time, keep the
    # start (last entry at that time) so the new infusion takes effect.
    result = result.drop_duplicates(subset=["ID", "TIME"], keep="last")

    return result.set_index(["ID", "TIME"]).sort_index()


def extract_boluses(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=['RATE','AMT'])
    boluses = df.query("AMT > 0 & RATE == 0")
    if boluses.empty:
        empty = pd.DataFrame(columns=["AMT"])
        empty.index = pd.MultiIndex.from_tuples([], names=["ID", "TIME"])
        return empty
    result = boluses[["ID", "TIME", "AMT"]].copy()
    return result.set_index(["ID", "TIME"]).sort_index()


def read_nonmem_dataset(
    filepath: str | Path | IO[str],
    covariates: list[str] | None = None,
    tv_covariates: list[str] | None = None,
    sep: str = r"\s+",
    dv_col: str = "DV",
    filter: str | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame]:
    df = pd.read_csv(filepath, sep=sep)
    df = df.rename(columns={"@ID": "ID"})
    if filter:
        df = df.query(filter)
    df = df.apply(pdtonum)
    df = df.sort_values(["ID", "TIME"])

    if "EVID" in df.columns:
        obs_record = df["EVID"] == 0
        infu_record = df["EVID"] == 1
        rate = extract_rates(df[infu_record])
        bolus = extract_boluses(df[infu_record])
        dv = df[obs_record].set_index(["ID", "TIME"])[dv_col]
    else:
        rate = extract_rates(df)
        bolus = extract_boluses(df)
        dv = df.set_index(["ID", "TIME"])[dv_col].dropna()

    dv = dv[~dv.index.duplicated(keep="first")]

    if covariates:
        covar = df.groupby("ID")[covariates].max()
    else:
        covar = pd.DataFrame([])

    if tv_covariates:
        tv_covar = df.set_index("ID")[tv_covariates]
    else:
        tv_covar = pd.DataFrame([])

    subj = df["ID"].unique()

    return {
        "subj": subj,
        "n_subj": len(subj),
        "rate": rate,
        "dv": dv,
        "covariates": covar,
        "tv_covariates": tv_covar,
        "bolus": bolus,
        "subj_idx": {bid: i for i, bid in enumerate(subj)},
    }


if __name__ == "__main__":
    from importlib import resources

    from pmxmc import assets

    with resources.open_text(assets, "eleveld.csv") as fd:
        res = read_nonmem_dataset(
            fd,
            sep=" ",
            covariates=["WGT", "HGT"],
            tv_covariates=["DVTY"],
            filter="STDY==13",  # Schnider
        )
    for k, v in res.items():
        print(k)
        print(v)
        print()

    dv = res["dv"]
    print(dv)
    print(dv.xs(354, level="ID"))
