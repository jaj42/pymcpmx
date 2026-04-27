import numpy as np
from typing import IO
import polars as pl
import pytensor.tensor as pt

def extract_rate_schedule(df: pl.DataFrame) -> pl.DataFrame:
    """Infusion start/end records as (ID, TIME, RATE)."""
    doses = df.filter((pl.col("EVID") != 0) & (pl.col("RATE") > 0))
    starts = doses.select(
        pl.col('OCCID'),
        pl.col("TIME"),
        pl.col("RATE").cast(pl.Float64),
    )
    ends = doses.select(
        pl.col('OCCID'),
        (pl.col("TIME") + pl.col("AMT") / pl.col("RATE")).alias("TIME"),
        pl.lit(0.0).alias("RATE"),
    )
    # return (
    #     pl.concat([starts, ends])
    #     .sort(['OCCID', "TIME"])
    #     .unique(subset=['OCCID', "TIME"], keep="last", maintain_order=True)
    # )
    return (
    pl.concat([ends, starts])  # ends first, then starts
    .sort(['OCCID', "TIME"])
    .unique(subset=['OCCID', "TIME"], keep="last", maintain_order=True)
)


def extract_bolus_schedule(df: pl.DataFrame) -> pl.DataFrame:
    """Bolus dose records as (ID, TIME, AMT)."""
    return (
        df.filter((pl.col("EVID") != 0) & (pl.col("RATE") == 0) & (pl.col("AMT") > 0))
        .select(pl.col('OCCID'), pl.col("TIME"), pl.col("AMT").cast(pl.Float64))
        .sort(['OCCID', "TIME"])
    )


def build_time_grid(meas_time, infu_time, infu_rate, bolus_time, bolus_amt, tbeg, tend):
    """Unified time grid with vectorised rate lookup and bolus placement."""
    all_times = np.unique(np.concatenate([infu_time, bolus_time, meas_time]))
    all_times = all_times[(all_times >= tbeg) & (all_times <= tend)]
    dts = np.diff(all_times)
    query = all_times[:-1]

    if len(infu_time) > 0:
        idx = np.searchsorted(infu_time[1:], query, side="right")
        idx = np.clip(idx, 0, len(infu_rate) - 1)
        rates = np.where(query < infu_time[0], 0.0, infu_rate[idx])
    else:
        rates = np.zeros(len(dts))

    boluses = np.zeros(len(dts))
    for bt, ba in zip(bolus_time, bolus_amt):
        hits = np.where(query == bt)[0]
        if len(hits):
            boluses[hits[0]] += ba

    meas_idx = np.where(np.isin(all_times, meas_time))[0]
    return dts, rates, boluses, meas_idx


def read_nonmem_dataset(
    filepath: str | IO[str],
    covariates: list[str] | None = None,
    sep: str = r" ",
    dv_col: str = "DV",
):
    df = pl.read_csv(filepath, separator=sep,infer_schema_length=None)
    if "@ID" in df.columns:
        df = df.rename({"@ID": 'ID'})

    df = (
        # df.with_columns(pl.col("EVID").ge(3).cum_sum().over('ID').alias("OCC"))
        df.with_columns((pl.col("EVID") == 4).cum_sum().over('ID').alias("OCC"))
        .sort(['ID', 'OCC',"TIME"])
    )
    df = df.with_columns(df.select("ID", "OCC").hash_rows().alias("OCCID"))
    occdf = df.select("OCCID", "ID").unique(("OCCID"))
    occdf = occdf.to_dict()
    occid_map = dict(zip(occdf["OCCID"], occdf["ID"]))

    rate_sched = extract_rate_schedule(df)
    bolus_sched = extract_bolus_schedule(df)

    obs = (
        df.filter(pl.col("EVID") == 0)
        .select('OCCID', "TIME", dv_col)
        .unique(subset=['OCCID', "TIME"], maintain_order=True)
    )

    # occ_idx_map = {sid: i for i, sid in enumerate(set(occid_map.values()))}
    occ_idx_map = {sid: i for i, sid in enumerate(sorted(set(occid_map.values())))}

    id_indices=[]
    dts_list=[]
    rates_list=[]
    boluses_list=[]
    meas_idx_list=[]
    meas_dv_list = []
    n_meas_indiv_list =[]

    for occ_id in occid_map.keys():
        obs_np = obs.filter(pl.col('OCCID') == occ_id).to_numpy()
        meas_time = obs_np[:,1]
        if len(meas_time) == 0:
            continue
        meas_dv = obs_np[:,2]
        meas_dv_list.append(meas_dv)

        tbeg, tend = meas_time[0], meas_time[-1]
        infu_time_full = rate_sched.filter(pl.col('OCCID') == occ_id)["TIME"].to_numpy()
        bolus_time_full = bolus_sched.filter(pl.col('OCCID') == occ_id)["TIME"].to_numpy()
        if len(infu_time_full):
            tbeg = min(tbeg, infu_time_full[0])
        if len(bolus_time_full):
            tbeg = min(tbeg, bolus_time_full[0])

        occ_rate = rate_sched.filter(
            (pl.col('OCCID') == occ_id)
            # & (pl.col("TIME") >= tbeg)
            # & (pl.col("TIME") <= tend)
        )
        occ_bolus = bolus_sched.filter(
            (pl.col('OCCID') == occ_id)
            # & (pl.col("TIME") >= tbeg)
            # & (pl.col("TIME") <= tend)
        )

        dts, rates, boluses, meas_idx = build_time_grid(
            meas_time,
            occ_rate["TIME"].to_numpy(),
            occ_rate["RATE"].to_numpy(),
            occ_bolus["TIME"].to_numpy(),
            occ_bolus["AMT"].to_numpy(),
            tbeg,
            tend,
        )
        id_indices.append(occ_idx_map[int(occid_map[occ_id])])
        dts_list.append(dts)
        rates_list.append(rates)
        boluses_list.append(boluses)
        meas_idx_list.append(meas_idx)
        n_meas_indiv_list.append(len(meas_idx))

    n_occ = len(dts_list)
    max_steps = max(len(d) for d in dts_list)
    max_meas = max(len(m) for m in meas_idx_list)

    dts_padded = np.zeros((n_occ, max_steps))
    rates_padded = np.zeros((n_occ, max_steps))
    boluses_padded = np.zeros((n_occ, max_steps))
    meas_idx_padded = np.zeros((n_occ, max_meas), dtype=np.int32)

    for i, (d, r, b, mi) in enumerate(
        zip(dts_list, rates_list, boluses_list, meas_idx_list)
    ):
        dts_padded[i, : len(d)] = d
        rates_padded[i, : len(r)] = r
        boluses_padded[i, : len(b)] = b
        meas_idx_padded[i, : len(mi)] = mi

    valid_flat_indices = np.concatenate(
        [i * max_meas + np.arange(n) for i, n in enumerate(n_meas_indiv_list)]
    )

    return {
        'id':np.array(id_indices),
        'dt':pt.as_tensor_variable(dts_padded),
        'rate':pt.as_tensor_variable(rates_padded),
        'bolus':pt.as_tensor_variable(boluses_padded),
        'meas_idx':pt.as_tensor_variable(meas_idx_padded),
        'valid_idx':valid_flat_indices,
        'dv': np.concatenate(meas_dv_list),
        'occid_map': occid_map,
        }


if __name__ == "__main__":
    from pmxmc import assets
    from importlib import resources
    # with resources.open_text(assets, "eleveld.csv") as fd:
    #     ds= read_nonmem_dataset(fd, sep=" ", dv_col="DV")
    with resources.open_text(assets, "schnider.csv") as fd:
        ds= read_nonmem_dataset(fd,sep=',',dv_col='CP')

    print(f"{ds['id'].shape=}")
    print(f"{ds['dt'].type=}")
    print(f"{ds['rate'].type=}")
    print(f"{ds['bolus'].type=}")
    print(f"{ds['meas_idx'].type=}")
    print(f"{ds['valid_idx'].shape=}")
    print(f"{ds['dv'].shape=}")
    print(f"{ds['occid_map']}")
