import polars as pl
import numpy as np
import matplotlib
matplotlib.rcParams["text.usetex"] = True
matplotlib.rcParams["font.family"] = "serif"
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

src = "data/SPX/otm/full.parquet"
out = Path("out")
out.mkdir(exist_ok=True)

N_MON    = 6
OTM_PUT  = 2
OTM_CALL = 5
ROLL     = 21

mat_labels = {
    1: r"$[7,\,45]$ days",
    2: r"$(45,\,90]$ days",
    3: r"$(90,\,180]$ days",
    4: r"$(180,\,360]$ days",
}

df = pl.read_parquet(src)

daily = (
    df.with_columns([
        pl.col("DATE").str.to_date("%Y%m%d").alias("date"),
        (pl.col("bucket_idx") // N_MON + 1).alias("mat_bucket"),
        (pl.col("bucket_idx") % N_MON + 1).alias("mon_bucket"),
        pl.col("logIV").exp().alias("IV"),
        pl.col("moneyness").log(base=np.e).alias("log_moneyness"),
    ])
    .group_by(["date", "mat_bucket", "mon_bucket"])
    .agg([
        pl.col("IV").mean().alias("IV_mean"),
        pl.col("log_moneyness").mean().alias("lm_mean"),
    ])
)

put_side = (
    daily.filter(pl.col("mon_bucket") == OTM_PUT)
    .select(["date", "mat_bucket",
             pl.col("IV_mean").alias("IV_put"),
             pl.col("lm_mean").alias("lm_put")])
)
call_side = (
    daily.filter(pl.col("mon_bucket") == OTM_CALL)
    .select(["date", "mat_bucket",
             pl.col("IV_mean").alias("IV_call"),
             pl.col("lm_mean").alias("lm_call")])
)

skew_df = (
    put_side.join(call_side, on=["date", "mat_bucket"], how="inner")
    .with_columns(
        ((pl.col("IV_call") - pl.col("IV_put")) /
         (pl.col("lm_call") - pl.col("lm_put"))).alias("skew")
    )
)

parts = []
for mat in sorted(mat_labels):
    sub = (skew_df.filter(pl.col("mat_bucket") == mat)
           .sort("date")
           .with_columns(pl.col("skew").rolling_mean(window_size=ROLL).alias("skew_roll")))
    parts.append(sub)
skew_df = pl.concat(parts)

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)

for ax, mat, color in zip(axes.flat, sorted(mat_labels), colors):
    sub = skew_df.filter(pl.col("mat_bucket") == mat).to_pandas()
    ax.plot(sub["date"], sub["skew"],      color=color, alpha=0.2, linewidth=0.5)
    ax.plot(sub["date"], sub["skew_roll"], color=color, linewidth=1.5)
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_title(mat_labels[mat])
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(r"%Y"))

for ax in axes[:, 0]:
    ax.set_ylabel(r"$\partial \mathrm{IV} / \partial \log K$")

fig.suptitle(
    r"Volatility skew $\partial \mathrm{IV}/\partial \log K\,\big|_{\log(K/S)=0}$ "
    r"(OTM put/call proxy, " + str(ROLL) + r"-day rolling mean)",
    fontsize=11,
)
fig.tight_layout()
fig.savefig(out / "skew.pdf", bbox_inches="tight")
print("saved to out/skew.pdf")
