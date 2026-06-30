from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 36,
})

INPUT_SD = Path("out/SPX/otm/full_performance_fixed/b_ffSD_K5.parquet")
INPUT_SS = Path("out/SPX/otm/full_performance_fixed/b_ffSS_K5.parquet")
OUTPUT = Path("plot/SPX/otm/full_performance_fixed/bis_ffSD_ffSS_K5_over_time.pdf")

N_MATURITIES = 6
N_COEFS = 4

COL_LABELS = [
    r" -- intercept",
    r" -- moneyness",
    r" -- maturity",
    r" -- $\omega_x$",
]

CRISIS_PERIODS = [
    (pd.Timestamp("2010-04-01"), pd.Timestamp("2012-07-01")),
    (pd.Timestamp("2015-08-01"), pd.Timestamp("2016-02-29")),
    (pd.Timestamp("2020-02-20"), pd.Timestamp("2020-06-01")),
    (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
]

IS_END = pd.Timestamp("2017-01-09")

COLOR_SD_IS  = "steelblue"
COLOR_SD_OOS = "firebrick"
COLOR_SS_IS  = "#0a2a6e"
COLOR_SS_OOS = "darkorange"

FMT = mticker.FormatStrFormatter("%.1f")


def load(path: Path) -> pl.DataFrame:
    return (
        pl.read_parquet(path)
        .with_columns(pl.col("date").str.strptime(pl.Date, "%Y%m%d"))
        .sort("date")
    )


df_sd = load(INPUT_SD)
df_ss = load(INPUT_SS)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with PdfPages(OUTPUT) as pdf:
    fig, axes = plt.subplots(
        N_MATURITIES, N_COEFS,
        figsize=(10 * N_COEFS, 6 * N_MATURITIES),
        sharex=True,
    )

    for t in range(N_MATURITIES):
        for c in range(N_COEFS):
            ax = axes[t, c]
            ax2 = ax.twinx()

            for df, c_is, c_oos, which_ax in [
                (df_sd, COLOR_SD_IS, COLOR_SD_OOS, ax),
                (df_ss, COLOR_SS_IS, COLOR_SS_OOS, ax2),
            ]:
                sub = df.filter((pl.col("k") == t) & (pl.col("j") == c)).sort("date")
                dates = sub["date"].to_pandas()
                vals = sub["b"].to_numpy()
                is_is = sub["is_insample"].to_numpy()
                which_ax.plot(dates, np.where(is_is, vals, np.nan), color=c_is, linewidth=0.8)
                which_ax.plot(dates, np.where(~is_is, vals, np.nan), color=c_oos, linewidth=0.8)
                which_ax.yaxis.set_major_formatter(FMT)

            for start, end in CRISIS_PERIODS:
                ax.axvspan(start, end, color="gray", alpha=0.15, zorder=0)

            ax.axvline(IS_END, color="black", linestyle="--", linewidth=1.2, zorder=3)

            ax.set_title(
                r"$b_{" + str(t) + r"}$" + COL_LABELS[c],
                fontsize=36,
            )
            ax.tick_params(axis="x", labelsize=28)
            ax.tick_params(axis="y", labelsize=28, colors=COLOR_SD_IS)
            ax2.tick_params(axis="y", labelsize=28, colors=COLOR_SS_IS)
            if t == N_MATURITIES - 1:
                ax.tick_params(axis="x", rotation=30)

    handles = [
        mpatches.Patch(color=COLOR_SD_IS,  label=r"Mk-SD -- in-sample"),
        mpatches.Patch(color=COLOR_SD_OOS, label=r"Mk-SD -- out-of-sample"),
        mpatches.Patch(color=COLOR_SS_IS,  label=r"Mk-SS -- in-sample"),
        mpatches.Patch(color=COLOR_SS_OOS, label=r"Mk-SS -- out-of-sample"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        fontsize=36,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    pdf.savefig(fig)
    plt.close(fig)

print(f"Saved → {OUTPUT}")
