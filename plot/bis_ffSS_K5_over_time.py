from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.backends.backend_pdf import PdfPages

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 28,
})

INPUT = Path("out/SPX/otm/full_performance_fixed/b_ffSS_K5.parquet")
OUTPUT = Path("plot/SPX/otm/full_performance_fixed/bis_ffSS_K5_over_time.pdf")

N_MATURITIES = 6
N_COEFS = 4

COL_LABELS = [
    r" -- intercept",
    r" -- moneyness",
    r" -- maturity",
    r" -- $-\omega$",
]

CRISIS_PERIODS = [
    (pd.Timestamp("2010-04-01"), pd.Timestamp("2012-07-01")),
    (pd.Timestamp("2015-08-01"), pd.Timestamp("2016-02-29")),
    (pd.Timestamp("2020-02-20"), pd.Timestamp("2020-06-01")),
    (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
]

df = (
    pl.read_parquet(INPUT)
    .with_columns(pl.col("date").str.strptime(pl.Date, "%Y%m%d"))
    .sort("date")
)

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
            sub = (
                df.filter((pl.col("k") == t) & (pl.col("j") == c))
                .sort("date")
            )
            dates = sub["date"].to_pandas()
            vals = sub["b"].to_numpy()
            is_is = sub["is_insample"].to_numpy()

            insample_vals = np.where(is_is, vals, np.nan)
            outsample_vals = np.where(~is_is, vals, np.nan)

            for start, end in CRISIS_PERIODS:
                ax.axvspan(start, end, color="gray", alpha=0.15, zorder=0)

            ax.plot(dates, insample_vals, color="steelblue", linewidth=0.6)
            ax.plot(dates, outsample_vals, color="firebrick", linewidth=0.6)

            ax.set_title(
                r"$b_{" + str(t) + r"}$" + COL_LABELS[c],
                fontsize=28,
            )
            ax.tick_params(axis="both", labelsize=22)
            if t == N_MATURITIES - 1:
                ax.tick_params(axis="x", rotation=30)

    is_patch = mpatches.Patch(color="steelblue", label="In-sample")
    oos_patch = mpatches.Patch(color="firebrick", label="Out-of-sample")
    fig.legend(
        handles=[is_patch, oos_patch],
        loc="lower center",
        ncol=2,
        fontsize=28,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.tight_layout(rect=[0, 0.03, 1, 1])
    pdf.savefig(fig)
    plt.close(fig)

print(f"Saved → {OUTPUT}")
