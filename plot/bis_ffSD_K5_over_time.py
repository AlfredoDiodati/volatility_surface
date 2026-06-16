from __future__ import annotations

from pathlib import Path
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

INPUT = Path("out/SPX/otm/full_performance/bis_ffSD_K5.parquet")
OUTPUT = Path("plot/SPX/otm/full_performance/bis_ffSD_K5_over_time.pdf")

N_MATURITIES = 6
N_COEFS = 4

df = pl.read_parquet(INPUT).with_columns(
    pl.col("date").str.strptime(pl.Date, "%Y%m%d").alias("date")
)

dates = df["date"].to_pandas()

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

with PdfPages(OUTPUT) as pdf:
    fig, axes = plt.subplots(
        N_MATURITIES, N_COEFS,
        figsize=(4 * N_COEFS, 2.5 * N_MATURITIES),
        sharex=True,
    )
    fig.suptitle("bis_ffSD_K5 — betas over time", fontsize=12)

    for t in range(N_MATURITIES):
        for c in range(N_COEFS):
            ax = axes[t, c]
            ax.plot(dates, df[f"b_T_{t}_{c}"].to_numpy(), linewidth=0.6)
            ax.set_title(f"T={t}, β{c}", fontsize=8)
            ax.tick_params(axis="both", labelsize=6)
            if t == N_MATURITIES - 1:
                ax.tick_params(axis="x", rotation=30)

    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

print(f"Saved → {OUTPUT}")
