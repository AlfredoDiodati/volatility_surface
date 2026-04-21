import json
import os
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from scipy import stats

from scaling.scaling_reg import moment_scaling, _make_time_lags

PARQUET_PATH = "data/SPX/put/bucket.parquet"
PARAMS_PATH  = "out/SPX/put/params_adjSD.json"
OUTPUT_PATH  = "out/SPX/put/ols_betas_omega.json"
PLOT_DIR     = "plot/SPX/put/ols_betas_omega"

FACTOR_COLS = ["level", "moneyness", "maturity"]
BETA_NAMES  = ["level", "moneyness", "maturity", "omega"]

MIN_SCALE   = 1.0
MAX_SCALE   = 126.0
TICK_DAYS   = np.array([1, 5, 21, 63, 126])
TICK_LABELS = ["1d", "1 week", "1 month", "3 months", "6 months"]


def plot_scaling_q1(beta_series, name, out_dir):
    scaling = moment_scaling(beta_series, MIN_SCALE, MAX_SCALE, np.array([1.0]))
    delta_ts = scaling["delta_ts"]
    y = np.exp(scaling[1.0]["shifted_power_var"])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator(TICK_DAYS))
    ax.xaxis.set_major_formatter(FixedFormatter(TICK_LABELS))
    for x in TICK_DAYS:
        ax.axvline(x, linestyle="--", linewidth=0.6, color="black")
    ax.loglog(delta_ts, y, color="steelblue", linewidth=1.5)
    ax.set_xlabel(r"$\Delta t$")
    ax.set_ylabel(r"$S(1, \Delta t)$")
    ax.set_title(f"Moment scaling (q=1) — beta: {name}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"scaling_{name}.pdf"))
    plt.close(fig)


def rs_analysis(series, min_n=10, max_n=None, n_points=20):
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    T = len(x)
    if max_n is None:
        max_n = T // 2
    ns = np.unique(np.round(np.logspace(np.log10(min_n), np.log10(max_n), n_points)).astype(int))
    rs_vals = []
    valid_ns = []
    for n in ns:
        n_blocks = T // n
        if n_blocks < 2:
            continue
        rs_block = []
        for b in range(n_blocks):
            sub = x[b * n:(b + 1) * n]
            mean_sub = sub.mean()
            cumdev = np.cumsum(sub - mean_sub)
            r = cumdev.max() - cumdev.min()
            s = sub.std(ddof=1)
            if s > 0:
                rs_block.append(r / s)
        if rs_block:
            rs_vals.append(np.mean(rs_block))
            valid_ns.append(n)
    valid_ns = np.array(valid_ns, dtype=float)
    rs_vals = np.array(rs_vals)
    slope, intercept, *_ = stats.linregress(np.log(valid_ns), np.log(rs_vals))
    return slope, valid_ns, rs_vals, intercept


def plot_hurst_rs(beta_series, name, out_dir):
    h, ns, rs_vals, intercept = rs_analysis(beta_series)
    fitted = np.exp(intercept) * ns ** h

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(ns, rs_vals, "o", color="steelblue", markersize=5, label="R/S")
    ax.loglog(ns, fitted, "--", color="firebrick", linewidth=1.5, label=f"H = {h:.4f}")
    ax.set_xlabel("n")
    ax.set_ylabel("R/S(n)")
    ax.set_title(f"R/S analysis — beta: {name}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"hurst_{name}.pdf"))
    plt.close(fig)
    return h


def main():
    with open(PARAMS_PATH) as f:
        params = json.load(f)
    omega = np.array(params["omega"])

    raw = (
        pl.read_parquet(PARQUET_PATH)
        .with_columns(pl.col("DATE").cast(pl.Utf8))
    )
    bucket_cols = sorted([c for c in raw.columns if c.startswith("bucket_")])
    raw = raw.with_columns(
        pl.max_horizontal(
            [pl.when(pl.col(c)).then(i + 1).otherwise(0) for i, c in enumerate(bucket_cols)]
        ).alias("bucket_idx")
    )

    dates = raw["DATE"].unique(maintain_order=True).sort().to_list()

    results = {}
    for date in dates:
        day = raw.filter(pl.col("DATE") == date)
        y = day["logIV"].to_numpy()
        Z = day[FACTOR_COLS].to_numpy()
        bucket_idx = day["bucket_idx"].to_numpy().astype(int)
        omega_col = omega[bucket_idx].reshape(-1, 1)
        X = np.hstack([Z, omega_col])
        mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        if mask.sum() < X.shape[1]:
            results[date] = None
            continue
        beta, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
        results[date] = beta.tolist()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} dates to {OUTPUT_PATH}")

    valid_dates = [d for d in dates if results[d] is not None]
    beta_matrix = np.array([results[d] for d in valid_dates])

    os.makedirs(PLOT_DIR, exist_ok=True)

    hurst_summary = {}
    for i, name in enumerate(BETA_NAMES):
        series = beta_matrix[:, i]
        print(f"Processing beta: {name}")
        plot_scaling_q1(series, name, PLOT_DIR)
        h = plot_hurst_rs(series, name, PLOT_DIR)
        hurst_summary[name] = {"hurst_rs": float(h)}
        print(f"  H_RS={h:.4f}")

    summary_path = os.path.join(PLOT_DIR, "hurst_summary.json")
    with open(summary_path, "w") as f:
        json.dump(hurst_summary, f, indent=2)
    print(f"Hurst summary saved to {summary_path}")


if __name__ == "__main__":
    main()
