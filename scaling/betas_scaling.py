import json
import os
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from scipy import stats

from scaling.scaling_reg import moment_scaling

PARQUET_PATH = "data/SPX/put/bucket.parquet"
OUT_ROOT     = "out/SPX/put"
PLOT_ROOT    = "plot/SPX/put/betas_scaling"

FACTOR_COLS = ["level", "moneyness", "maturity"]
BETA_NAMES  = ["level", "moneyness", "maturity", "omega"]

MIN_SCALE   = 1.0
MAX_SCALE   = 126.0
TICK_DAYS   = np.array([1, 5, 21, 63, 126])
TICK_LABELS = ["1d", "1 week", "1 month", "3 months", "6 months"]


def find_json_with_betas(root):
    matches = []
    for dirpath, _, files in os.walk(root):
        for fname in sorted(files):
            if not fname.endswith(".json"):
                continue
            path = os.path.join(dirpath, fname)
            with open(path) as f:
                try:
                    d = json.load(f)
                except Exception:
                    continue
            if not isinstance(d, dict):
                continue
            if "betas" not in d:
                continue
            omega_key = "omega_load" if "omega_load" in d else ("omega" if "omega" in d else None)
            if omega_key is None:
                continue
            matches.append((path, d, omega_key))
    return matches


def plot_folder(json_path):
    rel = os.path.relpath(json_path, OUT_ROOT)
    rel_no_ext = os.path.splitext(rel)[0]
    return os.path.join(PLOT_ROOT, rel_no_ext)


def plot_scaling_q1(beta_series, name, out_dir):
    scaling = moment_scaling(beta_series, MIN_SCALE, MAX_SCALE, np.array([1.0]))
    delta_ts = scaling["delta_ts"]
    y = np.exp(scaling[1.0]["shifted_power_var"])

    log_t = np.log(delta_ts)
    log_y = np.log(y)
    good = np.isfinite(log_y)
    slope, intercept, *_ = stats.linregress(log_t[good], log_y[good])
    y_fit = np.exp(intercept + slope * log_t)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(FixedLocator(TICK_DAYS))
    ax.xaxis.set_major_formatter(FixedFormatter(TICK_LABELS))
    for x in TICK_DAYS:
        ax.axvline(x, linestyle="--", linewidth=0.6, color="black")
    ax.loglog(delta_ts, y, color="steelblue", linewidth=1.5, label="data")
    ax.loglog(delta_ts, y_fit, color="firebrick", linewidth=1.5, linestyle="--",
              label=f"OLS slope = {slope:.3f}")
    ax.legend()
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


def process(json_path, d, omega_key):
    out_dir = plot_folder(json_path)
    os.makedirs(out_dir, exist_ok=True)

    betas = np.array(d["betas"])
    omega = np.array(d[omega_key])

    hurst_summary = {}
    for i, name in enumerate(BETA_NAMES):
        series = betas[:, i]
        plot_scaling_q1(series, name, out_dir)
        h = plot_hurst_rs(series, name, out_dir)
        hurst_summary[name] = {"hurst_rs": float(h)}

    with open(os.path.join(out_dir, "hurst_summary.json"), "w") as f:
        json.dump(hurst_summary, f, indent=2)


def main():
    matches = find_json_with_betas(OUT_ROOT)
    print(f"Found {len(matches)} JSON files with betas")
    for json_path, d, omega_key in matches:
        rel = os.path.relpath(json_path, OUT_ROOT)
        print(f"  Processing {rel} (omega_key={omega_key})")
        process(json_path, d, omega_key)
        print(f"    -> plots saved to {plot_folder(json_path)}")
    print("Done.")


if __name__ == "__main__":
    main()
