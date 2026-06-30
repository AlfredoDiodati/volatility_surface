from __future__ import annotations

import json
from pathlib import Path

import jax
from _device import get_device as _get_device
jax.config.update("jax_default_device", _get_device())
import jax.numpy as jnp
import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.backends.backend_pdf import PdfPages

from models.ff_SD import _solve_weights_ff as ffSD_weights
from models.ff_SS import _solve_weights_ff as ffSS_weights
from models.adjSD import _filter as adjSD_filter
from models.lmSD import _filter as lmSD_filter
from models.ss import fit_collapsed as ss_fit_collapsed

matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 36,
})

PARQUET_PATH = Path("data/SPX/otm/full.parquet")
OUTPUT_DIR = Path("out/SPX/otm/full_performance_fixed")
OUTPUT = Path("plot/SPX/otm/full_performance_fixed/betas_over_time.pdf")

FACTOR_LOADING_COLS = ["level", "moneyness", "maturity"]
P_BASE = 3
P = P_BASE + 1
K = 5

CRISIS_PERIODS = [
    (pd.Timestamp("2010-04-01"), pd.Timestamp("2012-07-01")),
    (pd.Timestamp("2015-08-01"), pd.Timestamp("2016-02-29")),
    (pd.Timestamp("2020-02-20"), pd.Timestamp("2020-06-01")),
    (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
]

IS_END = pd.Timestamp("2017-01-09")

COL_LABELS = [r"intercept", r"moneyness", r"maturity", r"$\omega_x$"]

MODELS = {
    r"SS":     "dimgray",
    r"SD":     "#2ca02c",
    r"FI-SD":  "#9467bd",
    r"Mk-SD":  "steelblue",
    r"Mk-SS":  "#0a2a6e",
}


def _load_data():
    raw = (
        pl.read_parquet(PARQUET_PATH)
        .with_columns(pl.col("DATE").cast(pl.Utf8))
        .sort(["DATE", *FACTOR_LOADING_COLS])
        .with_columns(
            pl.int_range(pl.len()).over("DATE").cast(pl.Int32).alias("row_in_date")
        )
    )
    dates = raw["DATE"].unique(maintain_order=True).sort().to_list()
    T = len(dates)
    max_n = int(raw.group_by("DATE").len()["len"].max())

    raw = raw.with_columns(
        pl.col("DATE").replace_strict(
            {d: i for i, d in enumerate(dates)}, return_dtype=pl.Int32
        ).alias("date_idx")
    )

    t_idx = jnp.array(raw["date_idx"], dtype=jnp.int32)
    n_idx = jnp.array(raw["row_in_date"], dtype=jnp.int32)
    y_vals = jnp.array(raw["logIV"])
    factor_vals = jnp.stack([jnp.array(raw[c]) for c in FACTOR_LOADING_COLS], axis=1)
    bucket_vals = jnp.array(raw["bucket_idx"], dtype=jnp.float32)

    y_cube = jnp.full((T, max_n), jnp.nan).at[t_idx, n_idx].set(y_vals)
    Z_cube = jnp.zeros((T, max_n, P)).at[t_idx, n_idx].set(
        jnp.concatenate([factor_vals, bucket_vals[:, None]], axis=-1)
    )
    return y_cube, Z_cube, dates


def _betas_from_b(parquet_name, params_name, weight_fn, phi_key, phi_is_scalar):
    p = json.load(open(OUTPUT_DIR / params_name))
    b_df = (
        pl.read_parquet(OUTPUT_DIR / parquet_name)
        .with_columns(pl.col("date").str.strptime(pl.Date, "%Y%m%d"))
        .sort(["date", "k", "j"])
    )
    dates_pd = b_df["date"].unique(maintain_order=True).sort().to_pandas()

    beta_bar = np.array(p["beta_bar"])
    eta = jnp.array(p["eta"])
    phi_val = jnp.full(P, p[phi_key]) if phi_is_scalar else jnp.array(p[phi_key])
    ws, _ = weight_fn(eta, phi_val, K)
    ws_np = np.array(ws)

    n_dates = len(dates_pd)
    b_arr = b_df.select("b").to_numpy().reshape(n_dates, K + 1, P)
    betas = beta_bar + (ws_np[np.newaxis] * b_arr).sum(axis=1)
    return dates_pd, betas


def _betas_adjSD(y_masked, base_cov, bucket_idx, mask_f, dates):
    p = json.load(open(OUTPUT_DIR / "params_adjSD.json"))
    params = {
        "beta_bar": jnp.array(p["beta_bar"]),
        "B": jnp.diag(jnp.array(p["B_diag"])),
        "A": jnp.diag(jnp.array(p["A_diag"])),
        "sigma2": p["sigma2"],
        "C": jnp.array(p["C_diag"]),
        "nu": p["nu"],
        "omega": jnp.array(p["omega"]),
    }
    betas, _, _ = adjSD_filter(
        y_masked, base_cov, bucket_idx, mask_f, params, params["beta_bar"]
    )
    return pd.to_datetime(dates, format="%Y%m%d"), np.array(betas)


def _betas_lmSD(y_masked, base_cov, bucket_idx, mask_f, dates):
    p = json.load(open(OUTPUT_DIR / "params_lmSD.json"))
    T = y_masked.shape[0]
    params = {
        "beta_bar": jnp.array(p["beta_bar"]),
        "A": jnp.diag(jnp.array(p["A_diag"])),
        "d": jnp.array(p["d"]),
        "sigma2": p["sigma2"],
        "C": jnp.array(p["C_diag"]),
        "nu": p["nu"],
        "omega": jnp.array(p["omega"]),
    }
    state0 = (params["beta_bar"], jnp.zeros((T, P)), jnp.asarray(0, jnp.int32))
    betas, _, _, _, _ = lmSD_filter(
        y_masked, base_cov, bucket_idx, mask_f, params, state0
    )
    return pd.to_datetime(dates, format="%Y%m%d"), np.array(betas)


def _betas_ss(y_cube, Z_cube, dates):
    p = json.load(open(OUTPUT_DIR / "params_ss.json"))
    bar_beta = jnp.array(p["bar_beta"])
    sigma2 = float(p["sigma2"])
    ig = {
        "Q_param": jnp.diag(jnp.array(p["Q_diag"])),
        "H_param": sigma2 * jnp.eye(1),
        "B": jnp.diag(jnp.array(p["B_diag"])),
        "bar_beta": bar_beta,
        "omega": jnp.array(p["omega"]),
    }
    init = (
        bar_beta,
        10.0 * jnp.eye(P),
        jnp.zeros((P, P)),
        ig["B"],
        sigma2 * jnp.eye(P),
        jnp.eye(P),
        ig["Q_param"],
        jnp.asarray(0, jnp.int32),
    )
    result = ss_fit_collapsed(y_cube, Z_cube, ig, init, maxiter=0)
    att = np.array(result["att"])
    if float(p["omega"][1]) > 0:
        att = att.copy()
        att[:, 3] *= -1
    return pd.to_datetime(dates, format="%Y%m%d"), att


print("Loading data...", flush=True)
y_cube, Z_cube, dates = _load_data()
mask_bool = ~jnp.isnan(y_cube)
y_masked = jnp.where(mask_bool, y_cube, 0.0)
mask_f = mask_bool.astype(float)
base_cov = Z_cube[:, :, :-1]
bucket_idx = Z_cube[:, :, -1].astype(jnp.int32)
print("Running SS filter...", flush=True)
dates_ss, betas_ss = _betas_ss(y_cube, Z_cube, dates)

print("Running SD filter...", flush=True)
dates_sd, betas_sd = _betas_adjSD(y_masked, base_cov, bucket_idx, mask_f, dates)

print("Running FI-SD filter...", flush=True)
dates_fisd, betas_fisd = _betas_lmSD(y_masked, base_cov, bucket_idx, mask_f, dates)

print("Reconstructing Mk-SD betas...", flush=True)
dates_mksd, betas_mksd = _betas_from_b(
    f"b_ffSD_K{K}.parquet", f"params_ffSD_K{K}.json",
    ffSD_weights, "phi", phi_is_scalar=True,
)

print("Reconstructing Mk-SS betas...", flush=True)
dates_mkss, betas_mkss = _betas_from_b(
    f"b_ffSS_K{K}.parquet", f"params_ffSS_K{K}.json",
    ffSS_weights, "alpha", phi_is_scalar=True,
)

all_series = [
    (dates_ss,   betas_ss,   r"SS",    MODELS[r"SS"]),
    (dates_sd,   betas_sd,   r"SD",    MODELS[r"SD"]),
    (dates_fisd, betas_fisd, r"FI-SD", MODELS[r"FI-SD"]),
    (dates_mksd, betas_mksd, r"Mk-SD", MODELS[r"Mk-SD"]),
    (dates_mkss, betas_mkss, r"Mk-SS", MODELS[r"Mk-SS"]),
]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

print("Plotting...", flush=True)
with PdfPages(OUTPUT) as pdf:
    fig, axes = plt.subplots(4, 1, figsize=(20, 8 * 4), sharex=True)

    for j, (ax, label) in enumerate(zip(axes, COL_LABELS)):
        for start, end in CRISIS_PERIODS:
            ax.axvspan(start, end, color="gray", alpha=0.15, zorder=0)
        ax.axvline(IS_END, color="black", linestyle="--", linewidth=1.2, zorder=3)

        for dates_m, betas_m, name, color in all_series:
            ax.plot(dates_m, betas_m[:, j], color=color, linewidth=0.8, label=name)

        ax.set_title(label, fontsize=36)
        ax.tick_params(axis="both", labelsize=28)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
        if j == 3:
            ax.tick_params(axis="x", rotation=30)

    handles = [
        mpatches.Patch(color=color, label=name)
        for _, _, name, color in all_series
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=5,
        fontsize=36,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    pdf.savefig(fig)
    plt.close(fig)

print(f"Saved → {OUTPUT}")
