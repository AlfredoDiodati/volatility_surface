import json
import os
import numpy as np
import jax.numpy as jnp
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from scipy.linalg import solve_discrete_lyapunov
from scipy.special import gamma as gamma_func

from models.ss import fit_collapsed
from scaling.scaling_reg import moment_scaling, _make_time_lags

PARQUET_PATH       = "data/SPX/put/bucket.parquet"
BUCKET_MATRIX_PATH = "data/SPX/put/bucket_matrix.parquet"
OUTPUT_PATH        = "out/SPX/put/params.json"
PLOT_BASE          = "plot/SPX/put/scaling"

FACTOR_LOADING_COLS = ["level", "moneyness", "maturity"]
P_BASE = len(FACTOR_LOADING_COLS)
P      = P_BASE + 1

MIN_SCALE = 1.0
MAX_SCALE = 126.0
MOMENTS   = np.arange(1, 9) / 2
TICK_DAYS   = np.array([1, 5, 21, 63, 126])
TICK_LABELS = ["1d", "1 week", "1 month", "3 months", "6 months"]


def load_and_reshape(path):
    raw = (
        pl.read_parquet(path)
        .with_columns(pl.col("DATE").cast(pl.Utf8))
    )
    bucket_cols = sorted([c for c in raw.columns if c.startswith("bucket_")])
    n_buckets   = len(bucket_cols) + 1

    raw = raw.with_columns(
        pl.max_horizontal(
            [pl.when(pl.col(c)).then(i + 1).otherwise(0) for i, c in enumerate(bucket_cols)]
        ).alias("bucket_idx")
    ).sort(["DATE", *FACTOR_LOADING_COLS])

    dates = raw["DATE"].unique(maintain_order=True).sort().to_list()
    T     = len(dates)
    max_n = int(raw.group_by("DATE").len()["len"].max())

    y_cube = np.full((T, max_n), np.nan, dtype=np.float64)
    Z_cube = np.zeros((T, max_n, P_BASE + 1), dtype=np.float64)

    for t, date in enumerate(dates):
        slice_t = raw.filter(pl.col("DATE") == date).sort(FACTOR_LOADING_COLS)
        n_t = len(slice_t)
        y_cube[t, :n_t]          = slice_t["logIV"].to_numpy()
        Z_cube[t, :n_t, :P_BASE] = slice_t[FACTOR_LOADING_COLS].to_numpy()
        Z_cube[t, :n_t, P_BASE]  = slice_t["bucket_idx"].to_numpy().astype(float)

    return y_cube, Z_cube, n_buckets, bucket_cols, dates


def pooled_ols_beta(y_cube, Z_cube):
    T, max_n, _ = Z_cube.shape
    X    = Z_cube[:, :, :P_BASE].reshape(T * max_n, P_BASE)
    y    = y_cube.reshape(T * max_n)
    mask = ~np.isnan(y)
    beta_ols, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
    return beta_ols


def residual_variance(y_cube, Z_cube, beta):
    X    = Z_cube[:, :, :P_BASE].reshape(-1, P_BASE)
    y    = y_cube.reshape(-1)
    mask = ~np.isnan(y)
    return float(np.var(y[mask] - X[mask] @ beta))


def build_initial_guess(beta_ols, sigma2, n_buckets):
    B        = 0.95 * np.eye(P, dtype=np.float64)
    bar_beta = np.append(beta_ols, 0.0)
    Q_param  = np.diag(np.full(P, 1e-3, dtype=np.float64))
    H_param  = np.array([[sigma2]], dtype=np.float64)
    omega    = np.zeros(n_buckets, dtype=np.float64)
    omega[1:] = 1e-2
    return {"Q_param": Q_param, "H_param": H_param, "B": B, "bar_beta": bar_beta, "omega": omega}


def build_jax_initial_guess(initial_guess):
    return {k: jnp.array(v) for k, v in initial_guess.items()}


def build_initialization(beta_ols, sigma2):
    a1   = jnp.array(np.append(beta_ols, 0.0), dtype=float)
    P1   = 10.0 * jnp.eye(P, dtype=float)
    Z0   = jnp.zeros((P, P), dtype=float)
    T0   = 0.95 * jnp.eye(P, dtype=float)
    H0   = sigma2 * jnp.eye(P, dtype=float)
    R0   = jnp.eye(P, dtype=float)
    Q0   = jnp.diag(jnp.full(P, 1e-3, dtype=float))
    idx0 = jnp.asarray(0, dtype=jnp.int32)
    return (a1, P1, Z0, T0, H0, R0, Q0, idx0)


def serialize_params(fit_output):
    serializable = {}
    for key, value in fit_output.items():
        if hasattr(value, "tolist"):
            serializable[key] = value.tolist()
        else:
            serializable[key] = value
    return serializable


def build_per_bucket_loadings(parquet_path, bucket_cols, dates, omega):
    raw = (
        pl.read_parquet(parquet_path)
        .with_columns(pl.col("DATE").cast(pl.Utf8))
    )
    raw = raw.with_columns(
        pl.max_horizontal(
            [pl.when(pl.col(c)).then(i + 1).otherwise(0) for i, c in enumerate(bucket_cols)]
        ).alias("bucket_idx")
    ).sort(["DATE", *FACTOR_LOADING_COLS])

    T            = len(dates)
    date_to_idx  = {d: i for i, d in enumerate(dates)}
    bucket_names = ["ref"] + [c.replace("bucket_", "") for c in bucket_cols]

    loadings_by_bucket = {name: np.full((T, P), np.nan) for name in bucket_names}

    for k, name in enumerate(bucket_names):
        subset = raw.filter(pl.col("bucket_idx") == k)
        if len(subset) == 0:
            continue
        subset_dates = subset["DATE"].to_list()
        t_indices    = np.array([date_to_idx[d] for d in subset_dates], dtype=int)
        cont         = subset[FACTOR_LOADING_COLS].to_numpy()
        omega_col    = np.full((len(subset), 1), float(omega[k]))
        loadings_by_bucket[name][t_indices] = np.hstack([cont, omega_col])

    return loadings_by_bucket, bucket_names


def solve_lyapunov(B, C):
    return solve_discrete_lyapunov(B, C)


def gaussian_absolute_moment_const(q):
    return (2.0 ** (q / 2.0)) * gamma_func((q + 1.0) / 2.0) / np.sqrt(np.pi)


def model_implied_partition(B, P_infty, sigma2, loading_series, delta_ts, moments):
    T      = loading_series.shape[0]
    log_t  = np.log(delta_ts)
    n_q    = len(moments)
    n_dt   = len(delta_ts)
    pv_mat = np.full((n_q, n_dt), np.nan)

    for i_dt, delta_t in enumerate(delta_ts):
        dt   = int(delta_t)
        B_dt = np.linalg.matrix_power(B, dt)
        BP   = B_dt @ P_infty
        N    = T // dt

        gammas = []
        for k in range(N - 1):
            t      = k * dt
            t_next = (k + 1) * dt
            m_t    = loading_series[t]
            m_next = loading_series[t_next]
            if np.any(np.isnan(m_t)) or np.any(np.isnan(m_next)):
                continue
            gamma = (
                m_next @ P_infty @ m_next
                - 2.0 * m_next @ BP @ m_t
                + m_t @ P_infty @ m_t
                + 2.0 * sigma2
            )
            gammas.append(max(gamma, 1e-16))

        if len(gammas) == 0:
            continue

        gammas = np.array(gammas)
        for i_q, q in enumerate(moments):
            pv_mat[i_q, i_dt] = gaussian_absolute_moment_const(q) * np.mean(gammas ** (q / 2.0))

    out = {}
    for i_q, q in enumerate(moments):
        log_pv = np.log(pv_mat[i_q])
        good   = np.isfinite(log_pv)
        if good.sum() < 2:
            out[q] = {"holder": np.nan, "log_power_var": log_pv,
                      "shifted_power_var": log_pv, "intercept": np.nan}
            continue
        lt        = log_t[good] - log_t[good].mean()
        lp        = log_pv[good] - log_pv[good].mean()
        holder    = np.sum(lt * lp) / np.sum(lt ** 2) - 1.0
        intercept = log_pv[good].mean() - holder * log_t[good].mean()
        out[q] = {
            "log_power_var":     log_pv,
            "shifted_power_var": log_pv - intercept,
            "intercept":         intercept,
            "holder":            holder,
        }

    out["delta_ts"] = delta_ts
    out["log_t"]    = log_t
    return out


def make_panel_plots(empirical_scalings, model_scalings, moments, group_dim, cross_dim, subfolder):
    group_labels = sorted(set(label.split("_")[group_dim] for label in empirical_scalings))
    cross_labels = sorted(set(label.split("_")[cross_dim] for label in empirical_scalings))

    blue_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(cross_labels)))
    holder_bm   = moments / 2.0 - 1.0

    fig, axes = plt.subplots(1, len(group_labels), figsize=(4 * len(group_labels), 4))
    if len(group_labels) == 1:
        axes = [axes]

    for g_idx, group in enumerate(group_labels):
        ax = axes[g_idx]
        ax.plot(moments, holder_bm, color="black", linestyle="--", linewidth=1.5, zorder=5)

        for c_idx, cross in enumerate(cross_labels):
            label = (f"{group}_{cross}" if group_dim == 0 else f"{cross}_{group}")
            if label not in empirical_scalings:
                continue
            emp_tau   = np.array([empirical_scalings[label][q]["holder"] for q in moments])
            model_tau = np.array([model_scalings[label][q]["holder"] for q in moments])
            color     = blue_colors[c_idx]
            ax.plot(moments, emp_tau,   color=color, linewidth=1.5,  alpha=0.9,  label=cross)
            ax.plot(moments, model_tau, color=color, linewidth=1.5,  alpha=0.9,  linestyle="--")

        dim_name  = "Maturity" if group_dim == 0 else "Moneyness"
        cross_name = "Moneyness" if group_dim == 0 else "Maturity"
        ax.set_title(f"{dim_name}: {group}", fontsize=11)
        ax.set_xlabel(r"$q$", fontsize=10)
        ax.set_ylabel(r"$\tau(q)$", fontsize=10)
        ax.set_xlim((moments[0], moments[-1]))
        ax.set_ylim((-0.8, 1.0))
        ax.legend(title=cross_name, fontsize=8)
        ax.grid(True, alpha=0.3)

    dim_name_title = "Maturity" if group_dim == 0 else "Moneyness"
    fig.suptitle(
        rf"$\tau(q)$ by {dim_name_title} — solid: data, dashed: model",
        fontsize=12
    )
    plt.tight_layout()
    fname = "by_maturity" if group_dim == 0 else "by_moneyness"
    plt.savefig(os.path.join(subfolder, f"panel_tau_{fname}_model_vs_data.pdf"))
    plt.close()


def make_partition_plots(empirical_scaling, model_scaling, label):
    delta_ts = empirical_scaling["delta_ts"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, scaling, title in zip(
        axes,
        [empirical_scaling, model_scaling],
        ["Data", "Model-implied"]
    ):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_locator(FixedLocator(TICK_DAYS))
        ax.xaxis.set_major_formatter(FixedFormatter(TICK_LABELS))
        for x in TICK_DAYS:
            ax.axvline(x, linestyle="--", linewidth=0.6, color="black")

        for q in MOMENTS:
            y = np.exp(scaling[q]["shifted_power_var"])
            line, = ax.loglog(delta_ts, y)
            ax.text(delta_ts[-1] * 1.01, y[-1], f"q={q}",
                    color=line.get_color(), va="center", fontsize=8)

        ax.set_xlabel(r"$\Delta t$")
        ax.set_ylabel(r"$S(q, \Delta t)$")
        ax.set_title(f"{title} — {label}")
        ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    return fig


def main():
    print("Loading data...")
    y_cube, Z_cube, n_buckets, bucket_cols, dates = load_and_reshape(PARQUET_PATH)
    T, max_n = y_cube.shape
    print(f"  T={T}, max_n={max_n}, n_buckets={n_buckets}")

    if os.path.exists(OUTPUT_PATH):
        print("Found existing params, loading...")
        with open(OUTPUT_PATH) as f:
            raw_params = json.load(f)
        fit_output = {k: np.array(v) if isinstance(v, list) else v
                      for k, v in raw_params.items()}
    else:
        print("No params found, fitting model...")
        beta_ols = pooled_ols_beta(y_cube, Z_cube)
        sigma2   = residual_variance(y_cube, Z_cube, beta_ols)

        initial_guess  = build_jax_initial_guess(build_initial_guess(beta_ols, sigma2, n_buckets))
        initialization = build_initialization(beta_ols, sigma2)

        fit_output = fit_collapsed(
            data=jnp.array(y_cube),
            covariates=jnp.array(Z_cube),
            initial_guess=initial_guess,
            initialization=initialization,
            opt_options={"learning_rate": 1e-3, "tol": 1e-6},
            maxiter=20_000,
        )
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(serialize_params(fit_output), f, indent=2)
        print(f"  Saved to {OUTPUT_PATH}")

    B       = np.array(fit_output["B"])
    C       = np.array(fit_output["Q_param"])
    sigma2  = float(np.array(fit_output["H_param"]).ravel()[0])
    omega   = np.array(fit_output["omega"])

    print("Solving Lyapunov equation...")
    P_infty = solve_lyapunov(B, C)

    print("Building per-bucket loading series...")
    loadings_by_bucket, bucket_names = build_per_bucket_loadings(
        PARQUET_PATH, bucket_cols, dates, omega
    )

    print("Loading bucket matrix for empirical scaling...")
    bucket_matrix      = pl.read_parquet(BUCKET_MATRIX_PATH)
    matrix_cols        = [c for c in bucket_matrix.columns if c != "DATE"]

    stripped_names     = set(bucket_names) - {"ref"}
    ref_candidates     = set(matrix_cols) - stripped_names
    ref_label          = ref_candidates.pop() if len(ref_candidates) == 1 else None

    label_map = {name: name for name in bucket_names if name != "ref"}
    if ref_label is not None:
        label_map["ref"] = ref_label

    delta_ts = _make_time_lags(MIN_SCALE, MAX_SCALE)

    print("Computing empirical and model-implied scaling...")
    empirical_scalings = {}
    model_scalings     = {}

    os.makedirs(PLOT_BASE, exist_ok=True)

    for bucket_name in bucket_names:
        matrix_label = label_map.get(bucket_name)
        if matrix_label is None or matrix_label not in matrix_cols:
            print(f"  skipping {bucket_name}: no matching column in bucket matrix")
            continue

        col         = bucket_matrix[matrix_label].to_numpy().astype(float)
        col[col == 0] = np.nan
        emp_scaling = moment_scaling(col, MIN_SCALE, MAX_SCALE, MOMENTS)
        loading_series = loadings_by_bucket[bucket_name]
        mdl_scaling    = model_implied_partition(B, P_infty, sigma2, loading_series, delta_ts, MOMENTS)

        empirical_scalings[matrix_label] = emp_scaling
        model_scalings[matrix_label]     = mdl_scaling

        fig = make_partition_plots(emp_scaling, mdl_scaling, matrix_label)
        fig.savefig(os.path.join(PLOT_BASE, f"partition_{matrix_label}.pdf"))
        plt.close(fig)

    print("Making panel plots...")
    make_panel_plots(empirical_scalings, model_scalings, MOMENTS,
                     group_dim=0, cross_dim=1, subfolder=PLOT_BASE)
    make_panel_plots(empirical_scalings, model_scalings, MOMENTS,
                     group_dim=1, cross_dim=0, subfolder=PLOT_BASE)

    print("Done.")


if __name__ == "__main__":
    main()