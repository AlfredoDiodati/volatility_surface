import json
import os
import numpy as np
import jax.numpy as jnp
import polars as pl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter
from scipy.special import gamma as gamma_func

from models.f_SD import fit, _solve_weights
from scaling.scaling_reg import moment_scaling, _make_time_lags

PARQUET_PATH       = "data/SPX/put/bucket.parquet"
BUCKET_MATRIX_PATH = "data/SPX/put/bucket_matrix.parquet"
OUTPUT_DIR         = "out/SPX/put/fSD"
PLOT_DIR           = "plot/SPX/put/fSD"

FACTOR_LOADING_COLS = ["level", "moneyness", "maturity"]
P_BASE  = len(FACTOR_LOADING_COLS)
P_TILDE = P_BASE
P_FULL  = P_TILDE + 1

K_VALUES = [3, 5, 8, 10, 15, 20, 30]

ETA   = 0.4
RHO_K = 0.999

MIN_SCALE   = 1.0
MAX_SCALE   = 126.0
MOMENTS     = np.arange(1, 9) / 2
TICK_DAYS   = np.array([1, 5, 21, 63, 126])
TICK_LABELS = ["1d", "1 week", "1 month", "3 months", "6 months"]

def output_path(k):
    return os.path.join(OUTPUT_DIR, f"K{k}", "params.json")

def plot_base(k):
    return os.path.join(PLOT_DIR, f"K{k}")

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
    omega_load = np.zeros(n_buckets)
    omega_load[1:] = 1e-2
    return {
        "beta_bar":   jnp.array(beta_ols),
        "B":          jnp.array(0.95 * np.eye(P_TILDE, dtype=np.float64)),
        "A":          jnp.array(0.05 * np.eye(P_TILDE, dtype=np.float64)),
        "sigma2":     jnp.array(sigma2),
        "sigma_0":    jnp.array(0.1),
        "omega_load": jnp.array(omega_load),
        "eta":        jnp.array(ETA),
        "rho_K":      jnp.array(RHO_K),
        "C":          jnp.array(1e-3 * np.eye(P_FULL, dtype=np.float64)),
        "nu":         jnp.array(10.0),
    }

def warm_start_guess(prev_params):
    return {
        "beta_bar":   jnp.array(np.array(prev_params["beta_bar"])),
        "B":          jnp.array(np.array(prev_params["B"])),
        "A":          jnp.array(np.array(prev_params["A"])),
        "sigma2":     jnp.array(float(np.array(prev_params["sigma2"]).ravel()[0])),
        "sigma_0":    jnp.array(float(np.array(prev_params["sigma_0"]).ravel()[0])),
        "omega_load": jnp.array(np.array(prev_params["omega_load"])),
        "eta":        jnp.array(float(np.array(prev_params["eta"]).ravel()[0])),
        "rho_K":      jnp.array(float(np.array(prev_params["rho_K"]).ravel()[0])),
        "C":          jnp.array(np.array(prev_params["C"])),
        "nu":         jnp.array(float(np.array(prev_params["nu"]).ravel()[0])),
    }

def serialize_params(fit_output):
    serializable = {}
    for key, value in fit_output.items():
        if hasattr(value, "tolist"):
            serializable[key] = value.tolist()
        elif hasattr(value, "item"):
            serializable[key] = value.item()
        else:
            serializable[key] = value
    return serializable

def load_params(path):
    with open(path) as f:
        raw = json.load(f)
    return {k: np.array(v) if isinstance(v, list) else v for k, v in raw.items()}

def build_per_bucket_loadings(parquet_path, bucket_cols, dates, omega_load):
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

    loadings_by_bucket = {name: np.full((T, P_FULL), np.nan) for name in bucket_names}

    for k, name in enumerate(bucket_names):
        subset = raw.filter(pl.col("bucket_idx") == k)
        if len(subset) == 0:
            continue
        subset_dates = subset["DATE"].to_list()
        t_indices    = np.array([date_to_idx[d] for d in subset_dates], dtype=int)
        cont         = subset[FACTOR_LOADING_COLS].to_numpy()
        omega_col    = np.full((len(subset), 1), float(omega_load[k]))
        loadings_by_bucket[name][t_indices] = np.hstack([cont, omega_col])

    return loadings_by_bucket, bucket_names

def student_t_absolute_moment_const(q, nu):
    return (
        (nu - 2.0) ** (q / 2.0)
        * gamma_func((q + 1.0) / 2.0)
        * gamma_func((nu - q) / 2.0)
        / (np.sqrt(np.pi) * gamma_func(nu / 2.0))
    )

def model_implied_partition(sigma2, nu, loading_series, betas, delta_ts, moments):
    T      = loading_series.shape[0]
    log_t  = np.log(delta_ts)
    n_q    = len(moments)
    n_dt   = len(delta_ts)
    pv_mat = np.full((n_q, n_dt), np.nan)

    for i_dt, delta_t in enumerate(delta_ts):
        dt   = int(delta_t)
        N    = T // dt

        gammas = []
        for k in range(N - 1):
            t      = k * dt
            t_next = (k + 1) * dt
            m_t    = loading_series[t]
            m_next = loading_series[t_next]
            if np.any(np.isnan(m_t)) or np.any(np.isnan(m_next)):
                continue
            hat_y_t    = float(m_t    @ betas[t])
            hat_y_next = float(m_next @ betas[t_next])
            gamma = (hat_y_next - hat_y_t) ** 2 + 2.0 * sigma2
            gammas.append(max(gamma, 1e-16))

        if len(gammas) == 0:
            continue

        gammas = np.array(gammas)
        for i_q, q in enumerate(moments):
            if q >= nu:
                continue
            pv_mat[i_q, i_dt] = student_t_absolute_moment_const(q, nu) * np.mean(gammas ** (q / 2.0))

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
            ax.plot(moments, emp_tau,   color=color, linewidth=1.5, alpha=0.9, label=cross)
            ax.plot(moments, model_tau, color=color, linewidth=1.5, alpha=0.9, linestyle="--")

        dim_name   = "Maturity" if group_dim == 0 else "Moneyness"
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
        rf"$\tau(q)$ by {dim_name_title} — solid: data, dashed: model (K={subfolder.split('K')[-1]})",
        fontsize=12
    )
    plt.tight_layout()
    fname = "by_maturity" if group_dim == 0 else "by_moneyness"
    plt.savefig(os.path.join(subfolder, f"panel_tau_{fname}_model_vs_data.pdf"))
    plt.close()

def make_partition_plots(empirical_scaling, model_scaling, label, moments):
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

        for q in moments:
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

def run_scaling(fit_output, bucket_cols, dates, bucket_matrix, matrix_cols, label_map,
                delta_ts, valid_moments):
    sigma2     = float(np.array(fit_output["sigma2"]).ravel()[0])
    nu         = float(np.array(fit_output["nu"]).ravel()[0])
    omega_load = np.array(fit_output["omega_load"])
    betas      = np.array(fit_output["betas"])

    loadings_by_bucket, bucket_names = build_per_bucket_loadings(
        PARQUET_PATH, bucket_cols, dates, omega_load
    )

    empirical_scalings = {}
    model_scalings     = {}

    for bucket_name in bucket_names:
        matrix_label = label_map.get(bucket_name)
        if matrix_label is None or matrix_label not in matrix_cols:
            continue

        col           = bucket_matrix[matrix_label].to_numpy().astype(float)
        col[col == 0] = np.nan
        emp_scaling   = moment_scaling(col, MIN_SCALE, MAX_SCALE, valid_moments)
        loading_series = loadings_by_bucket[bucket_name]
        mdl_scaling    = model_implied_partition(
            sigma2, nu, loading_series, betas, delta_ts, valid_moments
        )

        empirical_scalings[matrix_label] = emp_scaling
        model_scalings[matrix_label]     = mdl_scaling

    return empirical_scalings, model_scalings

def main():
    print("Loading data...")
    y_cube, Z_cube, n_buckets, bucket_cols, dates = load_and_reshape(PARQUET_PATH)
    T, max_n = y_cube.shape
    print(f"  T={T}, max_n={max_n}, n_buckets={n_buckets}")

    beta_ols = pooled_ols_beta(y_cube, Z_cube)
    sigma2   = residual_variance(y_cube, Z_cube, beta_ols)

    print("Loading bucket matrix for empirical scaling...")
    bucket_matrix = pl.read_parquet(BUCKET_MATRIX_PATH)
    matrix_cols   = [c for c in bucket_matrix.columns if c != "DATE"]

    bucket_names_ref = ["ref"] + [c.replace("bucket_", "") for c in bucket_cols]
    stripped_names   = set(bucket_names_ref) - {"ref"}
    ref_candidates   = set(matrix_cols) - stripped_names
    ref_label        = ref_candidates.pop() if len(ref_candidates) == 1 else None
    label_map        = {name: name for name in bucket_names_ref if name != "ref"}
    if ref_label is not None:
        label_map["ref"] = ref_label

    delta_ts = _make_time_lags(MIN_SCALE, MAX_SCALE)

    prev_params = None

    for k in K_VALUES:
        opath = output_path(k)
        pbase = plot_base(k)

        print(f"\n--- K={k} ---")
        print(f"  Solving weights (eta={ETA}, rho_K={RHO_K})...")
        w_tilde_norm, rho = _solve_weights(ETA, RHO_K, k)
        print(f"  rho={rho.round(4)}")
        print(f"  w_tilde_norm={w_tilde_norm.round(4)}")

        if os.path.exists(opath) and os.path.getsize(opath) > 0:
            print(f"  Found existing params, loading from {opath}")
            fit_output = load_params(opath)
        else:
            if prev_params is None:
                print(f"  Cold start")
                initial_guess = build_initial_guess(beta_ols, sigma2, n_buckets)
            else:
                print(f"  Warm start from K={K_VALUES[K_VALUES.index(k) - 1]}")
                initial_guess = warm_start_guess(prev_params)

            fit_output = fit(
                data=jnp.array(y_cube),
                covariates=jnp.array(Z_cube),
                initial_guess=initial_guess,
                K=k,
                opt_options={"learning_rate": 1e-3, "tol": 1e-6},
                maxiter=20_000,
            )
            
            os.makedirs(os.path.dirname(opath), exist_ok=True)
            with open(opath, "w") as f:
                json.dump(serialize_params(fit_output), f, indent=2)
            print(f"  Saved to {opath}")

        nu         = float(np.array(fit_output["nu"]).ravel()[0])
        sigma2_fit = float(np.array(fit_output["sigma2"]).ravel()[0])
        print(f"  nu={nu:.2f}, sigma2={sigma2_fit:.4f}")
        print(f"  log_lik={float(np.array(fit_output['log_likelihood']).ravel()[0]):.2f}")

        prev_params = fit_output

        valid_moments = MOMENTS[MOMENTS < nu]
        if len(valid_moments) < len(MOMENTS):
            print(f"  Dropping moments >= nu={nu:.2f}: {MOMENTS[MOMENTS >= nu]}")

        print(f"  Computing scaling...")
        empirical_scalings, model_scalings = run_scaling(
            fit_output, bucket_cols, dates, bucket_matrix, matrix_cols,
            label_map, delta_ts, valid_moments
        )

        os.makedirs(pbase, exist_ok=True)

        for matrix_label in empirical_scalings:
            fig = make_partition_plots(
                empirical_scalings[matrix_label],
                model_scalings[matrix_label],
                matrix_label, valid_moments
            )
            fig.savefig(os.path.join(pbase, f"partition_{matrix_label}.pdf"))
            plt.close(fig)

        make_panel_plots(empirical_scalings, model_scalings, valid_moments,
                         group_dim=0, cross_dim=1, subfolder=pbase)
        make_panel_plots(empirical_scalings, model_scalings, valid_moments,
                         group_dim=1, cross_dim=0, subfolder=pbase)

    print("\nDone.")

if __name__ == "__main__":
    main()