import argparse
import os
import shutil

if shutil.which("nvidia-smi") is not None:
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import numpy as np
import jax
import jax.numpy as jnp
import polars as pl

print(f"Running on: {jax.devices()}")

from models.ss import fit_collapsed, forecast as ss_forecast, forecast_rolling_h as ss_forecast_rolling_h
from models.adjSD import fit as adjSD_fit, forecast as adjSD_forecast, forecast_rolling_h as adjSD_forecast_rolling_h
from models.ff_SS import fit as ffSS_fit, forecast as ffSS_forecast, forecast_rolling_h as ffSS_forecast_rolling_h
from models.ff_SD import fit as ffSD_fit, forecast as ffSD_forecast, forecast_rolling_h as ffSD_forecast_rolling_h
from models.lmSD import fit as lmSD_fit, forecast as lmSD_forecast, forecast_rolling_h as lmSD_forecast_rolling_h
from models.MSMSD import fit as msmSD_fit, forecast as msmSD_forecast, forecast_rolling_h as msmSD_forecast_rolling_h
from fit._forecast_metrics import compute_mse, compute_mae, compute_aic, compute_bic

PARQUET_PATH = "data/SPX/otm/full.parquet"
OUTPUT_DIR = "out/SPX/otm/full_performance_fixed"
FACTOR_LOADING_COLS = ["level", "moneyness", "maturity"]
P_BASE = 3
P = P_BASE + 1

MAXITER = 200
TOL = 1e-2
LR = 1.0
Q_ALPHA = -1.6448536269514722
ALPHA = 0.05
FFSS_K_VALUES = [1, 2, 3, 5, 10]
FFSD_K_VALUES = FFSS_K_VALUES
MSMSD_K_VALUES = FFSS_K_VALUES
FCST_HORIZONS = (5, 22, 66, 260)
_MAX_FCST_H = max(FCST_HORIZONS)


def load_and_reshape(path):
    raw = (
        pl.read_parquet(path)
        .with_columns(pl.col("DATE").cast(pl.Utf8))
        .sort(["DATE", *FACTOR_LOADING_COLS])
        .with_columns(
            pl.int_range(pl.len()).over("DATE").cast(pl.Int32).alias("row_in_date")
        )
    )

    dates = raw["DATE"].unique(maintain_order=True).sort().to_list()
    T = len(dates)
    max_n = int(raw.group_by("DATE").len()["len"].max())
    n_buckets = int(raw["bucket_idx"].max()) + 1

    raw = raw.with_columns(
        pl.col("DATE").replace_strict(
            {d: i for i, d in enumerate(dates)}, return_dtype=pl.Int32
        ).alias("date_idx")
    )

    t_idx = jnp.array(raw["date_idx"], dtype=jnp.int32)
    n_idx = jnp.array(raw["row_in_date"], dtype=jnp.int32)
    y_vals = jnp.array(raw["logIV"])
    factor_vals = jnp.stack([jnp.array(raw[c]) for c in FACTOR_LOADING_COLS], axis=1)
    bucket_vals = jnp.array(raw["bucket_idx"], dtype=jnp.float64)

    y_cube = jnp.full((T, max_n), jnp.nan).at[t_idx, n_idx].set(y_vals)
    Z_cube = jnp.zeros((T, max_n, P_BASE + 1)).at[t_idx, n_idx].set(
        jnp.concatenate([factor_vals, bucket_vals[:, None]], axis=-1)
    )
    return y_cube, Z_cube, n_buckets, dates


def _ols_beta(y_win, Z_win):
    X = np.asarray(Z_win[:, :, :P_BASE]).reshape(-1, P_BASE)
    y = np.asarray(y_win).reshape(-1)
    mask = ~np.isnan(y)
    X_m = np.where(mask[:, None], X, 0.0)
    y_m = np.where(mask, y, 0.0)
    return np.linalg.solve(X_m.T @ X_m, X_m.T @ y_m)


def _sigma2(y_win, Z_win, beta):
    X = np.asarray(Z_win[:, :, :P_BASE]).reshape(-1, P_BASE)
    y = np.asarray(y_win).reshape(-1)
    mask = ~np.isnan(y)
    resid = np.where(mask, y - X @ beta, 0.0)
    return float(np.sum(resid ** 2) / np.sum(mask))


def _base_init(y_train, Z_train, n_buckets):
    beta_ols = _ols_beta(y_train, Z_train)
    sig2 = _sigma2(y_train, Z_train, beta_ols)
    omega = jnp.concatenate([jnp.zeros(1), jnp.full(n_buckets - 1, 1e-2)])
    return beta_ols, sig2, omega


def _ss_ig(beta_ols, sig2, omega):
    beta_bar = jnp.append(beta_ols, 0.0)
    ig = {
        "Q_param": jnp.diag(jnp.full(P, 1e-3)),
        "H_param": sig2 * jnp.eye(1),
        "B": 0.95 * jnp.eye(P),
        "bar_beta": beta_bar,
        "omega": omega,
    }
    init = (beta_bar, 10.0 * jnp.eye(P), jnp.zeros((P, P)),
            0.95 * jnp.eye(P), sig2 * jnp.eye(P),
            jnp.eye(P), jnp.diag(jnp.full(P, 1e-3)), jnp.asarray(0, jnp.int32))
    return ig, init


def _adjSD_ig(beta_ols, sig2, omega):
    return {
        "beta_bar": jnp.append(beta_ols, 0.0),
        "B": 0.95 * jnp.eye(P),
        "A": 0.05 * jnp.eye(P),
        "sigma2": sig2,
        "omega": omega,
        "C": jnp.full(P, 1e-3),
        "nu": jnp.array(10.0),
    }


def _ffSS_ig(beta_ols, sig2, omega):
    return {
        "beta_bar": jnp.append(beta_ols, 0.0),
        "sigma2": sig2,
        "Q_param": 1e-3 * jnp.eye(P),
        "omega": omega,
        "eta": jnp.full(P, 0.06),
        "alpha": jnp.array(1.5),
    }


def _ffSD_ig(beta_ols, sig2, omega):
    return {
        "beta_bar": jnp.append(beta_ols, 0.0),
        "A": 0.05 * jnp.eye(P),
        "sigma2": sig2,
        "omega_load": omega,
        "eta": jnp.full(P, 0.4),
        "alpha": jnp.full(P, 1.4255056381225586),
        "C": 1e-3 * jnp.eye(P),
        "nu": jnp.array(10.0),
    }


def _lmSD_ig(beta_ols, sig2, omega):
    return {
        "beta_bar": jnp.append(beta_ols, 0.0),
        "A": 0.05 * jnp.eye(P),
        "d": jnp.full(P, 0.4),
        "sigma2": sig2,
        "omega": omega,
        "C": jnp.full(P, 1e-3),
        "nu": jnp.array(10.0),
    }


def _msmSD_ig(beta_ols, sig2, omega):
    return {
        "beta_bar": beta_ols,
        "B": 0.95 * jnp.eye(P_BASE),
        "A": 0.05 * jnp.eye(P_BASE),
        "sigma2": sig2,
        "sigma_0": jnp.array(0.1),
        "omega_load": omega,
        "C": 1e-3 * jnp.eye(P),
        "nu": jnp.array(10.0),
        "m0": jnp.array(1.5),
        "gamma_K": jnp.array(0.5),
        "b": jnp.array(2.0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    selected = set(args.models) if args.models else None
    suffix = args.suffix

    print("Loading data...")
    y_jax, Z_jax, n_buckets, dates = load_and_reshape(PARQUET_PATH)
    T, max_n = y_jax.shape
    train_size = T // 2
    n_test = T - train_size
    print(f"  T={T}, train_size={train_size}, n_test={n_test}, max_n={max_n}, n_buckets={n_buckets}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    y_train = y_jax[:train_size]
    Z_train = Z_jax[:train_size]
    y_test = y_jax[train_size:]
    Z_test = Z_jax[train_size:]
    test_dates = dates[train_size:]

    beta_ols, sig2, omega = _base_init(y_train, Z_train, n_buckets)

    Z_padded = jnp.concatenate([Z_jax, jnp.tile(Z_jax[-1:], (_MAX_FCST_H, 1, 1))], axis=0)
    Z_test_ext = Z_padded[train_size:]

    mask_test = ~jnp.isnan(y_test)
    total_obs = int(jnp.sum(mask_test))

    n_params_map = {
        "ss": 3 * P + n_buckets,
        "adjSD": 4 * P + n_buckets + 1,
        "lmSD": 4 * P + n_buckets + 1,
        **{f"ffSS_K{k}": 3 * P + n_buckets + 1 for k in FFSS_K_VALUES},
        **{f"ffSD_K{k}": 4 * P + n_buckets + 1 for k in FFSD_K_VALUES},
        **{f"msmSD_K{k}": 3 * P_BASE + P + n_buckets + 4 for k in MSMSD_K_VALUES},
    }

    step_frames = []
    agg_rows = []

    def _save_1step(name, y_hat, P_mean, VaR, oos_ll, train_ll, niter, converged):
        pl.DataFrame({
            "date": test_dates,
            "predictions": y_hat.tolist(),
        }).write_parquet(os.path.join(OUTPUT_DIR, f"predictions_{name}{suffix}.parquet"))

        n_params = n_params_map[name]
        total_oos_ll = float(jnp.sum(oos_ll))
        mse = float(compute_mse(y_test, y_hat, mask_test))
        mae = float(compute_mae(y_test, y_hat, mask_test))
        aic = float(compute_aic(jnp.array(total_oos_ll), n_params))
        bic = float(compute_bic(jnp.array(total_oos_ll), n_params, total_obs))

        step_frames.append(pl.DataFrame({
            "date": test_dates,
            "model": pl.Series([name] * n_test),
            "P_mean": P_mean.tolist(),
            "VaR": VaR.tolist(),
            "oos_loglik": oos_ll.tolist(),
            "train_loglik": pl.Series([float(train_ll)] * n_test),
            "n_params": pl.Series([n_params] * n_test, dtype=pl.Int32),
            "niter": pl.Series([int(niter)] * n_test, dtype=pl.Int32),
            "is_converged": pl.Series([bool(converged)] * n_test),
        }))
        agg_rows.append({
            "model": name, "mse": mse, "mae": mae,
            "total_oos_loglik": total_oos_ll,
            "aic": aic, "bic": bic,
            "n_params": n_params, "n_test_steps": n_test, "total_oos_obs": total_obs,
        })

    def _save_hstep(name, preds_h):
        frames = []
        for h_idx, h in enumerate(FCST_HORIZONS):
            n_valid = n_test - h
            if n_valid <= 0:
                continue
            frames.append(pl.DataFrame({
                "date": test_dates[:n_valid],
                "horizon": pl.Series([h] * n_valid, dtype=pl.Int32),
                "predictions": preds_h[h_idx][:n_valid].tolist(),
            }))
        pl.concat(frames).write_parquet(
            os.path.join(OUTPUT_DIR, f"predictions_h_{name}{suffix}.parquet")
        )

    def _skip(name):
        if selected is not None and name not in selected:
            return True
        path = os.path.join(OUTPUT_DIR, f"predictions_{name}{suffix}.parquet")
        if os.path.exists(path):
            print(f"  {name}: output exists, skipping.", flush=True)
            return True
        return False

    OPT = {"learning_rate": LR, "tol": TOL}

    print("Running models (single fit per model)...")

    if not _skip("ss"):
        print("  ss...", flush=True)
        ig, init = _ss_ig(beta_ols, sig2, omega)
        r = fit_collapsed(y_train, Z_train, ig, init, opt_options=OPT, maxiter=MAXITER)
        jax.effects_barrier()
        y_hat, P_mean, VaR, oos_ll = ss_forecast(r, Z_test, y_test, Q_ALPHA)
        _save_1step("ss", y_hat, P_mean, VaR, oos_ll,
                    r["loglikelihood"], r["niter"], r["is_converged"])
        print("  ss h-step...", flush=True)
        _save_hstep("ss", ss_forecast_rolling_h(r, Z_test_ext, y_test, FCST_HORIZONS))
        jax.clear_caches()

    if not _skip("adjSD"):
        print("  adjSD...", flush=True)
        r = adjSD_fit(y_train, Z_train, _adjSD_ig(beta_ols, sig2, omega),
                      opt_options=OPT, maxiter=MAXITER)
        jax.effects_barrier()
        y_hat, P_mean, VaR, oos_ll = adjSD_forecast(r, Z_test, y_test, ALPHA)
        _save_1step("adjSD", y_hat, P_mean, VaR, oos_ll,
                    r["log_likelihood"], r["niter"], r["is_converged"])
        print("  adjSD h-step...", flush=True)
        _save_hstep("adjSD", adjSD_forecast_rolling_h(r, Z_test_ext, y_test, FCST_HORIZONS))
        jax.clear_caches()

    if not _skip("lmSD"):
        print("  lmSD...", flush=True)
        r = lmSD_fit(y_train, Z_train, _lmSD_ig(beta_ols, sig2, omega),
                     opt_options=OPT, maxiter=MAXITER)
        jax.effects_barrier()
        y_hat, P_mean, VaR, oos_ll = lmSD_forecast(r, Z_test, y_test, ALPHA)
        _save_1step("lmSD", y_hat, P_mean, VaR, oos_ll,
                    r["log_likelihood"], r["niter"], r["is_converged"])
        print("  lmSD h-step...", flush=True)
        _save_hstep("lmSD", lmSD_forecast_rolling_h(r, Z_test_ext, y_test, FCST_HORIZONS))
        jax.clear_caches()

    for K in FFSS_K_VALUES:
        name = f"ffSS_K{K}"
        if _skip(name):
            continue
        print(f"  {name}...", flush=True)
        r = ffSS_fit(y_train, Z_train, _ffSS_ig(beta_ols, sig2, omega),
                     K=K, opt_options=OPT, maxiter=MAXITER)
        jax.effects_barrier()
        y_hat, P_mean, VaR, oos_ll = ffSS_forecast(r, Z_test, y_test, Q_ALPHA)
        _save_1step(name, y_hat, P_mean, VaR, oos_ll,
                    r["loglikelihood"], r["niter"], r["is_converged"])
        print(f"  {name} h-step...", flush=True)
        _save_hstep(name, ffSS_forecast_rolling_h(r, Z_test_ext, y_test, FCST_HORIZONS))
        jax.clear_caches()

    for K in FFSD_K_VALUES:
        name = f"ffSD_K{K}"
        if _skip(name):
            continue
        print(f"  {name}...", flush=True)
        r = ffSD_fit(y_train, Z_train, _ffSD_ig(beta_ols, sig2, omega),
                     K=K, opt_options=OPT, maxiter=MAXITER)
        jax.effects_barrier()
        y_hat, P_mean, VaR, oos_ll = ffSD_forecast(r, Z_test, y_test, K, ALPHA)
        _save_1step(name, y_hat, P_mean, VaR, oos_ll,
                    r["log_likelihood"], r["niter"], r["is_converged"])
        print(f"  {name} h-step...", flush=True)
        _save_hstep(name, ffSD_forecast_rolling_h(r, Z_test_ext, y_test, K, FCST_HORIZONS))
        jax.clear_caches()

    for K in MSMSD_K_VALUES:
        name = f"msmSD_K{K}"
        if _skip(name):
            continue
        print(f"  {name}...", flush=True)
        r = msmSD_fit(y_train, Z_train, _msmSD_ig(beta_ols, sig2, omega),
                      K=K, score_power=1.0, opt_options=OPT, maxiter=MAXITER)
        jax.effects_barrier()
        y_hat, P_mean, VaR, oos_ll = msmSD_forecast(r, Z_test, y_test, K, 1.0, ALPHA)
        _save_1step(name, y_hat, P_mean, VaR, oos_ll,
                    r["log_likelihood"], r["niter"], r["is_converged"])
        print(f"  {name} h-step...", flush=True)
        _save_hstep(name, msmSD_forecast_rolling_h(r, Z_test_ext, y_test, K, 1.0, FCST_HORIZONS))
        jax.clear_caches()

    if not step_frames:
        print("No models ran (check --models argument).")
        return

    new_step = pl.concat(step_frames)
    step_path = os.path.join(OUTPUT_DIR, f"step_results{suffix}.parquet")
    if os.path.exists(step_path):
        existing = pl.read_parquet(step_path)
        ran_models = new_step["model"].unique()
        existing = existing.filter(~pl.col("model").is_in(ran_models))
        new_step = pl.concat([existing, new_step])
    new_step.write_parquet(step_path)

    new_agg = pl.DataFrame(agg_rows)
    agg_path = os.path.join(OUTPUT_DIR, f"aggregate_metrics{suffix}.csv")
    if os.path.exists(agg_path):
        existing_agg = pl.read_csv(agg_path)
        ran_models = new_agg["model"].unique()
        existing_agg = existing_agg.filter(~pl.col("model").is_in(ran_models))
        new_agg = pl.concat([existing_agg, new_agg])
    new_agg.write_csv(agg_path)
    print("\nAggregate metrics:")
    print(new_agg)
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
