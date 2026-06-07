import argparse
import os
import shutil

if shutil.which("nvidia-smi") is not None:
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import jax
import jax.numpy as jnp
from jax import lax
import polars as pl

print(f"Running on: {jax.devices()}")

from models.ss import fit_collapsed, forecast as ss_forecast
from models.adjSD import fit as adjSD_fit, forecast as adjSD_forecast
from models.f_SD import fit as fSD_fit, forecast as fSD_forecast
from models.ff_SD import fit as ffSD_fit, forecast as ffSD_forecast
from models.lmSD import fit as lmSD_fit, forecast as lmSD_forecast
from models.MSMSD import fit as msmSD_fit, forecast as msmSD_forecast
from fit._forecast_metrics import compute_mse, compute_mae, compute_aic, compute_bic

PARQUET_PATH = "data/SPX/otm/full.parquet"
OUTPUT_DIR = "out/SPX/otm/full_performance"
FACTOR_LOADING_COLS = ["level", "moneyness", "maturity"]
P_BASE = 3
P = P_BASE + 1
P_FULL = P

TRAIN_SIZE = 500
TEST_SIZE = 10_000
MAXITER = 2000
Q_ALPHA = -1.6448536269514722
ALPHA = 0.05
FSD_K_VALUES   = [1, 2, 3, 5, 10]
FFSD_K_VALUES  = FSD_K_VALUES
MSMSD_K_VALUES = FSD_K_VALUES

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
    X = Z_win[:, :, :P_BASE].reshape(-1, P_BASE)
    y = y_win.reshape(-1)
    mask = ~jnp.isnan(y)
    X_m = jnp.where(mask[:, None], X, 0.0)
    y_m = jnp.where(mask, y, 0.0)
    return jnp.linalg.solve(X_m.T @ X_m, X_m.T @ y_m)

def _sigma2(y_win, Z_win, beta):
    X = Z_win[:, :, :P_BASE].reshape(-1, P_BASE)
    y = y_win.reshape(-1)
    mask = ~jnp.isnan(y)
    resid = jnp.where(mask, y - X @ beta, 0.0)
    return jnp.sum(resid ** 2) / jnp.sum(mask)

def make_ss_rolling(y_jax, Z_jax, n_buckets, train_size, max_n, q_alpha, maxiter):
    def step(carry, i):
        y_win = lax.dynamic_slice(y_jax, (i, 0), (train_size, max_n))
        Z_win = lax.dynamic_slice(Z_jax, (i, 0, 0), (train_size, max_n, P_BASE + 1))
        y_test = lax.dynamic_slice(y_jax, (i + train_size, 0), (1, max_n))
        Z_test = lax.dynamic_slice(Z_jax, (i + train_size, 0, 0), (1, max_n, P_BASE + 1))

        beta_ols = _ols_beta(y_win, Z_win)
        sig2 = _sigma2(y_win, Z_win, beta_ols)

        a1 = jnp.append(beta_ols, 0.0)
        P1 = 10.0 * jnp.eye(P)
        init = (a1, P1, jnp.zeros((P, P)), 0.95 * jnp.eye(P), sig2 * jnp.eye(P),
                jnp.eye(P), jnp.diag(jnp.full(P, 1e-3)), jnp.asarray(0, jnp.int32))

        ig = {"Q_param": carry[0], "H_param": carry[1], "B": carry[2],
              "bar_beta": carry[3], "omega": carry[4]}
        r = fit_collapsed(y_win, Z_win, ig, init,
                          opt_options={"learning_rate": 1e-3, "tol": 1e-4},
                          maxiter=maxiter)

        preds, P_mean, VaR, oos_ll = ss_forecast(r, Z_test, y_test, q_alpha)

        new_carry = (r["Q_param"], r["H_param"], r["B"], r["bar_beta"], r["omega"])
        return new_carry, (preds[0], P_mean[0], VaR[0], oos_ll[0], r["loglikelihood"], r["niter"], r["is_converged"])

    return step

def make_adjSD_rolling(y_jax, Z_jax, n_buckets, train_size, max_n, alpha, maxiter):
    def step(carry, i):
        y_win = lax.dynamic_slice(y_jax, (i, 0), (train_size, max_n))
        Z_win = lax.dynamic_slice(Z_jax, (i, 0, 0), (train_size, max_n, P_BASE + 1))
        y_test = lax.dynamic_slice(y_jax, (i + train_size, 0), (1, max_n))
        Z_test = lax.dynamic_slice(Z_jax, (i + train_size, 0, 0), (1, max_n, P_BASE + 1))

        ig = {"beta_bar": carry[0], "B": carry[1], "A": carry[2], "sigma2": carry[3],
              "omega": carry[4], "C": carry[5], "nu": carry[6]}
        r = adjSD_fit(y_win, Z_win, ig,
                      opt_options={"learning_rate": 1e-3, "tol": 1e-4},
                      maxiter=maxiter)

        preds, P_mean, VaR, oos_ll = adjSD_forecast(r, Z_test, y_test, alpha)

        new_carry = (r["beta_bar"], r["B"], r["A"], r["sigma2"], r["omega"], r["C"], r["nu"])
        return new_carry, (preds[0], P_mean[0], VaR[0], oos_ll[0], r["log_likelihood"], r["niter"], r["is_converged"])

    return step

def make_fSD_rolling(y_jax, Z_jax, n_buckets, train_size, max_n, alpha, maxiter, K):
    def step(carry, i):
        y_win = lax.dynamic_slice(y_jax, (i, 0), (train_size, max_n))
        Z_win = lax.dynamic_slice(Z_jax, (i, 0, 0), (train_size, max_n, P_BASE + 1))
        y_test = lax.dynamic_slice(y_jax, (i + train_size, 0), (1, max_n))
        Z_test = lax.dynamic_slice(Z_jax, (i + train_size, 0, 0), (1, max_n, P_BASE + 1))

        ig = {"beta_bar": carry[0], "B": carry[1], "A": carry[2], "sigma2": carry[3],
              "sigma_0": carry[4], "omega_load": carry[5], "eta": carry[6],
              "alpha": carry[7], "C": carry[8], "nu": carry[9]}
        r = fSD_fit(y_win, Z_win, ig, K=K, score_power=1.0,
                    opt_options={"learning_rate": 1e-3, "tol": 1e-4},
                    maxiter=maxiter)

        preds, P_mean, VaR, oos_ll = fSD_forecast(r, Z_test, y_test, K, 1.0, alpha)

        new_carry = (r["beta_bar"], r["B"], r["A"], r["sigma2"], r["sigma_0"],
                     r["omega_load"], r["eta"], r["alpha"], r["C"], r["nu"])
        return new_carry, (preds[0], P_mean[0], VaR[0], oos_ll[0], r["log_likelihood"], r["niter"], r["is_converged"], r["b_T"], r["beta_tilde_T"])

    return step

def make_ffSD_rolling(y_jax, Z_jax, n_buckets, train_size, max_n, alpha, maxiter, K):
    def step(carry, i):
        y_win = lax.dynamic_slice(y_jax, (i, 0), (train_size, max_n))
        Z_win = lax.dynamic_slice(Z_jax, (i, 0, 0), (train_size, max_n, P_BASE + 1))
        y_test = lax.dynamic_slice(y_jax, (i + train_size, 0), (1, max_n))
        Z_test = lax.dynamic_slice(Z_jax, (i + train_size, 0, 0), (1, max_n, P_BASE + 1))

        ig = {"beta_bar": carry[0], "A": carry[1], "sigma2": carry[2],
              "omega_load": carry[3], "eta": carry[4], "alpha": carry[5],
              "C": carry[6], "nu": carry[7]}
        r = ffSD_fit(y_win, Z_win, ig, K=K,
                     opt_options={"learning_rate": 1e-3, "tol": 1e-4},
                     maxiter=maxiter)

        preds, P_mean, VaR, oos_ll = ffSD_forecast(r, Z_test, y_test, K, alpha)

        new_carry = (r["beta_bar"], r["A"], r["sigma2"], r["omega_load"],
                     r["eta"], r["alpha"], r["C"], r["nu"])
        return new_carry, (preds[0], P_mean[0], VaR[0], oos_ll[0], r["log_likelihood"], r["niter"], r["is_converged"], r["b_T"])

    return step

def make_lmSD_rolling(y_jax, Z_jax, n_buckets, train_size, max_n, alpha, maxiter):
    def step(carry, i):
        y_win = lax.dynamic_slice(y_jax, (i, 0), (train_size, max_n))
        Z_win = lax.dynamic_slice(Z_jax, (i, 0, 0), (train_size, max_n, P_BASE + 1))
        y_test = lax.dynamic_slice(y_jax, (i + train_size, 0), (1, max_n))
        Z_test = lax.dynamic_slice(Z_jax, (i + train_size, 0, 0), (1, max_n, P_BASE + 1))

        ig = {"beta_bar": carry[0], "A": carry[1], "d": carry[2],
              "sigma2": carry[3], "omega": carry[4], "C": carry[5], "nu": carry[6]}
        r = lmSD_fit(y_win, Z_win, ig,
                     opt_options={"learning_rate": 1e-3, "tol": 1e-4},
                     maxiter=maxiter)

        preds, P_mean, VaR, oos_ll = lmSD_forecast(r, Z_test, y_test, alpha)

        new_carry = (r["beta_bar"], r["A"], r["d"], r["sigma2"],
                     r["omega"], r["C"], r["nu"])
        return new_carry, (preds[0], P_mean[0], VaR[0], oos_ll[0], r["log_likelihood"], r["niter"], r["is_converged"])

    return step

def make_msmSD_rolling(y_jax, Z_jax, n_buckets, train_size, max_n, alpha, maxiter, K):
    def step(carry, i):
        y_win = lax.dynamic_slice(y_jax, (i, 0), (train_size, max_n))
        Z_win = lax.dynamic_slice(Z_jax, (i, 0, 0), (train_size, max_n, P_BASE + 1))
        y_test = lax.dynamic_slice(y_jax, (i + train_size, 0), (1, max_n))
        Z_test = lax.dynamic_slice(Z_jax, (i + train_size, 0, 0), (1, max_n, P_BASE + 1))

        ig = {"beta_bar": carry[0], "B": carry[1], "A": carry[2], "sigma2": carry[3],
              "sigma_0": carry[4], "omega_load": carry[5], "C": carry[6], "nu": carry[7],
              "m0": carry[8], "gamma_K": carry[9], "b": carry[10]}
        r = msmSD_fit(y_win, Z_win, ig, K=K, score_power=1.0,
                      opt_options={"learning_rate": 1e-3, "tol": 1e-4},
                      maxiter=maxiter)

        preds, P_mean, VaR, oos_ll = msmSD_forecast(r, Z_test, y_test, K, 1.0, alpha)

        new_carry = (r["beta_bar"], r["B"], r["A"], r["sigma2"], r["sigma_0"],
                     r["omega_load"], r["C"], r["nu"], r["m0"], r["gamma_K"], r["b"])
        return new_carry, (preds[0], P_mean[0], VaR[0], oos_ll[0], r["log_likelihood"], r["niter"], r["is_converged"])

    return step

def _msmSD_cold_carry(y_jax, Z_jax, n_buckets):
    y_win = y_jax[:TRAIN_SIZE]
    Z_win = Z_jax[:TRAIN_SIZE]
    beta_ols = _ols_beta(y_win, Z_win)
    sig2 = _sigma2(y_win, Z_win, beta_ols)
    omega_load = jnp.concatenate([jnp.zeros(1), jnp.full(n_buckets - 1, 1e-2)])
    return (
        beta_ols,
        0.95 * jnp.eye(P_BASE),
        0.05 * jnp.eye(P_BASE),
        sig2,
        jnp.array(0.1),
        omega_load,
        1e-3 * jnp.eye(P_FULL),
        jnp.array(10.0),
        jnp.array(1.5),
        jnp.array(0.5),
        jnp.array(2.0),
    )

def _ss_cold_carry(y_jax, Z_jax, n_buckets):
    y_win = y_jax[:TRAIN_SIZE]
    Z_win = Z_jax[:TRAIN_SIZE]
    beta_ols = _ols_beta(y_win, Z_win)
    sig2 = _sigma2(y_win, Z_win, beta_ols)
    omega = jnp.concatenate([jnp.zeros(1), jnp.full(n_buckets - 1, 1e-2)])
    return (
        jnp.diag(jnp.full(P, 1e-3)),
        sig2 * jnp.eye(1),
        0.95 * jnp.eye(P),
        jnp.append(beta_ols, 0.0),
        omega,
    )

def _adjSD_cold_carry(y_jax, Z_jax, n_buckets):
    y_win = y_jax[:TRAIN_SIZE]
    Z_win = Z_jax[:TRAIN_SIZE]
    beta_ols = _ols_beta(y_win, Z_win)
    sig2 = _sigma2(y_win, Z_win, beta_ols)
    omega = jnp.concatenate([jnp.zeros(1), jnp.full(n_buckets - 1, 1e-2)])
    return (
        jnp.append(beta_ols, 0.0),
        0.95 * jnp.eye(P),
        0.05 * jnp.eye(P),
        sig2,
        omega,
        jnp.full(P, 1e-3),
        jnp.array(10.0),
    )

def _fSD_cold_carry(y_jax, Z_jax, n_buckets):
    y_win = y_jax[:TRAIN_SIZE]
    Z_win = Z_jax[:TRAIN_SIZE]
    beta_ols = _ols_beta(y_win, Z_win)
    sig2 = _sigma2(y_win, Z_win, beta_ols)
    omega_load = jnp.concatenate([jnp.zeros(1), jnp.full(n_buckets - 1, 1e-2)])
    return (
        beta_ols,
        0.95 * jnp.eye(P_BASE),
        0.05 * jnp.eye(P_FULL),
        sig2,
        jnp.array(0.1),
        omega_load,
        jnp.array(0.06251277029514313),
        jnp.array(1.4255056381225586),
        1e-3 * jnp.eye(P_FULL),
        jnp.array(10.0),
    )

def _ffSD_cold_carry(y_jax, Z_jax, n_buckets):
    y_win = y_jax[:TRAIN_SIZE]
    Z_win = Z_jax[:TRAIN_SIZE]
    beta_ols = _ols_beta(y_win, Z_win)
    sig2 = _sigma2(y_win, Z_win, beta_ols)
    omega_load = jnp.concatenate([jnp.zeros(1), jnp.full(n_buckets - 1, 1e-2)])
    return (
        jnp.append(beta_ols, 0.0),
        0.05 * jnp.eye(P_FULL),
        sig2,
        omega_load,
        jnp.full(P_FULL, 0.4),
        jnp.full(P_FULL, 1.4255056381225586),
        1e-3 * jnp.eye(P_FULL),
        jnp.array(10.0),
    )

def _lmSD_cold_carry(y_jax, Z_jax, n_buckets):
    y_win = y_jax[:TRAIN_SIZE]
    Z_win = Z_jax[:TRAIN_SIZE]
    beta_ols = _ols_beta(y_win, Z_win)
    sig2 = _sigma2(y_win, Z_win, beta_ols)
    omega = jnp.concatenate([jnp.zeros(1), jnp.full(n_buckets - 1, 1e-2)])
    return (
        jnp.append(beta_ols, 0.0),
        0.05 * jnp.eye(P),
        jnp.full(P, 0.4),
        sig2,
        omega,
        jnp.full(P, 1e-3),
        jnp.array(10.0),
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Subset of models to run. Omit to run all. Use for parallel execution across processes.",
    )
    parser.add_argument(
        "--suffix", default="",
        help="Suffix appended to output filenames (e.g. '_g1'). Use with --models for parallel runs.",
    )
    args = parser.parse_args()
    selected = set(args.models) if args.models else None
    suffix   = args.suffix

    print("Loading data...")
    y_jax, Z_jax, n_buckets, dates = load_and_reshape(PARQUET_PATH)
    T, max_n = y_jax.shape
    n_test = min(TEST_SIZE, T - TRAIN_SIZE)
    print(f"  T={T}, max_n={max_n}, n_buckets={n_buckets}, test_steps={n_test}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    indices = jnp.arange(n_test, dtype=jnp.int32)

    n_params_ss = 3 * P + n_buckets
    n_params_adjSD = 4 * P + n_buckets + 1
    n_params_lmSD = 4 * P + n_buckets + 1
    n_params_fSD = 3 * P_BASE + P_FULL + n_buckets + 4
    n_params_ffSD = 4 * P + n_buckets + 1
    n_params_msmSD = n_params_fSD  # equal count: beta_bar,B,A,sigma2,sigma_0,omega[1:],C,nu,m0,gamma_K,b
    n_params_map = {"ss": n_params_ss, "adjSD": n_params_adjSD, "lmSD": n_params_lmSD}
    n_params_map.update({f"fSD_K{k}": n_params_fSD for k in FSD_K_VALUES})
    n_params_map.update({f"ffSD_K{k}": n_params_ffSD for k in FFSD_K_VALUES})
    n_params_map.update({f"msmSD_K{k}": n_params_msmSD for k in MSMSD_K_VALUES})

    y_test_all = y_jax[TRAIN_SIZE:TRAIN_SIZE + n_test]
    mask_test_all = ~jnp.isnan(y_test_all)
    test_dates = dates[TRAIN_SIZE:TRAIN_SIZE + n_test]
    total_obs = int(jnp.sum(mask_test_all))

    step_frames = []
    agg_rows = []

    def run_and_save(name, step_fn, carry0, b_is_type=None):
        if selected is not None and name not in selected:
            return
        out_path = os.path.join(OUTPUT_DIR, f"predictions_{name}{suffix}.parquet")
        if os.path.exists(out_path):
            print(f"  {name}: output exists, skipping.", flush=True)
            return
        print(f"  {name}...", flush=True)
        _, out = jax.jit(lambda c, x: lax.scan(step_fn, c, x))(carry0, indices)
        jax.effects_barrier()
        if b_is_type == "fSD":
            y_hat, P_mean, VaR, oos_ll, train_ll, niter, converged, b_T, beta_tilde_T = out
        elif b_is_type == "ffSD":
            y_hat, P_mean, VaR, oos_ll, train_ll, niter, converged, b_T = out
        else: y_hat, P_mean, VaR, oos_ll, train_ll, niter, converged = out

        pl.DataFrame({
            "date": test_dates,
            "predictions": y_hat.tolist(),
        }).write_parquet(os.path.join(OUTPUT_DIR, f"predictions_{name}{suffix}.parquet"))

        n_params     = n_params_map[name]
        total_oos_ll = float(jnp.sum(oos_ll))
        mse = float(compute_mse(y_test_all, y_hat, mask_test_all))
        mae = float(compute_mae(y_test_all, y_hat, mask_test_all))
        aic = float(compute_aic(jnp.array(total_oos_ll), n_params))
        bic = float(compute_bic(jnp.array(total_oos_ll), n_params, total_obs))

        step_frames.append(pl.DataFrame({
            "date": test_dates,
            "model": pl.Series([name] * n_test),
            "P_mean": P_mean.tolist(),
            "VaR": VaR.tolist(),
            "oos_loglik": oos_ll.tolist(),
            "train_loglik": train_ll.tolist(),
            "n_params": pl.Series([n_params] * n_test, dtype=pl.Int32),
            "niter": niter.tolist(),
            "is_converged": converged.tolist(),
        }))
        agg_rows.append({
            "model": name, "mse": mse, "mae": mae,
            "total_oos_loglik": total_oos_ll,
            "aic": aic, "bic": bic,
            "n_params": n_params, "n_test_steps": n_test, "total_oos_obs": total_obs,
        })

        if b_is_type == "fSD":
            n_k = b_T.shape[1]
            n_p = beta_tilde_T.shape[1]
            bis_df = pl.DataFrame({
                "date": test_dates,
                **{f"b_T_{i}": b_T[:, i].tolist() for i in range(n_k)},
                **{f"beta_tilde_T_{j}": beta_tilde_T[:, j].tolist() for j in range(n_p)},
            })
            bis_df.write_parquet(os.path.join(OUTPUT_DIR, f"bis_{name}{suffix}.parquet"))
        elif b_is_type == "ffSD":
            n_k, n_p = b_T.shape[1], b_T.shape[2]
            bis_df = pl.DataFrame({
                "date": test_dates,
                **{f"b_T_{i}_{j}": b_T[:, i, j].tolist() for i in range(n_k) for j in range(n_p)},
            })
            bis_df.write_parquet(os.path.join(OUTPUT_DIR, f"bis_{name}{suffix}.parquet"))

        jax.clear_caches()

    print("Running models (one JIT per model)...")
    run_and_save("ss",
        make_ss_rolling(y_jax, Z_jax, n_buckets, TRAIN_SIZE, max_n, Q_ALPHA, MAXITER),
        _ss_cold_carry(y_jax, Z_jax, n_buckets))
    run_and_save("adjSD",
        make_adjSD_rolling(y_jax, Z_jax, n_buckets, TRAIN_SIZE, max_n, ALPHA, MAXITER),
        _adjSD_cold_carry(y_jax, Z_jax, n_buckets))
    run_and_save("lmSD",
        make_lmSD_rolling(y_jax, Z_jax, n_buckets, TRAIN_SIZE, max_n, ALPHA, MAXITER),
        _lmSD_cold_carry(y_jax, Z_jax, n_buckets))
    for K in FSD_K_VALUES:
        run_and_save(f"fSD_K{K}",
        make_fSD_rolling(y_jax, Z_jax, n_buckets, TRAIN_SIZE, max_n, ALPHA, MAXITER, K),
        _fSD_cold_carry(y_jax, Z_jax, n_buckets),
        b_is_type="fSD")
    for K in FFSD_K_VALUES:
        run_and_save(f"ffSD_K{K}",
        make_ffSD_rolling(y_jax, Z_jax, n_buckets, TRAIN_SIZE, max_n, ALPHA, MAXITER, K),
        _ffSD_cold_carry(y_jax, Z_jax, n_buckets),
        b_is_type="ffSD")
    for K in MSMSD_K_VALUES:
        run_and_save(f"msmSD_K{K}",
        make_msmSD_rolling(y_jax, Z_jax, n_buckets, TRAIN_SIZE, max_n, ALPHA, MAXITER, K),
        _msmSD_cold_carry(y_jax, Z_jax, n_buckets))

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
