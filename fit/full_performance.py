import os
import jax
import jax.numpy as jnp
from jax import lax
import polars as pl

import os

os.environ['JAX_ENABLE_X64'] = 'True'
# 2. Don't grab all the VRAM at once (helpful for 4GB cards)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
print(f"Running on: {jax.devices()}")

from models.ss import fit_collapsed, forecast as ss_forecast
from models.adjSD import fit as adjSD_fit, forecast as adjSD_forecast
from models.f_SD import fit as fSD_fit, forecast as fSD_forecast
from models.ff_SD import fit as ffSD_fit, forecast as ffSD_forecast
from models.if_SD import fit as ifSD_fit, forecast as ifSD_forecast
from models.lmSD import fit as lmSD_fit, forecast as lmSD_forecast
from fit._forecast_metrics import compute_mse, compute_mae, compute_aic, compute_bic

PARQUET_PATH = "data/SPX/otm/full.parquet"
FILTERED_PATH = "data/SPX/otm/filtered.parquet"
OUTPUT_DIR = "out/SPX/otm/full_performance"
FACTOR_LOADING_COLS = ["level", "moneyness", "maturity"]
P_BASE = 3
P = P_BASE + 1
P_FULL = P

TRAIN_SIZE = 500
TEST_SIZE = 250
Q_ALPHA = -1.6448536269514722  # scipy.stats.norm.ppf(0.05) — used for SS (Gaussian)
ALPHA = 0.05                   # probability level — used for t-distribution models
FSD_K_VALUES = [3, 10]
FFSD_K_VALUES = [3, 10]
IFSD_K_VALUES = [3, 100]

# 4 maturity bins × 6 moneyness bins = 24 buckets (matches bucket_performance.py)
N_MAT_BUCKETS = 4
N_MON_BUCKETS = 6


def load_and_reshape(path):
    # Load full options data
    raw = (
        pl.read_parquet(path)
        .with_columns(pl.col("DATE").cast(pl.Utf8))
    )

    # Recover bucket assignments from filtered.parquet, which retains
    # MATURITY_BUCKET and MONEYNESS_BUCKET from the original pre-processing.
    # full.parquet was created from filtered.parquet without sorting or filtering,
    # so (DATE, MONEYNESS, MATURITY) uniquely identifies each row in both files.
    bucket_key = (
        pl.scan_parquet(FILTERED_PATH)
        .select(["DATE", "MONEYNESS", "MATURITY", "MATURITY_BUCKET", "MONEYNESS_BUCKET"])
        .with_columns(pl.col("DATE").cast(pl.Utf8))
        .unique(subset=["DATE", "MONEYNESS", "MATURITY"])
        .collect()
    )

    # bucket_idx = (mat_bin - 1)*N_MON_BUCKETS + (mon_bin - 1)
    # This replicates the ordering used in bucket_performance.py:
    # mat1_mon1 → 0 (base), mat1_mon2 → 1, …, mat4_mon6 → 23
    raw = (
        raw
        .with_columns(
            (pl.col("maturity") * 255).round().cast(pl.Int32).alias("_mat_days")
        )
        .join(
            bucket_key
            .rename({"MONEYNESS": "moneyness", "MATURITY": "_mat_days"})
            .with_columns(pl.col("_mat_days").cast(pl.Int32)),
            on=["DATE", "moneyness", "_mat_days"],
            how="left",
        )
        .with_columns(
            (
                (pl.col("MATURITY_BUCKET").fill_null(1) - 1) * N_MON_BUCKETS
                + (pl.col("MONEYNESS_BUCKET").fill_null(1) - 1)
            ).cast(pl.Int32).alias("bucket_idx")
        )
        .drop(["_mat_days", "MATURITY_BUCKET", "MONEYNESS_BUCKET"])
        .sort(["DATE", *FACTOR_LOADING_COLS])
        .with_columns(
            pl.int_range(pl.len()).over("DATE").cast(pl.Int32).alias("row_in_date")
        )
    )

    dates = raw["DATE"].unique(maintain_order=True).sort().to_list()
    T = len(dates)
    max_n = int(raw.group_by("DATE").len()["len"].max())
    n_buckets = N_MAT_BUCKETS * N_MON_BUCKETS  # 24

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
                          opt_options={"learning_rate": 1e-3, "tol": 1e-6},
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
                      opt_options={"learning_rate": 1e-3, "tol": 1e-6},
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
                    opt_options={"learning_rate": 1e-3, "tol": 1e-6},
                    maxiter=maxiter)

        preds, P_mean, VaR, oos_ll = fSD_forecast(r, Z_test, y_test, K, 1.0, alpha)

        new_carry = (r["beta_bar"], r["B"], r["A"], r["sigma2"], r["sigma_0"],
                     r["omega_load"], r["eta"], r["alpha"], r["C"], r["nu"])
        return new_carry, (preds[0], P_mean[0], VaR[0], oos_ll[0], r["log_likelihood"], r["niter"], r["is_converged"])

    return step


def make_ifSD_rolling(y_jax, Z_jax, n_buckets, train_size, max_n, alpha, maxiter, K):
    def step(carry, i):
        y_win = lax.dynamic_slice(y_jax, (i, 0), (train_size, max_n))
        Z_win = lax.dynamic_slice(Z_jax, (i, 0, 0), (train_size, max_n, P_BASE + 1))
        y_test = lax.dynamic_slice(y_jax, (i + train_size, 0), (1, max_n))
        Z_test = lax.dynamic_slice(Z_jax, (i + train_size, 0, 0), (1, max_n, P_BASE + 1))

        ig = {"beta_bar": carry[0], "A": carry[1], "sigma2": carry[2],
              "omega_load": carry[3], "eta": carry[4], "alpha": carry[5],
              "a_midas": carry[6], "b_midas": carry[7], "C": carry[8], "nu": carry[9]}
        r = ifSD_fit(y_win, Z_win, ig, K=K, score_power=1.0,
                     opt_options={"learning_rate": 1e-3, "tol": 1e-6},
                     maxiter=maxiter)

        preds, P_mean, VaR, oos_ll = ifSD_forecast(r, Z_test, y_test, K, 1.0, alpha)

        new_carry = (r["beta_bar"], r["A"], r["sigma2"], r["omega_load"],
                     r["eta"], r["alpha"], r["a_midas"], r["b_midas"], r["C"], r["nu"])
        return new_carry, (preds[0], P_mean[0], VaR[0], oos_ll[0], r["log_likelihood"], r["niter"], r["is_converged"])

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
        r = ffSD_fit(y_win, Z_win, ig, K=K, score_power=1.0,
                     opt_options={"learning_rate": 1e-3, "tol": 1e-6},
                     maxiter=maxiter)

        preds, P_mean, VaR, oos_ll = ffSD_forecast(r, Z_test, y_test, K, 1.0, alpha)

        new_carry = (r["beta_bar"], r["A"], r["sigma2"], r["omega_load"],
                     r["eta"], r["alpha"], r["C"], r["nu"])
        return new_carry, (preds[0], P_mean[0], VaR[0], oos_ll[0], r["log_likelihood"], r["niter"], r["is_converged"])

    return step


def make_lmSD_rolling(y_jax, Z_jax, n_buckets, train_size, max_n, alpha, maxiter):
    def step(carry, i):
        y_win = lax.dynamic_slice(y_jax, (i, 0), (train_size, max_n))
        Z_win = lax.dynamic_slice(Z_jax, (i, 0, 0), (train_size, max_n, P_BASE + 1))
        y_test = lax.dynamic_slice(y_jax, (i + train_size, 0), (1, max_n))
        Z_test = lax.dynamic_slice(Z_jax, (i + train_size, 0, 0), (1, max_n, P_BASE + 1))

        ig = {"beta_bar": carry[0], "B": carry[1], "A": carry[2], "d": carry[3],
              "sigma2": carry[4], "omega": carry[5], "C": carry[6], "nu": carry[7]}
        r = lmSD_fit(y_win, Z_win, ig,
                     opt_options={"learning_rate": 1e-3, "tol": 1e-6},
                     maxiter=maxiter)

        preds, P_mean, VaR, oos_ll = lmSD_forecast(r, Z_test, y_test, alpha)

        new_carry = (r["beta_bar"], r["B"], r["A"], r["d"], r["sigma2"],
                     r["omega"], r["C"], r["nu"])
        return new_carry, (preds[0], P_mean[0], VaR[0], oos_ll[0], r["log_likelihood"], r["niter"], r["is_converged"])

    return step


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
        0.05 * jnp.eye(P_BASE),
        sig2,
        jnp.array(0.1),
        omega_load,
        jnp.array(0.06251277029514313),
        jnp.array(1.4255056381225586),
        1e-3 * jnp.eye(P_FULL),
        jnp.array(10.0),
    )


def _ifSD_cold_carry(y_jax, Z_jax, n_buckets):
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
        jnp.full(P_FULL, 1.2),
        jnp.ones(P_FULL) + 0.2,
        jnp.full(P_FULL, 5.0),
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
        0.95 * jnp.eye(P),
        0.05 * jnp.eye(P),
        jnp.full(P, 0.4),
        sig2,
        omega,
        jnp.full(P, 1e-3),
        jnp.array(10.0),
    )


def main():
    print("Loading data...")
    y_jax, Z_jax, n_buckets, dates = load_and_reshape(PARQUET_PATH)
    T, max_n = y_jax.shape
    n_test = min(TEST_SIZE, T - TRAIN_SIZE)
    print(f"  T={T}, max_n={max_n}, n_buckets={n_buckets}, test_steps={n_test}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    indices = jnp.arange(n_test, dtype=jnp.int32)

    # Parameter counts from each model's _invlink output length (P=4, verified):
    # SS:    H(1) + Q_diag(P) + B_diag(P) + omega[1:](n_b-1) + bar_beta(P) = 3P + n_b
    # adjSD: beta_bar(P) + B(P) + A(P) + sigma2(1) + omega[1:](n_b-1) + C(P) + nu(1) = 4P + n_b + 1
    # lmSD:  beta_bar(P) + B(P) + A(P) + d(P) + sigma2(1) + omega[1:](n_b-1) + C(P) + nu(1) = 5P + n_b + 1
    # fSD:   beta_bar(Pb) + B(Pb) + A(Pb) + sigma2(1) + sigma_0(1) + omega[1:](n_b-1) + eta(1) + alpha(1) + C(Pf) + nu(1)
    #        = 3*P_BASE + P_FULL + n_b + 4
    # ffSD:  beta_bar(P) + A(P) + sigma2(1) + omega[1:](n_b-1) + eta(P) + alpha(1) + C(P) + nu(1) = 4P + n_b + 2
    # ifSD:  beta_bar(P) + A(P) + sigma2(1) + omega[1:](n_b-1) + eta(P) + alpha(1) + a_midas(P) + b_midas(P) + C(P) + nu(1)
    #        = 6P + n_b + 2
    n_params_ss    = 3 * P + n_buckets
    n_params_adjSD = 4 * P + n_buckets + 1
    n_params_lmSD  = 5 * P + n_buckets + 1
    n_params_fSD   = 3 * P_BASE + P_FULL + n_buckets + 4
    n_params_ffSD  = 4 * P + n_buckets + 2
    n_params_ifSD  = 6 * P + n_buckets + 2
    n_params_map = {"ss": n_params_ss, "adjSD": n_params_adjSD, "lmSD": n_params_lmSD}
    n_params_map.update({f"fSD_K{k}": n_params_fSD for k in FSD_K_VALUES})
    n_params_map.update({f"ffSD_K{k}": n_params_ffSD for k in FFSD_K_VALUES})
    n_params_map.update({f"ifSD_K{k}": n_params_ifSD for k in IFSD_K_VALUES})

    ss_step    = make_ss_rolling(y_jax, Z_jax, n_buckets, TRAIN_SIZE, max_n, Q_ALPHA, 5000)
    adjSD_step = make_adjSD_rolling(y_jax, Z_jax, n_buckets, TRAIN_SIZE, max_n, ALPHA, 5000)
    fSD_steps  = [make_fSD_rolling(y_jax, Z_jax, n_buckets, TRAIN_SIZE, max_n, ALPHA, 5000, K) for K in FSD_K_VALUES]
    ffSD_steps = [make_ffSD_rolling(y_jax, Z_jax, n_buckets, TRAIN_SIZE, max_n, ALPHA, 5000, K) for K in FFSD_K_VALUES]
    ifSD_steps = [make_ifSD_rolling(y_jax, Z_jax, n_buckets, TRAIN_SIZE, max_n, ALPHA, 5000, K) for K in IFSD_K_VALUES]
    lmSD_step  = make_lmSD_rolling(y_jax, Z_jax, n_buckets, TRAIN_SIZE, max_n, ALPHA, 5000)

    ss_carry0    = _ss_cold_carry(y_jax, Z_jax, n_buckets)
    adjSD_carry0 = _adjSD_cold_carry(y_jax, Z_jax, n_buckets)
    fSD_carry0s  = [_fSD_cold_carry(y_jax, Z_jax, n_buckets) for _ in FSD_K_VALUES]
    ffSD_carry0s = [_ffSD_cold_carry(y_jax, Z_jax, n_buckets) for _ in FFSD_K_VALUES]
    ifSD_carry0s = [_ifSD_cold_carry(y_jax, Z_jax, n_buckets) for _ in IFSD_K_VALUES]
    lmSD_carry0  = _lmSD_cold_carry(y_jax, Z_jax, n_buckets)

    def run_all(carries, idx):
        ss_carry, adjSD_carry, fSD_carries, ffSD_carries, ifSD_carries, lmSD_carry = carries
        _, ss_out    = lax.scan(ss_step, ss_carry, idx)
        _, adjSD_out = lax.scan(adjSD_step, adjSD_carry, idx)
        fSD_outs  = [lax.scan(step, carry, idx)[1] for step, carry in zip(fSD_steps, fSD_carries)]
        ffSD_outs = [lax.scan(step, carry, idx)[1] for step, carry in zip(ffSD_steps, ffSD_carries)]
        ifSD_outs = [lax.scan(step, carry, idx)[1] for step, carry in zip(ifSD_steps, ifSD_carries)]
        _, lmSD_out  = lax.scan(lmSD_step, lmSD_carry, idx)
        return ss_out, adjSD_out, fSD_outs, ffSD_outs, ifSD_outs, lmSD_out

    print("Compiling and running all models in a single JIT...")
    ss_out, adjSD_out, fSD_outs, ffSD_outs, ifSD_outs, lmSD_out = jax.jit(run_all)(
        (ss_carry0, adjSD_carry0, fSD_carry0s, ffSD_carry0s, ifSD_carry0s, lmSD_carry0), indices
    )

    all_results = {"ss": ss_out, "adjSD": adjSD_out, "lmSD": lmSD_out}
    all_results.update({f"fSD_K{K}": out for K, out in zip(FSD_K_VALUES, fSD_outs)})
    all_results.update({f"ffSD_K{K}": out for K, out in zip(FFSD_K_VALUES, ffSD_outs)})
    all_results.update({f"ifSD_K{K}": out for K, out in zip(IFSD_K_VALUES, ifSD_outs)})

    y_test_all    = y_jax[TRAIN_SIZE:TRAIN_SIZE + n_test]
    mask_test_all = ~jnp.isnan(y_test_all)
    test_dates    = dates[TRAIN_SIZE:TRAIN_SIZE + n_test]
    total_obs     = int(jnp.sum(mask_test_all))

    model_names = (["ss", "adjSD", "lmSD"] + [f"fSD_K{k}" for k in FSD_K_VALUES]
                   + [f"ffSD_K{k}" for k in FFSD_K_VALUES] + [f"ifSD_K{k}" for k in IFSD_K_VALUES])
    step_frames = []
    agg_rows    = []

    for name in model_names:
        y_hat, P_mean, VaR, oos_ll, train_ll, niter, converged = all_results[name]
        n_params = n_params_map[name]

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

        pl.DataFrame({
            "date": test_dates,
            "predictions": y_hat.tolist(),
        }).write_parquet(os.path.join(OUTPUT_DIR, f"predictions_{name}.parquet"))

        total_oos_ll = float(jnp.sum(oos_ll))
        mse = float(compute_mse(y_test_all, y_hat, mask_test_all))
        mae = float(compute_mae(y_test_all, y_hat, mask_test_all))
        aic = float(compute_aic(jnp.array(total_oos_ll), n_params))
        bic = float(compute_bic(jnp.array(total_oos_ll), n_params, total_obs))
        agg_rows.append({
            "model": name, "mse": mse, "mae": mae,
            "total_oos_loglik": total_oos_ll,
            "aic": aic, "bic": bic,
            "n_params": n_params, "n_test_steps": n_test, "total_oos_obs": total_obs,
        })

    pl.concat(step_frames).write_parquet(os.path.join(OUTPUT_DIR, "step_results.parquet"))

    df_agg = pl.DataFrame(agg_rows)
    df_agg.write_csv(os.path.join(OUTPUT_DIR, "aggregate_metrics.csv"))
    print("\nAggregate metrics:")
    print(df_agg)
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
