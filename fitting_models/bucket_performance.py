import os
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import polars as pl
import pandas as pd
from scipy.stats import norm

from models.ss import fit_collapsed
from models.adjSD import fit as adjSD_fit, _filter as adjSD_filter
from models.f_SD import fit as fSD_fit, _filter as fSD_filter
from fitting_models._forecast_metrics import compute_mse, compute_mae, compute_aic, compute_bic

PARQUET_PATH = "data/SPX/put/bucket.parquet"
OUTPUT_DIR = "out/SPX/put/bucket_performance"
FACTOR_LOADING_COLS = ["level", "moneyness", "maturity"]
P_BASE = 3
P = P_BASE + 1
P_FULL = P

TRAIN_SIZE = 500
TEST_SIZE = 250
Q_ALPHA = float(norm.ppf(0.05))
FSD_K_VALUES = [3, 10, 25, 50, 100, 300]


def load_and_reshape(path):
    raw = (
        pl.read_parquet(path)
        .with_columns(pl.col("DATE").cast(pl.Utf8))
    )
    bucket_cols = sorted([c for c in raw.columns if c.startswith("bucket_")])
    n_buckets = len(bucket_cols) + 1
    raw = raw.with_columns(
        pl.max_horizontal(
            [pl.when(pl.col(c)).then(i + 1).otherwise(0) for i, c in enumerate(bucket_cols)]
        ).alias("bucket_idx")
    ).sort(["DATE", *FACTOR_LOADING_COLS])
    dates = raw["DATE"].unique(maintain_order=True).sort().to_list()
    T = len(dates)
    max_n = int(raw.group_by("DATE").len()["len"].max())
    y_cube = np.full((T, max_n), np.nan, dtype=np.float64)
    Z_cube = np.zeros((T, max_n, P_BASE + 1), dtype=np.float64)
    for t, date in enumerate(dates):
        s = raw.filter(pl.col("DATE") == date).sort(FACTOR_LOADING_COLS)
        n_t = len(s)
        y_cube[t, :n_t] = s["logIV"].to_numpy()
        Z_cube[t, :n_t, :P_BASE] = s[FACTOR_LOADING_COLS].to_numpy()
        Z_cube[t, :n_t, P_BASE] = s["bucket_idx"].to_numpy().astype(float)
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


def _forecast_and_oos(omega, B, Q, ct, sigma2, att_T, Ptt_T, Z_test, y_test, max_n, q_alpha):
    base = Z_test[:, :P_BASE]
    bidx = Z_test[:, P_BASE].astype(jnp.int32)
    Z_h = jnp.concatenate([base, omega[bidx][:, None]], axis=-1)
    a_pred = B @ att_T + ct
    P_pred = B @ Ptt_T @ B.T + Q
    y_hat = Z_h @ a_pred
    P_mean = y_hat.mean()
    z_sum = Z_h.sum(axis=0)
    F_sum = max_n * sigma2 + z_sum @ P_pred @ z_sum
    VaR = P_mean + q_alpha / max_n * jnp.sqrt(F_sum)
    mask = ~jnp.isnan(y_test)
    y_m = jnp.where(mask, y_test, 0.0)
    v = (y_m - Z_h @ a_pred) * mask
    F_vec = jnp.einsum("ip,pq,iq->i", Z_h, P_pred, Z_h) + sigma2
    N = jnp.sum(mask)
    oos_ll = -0.5 * (N * jnp.log(2 * jnp.pi) + jnp.sum(mask * jnp.log(F_vec)) + jnp.sum(mask * v ** 2 / F_vec))
    return y_hat, P_mean, VaR, oos_ll


def make_ss_rolling(y_jax, Z_jax, n_buckets, train_size, max_n, q_alpha, maxiter):
    def step(carry, i):
        y_win = lax.dynamic_slice(y_jax, (i, 0), (train_size, max_n))
        Z_win = lax.dynamic_slice(Z_jax, (i, 0, 0), (train_size, max_n, P_BASE + 1))
        y_test = lax.dynamic_slice(y_jax, (i + train_size, 0), (1, max_n))[0]
        Z_test = lax.dynamic_slice(Z_jax, (i + train_size, 0, 0), (1, max_n, P_BASE + 1))[0]

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

        y_hat, P_mean, VaR, oos_ll = _forecast_and_oos(
            r["omega"], r["B"], r["Q_param"], r["ct"], r["H_param"][0, 0],
            r["att"][-1], r["Ptt"][-1], Z_test, y_test, max_n, q_alpha)

        new_carry = (r["Q_param"], r["H_param"], r["B"], r["bar_beta"], r["omega"])
        return new_carry, (y_hat, P_mean, VaR, oos_ll, r["loglikelihood"], r["niter"], r["is_converged"])

    return step


def make_adjSD_rolling(y_jax, Z_jax, n_buckets, train_size, max_n, q_alpha, maxiter):
    def step(carry, i):
        y_win = lax.dynamic_slice(y_jax, (i, 0), (train_size, max_n))
        Z_win = lax.dynamic_slice(Z_jax, (i, 0, 0), (train_size, max_n, P_BASE + 1))
        y_test = lax.dynamic_slice(y_jax, (i + train_size, 0), (1, max_n))[0]
        Z_test = lax.dynamic_slice(Z_jax, (i + train_size, 0, 0), (1, max_n, P_BASE + 1))[0]

        ig = {"beta_bar": carry[0], "B": carry[1], "A": carry[2], "sigma2": carry[3],
              "omega": carry[4], "C": carry[5], "nu": carry[6]}
        r = adjSD_fit(y_win, Z_win, ig,
                      opt_options={"learning_rate": 1e-3, "tol": 1e-6},
                      maxiter=maxiter)

        base = Z_test[:, :P_BASE]
        bidx = Z_test[:, P_BASE].astype(jnp.int32)
        omega_col = r["omega"][bidx]
        Z_h = jnp.concatenate([base, omega_col[:, None]], axis=-1)
        y_hat = Z_h @ r["beta_T"]
        P_mean = y_hat.mean()
        z_sum = Z_h.sum(axis=0)
        F_sum = max_n * r["sigma2"] + (z_sum ** 2 * r["C"]).sum()
        VaR = P_mean + q_alpha / max_n * jnp.sqrt(F_sum)

        mask_t = ~jnp.isnan(y_test)
        y_m = jnp.where(mask_t, y_test, 0.0)
        _, lls, _ = adjSD_filter(y_m[None], base[None], bidx[None], mask_t[None].astype(float),
                                 r, state0=r["beta_T"])
        oos_ll = lls[0]

        new_carry = (r["beta_bar"], r["B"], r["A"], r["sigma2"], r["omega"], r["C"], r["nu"])
        return new_carry, (y_hat, P_mean, VaR, oos_ll, r["log_likelihood"], r["niter"], r["is_converged"])

    return step


def make_fSD_rolling(y_jax, Z_jax, n_buckets, train_size, max_n, q_alpha, maxiter, K):
    def step(carry, i):
        y_win = lax.dynamic_slice(y_jax, (i, 0), (train_size, max_n))
        Z_win = lax.dynamic_slice(Z_jax, (i, 0, 0), (train_size, max_n, P_BASE + 1))
        y_test = lax.dynamic_slice(y_jax, (i + train_size, 0), (1, max_n))[0]
        Z_test = lax.dynamic_slice(Z_jax, (i + train_size, 0, 0), (1, max_n, P_BASE + 1))[0]

        ig = {"beta_bar": carry[0], "B": carry[1], "A": carry[2], "sigma2": carry[3],
              "sigma_0": carry[4], "omega_load": carry[5], "eta": carry[6],
              "rho_K": carry[7], "C": carry[8], "nu": carry[9]}
        r = fSD_fit(y_win, Z_win, ig, K=K,
                    opt_options={"learning_rate": 1e-3, "tol": 1e-6},
                    maxiter=maxiter)

        base = Z_test[:, :P_BASE]
        bidx = Z_test[:, P_BASE].astype(jnp.int32)
        omega_col = r["omega_load"][bidx]
        Z_h = jnp.concatenate([base, omega_col[:, None]], axis=-1)
        sigma_T = r["sigma_0"] + jnp.sum(r["b_T"])
        beta_full_T = jnp.append(r["beta_tilde_T"], sigma_T)
        y_hat = Z_h @ beta_full_T
        P_mean = y_hat.mean()
        z_sum = Z_h.sum(axis=0)
        F_sum = max_n * r["sigma2"] + z_sum @ r["C"] @ z_sum
        VaR = P_mean + q_alpha / max_n * jnp.sqrt(F_sum)

        mask_t = ~jnp.isnan(y_test)
        y_m = jnp.where(mask_t, y_test, 0.0)
        _, lls, _ = fSD_filter(y_m[None], base[None], bidx[None], mask_t[None].astype(float),
                               r, K, state0=(r["b_T"], r["beta_tilde_T"]))
        oos_ll = lls[0]

        new_carry = (r["beta_bar"], r["B"], r["A"], r["sigma2"], r["sigma_0"],
                     r["omega_load"], r["eta"], r["rho_K"], r["C"], r["nu"])
        return new_carry, (y_hat, P_mean, VaR, oos_ll, r["log_likelihood"], r["niter"], r["is_converged"])

    return step


def _ss_cold_carry(y_win_np, Z_win_np, n_buckets):
    X = Z_win_np[:, :, :P_BASE].reshape(-1, P_BASE)
    y = y_win_np.reshape(-1)
    mask = ~np.isnan(y)
    beta_ols, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
    sig2 = float(np.var(y[mask] - X[mask] @ beta_ols))
    omega = np.zeros(n_buckets)
    omega[1:] = 1e-2
    return (
        jnp.array(np.diag(np.full(P, 1e-3))),
        jnp.array([[sig2]]),
        jnp.array(0.95 * np.eye(P)),
        jnp.array(np.append(beta_ols, 0.0)),
        jnp.array(omega),
    )


def _adjSD_cold_carry(y_win_np, Z_win_np, n_buckets):
    X = Z_win_np[:, :, :P_BASE].reshape(-1, P_BASE)
    y = y_win_np.reshape(-1)
    mask = ~np.isnan(y)
    beta_ols, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
    sig2 = float(np.var(y[mask] - X[mask] @ beta_ols))
    omega = np.zeros(n_buckets)
    omega[1:] = 1e-2
    return (
        jnp.array(np.append(beta_ols, 0.0)),
        jnp.array(0.95 * np.eye(P)),
        jnp.array(0.05 * np.eye(P)),
        jnp.array(sig2),
        jnp.array(omega),
        jnp.array(1e-3 * np.ones(P)),
        jnp.array(10.0),
    )


def _fSD_cold_carry(y_win_np, Z_win_np, n_buckets):
    X = Z_win_np[:, :, :P_BASE].reshape(-1, P_BASE)
    y = y_win_np.reshape(-1)
    mask = ~np.isnan(y)
    beta_ols, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
    sig2 = float(np.var(y[mask] - X[mask] @ beta_ols))
    omega_load = np.zeros(n_buckets)
    omega_load[1:] = 1e-2
    return (
        jnp.array(beta_ols),
        jnp.array(0.95 * np.eye(P_BASE)),
        jnp.array(0.05 * np.eye(P_BASE)),
        jnp.array(sig2),
        jnp.array(0.1),
        jnp.array(omega_load),
        jnp.array(0.4),
        jnp.array(0.999),
        jnp.array(1e-3 * np.eye(P_FULL)),
        jnp.array(10.0),
    )


def main():
    print("Loading data...")
    y_cube_np, Z_cube_np, n_buckets, dates = load_and_reshape(PARQUET_PATH)
    T, max_n = y_cube_np.shape
    n_test = min(TEST_SIZE, T - TRAIN_SIZE)
    print(f"  T={T}, max_n={max_n}, n_buckets={n_buckets}, test_steps={n_test}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    y_jax = jnp.array(y_cube_np)
    Z_jax = jnp.array(Z_cube_np)
    indices = jnp.arange(n_test, dtype=jnp.int32)

    first_win_y = y_cube_np[:TRAIN_SIZE]
    first_win_Z = Z_cube_np[:TRAIN_SIZE]

    n_params_ss = 1 + P + P + (n_buckets - 1) + P
    n_params_adjSD = P + P + P + 1 + (n_buckets - 1) + P + 1
    n_params_fSD = P_BASE * 3 + 1 + 1 + (n_buckets - 1) + 1 + 1 + P_FULL + 1

    all_results = {}

    print("Running SS rolling window...")
    ss_step = make_ss_rolling(y_jax, Z_jax, n_buckets, TRAIN_SIZE, max_n, Q_ALPHA, 5000)
    run_ss = jax.jit(lambda carry, idx: lax.scan(ss_step, carry, idx))
    ss_carry0 = _ss_cold_carry(first_win_y, first_win_Z, n_buckets)
    _, ss_out = run_ss(ss_carry0, indices)
    all_results["ss"] = ss_out

    print("Running adjSD rolling window...")
    adjSD_step = make_adjSD_rolling(y_jax, Z_jax, n_buckets, TRAIN_SIZE, max_n, Q_ALPHA, 5000)
    run_adjSD = jax.jit(lambda carry, idx: lax.scan(adjSD_step, carry, idx))
    adjSD_carry0 = _adjSD_cold_carry(first_win_y, first_win_Z, n_buckets)
    _, adjSD_out = run_adjSD(adjSD_carry0, indices)
    all_results["adjSD"] = adjSD_out

    for K in FSD_K_VALUES:
        print(f"Running fSD K={K} rolling window...")
        fSD_step = make_fSD_rolling(y_jax, Z_jax, n_buckets, TRAIN_SIZE, max_n, Q_ALPHA, 5000, K)
        run_fSD = jax.jit(lambda carry, idx, _step=fSD_step: lax.scan(_step, carry, idx))
        fSD_carry0 = _fSD_cold_carry(first_win_y, first_win_Z, n_buckets)
        _, fSD_out = run_fSD(fSD_carry0, indices)
        all_results[f"fSD_K{K}"] = fSD_out

    y_test_all = jnp.array(y_cube_np[TRAIN_SIZE:TRAIN_SIZE + n_test])
    mask_test_all = ~jnp.isnan(y_test_all)
    test_dates = dates[TRAIN_SIZE:TRAIN_SIZE + n_test]

    scalar_rows = []
    agg_rows = []
    model_names = ["ss", "adjSD"] + [f"fSD_K{k}" for k in FSD_K_VALUES]

    for name in model_names:
        y_hat, P_mean, VaR, oos_ll, train_ll, niter, converged = all_results[name]
        y_hat_np = np.array(y_hat)
        P_mean_np = np.array(P_mean)
        VaR_np = np.array(VaR)
        oos_ll_np = np.array(oos_ll)
        train_ll_np = np.array(train_ll)
        niter_np = np.array(niter)
        converged_np = np.array(converged)

        n_params = n_params_ss if name == "ss" else (n_params_adjSD if name == "adjSD" else n_params_fSD)

        for step in range(n_test):
            scalar_rows.append({
                "date": test_dates[step], "model": name,
                "P_mean": float(P_mean_np[step]),
                "VaR": float(VaR_np[step]),
                "oos_loglik": float(oos_ll_np[step]),
                "train_loglik": float(train_ll_np[step]),
                "n_params": n_params,
                "niter": int(niter_np[step]),
                "is_converged": bool(converged_np[step]),
            })

        np.save(os.path.join(OUTPUT_DIR, f"predictions_{name}.npy"), y_hat_np)

        mse = float(compute_mse(y_test_all, jnp.array(y_hat_np), mask_test_all))
        mae = float(compute_mae(y_test_all, jnp.array(y_hat_np), mask_test_all))
        total_oos_ll = float(oos_ll_np.sum())
        total_obs = int(jnp.sum(mask_test_all))
        aic = float(compute_aic(jnp.array(total_oos_ll), n_params))
        bic = float(compute_bic(jnp.array(total_oos_ll), n_params, total_obs))
        agg_rows.append({
            "model": name, "mse": mse, "mae": mae,
            "total_oos_loglik": total_oos_ll,
            "aic": aic, "bic": bic,
            "n_params": n_params, "n_test_steps": n_test, "total_oos_obs": total_obs,
        })

    pd.DataFrame(scalar_rows).to_parquet(os.path.join(OUTPUT_DIR, "step_results.parquet"), index=False)
    df_agg = pd.DataFrame(agg_rows)
    df_agg.to_csv(os.path.join(OUTPUT_DIR, "aggregate_metrics.csv"), index=False)
    print("\nAggregate metrics:")
    print(df_agg.to_string(index=False))
    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__": main()