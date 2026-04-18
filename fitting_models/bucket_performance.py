import os
import jax
import jax.numpy as jnp
from jax import lax
import polars as pl

from models.ss import fit_collapsed, forecast as ss_forecast
from models.adjSD import fit as adjSD_fit, forecast as adjSD_forecast
from models.f0_SD import fit as fSD_fit, forecast as fSD_forecast
from fitting_models._forecast_metrics import compute_mse, compute_mae, compute_aic, compute_bic

PARQUET_PATH = "data/SPX/put/bucket.parquet"
OUTPUT_DIR = "out/SPX/put/bucket_performance"
FACTOR_LOADING_COLS = ["level", "moneyness", "maturity"]
P_BASE = 3
P = P_BASE + 1
P_FULL = P

TRAIN_SIZE = 500
TEST_SIZE = 250
Q_ALPHA = -1.6448536269514722  # scipy.stats.norm.ppf(0.05)
FSD_K_VALUES = [3, 10, 25, 50, 100, 300]


def load_and_reshape(path):
    raw = (
        pl.read_parquet(path)
        .with_columns(pl.col("DATE").cast(pl.Utf8))
    )
    bucket_cols = sorted([c for c in raw.columns if c.startswith("bucket_")])
    n_buckets = len(bucket_cols) + 1
    raw = (
        raw
        .with_columns(
            pl.max_horizontal(
                [pl.when(pl.col(c)).then(i + 1).otherwise(0) for i, c in enumerate(bucket_cols)]
            ).alias("bucket_idx")
        )
        .sort(["DATE", *FACTOR_LOADING_COLS])
        .with_columns(
            pl.int_range(pl.len()).over("DATE").cast(pl.Int32).alias("row_in_date")
        )
    )
    dates = raw["DATE"].unique(maintain_order=True).sort().to_list()
    T = len(dates)
    max_n = int(raw.group_by("DATE").len()["len"].max())
    raw = raw.with_columns(
        pl.col("DATE").replace(
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


def make_adjSD_rolling(y_jax, Z_jax, n_buckets, train_size, max_n, q_alpha, maxiter):
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

        preds, P_mean, VaR, oos_ll = adjSD_forecast(r, Z_test, y_test, q_alpha)

        new_carry = (r["beta_bar"], r["B"], r["A"], r["sigma2"], r["omega"], r["C"], r["nu"])
        return new_carry, (preds[0], P_mean[0], VaR[0], oos_ll[0], r["log_likelihood"], r["niter"], r["is_converged"])

    return step


def make_fSD_rolling(y_jax, Z_jax, n_buckets, train_size, max_n, q_alpha, maxiter, K):
    def step(carry, i):
        y_win = lax.dynamic_slice(y_jax, (i, 0), (train_size, max_n))
        Z_win = lax.dynamic_slice(Z_jax, (i, 0, 0), (train_size, max_n, P_BASE + 1))
        y_test = lax.dynamic_slice(y_jax, (i + train_size, 0), (1, max_n))
        Z_test = lax.dynamic_slice(Z_jax, (i + train_size, 0, 0), (1, max_n, P_BASE + 1))

        ig = {"beta_bar": carry[0], "B": carry[1], "A": carry[2], "sigma2": carry[3],
              "sigma_0": carry[4], "omega_load": carry[5], "eta": carry[6],
              "rho_K": carry[7], "C": carry[8], "nu": carry[9]}
        r = fSD_fit(y_win, Z_win, ig, K=K,
                    opt_options={"learning_rate": 1e-3, "tol": 1e-6},
                    maxiter=maxiter)

        preds, P_mean, VaR, oos_ll = fSD_forecast(r, Z_test, y_test, K, q_alpha)

        new_carry = (r["beta_bar"], r["B"], r["A"], r["sigma2"], r["sigma_0"],
                     r["omega_load"], r["eta"], r["rho_K"], r["C"], r["nu"])
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
        jnp.array(0.4),
        jnp.array(0.999),
        1e-3 * jnp.eye(P_FULL),
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

    n_params_ss = 1 + P + P + (n_buckets - 1) + P
    n_params_adjSD = P + P + P + 1 + (n_buckets - 1) + P + 1
    n_params_fSD = P_BASE * 3 + 1 + 1 + (n_buckets - 1) + 1 + 1 + P_FULL + 1
    n_params_map = {"ss": n_params_ss, "adjSD": n_params_adjSD}
    n_params_map.update({f"fSD_K{k}": n_params_fSD for k in FSD_K_VALUES})

    ss_step = make_ss_rolling(y_jax, Z_jax, n_buckets, TRAIN_SIZE, max_n, Q_ALPHA, 5000)
    adjSD_step = make_adjSD_rolling(y_jax, Z_jax, n_buckets, TRAIN_SIZE, max_n, Q_ALPHA, 5000)
    fSD_steps = [make_fSD_rolling(y_jax, Z_jax, n_buckets, TRAIN_SIZE, max_n, Q_ALPHA, 5000, K) for K in FSD_K_VALUES]

    ss_carry0 = _ss_cold_carry(y_jax, Z_jax, n_buckets)
    adjSD_carry0 = _adjSD_cold_carry(y_jax, Z_jax, n_buckets)
    fSD_carry0s = [_fSD_cold_carry(y_jax, Z_jax, n_buckets) for _ in FSD_K_VALUES]

    def run_all(carries, idx):
        ss_carry, adjSD_carry, fSD_carries = carries
        _, ss_out = lax.scan(ss_step, ss_carry, idx)
        _, adjSD_out = lax.scan(adjSD_step, adjSD_carry, idx)
        fSD_outs = [lax.scan(step, carry, idx)[1] for step, carry in zip(fSD_steps, fSD_carries)]
        return ss_out, adjSD_out, fSD_outs

    print("Compiling and running all models in a single JIT...")
    ss_out, adjSD_out, fSD_outs = jax.jit(run_all)((ss_carry0, adjSD_carry0, fSD_carry0s), indices)

    all_results = {"ss": ss_out, "adjSD": adjSD_out}
    all_results.update({f"fSD_K{K}": out for K, out in zip(FSD_K_VALUES, fSD_outs)})

    y_test_all = y_jax[TRAIN_SIZE:TRAIN_SIZE + n_test]
    mask_test_all = ~jnp.isnan(y_test_all)
    test_dates = dates[TRAIN_SIZE:TRAIN_SIZE + n_test]
    total_obs = int(jnp.sum(mask_test_all))

    model_names = ["ss", "adjSD"] + [f"fSD_K{k}" for k in FSD_K_VALUES]
    step_frames = []
    agg_rows = []

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