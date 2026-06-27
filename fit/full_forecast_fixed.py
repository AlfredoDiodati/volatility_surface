import argparse
import json
import os

import numpy as np
import jax
import jax.numpy as jnp
import polars as pl

jax.config.update("jax_default_device", jax.devices("cpu")[0])
print(f"Running on: {jax.devices('cpu')[0]}")

from models.ss import fit_collapsed, forecast as ss_forecast, forecast_rolling_h as ss_forecast_rolling_h
from models.adjSD import fit as adjSD_fit, forecast as adjSD_forecast, forecast_rolling_h as adjSD_forecast_rolling_h
from models.ff_SS import fit as ffSS_fit, forecast as ffSS_forecast, forecast_rolling_h as ffSS_forecast_rolling_h, eval_and_refit_k as ffSS_eval_and_refit_k
from models.ff_SD import fit as ffSD_fit, forecast as ffSD_forecast, forecast_rolling_h as ffSD_forecast_rolling_h, eval_and_refit_k as ffSD_eval_and_refit_k
from models.lmSD import fit as lmSD_fit, forecast as lmSD_forecast, forecast_rolling_h as lmSD_forecast_rolling_h
from models.MSMSD import fit as msmSD_fit, forecast as msmSD_forecast, forecast_rolling_h as msmSD_forecast_rolling_h, eval_and_refit_k as msmSD_eval_and_refit_k
from fit._forecast_metrics import compute_mse, compute_mae, compute_aic, compute_bic

PARQUET_PATH = "data/SPX/otm/full.parquet"
OUTPUT_DIR = "out/SPX/otm/full_performance_fixed"
FACTOR_LOADING_COLS = ["level", "moneyness", "maturity"]
P_BASE = 3
P = P_BASE + 1

MAXITER = 5000
TOL = 1e-4
LR = 1.0
Q_ALPHA = -1.6448536269514722
ALPHA = 0.05
FFSS_K_VALUES = [1, 2, 3, 5, 10]
FFSD_K_VALUES = FFSS_K_VALUES
MSMSD_K_VALUES = FFSS_K_VALUES
FCST_HORIZONS = (5, 22, 66, 260)
_MAX_FCST_H = max(FCST_HORIZONS)

OPT = {"learning_rate": LR, "tol": TOL}


def _run_ss(y_tr, Z_tr, ig, init, Z_te, y_te, Z_te_ext):
    r = fit_collapsed(y_tr, Z_tr, ig, init, opt_options=OPT, maxiter=MAXITER)
    y_hat, P_mean, VaR, oos_ll = ss_forecast(r, Z_te, y_te, Q_ALPHA)
    preds_h = ss_forecast_rolling_h(r, Z_te_ext, y_te, FCST_HORIZONS)
    return r, y_hat, P_mean, VaR, oos_ll, preds_h


def _run_adjSD(y_tr, Z_tr, ig, Z_te, y_te, Z_te_ext):
    r = adjSD_fit(y_tr, Z_tr, ig, opt_options=OPT, maxiter=MAXITER)
    y_hat, P_mean, VaR, oos_ll = adjSD_forecast(r, Z_te, y_te, ALPHA)
    preds_h = adjSD_forecast_rolling_h(r, Z_te_ext, y_te, FCST_HORIZONS)
    return r, y_hat, P_mean, VaR, oos_ll, preds_h


def _run_lmSD(y_tr, Z_tr, ig, Z_te, y_te, Z_te_ext):
    r = lmSD_fit(y_tr, Z_tr, ig, opt_options=OPT, maxiter=MAXITER)
    y_hat, P_mean, VaR, oos_ll = lmSD_forecast(r, Z_te, y_te, ALPHA)
    preds_h = lmSD_forecast_rolling_h(r, Z_te_ext, y_te, FCST_HORIZONS)
    return r, y_hat, P_mean, VaR, oos_ll, preds_h


def _run_ffSS_impl(y_tr, Z_tr, ig, K, Z_te, y_te, Z_te_ext):
    r = ffSS_fit(y_tr, Z_tr, ig, K, opt_options=OPT, maxiter=MAXITER)
    y_hat, P_mean, VaR, oos_ll, b_oos = ffSS_forecast(r, Z_te, y_te, Q_ALPHA)
    preds_h = ffSS_forecast_rolling_h(r, Z_te_ext, y_te, FCST_HORIZONS)
    return r, y_hat, P_mean, VaR, oos_ll, preds_h, b_oos


def _run_ffSD_impl(y_tr, Z_tr, ig, K, Z_te, y_te, Z_te_ext):
    r = ffSD_fit(y_tr, Z_tr, ig, K, opt_options=OPT, maxiter=MAXITER)
    y_hat, P_mean, VaR, oos_ll, b_oos = ffSD_forecast(r, Z_te, y_te, K, ALPHA)
    preds_h = ffSD_forecast_rolling_h(r, Z_te_ext, y_te, K, FCST_HORIZONS)
    return r, y_hat, P_mean, VaR, oos_ll, preds_h, b_oos


def _run_msmSD_impl(y_tr, Z_tr, ig, K, Z_te, y_te, Z_te_ext):
    r = msmSD_fit(y_tr, Z_tr, ig, K, 1.0, opt_options=OPT, maxiter=MAXITER)
    y_hat, P_mean, VaR, oos_ll = msmSD_forecast(r, Z_te, y_te, K, 1.0, ALPHA)
    preds_h = msmSD_forecast_rolling_h(r, Z_te_ext, y_te, K, 1.0, FCST_HORIZONS)
    return r, y_hat, P_mean, VaR, oos_ll, preds_h


_run_ss = jax.jit(_run_ss)
_run_adjSD = jax.jit(_run_adjSD)
_run_lmSD = jax.jit(_run_lmSD)
_run_ffSS = jax.jit(_run_ffSS_impl, static_argnames=("K",))
_run_ffSD = jax.jit(_run_ffSD_impl, static_argnames=("K",))
_run_msmSD = jax.jit(_run_msmSD_impl, static_argnames=("K",))


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
    X = np.asarray(Z_train[:, :, :P_BASE]).reshape(-1, P_BASE)
    y = np.asarray(y_train).reshape(-1)
    bidx = np.asarray(Z_train[:, :, -1]).reshape(-1).astype(int)
    mask = ~np.isnan(y)
    resid = np.where(mask, y - X @ beta_ols, np.nan)
    mu = np.array([
        float(np.nanmean(resid[bidx == k])) if np.any((bidx == k) & mask) else 0.0
        for k in range(n_buckets)
    ])
    mu = np.where(np.isnan(mu), 0.0, mu)
    omega = jnp.array(mu - mu[0])
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
        "beta_bar": jnp.append(beta_ols, 1.0),
        "B": 0.95 * jnp.eye(P),
        "A": 0.05 * jnp.eye(P),
        "sigma2": sig2,
        "omega": omega,
        "C": jnp.full(P, 1e-3),
        "nu": jnp.array(10.0),
    }


def _ffSS_ig(beta_ols, sig2, omega):
    return {
        "beta_bar": jnp.append(beta_ols, 1.0),
        "sigma2": sig2,
        "Q_param": 1e-3 * jnp.eye(P),
        "omega": omega,
        "eta": jnp.full(P, 0.06),
        "alpha": jnp.array(1.5),
    }


def _ffSD_ig(beta_ols, sig2, omega):
    return {
        "beta_bar": jnp.append(beta_ols, 1.0),
        "A": 0.05 * jnp.eye(P),
        "sigma2": sig2,
        "omega_load": omega,
        "eta": jnp.full(P, 0.4),
        "phi": jnp.full(P, 3.0),
        "C": 1e-3 * jnp.eye(P),
        "nu": jnp.array(10.0),
    }


def _lmSD_ig(beta_ols, sig2, omega):
    return {
        "beta_bar": jnp.append(beta_ols, 1.0),
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
        **{f"ffSS_K{k}": 3 * P + n_buckets for k in FFSS_K_VALUES},
        **{f"ffSD_K{k}": 4 * P + n_buckets + 1 for k in FFSD_K_VALUES},
        **{f"msmSD_K{k}": 3 * P_BASE + P + n_buckets + 4 for k in MSMSD_K_VALUES},
    }

    step_frames = []
    agg_rows = []

    def _save_1step(name, y_hat, P_mean, VaR, oos_ll, train_ll, niter, converged):
        pl.DataFrame({
            "date": test_dates,
            "predictions": pl.Series(np.asarray(y_hat)),
        }).write_parquet(os.path.join(OUTPUT_DIR, f"predictions_{name}{suffix}.parquet"))

        n_params = n_params_map[name]
        total_oos_ll = float(jnp.sum(oos_ll))
        mse = float(compute_mse(y_test, y_hat, mask_test))
        mae = float(compute_mae(y_test, y_hat, mask_test))
        aic = float(compute_aic(jnp.array(total_oos_ll), n_params))
        bic = float(compute_bic(jnp.array(total_oos_ll), n_params, total_obs))

        step_df = pl.DataFrame({
            "date": test_dates,
            "model": pl.Series([name] * n_test),
            "P_mean": pl.Series(np.asarray(P_mean)),
            "VaR": pl.Series(np.asarray(VaR)),
            "oos_loglik": pl.Series(np.asarray(oos_ll)),
            "train_loglik": pl.Series([float(train_ll)] * n_test),
            "n_params": pl.Series([n_params] * n_test, dtype=pl.Int32),
            "niter": pl.Series([int(niter)] * n_test, dtype=pl.Int32),
            "is_converged": pl.Series([bool(converged)] * n_test),
        })
        step_df.write_parquet(os.path.join(OUTPUT_DIR, f"step_{name}{suffix}.parquet"))
        step_frames.append(step_df)
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
                "predictions": pl.Series(np.asarray(preds_h[h_idx][:n_valid])),
            }))
        pl.concat(frames).write_parquet(
            os.path.join(OUTPUT_DIR, f"predictions_h_{name}{suffix}.parquet")
        )

    def _save_params(name, r):
        def _a(x): return np.asarray(x).tolist()
        if "H_param" in r:
            d = {
                "B_diag": _a(np.diag(r["B"])),
                "Q_diag": _a(np.diag(r["Q_param"])),
                "sigma2": float(r["H_param"][0, 0]),
                "bar_beta": _a(r["bar_beta"]),
                "omega": _a(r["omega"]),
            }
        elif "alpha" in r:
            d = {
                "beta_bar": _a(r["beta_bar"]),
                "sigma2": float(r["sigma2"]),
                "Q_diag": _a(np.diag(r["Q_param"])),
                "eta": _a(r["eta"]),
                "alpha": float(r["alpha"]),
                "omega": _a(r["omega"]),
            }
        elif "phi" in r:
            d = {
                "beta_bar": _a(r["beta_bar"]),
                "A_diag": _a(np.diag(r["A"])),
                "sigma2": float(r["sigma2"]),
                "eta": _a(r["eta"]),
                "phi": float(r["phi"][0]),
                "C_diag": _a(np.diag(r["C"])),
                "nu": float(r["nu"]),
                "omega_load": _a(r["omega_load"]),
            }
        elif "d" in r:
            d = {
                "beta_bar": _a(r["beta_bar"]),
                "A_diag": _a(np.diag(r["A"])),
                "d": _a(r["d"]),
                "sigma2": float(r["sigma2"]),
                "C_diag": _a(r["C"]),
                "nu": float(r["nu"]),
                "omega": _a(r["omega"]),
            }
        elif "m0" in r:
            d = {
                "beta_bar": _a(r["beta_bar"]),
                "B_diag": _a(np.diag(r["B"])),
                "A_diag": _a(np.diag(r["A"])),
                "sigma2": float(r["sigma2"]),
                "sigma_0": float(r["sigma_0"]),
                "C_diag": _a(np.diag(r["C"])),
                "nu": float(r["nu"]),
                "m0": float(r["m0"]),
                "gamma_K": float(r["gamma_K"]),
                "b": float(r["b"]),
                "omega_load": _a(r["omega_load"]),
            }
        else:
            d = {
                "beta_bar": _a(r["beta_bar"]),
                "B_diag": _a(np.diag(r["B"])),
                "A_diag": _a(np.diag(r["A"])),
                "sigma2": float(r["sigma2"]),
                "C_diag": _a(r["C"]),
                "nu": float(r["nu"]),
                "omega": _a(r["omega"]),
            }
        with open(os.path.join(OUTPUT_DIR, f"params_{name}{suffix}.json"), "w") as f:
            json.dump(d, f, indent=2)

    def _save_b(name, r, b_oos):
        is_ffss = "att" in r
        if is_ffss:
            K_p1 = r["att"].shape[1] // P
            b_ins = np.asarray(r["att"]).reshape(train_size, K_p1, P)
            b_oo = np.asarray(b_oos).reshape(n_test, K_p1, P)
        else:
            b_ins = np.asarray(r["b_hist"])
            b_oo = np.asarray(b_oos)
        K_p1 = b_ins.shape[1]

        def _to_df(b_arr, date_list, is_ins):
            T_ = b_arr.shape[0]
            date_rep = np.repeat(np.array(date_list), K_p1 * P).tolist()
            k_rep = np.tile(np.repeat(np.arange(K_p1), P), T_)
            j_rep = np.tile(np.arange(P), T_ * K_p1)
            return pl.DataFrame({
                "date": date_rep,
                "is_insample": pl.Series([is_ins] * (T_ * K_p1 * P)),
                "k": pl.Series(k_rep, dtype=pl.Int32),
                "j": pl.Series(j_rep, dtype=pl.Int32),
                "b": pl.Series(b_arr.reshape(-1).tolist()),
            })

        pl.concat([
            _to_df(b_ins, dates[:train_size], True),
            _to_df(b_oo, test_dates, False),
        ]).write_parquet(os.path.join(OUTPUT_DIR, f"b_{name}{suffix}.parquet"))

    def _skip(name, extra_paths=()):
        if selected is not None and name not in selected:
            return True
        pred_path = os.path.join(OUTPUT_DIR, f"predictions_{name}{suffix}.parquet")
        step_path = os.path.join(OUTPUT_DIR, f"step_{name}{suffix}.parquet")
        params_path = os.path.join(OUTPUT_DIR, f"params_{name}{suffix}.json")
        if all(os.path.exists(p) for p in [pred_path, step_path, params_path, *extra_paths]):
            print(f"  {name}: output exists, skipping.", flush=True)
            return True
        return False

    print("Running models (single fit per model)...")

    if not _skip("ss"):
        print("  ss...", flush=True)
        ig, init = _ss_ig(beta_ols, sig2, omega)
        r, y_hat, P_mean, VaR, oos_ll, preds_h = _run_ss(
            y_train, Z_train, ig, init, Z_test, y_test, Z_test_ext)
        jax.effects_barrier()
        _save_1step("ss", y_hat, P_mean, VaR, oos_ll,
                    r["loglikelihood"], r["niter"], r["is_converged"])
        _save_params("ss", r)
        print("  ss h-step...", flush=True)
        _save_hstep("ss", preds_h)
        jax.clear_caches()

    if not _skip("adjSD"):
        print("  adjSD...", flush=True)
        r, y_hat, P_mean, VaR, oos_ll, preds_h = _run_adjSD(
            y_train, Z_train, _adjSD_ig(beta_ols, sig2, omega), Z_test, y_test, Z_test_ext)
        jax.effects_barrier()
        _save_1step("adjSD", y_hat, P_mean, VaR, oos_ll,
                    r["log_likelihood"], r["niter"], r["is_converged"])
        _save_params("adjSD", r)
        print("  adjSD h-step...", flush=True)
        _save_hstep("adjSD", preds_h)
        jax.clear_caches()

    if not _skip("lmSD"):
        print("  lmSD...", flush=True)
        r, y_hat, P_mean, VaR, oos_ll, preds_h = _run_lmSD(
            y_train, Z_train, _lmSD_ig(beta_ols, sig2, omega), Z_test, y_test, Z_test_ext)
        jax.effects_barrier()
        _save_1step("lmSD", y_hat, P_mean, VaR, oos_ll,
                    r["log_likelihood"], r["niter"], r["is_converged"])
        _save_params("lmSD", r)
        print("  lmSD h-step...", flush=True)
        _save_hstep("lmSD", preds_h)
        jax.clear_caches()

    fits = {}
    for K in FFSS_K_VALUES:
        name = f"ffSS_K{K}"
        b_path = os.path.join(OUTPUT_DIR, f"b_{name}{suffix}.parquet")
        if _skip(name, [b_path]):
            continue
        print(f"  {name}...", flush=True)
        fits[K] = _run_ffSS(
            y_train, Z_train, _ffSS_ig(beta_ols, sig2, omega), K, Z_test, y_test, Z_test_ext)
        jax.effects_barrier()

    if len(fits) > 1:
        finite_ks = [k for k in fits if np.isfinite(float(fits[k][0]["loglikelihood"]))]
        best_K = max(finite_ks, key=lambda k: float(fits[k][0]["loglikelihood"]), default=None) if finite_ks else None
        r_best = fits[best_K][0] if best_K is not None else None
        for K in list(fits):
            if K == best_K or r_best is None:
                continue
            r_K = fits[K][0]
            ll_K_own = float(r_K["loglikelihood"])
            ll_best_in_K, r_adapted = ffSS_eval_and_refit_k(r_best, y_train, Z_train, K)
            if np.isfinite(ll_best_in_K) and (not np.isfinite(ll_K_own) or ll_best_in_K > ll_K_own):
                print(f"  ffSS_K{K}: adopting K={best_K} params (LL {ll_best_in_K:.1f} > {ll_K_own:.1f})", flush=True)
                y_hat, P_mean, VaR, oos_ll, b_oos = ffSS_forecast(r_adapted, Z_test, y_test, Q_ALPHA)
                preds_h = ffSS_forecast_rolling_h(r_adapted, Z_test_ext, y_test, FCST_HORIZONS)
                fits[K] = (r_adapted, y_hat, P_mean, VaR, oos_ll, preds_h, b_oos)

    for K, (r, y_hat, P_mean, VaR, oos_ll, preds_h, b_oos) in fits.items():
        name = f"ffSS_K{K}"
        _save_1step(name, y_hat, P_mean, VaR, oos_ll,
                    r["loglikelihood"], r["niter"], r["is_converged"])
        _save_params(name, r)
        _save_b(name, r, b_oos)
        print(f"  {name} h-step...", flush=True)
        _save_hstep(name, preds_h)
    jax.clear_caches()

    fits = {}
    for K in FFSD_K_VALUES:
        name = f"ffSD_K{K}"
        b_path = os.path.join(OUTPUT_DIR, f"b_{name}{suffix}.parquet")
        if _skip(name, [b_path]):
            continue
        print(f"  {name}...", flush=True)
        fits[K] = _run_ffSD(
            y_train, Z_train, _ffSD_ig(beta_ols, sig2, omega), K, Z_test, y_test, Z_test_ext)
        jax.effects_barrier()

    if len(fits) > 1:
        finite_ks = [k for k in fits if np.isfinite(float(fits[k][0]["log_likelihood"]))]
        best_K = max(finite_ks, key=lambda k: float(fits[k][0]["log_likelihood"]), default=None) if finite_ks else None
        r_best = fits[best_K][0] if best_K is not None else None
        for K in list(fits):
            if K == best_K or r_best is None:
                continue
            r_K = fits[K][0]
            ll_K_own = float(r_K["log_likelihood"])
            ll_best_in_K, r_adapted = ffSD_eval_and_refit_k(r_best, y_train, Z_train, K)
            if np.isfinite(ll_best_in_K) and (not np.isfinite(ll_K_own) or ll_best_in_K > ll_K_own):
                print(f"  ffSD_K{K}: adopting K={best_K} params (LL {ll_best_in_K:.1f} > {ll_K_own:.1f})", flush=True)
                y_hat, P_mean, VaR, oos_ll, b_oos = ffSD_forecast(r_adapted, Z_test, y_test, K, ALPHA)
                preds_h = ffSD_forecast_rolling_h(r_adapted, Z_test_ext, y_test, K, FCST_HORIZONS)
                fits[K] = (r_adapted, y_hat, P_mean, VaR, oos_ll, preds_h, b_oos)

    for K, (r, y_hat, P_mean, VaR, oos_ll, preds_h, b_oos) in fits.items():
        name = f"ffSD_K{K}"
        _save_1step(name, y_hat, P_mean, VaR, oos_ll,
                    r["log_likelihood"], r["niter"], r["is_converged"])
        _save_params(name, r)
        _save_b(name, r, b_oos)
        print(f"  {name} h-step...", flush=True)
        _save_hstep(name, preds_h)
    jax.clear_caches()

    fits = {}
    for K in MSMSD_K_VALUES:
        name = f"msmSD_K{K}"
        if _skip(name):
            continue
        print(f"  {name}...", flush=True)
        fits[K] = _run_msmSD(
            y_train, Z_train, _msmSD_ig(beta_ols, sig2, omega), K, Z_test, y_test, Z_test_ext)
        jax.effects_barrier()

    if len(fits) > 1:
        finite_ks = [k for k in fits if np.isfinite(float(fits[k][0]["log_likelihood"]))]
        best_K = max(finite_ks, key=lambda k: float(fits[k][0]["log_likelihood"]), default=None) if finite_ks else None
        r_best = fits[best_K][0] if best_K is not None else None
        for K in list(fits):
            if K == best_K or r_best is None:
                continue
            r_K = fits[K][0]
            ll_K_own = float(r_K["log_likelihood"])
            ll_best_in_K, r_adapted = msmSD_eval_and_refit_k(r_best, y_train, Z_train, K, 1.0)
            if np.isfinite(ll_best_in_K) and (not np.isfinite(ll_K_own) or ll_best_in_K > ll_K_own):
                print(f"  msmSD_K{K}: adopting K={best_K} params (LL {ll_best_in_K:.1f} > {ll_K_own:.1f})", flush=True)
                y_hat, P_mean, VaR, oos_ll = msmSD_forecast(r_adapted, Z_test, y_test, K, 1.0, ALPHA)
                preds_h = msmSD_forecast_rolling_h(r_adapted, Z_test_ext, y_test, K, 1.0, FCST_HORIZONS)
                fits[K] = (r_adapted, y_hat, P_mean, VaR, oos_ll, preds_h)

    for K, (r, y_hat, P_mean, VaR, oos_ll, preds_h) in fits.items():
        name = f"msmSD_K{K}"
        _save_1step(name, y_hat, P_mean, VaR, oos_ll,
                    r["log_likelihood"], r["niter"], r["is_converged"])
        _save_params(name, r)
        print(f"  {name} h-step...", flush=True)
        _save_hstep(name, preds_h)
    jax.clear_caches()

    if not step_frames:
        print("No new models ran; step_results.parquet not updated.")
        print(f"\nSaved to {OUTPUT_DIR}")
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
