import os
import shutil
import functools

if shutil.which("nvidia-smi") is not None:
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import json
import jax
import jax.numpy as jnp

from models.lmSD  import simulate
from models.lmSD  import fit as lmSD_fit,  forecast as lmSD_forecast
from models.adjSD import fit as adjSD_fit, forecast as adjSD_forecast
from models.ss    import fit_collapsed as ss_fit, forecast as ss_forecast
from models.ff_SD import fit as ff_SD_fit, forecast as ff_SD_forecast
from models.f_SD  import fit as f_SD_fit,  forecast as f_SD_forecast
from models.MSMSD import fit as msmsd_fit, forecast as msmsd_forecast
from fit._forecast_metrics import compute_mse, compute_mae, compute_aic, compute_bic

print(f"Running on: {jax.devices()}")

PARAMS_PATH = "out/SPX/otm/params_lmSD.json"
OUTPUT_DIR  = "out/SPX/mc/simulate_lmSD"
SCORE_POWER = 1.0
ALPHA = 0.05
MAXITER = 2000
K_VALUES = [1, 2, 3, 5, 10]
N_BUCKETS = 4

MONEYNESS = jnp.array([0.9, 0.98, 1.05, 1.15, 1.3, 1.5])
MATURITY  = jnp.array([10, 50, 100, 180]) / 255.0

HORIZONS = [400, 2000]
SIGMA2_SCALES = [1.0, 10.0, 0.1]
SCALE_TEX = {1.0: r"\sigma^2", 10.0: r"10\,\sigma^2", 0.1: r"\sigma^2/10"}

_P_FF = 4
_P_TILDE = 3
_P_FULL = 4

NP_FF    = _P_FF + _P_FF + 1 + (N_BUCKETS - 1) + _P_FF + 1 + _P_FF + 1
NP_F     = _P_TILDE + _P_TILDE + _P_FULL + 1 + 1 + (N_BUCKETS - 1) + 1 + 1 + _P_FULL + 1
NP_LMSD  = 5 * _P_FF + N_BUCKETS + 1   # beta_bar, B, A, d, C, sigma2, omega[1:], nu
NP_ADJSD = 4 * _P_FF + N_BUCKETS + 1   # beta_bar, B, A, C, sigma2, omega[1:], nu
NP_SS    = 3 * _P_FF + N_BUCKETS       # H, Q, B, omega[1:], bar_beta
NP_MSMSD = 3 * _P_TILDE + _P_FULL + (N_BUCKETS - 1) + 6  # beta_bar, B, A, sigma2, sigma_0, omega[1:], C, nu, m0, gamma_K, b

def load_params(path):
    with open(path) as f: raw = json.load(f)
    return {
        "beta_bar": jnp.array(raw["beta_bar"]),
        "B": jnp.array(raw["B"]),
        "A": jnp.array(raw["A"]),
        "d": jnp.array(raw["d"]),
        "sigma2": jnp.array(raw["sigma2"]),
        "omega": jnp.array(raw["omega"]),
        "C": jnp.array(raw["C"]),
        "nu": jnp.array(raw["nu"]),
    }

def make_Z_fixed():
    mon_g, mat_g = jnp.meshgrid(MONEYNESS, MATURITY, indexing="ij")
    _, midx_g = jnp.meshgrid(jnp.arange(len(MONEYNESS)), jnp.arange(len(MATURITY)), indexing="ij")
    N = mon_g.size
    return jnp.stack([jnp.ones(N), mon_g.ravel(), mat_g.ravel(), midx_g.ravel().astype(float)], axis=1)

def _ols(y_train, M):
    T, N = y_train.shape
    X = jnp.broadcast_to(M[None], (T, N, M.shape[1])).reshape(-1, M.shape[1])
    return jnp.linalg.lstsq(X, y_train.reshape(-1), rcond=None)[0]

def cold_ffSD(y_train, Z_fixed):
    M = Z_fixed[:, :3]
    beta3 = _ols(y_train, M)
    resid = y_train - (M @ beta3)
    return {
        "beta_bar": jnp.append(beta3, 0.0),
        "A": jnp.diag(jnp.full(_P_FF, 0.1)),
        "sigma2": jnp.var(resid),
        "omega_load": jnp.concatenate([jnp.zeros(1), jnp.full(N_BUCKETS - 1, 1e-2)]),
        "eta": jnp.full(_P_FF, 0.06),
        "alpha": jnp.full(_P_FF, 1.5),
        "C": jnp.diag(jnp.full(_P_FF, 1e-3)),
        "nu": jnp.array(10.0),
    }

def cold_fSD(y_train, Z_fixed):
    M = Z_fixed[:, :3]
    beta_bar = _ols(y_train, M)
    resid = y_train - (M @ beta_bar)
    return {
        "beta_bar":   beta_bar,
        "B":          jnp.diag(jnp.full(_P_TILDE, 0.95)),
        "A":          jnp.diag(jnp.full(_P_FULL, 0.1)),
        "sigma2":     jnp.var(resid),
        "sigma_0":    jnp.array(0.0),
        "omega_load": jnp.concatenate([jnp.zeros(1), jnp.full(N_BUCKETS - 1, 1e-2)]),
        "eta":        jnp.array(0.06),
        "alpha":      jnp.array(1.5),
        "C":          jnp.diag(jnp.full(_P_FULL, 1e-3)),
        "nu":         jnp.array(10.0),
    }

def cold_lmSD(y_train, Z_fixed):
    M = Z_fixed[:, :3]
    beta3 = _ols(y_train, M)
    resid = y_train - (M @ beta3)
    return {
        "beta_bar": jnp.append(beta3, 0.0),
        "B":        jnp.diag(jnp.full(_P_FF, 0.95)),
        "A":        jnp.diag(jnp.full(_P_FF, 0.05)),
        "d":        jnp.full(_P_FF, 0.3),
        "sigma2":   jnp.var(resid),
        "omega":    jnp.concatenate([jnp.zeros(1), jnp.full(N_BUCKETS - 1, 1e-2)]),
        "C":        jnp.full(_P_FF, 1e-3),
        "nu":       jnp.array(10.0),
    }

def warm_lmSD(y_train, Z_fixed, params_true):
    """Warm start from the true DGP params; only beta_bar and sigma2 are data-driven."""
    M = Z_fixed[:, :3]
    beta3 = _ols(y_train, M)
    resid = y_train - (M @ beta3)
    return {
        "beta_bar": jnp.append(beta3, 0.0),
        "B":        params_true["B"],
        "A":        params_true["A"],
        "d":        params_true["d"],
        "sigma2":   jnp.var(resid),
        "omega":    params_true["omega"][:N_BUCKETS],
        "C":        params_true["C"],
        "nu":       params_true["nu"],
    }

def cold_msmSD(y_train, Z_fixed):
    M = Z_fixed[:, :3]
    beta_bar = _ols(y_train, M)
    resid = y_train - (M @ beta_bar)
    return {
        "beta_bar":   beta_bar,
        "B":          jnp.diag(jnp.full(_P_TILDE, 0.95)),
        "A":          jnp.diag(jnp.full(_P_TILDE, 0.05)),
        "sigma2":     jnp.var(resid),
        "sigma_0":    jnp.array(0.1),
        "omega_load": jnp.concatenate([jnp.zeros(1), jnp.full(N_BUCKETS - 1, 1e-2)]),
        "C":          jnp.diag(jnp.full(_P_FULL, 1e-3)),
        "nu":         jnp.array(10.0),
        "m0":         jnp.array(1.5),
        "gamma_K":    jnp.array(0.5),
        "b":          jnp.array(2.0),
    }

def cold_adjSD(y_train, Z_fixed):
    M = Z_fixed[:, :3]
    beta3 = _ols(y_train, M)
    resid = y_train - (M @ beta3)
    return {
        "beta_bar": jnp.append(beta3, 0.0),
        "B":        jnp.diag(jnp.full(_P_FF, 0.95)),
        "A":        jnp.diag(jnp.full(_P_FF, 0.05)),
        "sigma2":   jnp.var(resid),
        "omega":    jnp.concatenate([jnp.zeros(1), jnp.full(N_BUCKETS - 1, 1e-2)]),
        "C":        jnp.full(_P_FF, 1e-3),
        "nu":       jnp.array(10.0),
    }

def cold_ss(y_train, Z_fixed):
    M = Z_fixed[:, :3]
    beta3 = _ols(y_train, M)
    resid = y_train - (M @ beta3)
    return {
        "Q_param":  jnp.diag(jnp.full(_P_FF, 1e-3)),
        "H_param":  jnp.var(resid) * jnp.eye(1),
        "B":        jnp.diag(jnp.full(_P_FF, 0.95)),
        "bar_beta": jnp.append(beta3, 0.0),
        "omega":    jnp.concatenate([jnp.zeros(1), jnp.full(N_BUCKETS - 1, 1e-2)]),
    }

def cold_ss_init(y_train, Z_fixed):
    M = Z_fixed[:, :3]
    beta3 = _ols(y_train, M)
    resid = y_train - (M @ beta3)
    sigma2 = jnp.var(resid)
    a1   = jnp.append(beta3, 0.0)
    P1   = 10.0 * jnp.eye(_P_FF)
    Z0   = jnp.zeros((_P_FF, _P_FF))
    T0   = 0.95 * jnp.eye(_P_FF)
    H0   = sigma2 * jnp.eye(_P_FF)
    R0   = jnp.eye(_P_FF)
    Q0   = jnp.diag(jnp.full(_P_FF, 1e-3))
    idx0 = jnp.asarray(0, dtype=jnp.int32)
    return (a1, P1, Z0, T0, H0, R0, Q0, idx0)

_sim_jit = jax.jit(simulate, static_argnames=("horizon", "score_buf_size"))

@functools.partial(jax.jit, static_argnames=("K",))
def _ff_fit(data, cov, ig, K):
    return ff_SD_fit(data, cov, ig, K=K,
                     opt_options={"learning_rate": 1e-3, "tol": 1e-4}, maxiter=MAXITER)

@functools.partial(jax.jit, static_argnames=("K",))
def _f_fit(data, cov, ig, K):
    return f_SD_fit(data, cov, ig, K=K, score_power=SCORE_POWER,
                    opt_options={"learning_rate": 1e-3, "tol": 1e-4}, maxiter=MAXITER)

@functools.partial(jax.jit, static_argnames=("K",))
def _msmsd_fit(data, cov, ig, K):
    return msmsd_fit(data, cov, ig, K=K, score_power=SCORE_POWER,
                     opt_options={"learning_rate": 1e-3, "tol": 1e-4}, maxiter=MAXITER)

_lmSD_fit = jax.jit(lambda data, cov, ig: lmSD_fit(
    data, cov, ig, opt_options={"learning_rate": 1e-3, "tol": 1e-4}, maxiter=MAXITER))

_lmSD_oracle = jax.jit(lambda data, cov, ig: lmSD_fit(
    data, cov, ig, opt_options={"learning_rate": 1e-3, "tol": 1e-4}, maxiter=0))

_adjSD_fit = jax.jit(lambda data, cov, ig: adjSD_fit(
    data, cov, ig, opt_options={"learning_rate": 1e-3, "tol": 1e-4}, maxiter=MAXITER))

_ss_fit = jax.jit(lambda data, cov, ig, init: ss_fit(
    data, cov, ig, init, opt_options={"learning_rate": 1e-3, "tol": 1e-4}, maxiter=MAXITER))

def _metrics(y_test, preds, oos_ll, n_params):
    mask = jnp.ones(y_test.shape, dtype=bool)
    mse = float(compute_mse(y_test, preds, mask))
    mae = float(compute_mae(y_test, preds, mask))
    tot_ll = float(jnp.sum(oos_ll))
    n_obs = int(y_test.size)
    aic = float(compute_aic(jnp.array(tot_ll), n_params))
    bic = float(compute_bic(jnp.array(tot_ll), n_params, n_obs))
    return mse, mae, tot_ll, aic, bic

def make_table(T, results):
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"$\sigma^2$ & Model & MSE & MAE & OOS LL & AIC & BIC \\",
        r"\midrule",
    ]
    for si, scale in enumerate(SIGMA2_SCALES):
        if si > 0: lines.append(r"\midrule")
        first_in_group = True
        for K in K_VALUES:
            for tag, label in [
                ("ffSD",  r"\texttt{ff-SD}"),
                ("fSD",   r"\texttt{f-SD}"),
                ("msmSD", r"\texttt{msm-SD}"),
            ]:
                mse, mae, ll, aic, bic = results[(T, scale, tag, K)]
                s = f"${SCALE_TEX[scale]}$" if first_in_group else ""
                first_in_group = False
                lines.append(
                    f"    {s} & {label} $K\\!=\\!{K}$"
                    f" & {mse:.3e} & {mae:.3e} & {ll:.1f} & {aic:.1f} & {bic:.1f} \\\\"
                )
        for tag, label in [
            ("lmSD",        r"\texttt{lmSD}$^{\star}$"),
            ("lmSD_oracle", r"\texttt{lmSD}$^{\dagger}$"),
            ("adjSD",       r"\texttt{adj-SD}"),
            ("SS",          r"\texttt{SS}"),
        ]:
            mse, mae, ll, aic, bic = results[(T, scale, tag, None)]
            s = f"${SCALE_TEX[scale]}$" if first_in_group else ""
            first_in_group = False
            lines.append(
                f"    {s} & {label}"
                f" & {mse:.3e} & {mae:.3e} & {ll:.1f} & {aic:.1f} & {bic:.1f} \\\\"
            )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        (
            rf"\caption{{One-step-ahead predictive performance on data simulated from the lmSD model,"
            rf" $T={T}$ (first $T/2$ for training, second $T/2$ for evaluation)."
            rf" $\star$: warm-started from true DGP parameters; $\dagger$: oracle (true params, no fitting).}}"
        ),
        rf"\label{{tab:sim_lmSD_T{T}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)

RESULTS_PATH = os.path.join(OUTPUT_DIR, "results.json")

def _results_key(T, scale, tag, K):
    return json.dumps([T, scale, tag, K], sort_keys=True)

def load_results():
    if not os.path.exists(RESULTS_PATH):
        return {}
    with open(RESULTS_PATH) as f:
        raw = json.load(f)
    results = {}
    for key_str, metrics in raw.items():
        T, scale, tag, K = json.loads(key_str)
        results[(T, scale, tag, K)] = tuple(metrics)
    return results

def save_results(results):
    serializable = {_results_key(T, scale, tag, K): list(v)
                    for (T, scale, tag, K), v in results.items()}
    with open(RESULTS_PATH, "w") as f:
        json.dump(serializable, f, indent=2)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    params_base = load_params(PARAMS_PATH)
    Z_fixed     = make_Z_fixed()
    sigma2_base = params_base["sigma2"]

    key = jax.random.PRNGKey(42)
    results = load_results()

    for T in HORIZONS:
        T_half = T // 2
        Z_cube = jnp.broadcast_to(Z_fixed[None], (T_half, Z_fixed.shape[0], Z_fixed.shape[1]))

        for scale in SIGMA2_SCALES:
            # Determine which models still need fitting for this (T, scale) combo
            needs_ff    = [K for K in K_VALUES if (T, scale, "ffSD",  K) not in results]
            needs_f     = [K for K in K_VALUES if (T, scale, "fSD",   K) not in results]
            needs_msmsd = [K for K in K_VALUES if (T, scale, "msmSD", K) not in results]
            needs_lmsd    = (T, scale, "lmSD",        None) not in results
            needs_oracle  = (T, scale, "lmSD_oracle", None) not in results
            needs_adjsd   = (T, scale, "adjSD",       None) not in results
            needs_ss      = (T, scale, "SS",           None) not in results

            if not (needs_ff or needs_f or needs_msmsd or needs_lmsd or
                    needs_oracle or needs_adjsd or needs_ss):
                print(f"T={T:5d}  scale={scale:.1f}  all models cached, skipping simulation.", flush=True)
                continue

            params = params_base | {"sigma2": sigma2_base * scale}
            key, subkey = jax.random.split(key)
            y_sim, _ = _sim_jit(params, Z_fixed, horizon=T, key=subkey, score_buf_size=T)
            jax.effects_barrier()

            y_train = y_sim[:T_half]
            y_test  = y_sim[T_half:]

            print(f"T={T:5d}  scale={scale:.1f}  simulated, fitting...", flush=True)

            for K in K_VALUES:
                if K in needs_ff:
                    ig_ff = cold_ffSD(y_train, Z_fixed)
                    r_ff  = _ff_fit(y_train, Z_cube, ig_ff, K)
                    preds_ff, _, _, oos_ll_ff = ff_SD_forecast(r_ff, Z_cube, y_test, K, ALPHA)
                    jax.effects_barrier()
                    results[(T, scale, "ffSD", K)] = _metrics(y_test, preds_ff, oos_ll_ff, NP_FF)
                    mse, mae, ll, *_ = results[(T, scale, "ffSD", K)]
                    print(f"  ff-SD K={K}  MSE={mse:.3e}  MAE={mae:.3e}  LL={ll:.1f}", flush=True)
                else:
                    mse, mae, ll, *_ = results[(T, scale, "ffSD", K)]
                    print(f"  ff-SD K={K}  MSE={mse:.3e}  MAE={mae:.3e}  LL={ll:.1f}  (cached)", flush=True)

                if K in needs_f:
                    ig_f = cold_fSD(y_train, Z_fixed)
                    r_f  = _f_fit(y_train, Z_cube, ig_f, K)
                    preds_f, _, _, oos_ll_f = f_SD_forecast(r_f, Z_cube, y_test, K, SCORE_POWER, ALPHA)
                    jax.effects_barrier()
                    results[(T, scale, "fSD", K)] = _metrics(y_test, preds_f, oos_ll_f, NP_F)
                    mse, mae, ll, *_ = results[(T, scale, "fSD", K)]
                    print(f"  f-SD  K={K}  MSE={mse:.3e}  MAE={mae:.3e}  LL={ll:.1f}", flush=True)
                else:
                    mse, mae, ll, *_ = results[(T, scale, "fSD", K)]
                    print(f"  f-SD  K={K}  MSE={mse:.3e}  MAE={mae:.3e}  LL={ll:.1f}  (cached)", flush=True)

                if K in needs_msmsd:
                    ig_msmsd = cold_msmSD(y_train, Z_fixed)
                    r_msmsd  = _msmsd_fit(y_train, Z_cube, ig_msmsd, K)
                    preds_msmsd, _, _, oos_ll_msmsd = msmsd_forecast(r_msmsd, Z_cube, y_test, K, SCORE_POWER, ALPHA)
                    jax.effects_barrier()
                    results[(T, scale, "msmSD", K)] = _metrics(y_test, preds_msmsd, oos_ll_msmsd, NP_MSMSD)
                    mse, mae, ll, *_ = results[(T, scale, "msmSD", K)]
                    print(f"  msm-SD K={K}  MSE={mse:.3e}  MAE={mae:.3e}  LL={ll:.1f}", flush=True)
                else:
                    mse, mae, ll, *_ = results[(T, scale, "msmSD", K)]
                    print(f"  msm-SD K={K}  MSE={mse:.3e}  MAE={mae:.3e}  LL={ll:.1f}  (cached)", flush=True)

            if needs_lmsd:
                ig_lmsd = warm_lmSD(y_train, Z_fixed, params_base)
                r_lmsd  = _lmSD_fit(y_train, Z_cube, ig_lmsd)
                preds_lmsd, _, _, oos_ll_lmsd = lmSD_forecast(r_lmsd, Z_cube, y_test, ALPHA)
                jax.effects_barrier()
                results[(T, scale, "lmSD", None)] = _metrics(y_test, preds_lmsd, oos_ll_lmsd, NP_LMSD)
                mse, mae, ll, *_ = results[(T, scale, "lmSD", None)]
                print(f"  lmSD     MSE={mse:.3e}  MAE={mae:.3e}  LL={ll:.1f}", flush=True)
            else:
                mse, mae, ll, *_ = results[(T, scale, "lmSD", None)]
                print(f"  lmSD     MSE={mse:.3e}  MAE={mae:.3e}  LL={ll:.1f}  (cached)", flush=True)

            if needs_oracle:
                ig_oracle = warm_lmSD(y_train, Z_fixed, params_base)
                r_oracle  = _lmSD_oracle(y_train, Z_cube, ig_oracle)
                preds_oracle, _, _, oos_ll_oracle = lmSD_forecast(r_oracle, Z_cube, y_test, ALPHA)
                jax.effects_barrier()
                results[(T, scale, "lmSD_oracle", None)] = _metrics(y_test, preds_oracle, oos_ll_oracle, NP_LMSD)
                mse, mae, ll, *_ = results[(T, scale, "lmSD_oracle", None)]
                print(f"  lmSD†    MSE={mse:.3e}  MAE={mae:.3e}  LL={ll:.1f}", flush=True)
            else:
                mse, mae, ll, *_ = results[(T, scale, "lmSD_oracle", None)]
                print(f"  lmSD†    MSE={mse:.3e}  MAE={mae:.3e}  LL={ll:.1f}  (cached)", flush=True)

            if needs_adjsd:
                ig_adjsd = cold_adjSD(y_train, Z_fixed)
                r_adjsd  = _adjSD_fit(y_train, Z_cube, ig_adjsd)
                preds_adjsd, _, _, oos_ll_adjsd = adjSD_forecast(r_adjsd, Z_cube, y_test, ALPHA)
                jax.effects_barrier()
                results[(T, scale, "adjSD", None)] = _metrics(y_test, preds_adjsd, oos_ll_adjsd, NP_ADJSD)
                mse, mae, ll, *_ = results[(T, scale, "adjSD", None)]
                print(f"  adjSD    MSE={mse:.3e}  MAE={mae:.3e}  LL={ll:.1f}", flush=True)
            else:
                mse, mae, ll, *_ = results[(T, scale, "adjSD", None)]
                print(f"  adjSD    MSE={mse:.3e}  MAE={mae:.3e}  LL={ll:.1f}  (cached)", flush=True)

            if needs_ss:
                ig_ss   = cold_ss(y_train, Z_fixed)
                init_ss = cold_ss_init(y_train, Z_fixed)
                r_ss    = _ss_fit(y_train, Z_cube, ig_ss, init_ss)
                preds_ss, _, _, oos_ll_ss = ss_forecast(r_ss, Z_cube, y_test, ALPHA)
                jax.effects_barrier()
                results[(T, scale, "SS", None)] = _metrics(y_test, preds_ss, oos_ll_ss, NP_SS)
                mse, mae, ll, *_ = results[(T, scale, "SS", None)]
                print(f"  SS       MSE={mse:.3e}  MAE={mae:.3e}  LL={ll:.1f}", flush=True)
            else:
                mse, mae, ll, *_ = results[(T, scale, "SS", None)]
                print(f"  SS       MSE={mse:.3e}  MAE={mae:.3e}  LL={ll:.1f}  (cached)", flush=True)

            save_results(results)

    tex = "\n\n".join(make_table(T, results) for T in HORIZONS)
    out_path = os.path.join(OUTPUT_DIR, "tables.tex")
    with open(out_path, "w") as f:
        f.write(tex)
    print(f"\nLaTeX tables saved to {out_path}")

if __name__ == "__main__": main()