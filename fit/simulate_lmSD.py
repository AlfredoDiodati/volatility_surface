import os
import shutil
import functools

if shutil.which("nvidia-smi") is not None:
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import json
import jax
import jax.numpy as jnp

from models.lmSD import simulate
from models.ff_SD import fit as ff_SD_fit, forecast as ff_SD_forecast
from models.f_SD  import fit as f_SD_fit,  forecast as f_SD_forecast
from fit._forecast_metrics import compute_mse, compute_mae, compute_aic, compute_bic

print(f"Running on: {jax.devices()}")

PARAMS_PATH = "out/SPX/otm/params_lmSD.json"
OUTPUT_DIR  = "out/SPX/mc/simulate_lmSD"
SCORE_POWER = 1.0
ALPHA = 0.05
MAXITER = 2000
K_VALUES = [3, 10]
N_BUCKETS = 4

MONEYNESS = jnp.array([0.9, 0.98, 1.05, 1.15, 1.3, 1.5])
MATURITY  = jnp.array([10, 50, 100, 180]) / 255.0

HORIZONS = [400, 2000, 7000]
SIGMA2_SCALES = [1.0, 10.0, 0.1]
SCALE_TEX = {1.0: r"\sigma^2", 10.0: r"10\,\sigma^2", 0.1: r"\sigma^2/10"}

_P_FF = 4
_P_TILDE = 3
_P_FULL = 4

NP_FF = _P_FF + _P_FF + 1 + (N_BUCKETS - 1) + _P_FF + 1 + _P_FF + 1
NP_F = _P_TILDE + _P_TILDE + _P_TILDE + 1 + 1 + (N_BUCKETS - 1) + 1 + 1 + _P_FULL + 1

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
        "A":          jnp.diag(jnp.full(_P_TILDE, 0.1)),
        "sigma2":     jnp.var(resid),
        "sigma_0":    jnp.array(0.0),
        "omega_load": jnp.concatenate([jnp.zeros(1), jnp.full(N_BUCKETS - 1, 1e-2)]),
        "eta":        jnp.array(0.06),
        "alpha":      jnp.array(1.5),
        "C":          jnp.diag(jnp.full(_P_FULL, 1e-3)),
        "nu":         jnp.array(10.0),
    }

_sim_jit = jax.jit(simulate, static_argnames=("horizon", "score_buf_size"))

@functools.partial(jax.jit, static_argnames=("K",))
def _ff_fit(data, cov, ig, K):
    return ff_SD_fit(data, cov, ig, K=K, score_power=SCORE_POWER,
                     opt_options={"learning_rate": 1e-3, "tol": 1e-4}, maxiter=MAXITER)

@functools.partial(jax.jit, static_argnames=("K",))
def _f_fit(data, cov, ig, K):
    return f_SD_fit(data, cov, ig, K=K, score_power=SCORE_POWER,
                    opt_options={"learning_rate": 1e-3, "tol": 1e-4}, maxiter=MAXITER)

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
            for tag, label in [("ffSD", r"\texttt{ff-SD}"), ("fSD", r"\texttt{f-SD}")]:
                mse, mae, ll, aic, bic = results[(T, scale, tag, K)]
                s = f"${SCALE_TEX[scale]}$" if first_in_group else ""
                first_in_group = False
                lines.append(
                    f"    {s} & {label} $K\\!=\\!{K}$"
                    f" & {mse:.3e} & {mae:.3e} & {ll:.1f} & {aic:.1f} & {bic:.1f} \\\\"
                )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        (
            rf"\caption{{One-step-ahead predictive performance on data simulated from the lmSD model,"
            rf" $T={T}$ (first $T/2$ for training, second $T/2$ for evaluation).}}"
        ),
        rf"\label{{tab:sim_lmSD_T{T}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    params_base = load_params(PARAMS_PATH)
    Z_fixed     = make_Z_fixed()
    sigma2_base = params_base["sigma2"]

    key = jax.random.PRNGKey(42)
    results = {}

    for T in HORIZONS:
        T_half = T // 2
        Z_cube = jnp.broadcast_to(Z_fixed[None], (T_half, Z_fixed.shape[0], Z_fixed.shape[1]))

        for scale in SIGMA2_SCALES:
            params = params_base | {"sigma2": sigma2_base * scale}
            key, subkey = jax.random.split(key)
            y_sim, _ = _sim_jit(params, Z_fixed, horizon=T, key=subkey, score_buf_size=T)
            jax.effects_barrier()

            y_train = y_sim[:T_half]
            y_test  = y_sim[T_half:]

            print(f"T={T:5d}  scale={scale:.1f}  simulated, fitting...", flush=True)

            for K in K_VALUES:
                ig_ff = cold_ffSD(y_train, Z_fixed)
                r_ff  = _ff_fit(y_train, Z_cube, ig_ff, K)
                preds_ff, _, _, oos_ll_ff = ff_SD_forecast(r_ff, Z_cube, y_test, K, SCORE_POWER, ALPHA)
                jax.effects_barrier()
                results[(T, scale, "ffSD", K)] = _metrics(y_test, preds_ff, oos_ll_ff, NP_FF)
                mse, mae, ll, *_ = results[(T, scale, "ffSD", K)]
                print(f"  ff-SD K={K}  MSE={mse:.3e}  MAE={mae:.3e}  LL={ll:.1f}", flush=True)

                ig_f = cold_fSD(y_train, Z_fixed)
                r_f  = _f_fit(y_train, Z_cube, ig_f, K)
                preds_f, _, _, oos_ll_f = f_SD_forecast(r_f, Z_cube, y_test, K, SCORE_POWER, ALPHA)
                jax.effects_barrier()
                results[(T, scale, "fSD", K)] = _metrics(y_test, preds_f, oos_ll_f, NP_F)
                mse, mae, ll, *_ = results[(T, scale, "fSD", K)]
                print(f"  f-SD  K={K}  MSE={mse:.3e}  MAE={mae:.3e}  LL={ll:.1f}", flush=True)

    tex = "\n\n".join(make_table(T, results) for T in HORIZONS)
    out_path = os.path.join(OUTPUT_DIR, "tables.tex")
    with open(out_path, "w") as f:
        f.write(tex)
    print(f"\nLaTeX tables saved to {out_path}")

if __name__ == "__main__": main()