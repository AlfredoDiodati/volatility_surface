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
from models.lmSD import fit as lmSD_fit, forecast as lmSD_forecast
from models.adjSD import fit as adjSD_fit, forecast as adjSD_forecast
from models.ss import fit_collapsed as ss_fit, forecast as ss_forecast
from models.ff_SD import fit as ff_SD_fit, forecast as ff_SD_forecast
from models.f_SD import fit as f_SD_fit, forecast as f_SD_forecast
from models.MSMSD import fit as msmsd_fit, forecast as msmsd_forecast
from fit._forecast_metrics import compute_mse, compute_mae, compute_aic, compute_bic

print(f"Running on: {jax.devices()}")


PARAMS_PATH = "out/SPX/otm/params_lmSD.json"
OUTPUT_DIR = "out/SPX/mc/simulate_lmSD"
SCORE_POWER = 1.0
ALPHA = 0.05
MAXITER = 10_000
K_VALUES = [1, 2, 3, 5, 10, 20, 30, 50]
K_VALUES_MSMSD = [1, 2, 3, 5, 10]
N_BUCKETS = 4

MONEYNESS = jnp.array([0.9, 0.98, 1.05, 1.15, 1.3, 1.5])
MATURITY = jnp.array([10, 50, 100, 180]) / 255.0

HORIZONS = [400, 2000]
SIGMA2_SCALES = [1.0, 10.0, 0.1]
SCALE_TEX = {1.0: r"\sigma^2", 10.0: r"10\,\sigma^2", 0.1: r"\sigma^2/10"}

LR = {
    "ffSD":  1e-4,
    "fSD":   1e-4,
    "msmSD": 1e-4,
    "lmSD":  1e-4,
    "adjSD": 1e-4,
    "SS":    1e-4,
}

_P_FF = 4
_P_TILDE = 3
_P_FULL = 4

NP_FF = _P_FF + _P_FF + 1 + (N_BUCKETS - 1) + _P_FF + 1 + _P_FF + 1
NP_F = _P_TILDE + _P_TILDE + _P_FULL + 1 + 1 + (N_BUCKETS - 1) + 1 + 1 + _P_FULL + 1
NP_LMSD = 5 * _P_FF + N_BUCKETS + 1
NP_ADJSD = 4 * _P_FF + N_BUCKETS + 1
NP_SS = 3 * _P_FF + N_BUCKETS
NP_MSMSD = 3 * _P_TILDE + _P_FULL + (N_BUCKETS - 1) + 6


def load_params(path):
    with open(path) as f:
        raw = json.load(f)
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
        "beta_bar": beta_bar,
        "B": jnp.diag(jnp.full(_P_TILDE, 0.95)),
        "A": jnp.diag(jnp.full(_P_FULL, 0.1)),
        "sigma2": jnp.var(resid),
        "sigma_0": jnp.array(0.0),
        "omega_load": jnp.concatenate([jnp.zeros(1), jnp.full(N_BUCKETS - 1, 1e-2)]),
        "eta": jnp.array(0.06),
        "alpha": jnp.array(1.5),
        "C": jnp.diag(jnp.full(_P_FULL, 1e-3)),
        "nu": jnp.array(10.0),
    }

def cold_lmSD(y_train, Z_fixed):
    M = Z_fixed[:, :3]
    beta3 = _ols(y_train, M)
    resid = y_train - (M @ beta3)
    return {
        "beta_bar": jnp.append(beta3, 0.0),
        "B": jnp.diag(jnp.full(_P_FF, 0.95)),
        "A": jnp.diag(jnp.full(_P_FF, 0.05)),
        "d": jnp.full(_P_FF, 0.3),
        "sigma2": jnp.var(resid),
        "omega": jnp.concatenate([jnp.zeros(1), jnp.full(N_BUCKETS - 1, 1e-2)]),
        "C": jnp.full(_P_FF, 1e-3),
        "nu": jnp.array(10.0),
    }

def true_lmSD(params_true):
    return {
        "beta_bar": params_true["beta_bar"],
        "B": params_true["B"],
        "A": params_true["A"],
        "d": params_true["d"],
        "sigma2": params_true["sigma2"],
        "omega": params_true["omega"][:N_BUCKETS],
        "C": params_true["C"],
        "nu": params_true["nu"],
    }

def warm_lmSD(y_train, Z_fixed, params_true):
    """Warm start from the true DGP params; only beta_bar and sigma2 are data-driven."""
    M = Z_fixed[:, :3]
    beta3 = _ols(y_train, M)
    resid = y_train - (M @ beta3)
    return {
        "beta_bar": jnp.append(beta3, 0.0),
        "B": params_true["B"],
        "A": params_true["A"],
        "d": params_true["d"],
        "sigma2": jnp.var(resid),
        "omega": params_true["omega"][:N_BUCKETS],
        "C": params_true["C"],
        "nu": params_true["nu"],
    }

def cold_msmSD(y_train, Z_fixed):
    M = Z_fixed[:, :3]
    beta_bar = _ols(y_train, M)
    resid = y_train - (M @ beta_bar)
    return {
        "beta_bar": beta_bar,
        "B": jnp.diag(jnp.full(_P_TILDE, 0.95)),
        "A": jnp.diag(jnp.full(_P_TILDE, 0.05)),
        "sigma2": jnp.var(resid),
        "sigma_0": jnp.array(0.1),
        "omega_load": jnp.concatenate([jnp.zeros(1), jnp.full(N_BUCKETS - 1, 1e-2)]),
        "C": jnp.diag(jnp.full(_P_FULL, 1e-3)),
        "nu": jnp.array(10.0),
        "m0": jnp.array(1.5),
        "gamma_K": jnp.array(0.5),
        "b": jnp.array(2.0),
    }

def cold_adjSD(y_train, Z_fixed):
    M = Z_fixed[:, :3]
    beta3 = _ols(y_train, M)
    resid = y_train - (M @ beta3)
    return {
        "beta_bar": jnp.append(beta3, 0.0),
        "B": jnp.diag(jnp.full(_P_FF, 0.95)),
        "A": jnp.diag(jnp.full(_P_FF, 0.05)),
        "sigma2": jnp.var(resid),
        "omega": jnp.concatenate([jnp.zeros(1), jnp.full(N_BUCKETS - 1, 1e-2)]),
        "C": jnp.full(_P_FF, 1e-3),
        "nu": jnp.array(10.0),
    }

def cold_ss(y_train, Z_fixed):
    M = Z_fixed[:, :3]
    beta3 = _ols(y_train, M)
    resid = y_train - (M @ beta3)
    return {
        "Q_param": jnp.diag(jnp.full(_P_FF, 1e-3)),
        "H_param": jnp.var(resid) * jnp.eye(1),
        "B": jnp.diag(jnp.full(_P_FF, 0.95)),
        "bar_beta": jnp.append(beta3, 0.0),
        "omega": jnp.concatenate([jnp.zeros(1), jnp.full(N_BUCKETS - 1, 1e-2)]),
    }

def cold_ss_init(y_train, Z_fixed):
    M = Z_fixed[:, :3]
    beta3 = _ols(y_train, M)
    resid = y_train - (M @ beta3)
    sigma2 = jnp.var(resid)
    a1 = jnp.append(beta3, 0.0)
    P1 = 10.0 * jnp.eye(_P_FF)
    Z0 = jnp.zeros((_P_FF, _P_FF))
    T0 = 0.95 * jnp.eye(_P_FF)
    H0 = sigma2 * jnp.eye(_P_FF)
    R0 = jnp.eye(_P_FF)
    Q0 = jnp.diag(jnp.full(_P_FF, 1e-3))
    idx0 = jnp.asarray(0, dtype=jnp.int32)
    return (a1, P1, Z0, T0, H0, R0, Q0, idx0)

_sim_jit = jax.jit(simulate, static_argnames=("horizon", "score_buf_size"))

@functools.partial(jax.jit, static_argnames=("K",))
def _ff_fit(data, cov, ig, K, lr):
    return ff_SD_fit(data, cov, ig, K=K,
                     opt_options={"learning_rate": lr, "tol": 1e-4}, maxiter=MAXITER)

@functools.partial(jax.jit, static_argnames=("K",))
def _f_fit(data, cov, ig, K, lr):
    return f_SD_fit(data, cov, ig, K=K, score_power=SCORE_POWER,
                    opt_options={"learning_rate": lr, "tol": 1e-4}, maxiter=MAXITER)

@functools.partial(jax.jit, static_argnames=("K",))
def _msmsd_fit(data, cov, ig, K, lr):
    return msmsd_fit(data, cov, ig, K=K, score_power=SCORE_POWER,
                     opt_options={"learning_rate": lr, "tol": 1e-4}, maxiter=MAXITER)

_lmSD_fit = jax.jit(lambda data, cov, ig, lr: lmSD_fit(
    data, cov, ig, opt_options={"learning_rate": lr, "tol": 1e-4}, maxiter=MAXITER))

_lmSD_oracle = jax.jit(lambda data, cov, ig, lr: lmSD_fit(
    data, cov, ig, opt_options={"learning_rate": lr, "tol": 1e-4}, maxiter=0))

_adjSD_fit = jax.jit(lambda data, cov, ig, lr: adjSD_fit(
    data, cov, ig, opt_options={"learning_rate": lr, "tol": 1e-4}, maxiter=MAXITER))

_ss_fit = jax.jit(lambda data, cov, ig, init, lr: ss_fit(
    data, cov, ig, init, opt_options={"learning_rate": lr, "tol": 1e-4}, maxiter=MAXITER))

def _metrics(y_test, preds, oos_ll, n_params):
    mask = jnp.ones(y_test.shape, dtype=bool)
    mse = float(compute_mse(y_test, preds, mask))
    mae = float(compute_mae(y_test, preds, mask))
    tot_ll = float(jnp.sum(oos_ll))
    n_obs = int(y_test.size)
    aic = float(compute_aic(jnp.array(tot_ll), n_params))
    bic = float(compute_bic(jnp.array(tot_ll), n_params, n_obs))
    mse_seq = ((y_test - preds) ** 2).mean(axis=1).tolist()
    mae_seq = jnp.abs(y_test - preds).mean(axis=1).tolist()
    ll_arr = jnp.array(oos_ll)
    ll_seq = (ll_arr.sum(axis=1) if ll_arr.ndim > 1 else ll_arr).tolist()
    return mse, mae, tot_ll, aic, bic, mse_seq, mae_seq, ll_seq

_N_METRICS = len(("mse", "mae", "tot_ll", "aic", "bic", "mse_seq", "mae_seq", "ll_seq"))
_DEFAULT_LR = 1e-3

def _is_cached(results, key, lr):
    if key not in results or len(results[key]) < _N_METRICS:
        return False
    v = results[key]
    stored_lr = v[_N_METRICS] if len(v) > _N_METRICS else _DEFAULT_LR
    return abs(stored_lr - lr) < 1e-12

def make_table(T, results):
    import math

    all_mse, all_mae = [], []
    all_keys = (
        [(t, k) for t in ("ffSD", "fSD") for k in K_VALUES]
        + [("msmSD", k) for k in K_VALUES_MSMSD]
        + [(t, None) for t in ("lmSD", "adjSD", "SS")]
    )
    for scale in SIGMA2_SCALES:
        for tag, K in all_keys:
            key = (T, scale, tag, K)
            if key in results:
                mse, mae, *_ = results[key]
                all_mse.append(mse)
                all_mae.append(mae)

    def _exp3(vals):
        """Nearest multiple-of-3 exponent so that scaled values are roughly O(1)."""
        if not vals: return 0
        mag = sum(abs(v) for v in vals) / len(vals)
        if mag == 0: return 0
        return 3 * round(math.floor(math.log10(mag)) / 3)

    mse_exp = _exp3(all_mse)
    mae_exp = _exp3(all_mae)
    mse_mult = 10.0 ** (-mse_exp)
    mae_mult = 10.0 ** (-mae_exp)

    def _col_hdr(name, exp): return name if exp == 0 else rf"{name} ($\times 10^{{{exp}}}$)"

    def _fmt(v):
        """Format a scaled value without scientific notation."""
        if v is None: return "--"
        av = abs(v)
        if av >= 10: return f"{v:.1f}"
        elif av >= 1: return f"{v:.2f}"
        else: return f"{v:.3f}"

    def _bench_seqs(scale):
        key = (T, scale, "lmSD", None)
        if key not in results or len(results[key]) < 8: return None, None, None
        r = results[key]
        return r[5], r[6], r[7]

    def _dm_stars(seq_m, seq_b, higher_better=False):
        if seq_m is None or seq_b is None: return ""
        n = min(len(seq_m), len(seq_b))
        if n < 2: return ""
        if higher_better: d = [seq_b[i] - seq_m[i] for i in range(n)]
        else: d = [seq_m[i] - seq_b[i] for i in range(n)]
        mean_d = sum(d) / n
        var_d = sum((x - mean_d) ** 2 for x in d) / (n - 1)
        if var_d <= 0: return ""
        stat = abs(mean_d) / (var_d ** 0.5 / n ** 0.5)
        if stat > 2.576: return "***"
        elif stat > 1.96: return "**"
        elif stat > 1.645: return "*"
        return ""

    def _get(scale, tag, K):
        key = (T, scale, tag, K)
        if key not in results:return None
        r = results[key]
        seqs = (r[5], r[6], r[7]) if len(r) >= 8 else (None, None, None)
        return r[0] * mse_mult, r[1] * mae_mult, r[2], seqs

    N_MET = 3

    def _cmidrule(i):
        lo = 3 + i * N_MET
        return rf"\cmidrule(lr){{{lo}-{lo + N_MET - 1}}}"

    mse_hdr = _col_hdr("MSE", mse_exp)
    mae_hdr = _col_hdr("MAE", mae_exp)
    col_spec = "ll" + "rrr" * len(SIGMA2_SCALES)

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    grp_hdrs = " & ".join(
        rf"\multicolumn{{{N_MET}}}{{c}}{{${SCALE_TEX[s]}$}}"
        for s in SIGMA2_SCALES
    )
    lines.append(rf"Model & $K$ & {grp_hdrs} \\")
    lines.append(" ".join(_cmidrule(i) for i in range(len(SIGMA2_SCALES))))

    metric_hdr = " & ".join(
        f"{mse_hdr} & {mae_hdr} & loglik" for _ in SIGMA2_SCALES
    )
    lines.append(rf"& & {metric_hdr} \\")
    lines.append(r"\midrule")

    def _row(model_cell, K_cell, tag, K, is_bench=False):
        cells = [model_cell, K_cell]
        for scale in SIGMA2_SCALES:
            got = _get(scale, tag, K)
            if got is None: cells += ["--", "--", "--"]
            else:
                mse_v, mae_v, ll_v, seqs = got
                if is_bench:
                    s_mse, s_mae, s_ll = "", "", ""
                else:
                    b_mse, b_mae, b_ll = _bench_seqs(scale)
                    s_mse = _dm_stars(seqs[0], b_mse)
                    s_mae = _dm_stars(seqs[1], b_mae)
                    s_ll = _dm_stars(seqs[2], b_ll, higher_better=True)
                cells += [_fmt(mse_v) + s_mse, _fmt(mae_v) + s_mae, f"{ll_v:.1f}" + s_ll]
        return "    " + " & ".join(cells) + r" \\"

    K_groups = [("ff-SD", "ffSD", K_VALUES), ("f-SD", "fSD", K_VALUES), ("msm-SD", "msmSD", K_VALUES_MSMSD)]
    for gi, (model_name, tag, k_list) in enumerate(K_groups):
        if gi > 0: lines.append(r"\addlinespace[3pt]")
        n = len(k_list)
        for ri, K in enumerate(k_list):
            model_cell = (
                rf"\multirow{{{n}}}{{*}}{{{model_name}}}" if ri == 0 else ""
            )
            lines.append(_row(model_cell, str(K), tag, K))

    lines.append(r"\midrule")
    fixed = [
        (r"lmSD$^{\star}$", "lmSD", None, True),
        ("adj-SD", "adjSD", None, False),
        ("SS", "SS", None, False),
    ]
    for model_name, tag, K, is_bench in fixed:
        lines.append(_row(model_name, "", tag, K, is_bench=is_bench))

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        (
            rf"\caption{{One-step-ahead predictive performance on data simulated"
            rf" from the lmSD model, $T={T}$"
            rf" (first $T/2$ for training, second $T/2$ for evaluation)."
            rf" $\star$: warm-started from true DGP parameters."
            rf" Stars denote significance of Diebold-Mariano test against lmSD$^{{\star}}$:"
            rf" $^{{*}}$10\%, $^{{**}}$5\%, $^{{***}}$1\%.}}"
        ),
        rf"\label{{tab:sim_lmSD_T{T}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)
RESULTS_PATH = os.path.join(OUTPUT_DIR, "results.json")

def _results_key(T, scale, tag, K):
    return json.dumps([T, scale, tag, K], sort_keys=True)

def load_results():
    if not os.path.exists(RESULTS_PATH): return {}
    with open(RESULTS_PATH) as f: raw = json.load(f)
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
    Z_fixed = make_Z_fixed()
    sigma2_base = params_base["sigma2"]

    key = jax.random.PRNGKey(42)
    results = load_results()

    for T in HORIZONS:
        T_half = T // 2
        Z_cube = jnp.broadcast_to(Z_fixed[None], (T_half, Z_fixed.shape[0], Z_fixed.shape[1]))

        for scale in SIGMA2_SCALES:
            needs_ff = [K for K in K_VALUES if not _is_cached(results, (T, scale, "ffSD", K), LR["ffSD"])]
            needs_f = [K for K in K_VALUES if not _is_cached(results, (T, scale, "fSD", K), LR["fSD"])]
            needs_msmsd = [K for K in K_VALUES_MSMSD if not _is_cached(results, (T, scale, "msmSD", K), LR["msmSD"])]
            needs_lmsd = not _is_cached(results, (T, scale, "lmSD", None), LR["lmSD"])
            needs_oracle = not _is_cached(results, (T, scale, "lmSD_oracle", None), LR["lmSD"])
            needs_adjsd = not _is_cached(results, (T, scale, "adjSD", None), LR["adjSD"])
            needs_ss = not _is_cached(results, (T, scale, "SS", None), LR["SS"])

            if not (needs_ff or needs_f or needs_msmsd or needs_lmsd or
                    needs_oracle or needs_adjsd or needs_ss):
                print(f"T={T:5d}  scale={scale:.1f}  all models cached, skipping simulation.", flush=True)
                continue

            params = params_base | {"sigma2": sigma2_base * scale}
            key, subkey = jax.random.split(key)
            y_sim, _ = _sim_jit(params, Z_fixed, horizon=T, key=subkey, score_buf_size=T)
            jax.effects_barrier()

            y_train = y_sim[:T_half]
            y_test = y_sim[T_half:]

            print(f"T={T:5d}  scale={scale:.1f}  simulated, fitting...", flush=True)

            for K in K_VALUES:
                if K in needs_ff:
                    ig_ff = cold_ffSD(y_train, Z_fixed)
                    r_ff = _ff_fit(y_train, Z_cube, ig_ff, K, LR["ffSD"])
                    preds_ff, _, _, oos_ll_ff = ff_SD_forecast(r_ff, Z_cube, y_test, K, ALPHA)
                    jax.effects_barrier()
                    results[(T, scale, "ffSD", K)] = _metrics(y_test, preds_ff, oos_ll_ff, NP_FF) + (LR["ffSD"], int(r_ff["niter"]), bool(r_ff["is_converged"]), MAXITER)
                    mse, mae, ll, *_ = results[(T, scale, "ffSD", K)]
                    print(f"  ff-SD K={K}  MSE={mse:.3e}  MAE={mae:.3e}  LL={ll:.1f}  iter={int(r_ff['niter'])}  conv={bool(r_ff['is_converged'])}", flush=True)
                    save_results(results)
                else:
                    mse, mae, ll, *_ = results[(T, scale, "ffSD", K)]
                    print(f"  ff-SD K={K}  MSE={mse:.3e}  MAE={mae:.3e}  LL={ll:.1f}  (cached)", flush=True)

                if K in needs_f:
                    ig_f = cold_fSD(y_train, Z_fixed)
                    r_f = _f_fit(y_train, Z_cube, ig_f, K, LR["fSD"])
                    preds_f, _, _, oos_ll_f = f_SD_forecast(r_f, Z_cube, y_test, K, SCORE_POWER, ALPHA)
                    jax.effects_barrier()
                    results[(T, scale, "fSD", K)] = _metrics(y_test, preds_f, oos_ll_f, NP_F) + (LR["fSD"], int(r_f["niter"]), bool(r_f["is_converged"]), MAXITER)
                    mse, mae, ll, *_ = results[(T, scale, "fSD", K)]
                    print(f"  f-SD  K={K}  MSE={mse:.3e}  MAE={mae:.3e}  LL={ll:.1f}  iter={int(r_f['niter'])}  conv={bool(r_f['is_converged'])}", flush=True)
                    save_results(results)
                else:
                    mse, mae, ll, *_ = results[(T, scale, "fSD", K)]
                    print(f"  f-SD  K={K}  MSE={mse:.3e}  MAE={mae:.3e}  LL={ll:.1f}  (cached)", flush=True)

            for K in K_VALUES_MSMSD:
                if K in needs_msmsd:
                    ig_msmsd = cold_msmSD(y_train, Z_fixed)
                    r_msmsd = _msmsd_fit(y_train, Z_cube, ig_msmsd, K, LR["msmSD"])
                    preds_msmsd, _, _, oos_ll_msmsd = msmsd_forecast(r_msmsd, Z_cube, y_test, K, SCORE_POWER, ALPHA)
                    jax.effects_barrier()
                    results[(T, scale, "msmSD", K)] = _metrics(y_test, preds_msmsd, oos_ll_msmsd, NP_MSMSD) + (LR["msmSD"], int(r_msmsd["niter"]), bool(r_msmsd["is_converged"]), MAXITER)
                    mse, mae, ll, *_ = results[(T, scale, "msmSD", K)]
                    print(f"  msm-SD K={K}  MSE={mse:.3e}  MAE={mae:.3e}  LL={ll:.1f}  iter={int(r_msmsd['niter'])}  conv={bool(r_msmsd['is_converged'])}", flush=True)
                    save_results(results)
                else:
                    mse, mae, ll, *_ = results[(T, scale, "msmSD", K)]
                    print(f"  msm-SD K={K}  MSE={mse:.3e}  MAE={mae:.3e}  LL={ll:.1f}  (cached)", flush=True)

            if needs_lmsd:
                ig_lmsd = warm_lmSD(y_train, Z_fixed, params_base)
                r_lmsd = _lmSD_fit(y_train, Z_cube, ig_lmsd, LR["lmSD"])
                preds_lmsd, _, _, oos_ll_lmsd = lmSD_forecast(r_lmsd, Z_cube, y_test, ALPHA)
                jax.effects_barrier()
                results[(T, scale, "lmSD", None)] = _metrics(y_test, preds_lmsd, oos_ll_lmsd, NP_LMSD) + (LR["lmSD"], int(r_lmsd["niter"]), bool(r_lmsd["is_converged"]), MAXITER)
                mse, mae, ll, *_ = results[(T, scale, "lmSD", None)]
                print(f"  lmSD     MSE={mse:.3e}  MAE={mae:.3e}  LL={ll:.1f}  iter={int(r_lmsd['niter'])}  conv={bool(r_lmsd['is_converged'])}", flush=True)
                save_results(results)
            else:
                mse, mae, ll, *_ = results[(T, scale, "lmSD", None)]
                print(f"  lmSD     MSE={mse:.3e}  MAE={mae:.3e}  LL={ll:.1f}  (cached)", flush=True)

            if needs_oracle:
                ig_oracle = true_lmSD(params)
                r_oracle = _lmSD_oracle(y_train, Z_cube, ig_oracle, LR["lmSD"])
                preds_oracle, _, _, oos_ll_oracle = lmSD_forecast(r_oracle, Z_cube, y_test, ALPHA)
                jax.effects_barrier()
                results[(T, scale, "lmSD_oracle", None)] = _metrics(y_test, preds_oracle, oos_ll_oracle, NP_LMSD) + (LR["lmSD"], int(r_oracle["niter"]), bool(r_oracle["is_converged"]), 0)
                mse, mae, ll, *_ = results[(T, scale, "lmSD_oracle", None)]
                print(f"  lmSD†    MSE={mse:.3e}  MAE={mae:.3e}  LL={ll:.1f}  iter={int(r_oracle['niter'])}  conv={bool(r_oracle['is_converged'])}", flush=True)
                save_results(results)
            else:
                mse, mae, ll, *_ = results[(T, scale, "lmSD_oracle", None)]
                print(f"  lmSD†    MSE={mse:.3e}  MAE={mae:.3e}  LL={ll:.1f}  (cached)", flush=True)

            if needs_adjsd:
                ig_adjsd = cold_adjSD(y_train, Z_fixed)
                r_adjsd = _adjSD_fit(y_train, Z_cube, ig_adjsd, LR["adjSD"])
                preds_adjsd, _, _, oos_ll_adjsd = adjSD_forecast(r_adjsd, Z_cube, y_test, ALPHA)
                jax.effects_barrier()
                results[(T, scale, "adjSD", None)] = _metrics(y_test, preds_adjsd, oos_ll_adjsd, NP_ADJSD) + (LR["adjSD"], int(r_adjsd["niter"]), bool(r_adjsd["is_converged"]), MAXITER)
                mse, mae, ll, *_ = results[(T, scale, "adjSD", None)]
                print(f"  adjSD    MSE={mse:.3e}  MAE={mae:.3e}  LL={ll:.1f}  iter={int(r_adjsd['niter'])}  conv={bool(r_adjsd['is_converged'])}", flush=True)
                save_results(results)
            else:
                mse, mae, ll, *_ = results[(T, scale, "adjSD", None)]
                print(f"  adjSD    MSE={mse:.3e}  MAE={mae:.3e}  LL={ll:.1f}  (cached)", flush=True)

            if needs_ss:
                ig_ss = cold_ss(y_train, Z_fixed)
                init_ss = cold_ss_init(y_train, Z_fixed)
                r_ss = _ss_fit(y_train, Z_cube, ig_ss, init_ss, LR["SS"])
                preds_ss, _, _, oos_ll_ss = ss_forecast(r_ss, Z_cube, y_test, ALPHA)
                jax.effects_barrier()
                results[(T, scale, "SS", None)] = _metrics(y_test, preds_ss, oos_ll_ss, NP_SS) + (LR["SS"], int(r_ss["niter"]), bool(r_ss["is_converged"]), MAXITER)
                mse, mae, ll, *_ = results[(T, scale, "SS", None)]
                print(f"  SS       MSE={mse:.3e}  MAE={mae:.3e}  LL={ll:.1f}  iter={int(r_ss['niter'])}  conv={bool(r_ss['is_converged'])}", flush=True)
                save_results(results)
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
