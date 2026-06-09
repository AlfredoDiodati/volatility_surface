import os
import shutil
import atexit
import subprocess
import functools

if shutil.which("nvidia-smi") is not None:
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import gc
import json
import numpy as np
import jax
import jax.numpy as jnp

from models.lmSD import simulate
from models.lmSD import fit as lmSD_fit, forecast as lmSD_forecast, forecast_rolling_h as lmSD_forecast_rolling_h
from models.adjSD import fit as adjSD_fit, forecast as adjSD_forecast, forecast_rolling_h as adjSD_forecast_rolling_h
from models.ss import fit_collapsed as ss_fit, forecast as ss_forecast, forecast_rolling_h as ss_forecast_rolling_h
from models.ff_SD import fit as ff_SD_fit, forecast as ff_SD_forecast, standard_errors as ff_SD_se, forecast_rolling_h as ff_SD_forecast_rolling_h
from models.ff_SS import fit as ff_SS_fit, forecast as ff_SS_forecast, forecast_rolling_h as ff_SS_forecast_rolling_h
from models.MSMSD import fit as msmsd_fit, forecast as msmsd_forecast, forecast_rolling_h as msmsd_forecast_rolling_h
from fit._forecast_metrics import compute_mse, compute_mae, compute_aic, compute_bic

print(f"Running on: {jax.devices()}")


PARAMS_PATH = "out/SPX/otm/params_lmSD.json"
OUTPUT_DIR = "out/SPX/mc/simulate_lmSD_sjit"
SCORE_POWER = 1.0
ALPHA = 0.05
MAXITER = 5000
TOL = 1e-2
SOLVER = "lbfgs"
K_VALUES = [1, 2, 3, 5, 10, 20, 30, 50]
K_VALUES_MSMSD = [1, 2, 3, 5, 10]
MONEYNESS = jnp.array([0.9, 0.98, 1.05, 1.15, 1.3, 1.5])
MATURITY = jnp.array([10, 50, 100, 180]) / 255.0
N_BUCKETS = len(MONEYNESS) * len(MATURITY)

FCST_HORIZONS = (5, 22, 66, 260)
_MAX_FCST_H = max(FCST_HORIZONS)

HORIZONS = [400, 2000]
SIGMA2_SCALES = [1.0, 10.0, 0.1]
SCALE_TEX = {1.0: r"\boldsymbol H", 10.0: r"10\,\boldsymbol H", 0.1: r"\boldsymbol H/10"}

LR = {
    "ffSD":  1.0,
    "ffSS":  1.0,
    "msmSD": 1.0,
    "lmSD":  1.0,
    "adjSD": 1.0,
    "SS":    1.0,
}

_P_FF = 4
_P_TILDE = 3
_P_FULL = 4

NP_FF = _P_FF + _P_FF + 1 + (N_BUCKETS - 1) + _P_FF + 1 + _P_FF + 1
NP_FFSS = _P_FF + 1 + _P_FF + (N_BUCKETS - 1) + _P_FF + 1
NP_LMSD = 4 * _P_FF + N_BUCKETS + 1
NP_ADJSD = 4 * _P_FF + N_BUCKETS + 1
NP_SS = 3 * _P_FF + N_BUCKETS
NP_MSMSD = 3 * _P_TILDE + _P_FULL + (N_BUCKETS - 1) + 6


def load_params(path):
    with open(path) as f:
        raw = json.load(f)
    return {
        "beta_bar": jnp.array(raw["beta_bar"]),
        "A": jnp.array(raw["A"]),
        "d": jnp.array(raw["d"]),
        "sigma2": jnp.array(raw["sigma2"]),
        "omega": jnp.array(raw["omega"]),
        "C": jnp.array(raw["C"]),
        "nu": jnp.array(raw["nu"]),
    }


def make_Z_fixed():
    mon_g, mat_g = jnp.meshgrid(MONEYNESS, MATURITY, indexing="ij")
    N = mon_g.size
    return jnp.stack([jnp.ones(N), mon_g.ravel(), mat_g.ravel(), jnp.arange(N).astype(float)], axis=1)

def _ols(y_train, M):
    return jnp.linalg.lstsq(M, y_train.mean(axis=0), rcond=None)[0]

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

def cold_ffSS(y_train, Z_fixed):
    M = Z_fixed[:, :3]
    beta3 = _ols(y_train, M)
    resid = y_train - (M @ beta3)
    return {
        "beta_bar": jnp.append(beta3, 0.0),
        "sigma2": jnp.var(resid),
        "Q_param": jnp.diag(jnp.full(_P_FF, 1e-3)),
        "omega": jnp.concatenate([jnp.zeros(1), jnp.full(N_BUCKETS - 1, 1e-2)]),
        "eta": jnp.full(_P_FF, 0.06),
        "alpha": jnp.array(1.5),
    }

def cold_lmSD(y_train, Z_fixed):
    M = Z_fixed[:, :3]
    beta3 = _ols(y_train, M)
    resid = y_train - (M @ beta3)
    return {
        "beta_bar": jnp.append(beta3, 0.0),
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
        "A": params_true["A"],
        "d": params_true["d"],
        "sigma2": params_true["sigma2"],
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
def _ffSD_ffSS_K(y_train, Z_cube, Z_cube_ext, y_test, ig_ffSD, ig_ffSS, K):
    r = ff_SD_fit(y_train, Z_cube, ig_ffSD, K=K,
                  opt_options={"learning_rate": LR["ffSD"], "tol": TOL}, maxiter=MAXITER)
    preds, _, _, oos_ll = ff_SD_forecast(r, Z_cube, y_test, K, ALPHA)
    preds_h = ff_SD_forecast_rolling_h(r, Z_cube_ext, y_test, K, FCST_HORIZONS)
    ffSD_out = (r, preds, oos_ll, preds_h)

    r = ff_SS_fit(y_train, Z_cube, ig_ffSS, K=K,
                  opt_options={"learning_rate": LR["ffSS"], "tol": TOL}, maxiter=MAXITER)
    preds, _, _, oos_ll = ff_SS_forecast(r, Z_cube, y_test, ALPHA)
    preds_h = ff_SS_forecast_rolling_h(r, Z_cube_ext, y_test, FCST_HORIZONS)
    ffSS_out = (r, preds, oos_ll, preds_h)

    return ffSD_out, ffSS_out

@functools.partial(jax.jit, static_argnames=("K",))
def _msmSD_K(y_train, Z_cube, Z_cube_ext, y_test, ig, K):
    r = msmsd_fit(y_train, Z_cube, ig, K=K, score_power=SCORE_POWER,
                  opt_options={"learning_rate": LR["msmSD"], "tol": TOL}, maxiter=MAXITER)
    preds, _, _, oos_ll = msmsd_forecast(r, Z_cube, y_test, K, SCORE_POWER, ALPHA)
    preds_h = msmsd_forecast_rolling_h(r, Z_cube_ext, y_test, K, SCORE_POWER, FCST_HORIZONS)
    return r, preds, oos_ll, preds_h

@jax.jit
def _others(y_train, Z_cube, Z_cube_ext, y_test, ig_oracle, ig_adjsd, ig_ss, ig_ss_init):
    r = lmSD_fit(y_train, Z_cube, ig_oracle,
                 opt_options={"learning_rate": LR["lmSD"], "tol": TOL}, maxiter=0)
    preds, _, _, oos_ll = lmSD_forecast(r, Z_cube, y_test, ALPHA)
    preds_h = lmSD_forecast_rolling_h(r, Z_cube_ext, y_test, FCST_HORIZONS)
    oracle_out = (r, preds, oos_ll, preds_h)

    r = adjSD_fit(y_train, Z_cube, ig_adjsd,
                  opt_options={"learning_rate": LR["adjSD"], "tol": TOL}, maxiter=MAXITER)
    preds, _, _, oos_ll = adjSD_forecast(r, Z_cube, y_test, ALPHA)
    preds_h = adjSD_forecast_rolling_h(r, Z_cube_ext, y_test, FCST_HORIZONS)
    adjsd_out = (r, preds, oos_ll, preds_h)

    r = ss_fit(y_train, Z_cube, ig_ss, ig_ss_init,
               opt_options={"learning_rate": LR["SS"], "tol": TOL}, maxiter=MAXITER)
    preds, _, _, oos_ll = ss_forecast(r, Z_cube, y_test, ALPHA)
    preds_h = ss_forecast_rolling_h(r, Z_cube_ext, y_test, FCST_HORIZONS)
    ss_out = (r, preds, oos_ll, preds_h)

    return oracle_out, adjsd_out, ss_out


def _fit_all_models(y_train, Z_cube, Z_cube_ext, y_test, ig_all, ig_ss_init):
    ffSD_out, ffSS_out = [], []
    for i, K in enumerate(K_VALUES):
        ffSD_k, ffSS_k = _ffSD_ffSS_K(
            y_train, Z_cube, Z_cube_ext, y_test,
            ig_all["ffSD"][i], ig_all["ffSS"][i], K)
        jax.block_until_ready((ffSD_k, ffSS_k))
        ffSD_out.append(ffSD_k)
        ffSS_out.append(ffSS_k)

    msmSD_out = []
    for i, K in enumerate(K_VALUES_MSMSD):
        r_k = _msmSD_K(y_train, Z_cube, Z_cube_ext, y_test, ig_all["msmSD"][i], K)
        jax.block_until_ready(r_k)
        msmSD_out.append(r_k)

    oracle_out, adjsd_out, ss_out = _others(
        y_train, Z_cube, Z_cube_ext, y_test,
        ig_all["lmSD_oracle"], ig_all["adjSD"], ig_all["SS"], ig_ss_init)
    jax.block_until_ready((oracle_out, adjsd_out, ss_out))

    return ffSD_out, ffSS_out, msmSD_out, oracle_out, adjsd_out, ss_out


def _metrics(y_test, preds, oos_ll, n_params):
    mask = jnp.ones(y_test.shape, dtype=bool)
    mse = float(compute_mse(y_test, preds, mask))
    mae = float(compute_mae(y_test, preds, mask))
    ll_arr = jnp.asarray(oos_ll)
    tot_ll = float(ll_arr.sum())
    n_obs = int(y_test.size)
    aic = float(compute_aic(ll_arr.sum(), n_params))
    bic = float(compute_bic(ll_arr.sum(), n_params, n_obs))
    mse_seq = ((y_test - preds) ** 2).mean(axis=1).tolist()
    mae_seq = jnp.abs(y_test - preds).mean(axis=1).tolist()
    ll_seq = (ll_arr.sum(axis=1) if ll_arr.ndim > 1 else ll_arr).tolist()
    return mse, mae, tot_ll, aic, bic, mse_seq, mae_seq, ll_seq

_N_METRICS = len(("mse", "mae", "tot_ll", "aic", "bic", "mse_seq", "mae_seq", "ll_seq"))

def _is_cached(results, key, lr, solver):
    v = results.get(key)
    if v is None or len(v) <= _N_METRICS:
        return False
    mse = v[0]
    if mse != mse:
        return False
    stored_lr = v[_N_METRICS]
    is_converged = v[_N_METRICS + 2] if len(v) > _N_METRICS + 2 else True
    stored_solver = v[_N_METRICS + 4] if len(v) > _N_METRICS + 4 else None
    return abs(stored_lr - lr) < 1e-12 and stored_solver == solver and is_converged

def _is_cached_h(results_h, key):
    v = results_h.get(key)
    return v is not None and len(v) == len(FCST_HORIZONS)

def _all_cached(results, results_h, T, scale):
    for K in K_VALUES:
        if not _is_cached(results, (T, scale, "ffSD", K), LR["ffSD"], SOLVER): return False
        if not _is_cached(results, (T, scale, "ffSS", K), LR["ffSS"], SOLVER): return False
        if not _is_cached_h(results_h, (T, scale, "ffSD", K)): return False
        if not _is_cached_h(results_h, (T, scale, "ffSS", K)): return False
    for K in K_VALUES_MSMSD:
        if not _is_cached(results, (T, scale, "msmSD", K), LR["msmSD"], SOLVER): return False
        if not _is_cached_h(results_h, (T, scale, "msmSD", K)): return False
    for tag, lr in [("lmSD_oracle", LR["lmSD"]), ("adjSD", LR["adjSD"]), ("SS", LR["SS"])]:
        if not _is_cached(results, (T, scale, tag, None), lr, SOLVER): return False
        if not _is_cached_h(results_h, (T, scale, tag, None)): return False
    return True


def _metrics_h(y_test_ext, preds_h_all, eval_horizons, T_half):
    results = []
    for h_idx, h in enumerate(eval_horizons):
        targets = jnp.asarray(y_test_ext[h - 1:T_half + h - 1])
        preds = jnp.asarray(preds_h_all[h_idx])
        err2 = (targets - preds) ** 2
        mse = float(err2.mean())
        mse_seq = err2.mean(axis=1).tolist()
        results.append((mse, mse_seq))
    return results


_IG_KEYS = {
    "ffSD":  ["beta_bar", "A", "sigma2", "omega_load", "eta", "alpha", "C", "nu"],
    "ffSS":  ["beta_bar", "sigma2", "Q_param", "omega", "eta", "alpha"],
    "msmSD": ["beta_bar", "B", "A", "sigma2", "sigma_0", "omega_load", "C", "nu", "m0", "gamma_K", "b"],
    "lmSD":  ["beta_bar", "A", "d", "sigma2", "omega", "C", "nu"],
    "adjSD": ["beta_bar", "B", "A", "sigma2", "omega", "C", "nu"],
    "SS":    ["Q_param", "H_param", "B", "bar_beta", "omega"],
}

def _results_key(T, scale, tag, K):
    return json.dumps([T, scale, tag, K])

def _cache_filename(T, scale, tag, K):
    s = int(round(scale * 10))
    k = "None" if K is None else str(K)
    return os.path.join(PARAMS_CACHE_DIR, f"T{T}__s{s}__m{tag}__K{k}.npz")

RESULTS_PATH = os.path.join(OUTPUT_DIR, "results.jsonl")
RESULTS_H_PATH = os.path.join(OUTPUT_DIR, "results_h.jsonl")
PARAMS_CACHE_DIR = os.path.join(OUTPUT_DIR, "params_cache")

def _load_jsonl(path, old_json_path):
    if os.path.exists(old_json_path) and not os.path.exists(path):
        with open(old_json_path) as f:
            raw = json.load(f)
        with open(path, "w") as out:
            for key_str, v in raw.items():
                out.write(json.dumps({"key": key_str, "value": v}) + "\n")
    if not os.path.exists(path):
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            T, scale, tag, K = json.loads(obj["key"])
            out[(T, scale, tag, K)] = obj["value"]
    return out

def load_results():
    raw = _load_jsonl(RESULTS_PATH, os.path.join(OUTPUT_DIR, "results.json"))
    return {k: tuple(v) for k, v in raw.items()}

def save_results(results, pk):
    with open(RESULTS_PATH, "a") as f:
        f.write(json.dumps({"key": _results_key(*pk), "value": list(results[pk])}) + "\n")

def load_results_h():
    return _load_jsonl(RESULTS_H_PATH, os.path.join(OUTPUT_DIR, "results_h.json"))

def save_results_h(results_h, pk):
    with open(RESULTS_H_PATH, "a") as f:
        f.write(json.dumps({"key": _results_key(*pk), "value": results_h[pk]}) + "\n")

def load_params_cache():
    os.makedirs(PARAMS_CACHE_DIR, exist_ok=True)
    old_json = os.path.join(OUTPUT_DIR, "params_cache.json")
    if os.path.exists(old_json) and not any(
        fn.endswith(".npz") for fn in os.listdir(PARAMS_CACHE_DIR)
    ):
        with open(old_json) as f:
            raw = json.load(f)
        cache = {}
        for key_str, params in raw.items():
            T, scale, tag, K = json.loads(key_str)
            ck = (T, scale, tag, K)
            cache[ck] = {k: np.array(v) for k, v in params.items()}
        for ck, params in cache.items():
            np.savez(_cache_filename(*ck), **params)
        return cache
    cache = {}
    for fname in os.listdir(PARAMS_CACHE_DIR):
        if not fname.endswith(".npz"):
            continue
        parts = fname[:-4].split("__")
        T = int(parts[0][1:])
        scale = int(parts[1][1:]) / 10.0
        tag = parts[2][1:]
        k_str = parts[3][1:]
        K = None if k_str == "None" else int(k_str)
        raw = np.load(os.path.join(PARAMS_CACHE_DIR, fname))
        cache[(T, scale, tag, K)] = dict(raw)
    return cache

def save_params_cache(cache, pk):
    np.savez(_cache_filename(*pk), **cache[pk])


def _build_ig(T, scale, y_train, Z_fixed, params_cache, params):
    def _get(tag, K, cold_fn):
        pk = (T, scale, tag, K)
        if pk in params_cache:
            return {k: jnp.array(v) for k, v in params_cache[pk].items()}
        return cold_fn(y_train, Z_fixed)

    return {
        "ffSD":  [_get("ffSD",  K, cold_ffSD)  for K in K_VALUES],
        "ffSS":  [_get("ffSS",  K, cold_ffSS)  for K in K_VALUES],
        "msmSD": [_get("msmSD", K, cold_msmSD) for K in K_VALUES_MSMSD],
        "lmSD_oracle": true_lmSD(params),
        "adjSD": _get("adjSD", None, cold_adjSD),
        "SS":    _get("SS",    None, cold_ss),
    }


def make_table(T, results):
    import math

    all_mse, all_mae = [], []
    all_keys = (
        [(t, k) for t in ("ffSD", "ffSS") for k in K_VALUES]
        + [("msmSD", k) for k in K_VALUES_MSMSD]
        + [(t, None) for t in ("adjSD", "SS")]
    )
    for scale in SIGMA2_SCALES:
        for tag, K in all_keys:
            key = (T, scale, tag, K)
            if key in results:
                mse, mae, *_ = results[key]
                all_mse.append(mse)
                all_mae.append(mae)

    def _exp3(vals):
        if not vals: return 0
        mag = sum(abs(v) for v in vals) / len(vals)
        if mag == 0: return 0
        return 3 * round(math.floor(math.log10(mag)) / 3)

    mse_exp = _exp3(all_mse)
    mse_mult = 10.0 ** (-mse_exp)
    mae_exp = -2
    mae_mult = 100.0
    ll_exp = 3

    def _fmt_mse(v):
        if v is None: return "--"
        av = abs(v)
        if av >= 10: return f"{v:.2f}"
        elif av >= 1: return f"{v:.3f}"
        else: return f"{v:.4f}"

    def _bench_seqs(scale):
        key = (T, scale, "lmSD_oracle", None)
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
        if stat > 2.576: return r"\rlap{$^{\ddagger}$}"
        elif stat > 1.96: return r"\rlap{$^{\dagger}$}"
        elif stat > 1.645: return r"\rlap{$^{\circ}$}"
        return ""

    def _get(scale, tag, K):
        key = (T, scale, tag, K)
        if key not in results: return None
        r = results[key]
        seqs = (r[5], r[6], r[7]) if len(r) >= 8 else (None, None, None)
        return r[0] * mse_mult, r[1] * mae_mult, r[2], seqs

    N_MET = 3

    def _cmidrule(i):
        lo = 3 + i * N_MET
        return rf"\cmidrule(lr){{{lo}-{lo + N_MET - 1}}}"

    col_spec = "ll" + "rrr" * len(SIGMA2_SCALES)

    def _scale_cell(exp):
        return rf"($\times 10^{{{exp}}}$)" if exp != 0 else ""

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\resizebox{\textwidth}{!}{",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    grp_hdrs = " & ".join(
        rf"\multicolumn{{{N_MET}}}{{c}}{{${SCALE_TEX[s]}$}}"
        for s in SIGMA2_SCALES
    )
    lines.append(rf"Model & $K$ & {grp_hdrs} \\")
    lines.append(" ".join(_cmidrule(i) for i in range(len(SIGMA2_SCALES))))

    metric_hdr = " & ".join("MSE & MAE & loglik" for _ in SIGMA2_SCALES)
    lines.append(rf"& & {metric_hdr} \\")

    scale_hdr = " & ".join(
        f"{_scale_cell(mse_exp)} & {_scale_cell(mae_exp)} & {_scale_cell(ll_exp)}" for _ in SIGMA2_SCALES
    )
    lines.append(rf"& & {scale_hdr} \\")
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
                cells += [_fmt_mse(mse_v) + s_mse, f"{mae_v:.1f}" + s_mae, f"{ll_v/1000:.2f}" + s_ll]
        return "    " + " & ".join(cells) + r" \\"

    K_groups = [("ff-SD", "ffSD", K_VALUES), ("ff-SS", "ffSS", K_VALUES), ("msm-SD", "msmSD", K_VALUES_MSMSD)]
    for gi, (model_name, tag, k_list) in enumerate(K_groups):
        if gi > 0: lines.append(r"\addlinespace[3pt]")
        n = len(k_list)
        for ri, K in enumerate(k_list):
            model_cell = rf"\multirow{{{n}}}{{*}}{{{model_name}}}" if ri == 0 else ""
            lines.append(_row(model_cell, str(K), tag, K))

    for model_name, tag in [("adj-SD", "adjSD"), ("SS", "SS")]:
        lines.append(r"\addlinespace[3pt]")
        lines.append(_row(model_name, "", tag, None))

    lines.append(r"\midrule")
    lines.append(_row("lmSD", "", "lmSD_oracle", None, is_bench=True))

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        (
            rf"\caption{{One-step-ahead predictive performance on data simulated"
            rf" from the lmSD model, $T={T}$"
            rf" (first $T/2$ for training, second $T/2$ for evaluation)."
            rf" Superscripts denote significance of Diebold-Mariano test against lmSD:"
            rf" $^{{\circ}}$10\%, $^{{\dagger}}$5\%, $^{{\ddagger}}$1\%.}}"
        ),
        rf"\label{{tab:sim_lmSD_T{T}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def make_table_h(T, results_h):
    import math

    n_h = len(FCST_HORIZONS)
    all_keys = (
        [(t, k) for t in ("ffSD", "ffSS") for k in K_VALUES]
        + [("msmSD", k) for k in K_VALUES_MSMSD]
        + [(t, None) for t in ("adjSD", "SS")]
    )

    all_mse = []
    for scale in SIGMA2_SCALES:
        for tag, K in all_keys:
            key = (T, scale, tag, K)
            if key in results_h:
                for mse, _ in results_h[key]:
                    all_mse.append(mse)

    def _exp3(vals):
        if not vals: return 0
        mag = sum(abs(v) for v in vals) / len(vals)
        if mag == 0: return 0
        return 3 * round(math.floor(math.log10(mag)) / 3)

    mse_exp = _exp3(all_mse)
    mse_mult = 10.0 ** (-mse_exp)

    def _fmt(v):
        if v is None: return "--"
        av = abs(v)
        if av >= 10: return f"{v:.2f}"
        elif av >= 1: return f"{v:.3f}"
        else: return f"{v:.4f}"

    def _dm_stars(seq_m, seq_b):
        if seq_m is None or seq_b is None: return ""
        n = min(len(seq_m), len(seq_b))
        if n < 2: return ""
        d = [seq_m[i] - seq_b[i] for i in range(n)]
        mean_d = sum(d) / n
        var_d = sum((x - mean_d) ** 2 for x in d) / (n - 1)
        if var_d <= 0: return ""
        stat = abs(mean_d) / (var_d ** 0.5 / n ** 0.5)
        if stat > 2.576: return r"\rlap{$^{\ddagger}$}"
        elif stat > 1.96: return r"\rlap{$^{\dagger}$}"
        elif stat > 1.645: return r"\rlap{$^{\circ}$}"
        return ""

    def _oracle_seqs(scale):
        key = (T, scale, "lmSD_oracle", None)
        if key not in results_h: return [None] * n_h
        return [seq for _, seq in results_h[key]]

    def _get(scale, tag, K):
        key = (T, scale, tag, K)
        return results_h.get(key)

    col_spec = "ll" + "r" * (n_h * len(SIGMA2_SCALES))
    grp_hdrs = " & ".join(
        rf"\multicolumn{{{n_h}}}{{c}}{{${SCALE_TEX[s]}$}}"
        for s in SIGMA2_SCALES
    )

    def _cmidrule(i):
        lo = 3 + i * n_h
        return rf"\cmidrule(lr){{{lo}-{lo + n_h - 1}}}"

    h_hdr = " & ".join(f"$h={h}$" for h in FCST_HORIZONS)
    h_hdrs = " & ".join(h_hdr for _ in SIGMA2_SCALES)
    scale_hdr = rf"($\times 10^{{{mse_exp}}}$)" if mse_exp != 0 else ""
    scale_hdrs = " & ".join(" & ".join(scale_hdr for _ in FCST_HORIZONS) for _ in SIGMA2_SCALES)

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\resizebox{\textwidth}{!}{",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf"Model & $K$ & {grp_hdrs} \\",
        " ".join(_cmidrule(i) for i in range(len(SIGMA2_SCALES))),
        rf"& & {h_hdrs} \\",
        rf"& & {scale_hdrs} \\",
        r"\midrule",
    ]

    def _row(model_cell, K_cell, tag, K, is_bench=False):
        cells = [model_cell, K_cell]
        for scale in SIGMA2_SCALES:
            got = _get(scale, tag, K)
            oracle_seqs = _oracle_seqs(scale)
            for h_idx in range(n_h):
                if got is None:
                    cells.append("--")
                else:
                    mse_v, seq = got[h_idx]
                    stars = "" if is_bench else _dm_stars(seq, oracle_seqs[h_idx])
                    cells.append(_fmt(mse_v * mse_mult) + stars)
        return "    " + " & ".join(cells) + r" \\"

    K_groups = [("ff-SD", "ffSD", K_VALUES), ("ff-SS", "ffSS", K_VALUES),
                ("msm-SD", "msmSD", K_VALUES_MSMSD)]
    for gi, (model_name, tag, k_list) in enumerate(K_groups):
        if gi > 0: lines.append(r"\addlinespace[3pt]")
        n = len(k_list)
        for ri, K in enumerate(k_list):
            model_cell = rf"\multirow{{{n}}}{{*}}{{{model_name}}}" if ri == 0 else ""
            lines.append(_row(model_cell, str(K), tag, K))

    for model_name, tag in [("adj-SD", "adjSD"), ("SS", "SS")]:
        lines.append(r"\addlinespace[3pt]")
        lines.append(_row(model_name, "", tag, None))

    lines.append(r"\midrule")
    lines.append(_row("lmSD", "", "lmSD_oracle", None, is_bench=True))
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        rf"\caption{{Multi-horizon point forecast MSE on data simulated from the lmSD model, $T={T}$."
        rf" Superscripts: $^{{\circ}}$10\%, $^{{\dagger}}$5\%, $^{{\ddagger}}$1\% DM test vs lmSD.}}",
        rf"\label{{tab:sim_h_lmSD_T{T}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def compute_wald_tests(params_base, Z_fixed):
    from scipy import stats as scipy_stats
    params_cache = load_params_cache()
    d_true = jnp.array(params_base["d"])
    sigma2_base = params_base["sigma2"]
    p = d_true.shape[0]

    test_results = {}
    key = jax.random.PRNGKey(42)

    for T in HORIZONS:
        T_half = T // 2
        Z_cube = jnp.broadcast_to(Z_fixed[None], (T_half, Z_fixed.shape[0], Z_fixed.shape[1]))

        for scale in SIGMA2_SCALES:
            key, subkey = jax.random.split(key)
            params_scaled = params_base | {"sigma2": sigma2_base * scale}
            y_sim, _ = _sim_jit(params_scaled, Z_fixed, horizon=T, key=subkey, score_buf_size=T)
            y_train = y_sim[:T_half]

            for K in K_VALUES:
                pk = (T, scale, "ffSD", K)
                if pk not in params_cache:
                    continue
                cached = params_cache[pk]
                se, cov_eta = ff_SD_se(cached, y_train, Z_cube, K)
                eta_hat = cached["eta"]
                d_hat = (1.0 - eta_hat) / 2.0
                cov_d = cov_eta / 4.0
                se_d = se["eta"] / 2.0
                diff = d_hat - d_true
                joint_stat = float(diff @ jnp.linalg.inv(cov_d) @ diff)
                joint_pval = float(scipy_stats.chi2.sf(joint_stat, df=p))
                marginal_z = [(float(d_hat[j]) - float(d_true[j])) / float(se_d[j]) for j in range(p)]
                marginal_pval = [float(2 * scipy_stats.norm.sf(abs(z))) for z in marginal_z]
                test_results[pk] = {
                    "joint_stat": joint_stat, "joint_pval": joint_pval,
                    "marginal_z": marginal_z, "marginal_pval": marginal_pval,
                }
                print(f"  wald ffSD T={T} scale={scale} K={K}: chi2={joint_stat:.2f} p={joint_pval:.3f}", flush=True)

    return test_results


def make_wald_table(T, test_results, p):
    from scipy import stats as scipy_stats

    def _sig(pval):
        if pval is None: return ""
        if pval < 0.01: return r"\rlap{$^{\ddagger}$}"
        if pval < 0.05: return r"\rlap{$^{\dagger}$}"
        if pval < 0.10: return r"\rlap{$^{\circ}$}"
        return ""

    def _zfmt(z, pval):
        if z is None: return "--"
        return f"{z:+.2f}{_sig(pval)}"

    def _chi2fmt(stat, pval):
        if stat is None: return "--"
        return f"{stat:.2f}{_sig(pval)}"

    N_MET = p + 1
    col_spec = "ll" + "r" * (N_MET * len(SIGMA2_SCALES))

    cmidrules = " ".join(
        rf"\cmidrule(lr){{{3 + i * N_MET}-{2 + (i + 1) * N_MET}}}"
        for i in range(len(SIGMA2_SCALES))
    )
    grp_hdrs = " & ".join(
        rf"\multicolumn{{{N_MET}}}{{c}}{{${SCALE_TEX[s]}$}}"
        for s in SIGMA2_SCALES
    )
    dim_hdrs = " & ".join(
        rf"$\chi^2({p})$ & " + " & ".join(rf"$d_{{{j+1}}}$" for j in range(p))
        for _ in SIGMA2_SCALES
    )

    lines = [
        r"\begin{table}[H]",
        r"\centering",
        r"\small",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf"Model & $K$ & {grp_hdrs} \\",
        cmidrules,
        rf"& & {dim_hdrs} \\",
        r"\midrule",
    ]

    def _row(model_cell, K_cell, tag, K):
        cells = [model_cell, K_cell]
        for scale in SIGMA2_SCALES:
            key = (T, scale, tag, K)
            r = test_results.get(key)
            if r is None:
                cells += ["--"] * N_MET
            else:
                cells.append(_chi2fmt(r["joint_stat"], r["joint_pval"]))
                for j in range(p):
                    cells.append(_zfmt(r["marginal_z"][j], r["marginal_pval"][j]))
        return "    " + " & ".join(cells) + r" \\"

    K_groups = [("ff-SD", "ffSD", K_VALUES)]
    for gi, (model_name, tag, k_list) in enumerate(K_groups):
        if gi > 0: lines.append(r"\addlinespace[3pt]")
        n = len(k_list)
        for ri, K in enumerate(k_list):
            model_cell = rf"\multirow{{{n}}}{{*}}{{{model_name}}}" if ri == 0 else ""
            lines.append(_row(model_cell, str(K), tag, K))

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{Wald tests for $\hat{{\eta}} = d_{{\mathrm{{true}}}}$ on data simulated from the lmSD model, $T={T}$."
        rf" $\chi^2({p})$: joint test; $d_j$: marginal $z$-test for dimension $j$."
        rf" Superscripts: $^{{\circ}}$10\%, $^{{\dagger}}$5\%, $^{{\ddagger}}$1\%.}}",
        rf"\label{{tab:wald_lmSD_T{T}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    params_base = load_params(PARAMS_PATH)
    Z_fixed = make_Z_fixed()
    sigma2_base = params_base["sigma2"]

    key = jax.random.PRNGKey(42)
    results = load_results()
    results_h = load_results_h()
    params_cache = load_params_cache()

    for T in HORIZONS:
        T_half = T // 2
        Z_cube = jnp.broadcast_to(Z_fixed[None], (T_half, Z_fixed.shape[0], Z_fixed.shape[1]))
        Z_cube_ext = jnp.broadcast_to(Z_fixed[None], (T_half + _MAX_FCST_H, Z_fixed.shape[0], Z_fixed.shape[1]))

        for scale in SIGMA2_SCALES:
            if _all_cached(results, results_h, T, scale):
                print(f"T={T:5d}  scale={scale:.1f}  all models cached, skipping simulation.", flush=True)
                continue

            params = params_base | {"sigma2": sigma2_base * scale}
            key, subkey = jax.random.split(key)
            y_sim, _ = _sim_jit(params, Z_fixed, horizon=T + _MAX_FCST_H, key=subkey, score_buf_size=T + _MAX_FCST_H)
            jax.effects_barrier()

            y_train = y_sim[:T_half]
            y_test = y_sim[T_half:T]
            y_test_ext = y_sim[T_half:]

            ig_all = _build_ig(T, scale, y_train, Z_fixed, params_cache, params)
            ig_ss_init = cold_ss_init(y_train, Z_fixed)

            print(f"T={T:5d}  scale={scale:.1f}  simulated, fitting all models in single JIT...", flush=True)

            ffSD_out, ffSS_out, msmSD_out, oracle_out, adjsd_out, ss_out = \
                _fit_all_models(y_train, Z_cube, Z_cube_ext, y_test, ig_all, ig_ss_init)
            jax.effects_barrier()

            for i, K in enumerate(K_VALUES):
                r, preds, oos_ll, preds_h = ffSD_out[i]
                pk = (T, scale, "ffSD", K)
                init = "warm" if pk in params_cache else "cold"
                results[pk] = _metrics(y_test, preds, oos_ll, NP_FF) + (LR["ffSD"], int(r["niter"]), bool(r["is_converged"]), MAXITER, SOLVER)
                params_cache[pk] = {k: np.asarray(r[k]) for k in _IG_KEYS["ffSD"]}
                mse, mae, ll, *_ = results[pk]
                print(f"  ff-SD K={K}  MSE={mse:.3e}  MAE={mae:.3e}  LL={ll:.1f}  iter={int(r['niter'])}  conv={bool(r['is_converged'])}  solver={SOLVER}  init={init}", flush=True)
                save_results(results, pk)
                save_params_cache(params_cache, pk)
                results_h[pk] = _metrics_h(y_test_ext, preds_h, FCST_HORIZONS, T_half)
                save_results_h(results_h, pk)

            for i, K in enumerate(K_VALUES):
                r, preds, oos_ll, preds_h = ffSS_out[i]
                pk = (T, scale, "ffSS", K)
                init = "warm" if pk in params_cache else "cold"
                results[pk] = _metrics(y_test, preds, oos_ll, NP_FFSS) + (LR["ffSS"], int(r["niter"]), bool(r["is_converged"]), MAXITER, SOLVER)
                params_cache[pk] = {k: np.asarray(r[k]) for k in _IG_KEYS["ffSS"]}
                mse, mae, ll, *_ = results[pk]
                print(f"  ff-SS K={K}  MSE={mse:.3e}  MAE={mae:.3e}  LL={ll:.1f}  iter={int(r['niter'])}  conv={bool(r['is_converged'])}  solver={SOLVER}  init={init}", flush=True)
                save_results(results, pk)
                save_params_cache(params_cache, pk)
                results_h[pk] = _metrics_h(y_test_ext, preds_h, FCST_HORIZONS, T_half)
                save_results_h(results_h, pk)

            for i, K in enumerate(K_VALUES_MSMSD):
                r, preds, oos_ll, preds_h = msmSD_out[i]
                pk = (T, scale, "msmSD", K)
                init = "warm" if pk in params_cache else "cold"
                results[pk] = _metrics(y_test, preds, oos_ll, NP_MSMSD) + (LR["msmSD"], int(r["niter"]), bool(r["is_converged"]), MAXITER, SOLVER)
                params_cache[pk] = {k: np.asarray(r[k]) for k in _IG_KEYS["msmSD"]}
                mse, mae, ll, *_ = results[pk]
                print(f"  msm-SD K={K}  MSE={mse:.3e}  MAE={mae:.3e}  LL={ll:.1f}  iter={int(r['niter'])}  conv={bool(r['is_converged'])}  solver={SOLVER}  init={init}", flush=True)
                save_results(results, pk)
                save_params_cache(params_cache, pk)
                results_h[pk] = _metrics_h(y_test_ext, preds_h, FCST_HORIZONS, T_half)
                save_results_h(results_h, pk)

            r, preds, oos_ll, preds_h = oracle_out
            pk = (T, scale, "lmSD_oracle", None)
            results[pk] = _metrics(y_test, preds, oos_ll, NP_LMSD) + (LR["lmSD"], int(r["niter"]), bool(r["is_converged"]), 0, SOLVER)
            mse, mae, ll, *_ = results[pk]
            print(f"  lmSD†    MSE={mse:.3e}  MAE={mae:.3e}  LL={ll:.1f}  iter={int(r['niter'])}  conv={bool(r['is_converged'])}  solver={SOLVER}  init=oracle", flush=True)
            save_results(results, pk)
            results_h[pk] = _metrics_h(y_test_ext, preds_h, FCST_HORIZONS, T_half)
            save_results_h(results_h, pk)

            r, preds, oos_ll, preds_h = adjsd_out
            pk = (T, scale, "adjSD", None)
            init = "warm" if pk in params_cache else "cold"
            results[pk] = _metrics(y_test, preds, oos_ll, NP_ADJSD) + (LR["adjSD"], int(r["niter"]), bool(r["is_converged"]), MAXITER, SOLVER)
            params_cache[pk] = {k: np.asarray(r[k]) for k in _IG_KEYS["adjSD"]}
            mse, mae, ll, *_ = results[pk]
            print(f"  adjSD    MSE={mse:.3e}  MAE={mae:.3e}  LL={ll:.1f}  iter={int(r['niter'])}  conv={bool(r['is_converged'])}  solver={SOLVER}  init={init}", flush=True)
            save_results(results, pk)
            save_params_cache(params_cache, pk)
            results_h[pk] = _metrics_h(y_test_ext, preds_h, FCST_HORIZONS, T_half)
            save_results_h(results_h, pk)

            r, preds, oos_ll, preds_h = ss_out
            pk = (T, scale, "SS", None)
            init = "warm" if pk in params_cache else "cold"
            results[pk] = _metrics(y_test, preds, oos_ll, NP_SS) + (LR["SS"], int(r["niter"]), bool(r["is_converged"]), MAXITER, SOLVER)
            params_cache[pk] = {k: np.asarray(r[k]) for k in _IG_KEYS["SS"]}
            mse, mae, ll, *_ = results[pk]
            print(f"  SS       MSE={mse:.3e}  MAE={mae:.3e}  LL={ll:.1f}  iter={int(r['niter'])}  conv={bool(r['is_converged'])}  solver={SOLVER}  init={init}", flush=True)
            save_results(results, pk)
            save_params_cache(params_cache, pk)
            results_h[pk] = _metrics_h(y_test_ext, preds_h, FCST_HORIZONS, T_half)
            save_results_h(results_h, pk)

            del y_sim, y_train, y_test, y_test_ext
            jax.effects_barrier()
            gc.collect()

    tex = "\n\n".join(make_table(T, results) for T in HORIZONS)
    out_path = os.path.join(OUTPUT_DIR, "MC_lmSD.tex")
    with open(out_path, "w") as f:
        f.write(tex)
    print(f"\nLaTeX tables saved to {out_path}")

    tex_h = "\n\n".join(make_table_h(T, results_h) for T in HORIZONS)
    out_path_h = os.path.join(OUTPUT_DIR, "MC_lmSD_h.tex")
    with open(out_path_h, "w") as f:
        f.write(tex_h)
    print(f"Multi-horizon tables saved to {out_path_h}")

    print("\nComputing Wald tests...", flush=True)
    p = int(params_base["d"].shape[0])
    test_results = compute_wald_tests(params_base, Z_fixed)
    wald_tex = "\n\n".join(make_wald_table(T, test_results, p) for T in HORIZONS)
    wald_path = os.path.join(OUTPUT_DIR, "wald_lmSD.tex")
    with open(wald_path, "w") as f:
        f.write(wald_tex)
    print(f"Wald test tables saved to {wald_path}")

def _inhibit_sleep():
    try:
        proc = subprocess.Popen(
            ["systemd-inhibit",
             "--what=sleep:idle:handle-lid-switch",
             "--who=MC_lmSD_sjit",
             "--why=Running Monte Carlo simulation",
             "--mode=block",
             "sleep", "infinity"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        atexit.register(proc.terminate)
    except FileNotFoundError:
        print("systemd-inhibit not found, sleep inhibition skipped.", flush=True)

if __name__ == "__main__":
    _inhibit_sleep()
    main()
