import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

jax.config.update("jax_default_device", jax.devices("cpu")[0])
print(f"Running on: {jax.devices('cpu')[0]}")

from models.ff_SD import _filter as _ffsd_filter
from models.lmSD import _filter as _lmsd_filter

PARQUET_PATH = "data/SPX/otm/full.parquet"
OUTPUT_DIR = "out/SPX/otm/full_performance_fixed"
FACTOR_LOADING_COLS = ["level", "moneyness", "maturity"]
P_BASE = 3
P = P_BASE + 1
FFSS_K_DEFAULT = 5
FFSD_K_VALUES = [1, 2, 3, 5, 10]



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
    Z_cube = jnp.zeros((T, max_n, P)).at[t_idx, n_idx].set(
        jnp.concatenate([factor_vals, bucket_vals[:, None]], axis=-1)
    )
    return y_cube, Z_cube, dates


def _load_params(name):
    with open(os.path.join(OUTPUT_DIR, f"params_{name}.json")) as f:
        return json.load(f)


def _load_train_lls():
    path = os.path.join(OUTPUT_DIR, "step_results.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"step_results.parquet not found in {OUTPUT_DIR}; run full_forecast_fixed.py first."
        )
    df = pl.read_parquet(path)
    first = df.group_by("model").agg(pl.first("train_loglik"))
    return dict(zip(first["model"].to_list(), first["train_loglik"].to_list()))


def _best_k(prefix, train_lls, k_values):
    return max(k_values, key=lambda k: train_lls.get(f"{prefix}_K{k}", float("-inf")))


def _lmsd_with_eta_d(lmsd_raw, ffsd_raw):
    eta = jnp.array(ffsd_raw["eta"])
    return {
        "beta_bar": jnp.array(lmsd_raw["beta_bar"]),
        "A": jnp.diag(jnp.array(lmsd_raw["A_diag"])),
        "d": (1.0 - eta) / 2.0,
        "sigma2": jnp.array(lmsd_raw["sigma2"]),
        "omega": jnp.array(lmsd_raw["omega"]),
        "C": jnp.array(lmsd_raw["C_diag"]),
        "nu": jnp.array(lmsd_raw["nu"]),
    }


def _ffss_to_ffsd(p):
    Q_diag = jnp.array(p["Q_diag"])
    return {
        "beta_bar": jnp.array(p["beta_bar"]),
        "A": jnp.diag(jnp.sqrt(Q_diag)),
        "sigma2": jnp.array(p["sigma2"]),
        "omega_load": jnp.array(p["omega"]),
        "eta": jnp.array(p["eta"]),
        "phi": jnp.full(P, float(p["alpha"])),
        "C": jnp.diag(jnp.full(P, 1e-3)),
        "nu": jnp.array(10.0),
    }


def _ffsd_to_lmsd(params):
    return {
        "beta_bar": params["beta_bar"],
        "A": params["A"],
        "d": (1.0 - params["eta"]) / 2.0,
        "sigma2": params["sigma2"],
        "omega": params["omega_load"],
        "C": jnp.diag(params["C"]),
        "nu": params["nu"],
    }


def _make_ffsd_runner(K):
    @jax.jit
    def _run(y_tr, Z_tr, params):
        mask_bool = ~jnp.isnan(y_tr)
        y_masked = jnp.where(mask_bool, y_tr, 0.0)
        mask_f = mask_bool.astype(float)
        base = Z_tr[:, :, :-1]
        bidx = Z_tr[:, :, -1].astype(jnp.int32)
        _, lls, _, _ = _ffsd_filter(
            y_masked, base, bidx, mask_f, params, K, jnp.zeros((K + 1, P))
        )
        return jnp.sum(lls)
    return _run


@jax.jit
def _run_lmsd(y_tr, Z_tr, params):
    mask_bool = ~jnp.isnan(y_tr)
    y_masked = jnp.where(mask_bool, y_tr, 0.0)
    mask_f = mask_bool.astype(float)
    base = Z_tr[:, :, :-1]
    bidx = Z_tr[:, :, -1].astype(jnp.int32)
    T = y_tr.shape[0]
    state0 = (params["beta_bar"], jnp.zeros((T, P)), jnp.asarray(0, jnp.int32))
    _, lls, _, _, _ = _lmsd_filter(y_masked, base, bidx, mask_f, params, state0)
    return jnp.sum(lls)


def _save_ffsd(name, params, ll):
    def a(x): return np.asarray(x).tolist()
    d = {
        "beta_bar": a(params["beta_bar"]),
        "A_diag": a(jnp.diag(params["A"])),
        "sigma2": float(params["sigma2"]),
        "eta": a(params["eta"]),
        "phi": float(params["phi"][0]),
        "C_diag": a(jnp.diag(params["C"])),
        "nu": float(params["nu"]),
        "omega_load": a(params["omega_load"]),
        "insample_ll": ll,
    }
    path = os.path.join(OUTPUT_DIR, f"params_{name}_xinit.json")
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    print(f"  Saved {path}")


def _save_lmsd(params, ll, suffix):
    def a(x): return np.asarray(x).tolist()
    d = {
        "beta_bar": a(params["beta_bar"]),
        "A_diag": a(jnp.diag(params["A"])),
        "d": a(params["d"]),
        "sigma2": float(params["sigma2"]),
        "C_diag": a(params["C"]),
        "nu": float(params["nu"]),
        "omega": a(params["omega"]),
        "insample_ll": ll,
    }
    path = os.path.join(OUTPUT_DIR, f"params_lmSD_xinit_{suffix}.json")
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    print(f"  Saved {path}")


def main():
    print("Loading data...")
    y_jax, Z_jax, dates = load_and_reshape(PARQUET_PATH)
    T = y_jax.shape[0]
    train_size = T // 2
    y_train = y_jax[:train_size]
    Z_train = Z_jax[:train_size]
    print(f"  T={T}, train_size={train_size}")

    print("Loading existing training log-likelihoods...")
    train_lls = _load_train_lls()

    best_ffsd_k = _best_k("ffSD", train_lls, FFSD_K_VALUES)
    print(f"  ffSS: using K={FFSS_K_DEFAULT} (default)")
    print(f"  Best ffSD: K={best_ffsd_k}  LL={train_lls.get(f'ffSD_K{best_ffsd_k}', float('nan')):.2f}")
    print(f"  lmSD:      LL={train_lls.get('lmSD', float('nan')):.2f}")

    ffss_raw = _load_params(f"ffSS_K{FFSS_K_DEFAULT}")
    ffsd_params = _ffss_to_ffsd(ffss_raw)

    ffsd_runners = {K: _make_ffsd_runner(K) for K in FFSD_K_VALUES}

    print(f"\nRunning ffSD filter (init from ffSS K={FFSS_K_DEFAULT}):")
    ffsd_lls = {}
    for K in FFSD_K_VALUES:
        ll = float(ffsd_runners[K](y_train, Z_train, ffsd_params))
        orig = train_lls.get(f"ffSD_K{K}", float("-inf"))
        tag = "  IMPROVED" if ll > orig else ""
        print(f"  K={K}: LL={ll:.2f}  (original {orig:.2f}){tag}", flush=True)
        ffsd_lls[K] = ll
        jax.clear_caches()

    best_k_from_ffss = max(ffsd_lls, key=ffsd_lls.get)
    best_ll_from_ffss = ffsd_lls[best_k_from_ffss]
    orig_best_ffsd_ll = train_lls.get(f"ffSD_K{best_ffsd_k}", float("-inf"))

    print(f"\n  Best ffSD from ffSS init: K={best_k_from_ffss}  LL={best_ll_from_ffss:.2f}")
    if best_ll_from_ffss > orig_best_ffsd_ll:
        print(f"  IMPROVED over original best ffSD (LL={orig_best_ffsd_ll:.2f}), saving.")
        _save_ffsd(f"ffSD_K{best_k_from_ffss}", ffsd_params, best_ll_from_ffss)
    else:
        print(f"  No improvement over original best ffSD (LL={orig_best_ffsd_ll:.2f}).")

    print(f"\nRunning lmSD filter (init from ffSD K={best_k_from_ffss} <- ffSS K={FFSS_K_DEFAULT}):")
    lmsd_params = _ffsd_to_lmsd(ffsd_params)
    lmsd_ll = float(_run_lmsd(y_train, Z_train, lmsd_params))
    orig_lmsd_ll = train_lls.get("lmSD", float("-inf"))
    tag = "  IMPROVED" if lmsd_ll > orig_lmsd_ll else ""
    print(f"  LL={lmsd_ll:.2f}  (original {orig_lmsd_ll:.2f}){tag}")

    if lmsd_ll > orig_lmsd_ll:
        _save_lmsd(lmsd_params, lmsd_ll, "ffss_chain")

    print(f"\nRunning lmSD filter (lmSD params, d replaced by eta from ffSD K={best_ffsd_k}):")
    lmsd_params_eta_d = _lmsd_with_eta_d(_load_params("lmSD"), _load_params(f"ffSD_K{best_ffsd_k}"))
    lmsd_ll_eta_d = float(_run_lmsd(y_train, Z_train, lmsd_params_eta_d))
    tag = "  IMPROVED" if lmsd_ll_eta_d > orig_lmsd_ll else ""
    print(f"  LL={lmsd_ll_eta_d:.2f}  (original {orig_lmsd_ll:.2f}){tag}")
    if lmsd_ll_eta_d > orig_lmsd_ll:
        _save_lmsd(lmsd_params_eta_d, lmsd_ll_eta_d, "ffsd")


if __name__ == "__main__":
    main()
