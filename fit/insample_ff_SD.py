import json
import os
import numpy as np
import jax.numpy as jnp
import polars as pl

from models.ff_SD import fit, _solve_weights_ff

PARQUET_PATH = "data/SPX/otm/bucket.parquet"
OUTPUT_DIR   = "out/SPX/otm/ffSD"

FACTOR_LOADING_COLS = ["level", "moneyness", "maturity"]
P_BASE = len(FACTOR_LOADING_COLS)
P_FULL = P_BASE + 1

K_VALUES = [3, 10, 80]

ETA_INIT   = 0.4
ALPHA_INIT = 1.2


def output_path(k):
    return os.path.join(OUTPUT_DIR, f"K{k}", "params.json")


def load_and_reshape(path):
    raw = (
        pl.read_parquet(path)
        .with_columns(pl.col("DATE").cast(pl.Utf8))
    )
    bucket_cols = sorted([c for c in raw.columns if c.startswith("bucket_")])
    n_buckets   = len(bucket_cols) + 1

    raw = raw.with_columns(
        pl.max_horizontal(
            [pl.when(pl.col(c)).then(i + 1).otherwise(0) for i, c in enumerate(bucket_cols)]
        ).alias("bucket_idx")
    ).sort(["DATE", *FACTOR_LOADING_COLS])

    dates = raw["DATE"].unique(maintain_order=True).sort().to_list()
    T     = len(dates)
    max_n = int(raw.group_by("DATE").len()["len"].max())

    y_cube = np.full((T, max_n), np.nan, dtype=np.float64)
    Z_cube = np.zeros((T, max_n, P_BASE + 1), dtype=np.float64)

    for t, date in enumerate(dates):
        slice_t = raw.filter(pl.col("DATE") == date).sort(FACTOR_LOADING_COLS)
        n_t = len(slice_t)
        y_cube[t, :n_t]          = slice_t["logIV"].to_numpy()
        Z_cube[t, :n_t, :P_BASE] = slice_t[FACTOR_LOADING_COLS].to_numpy()
        Z_cube[t, :n_t, P_BASE]  = slice_t["bucket_idx"].to_numpy().astype(float)

    return y_cube, Z_cube, n_buckets, bucket_cols, dates


def pooled_ols_beta(y_cube, Z_cube):
    T, max_n, _ = Z_cube.shape
    X    = Z_cube[:, :, :P_BASE].reshape(T * max_n, P_BASE)
    y    = y_cube.reshape(T * max_n)
    mask = ~np.isnan(y)
    beta_ols, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
    return beta_ols


def residual_variance(y_cube, Z_cube, beta):
    X    = Z_cube[:, :, :P_BASE].reshape(-1, P_BASE)
    y    = y_cube.reshape(-1)
    mask = ~np.isnan(y)
    return float(np.var(y[mask] - X[mask] @ beta))


def build_initial_guess(beta_ols, sigma2, n_buckets):
    omega_load = np.zeros(n_buckets)
    omega_load[1:] = 1e-2
    return {
        "beta_bar":   jnp.array(np.append(beta_ols, 0.0)),
        "A":          jnp.array(0.05 * np.eye(P_FULL, dtype=np.float64)),
        "sigma2":     jnp.array(sigma2),
        "omega_load": jnp.array(omega_load),
        "eta":        jnp.full(P_FULL, ETA_INIT),
        "alpha":      jnp.array([ALPHA_INIT]),
        "C":          jnp.array(1e-3 * np.eye(P_FULL, dtype=np.float64)),
        "nu":         jnp.array(10.0),
    }


def warm_start_guess(prev_params):
    return {
        "beta_bar":   jnp.array(np.array(prev_params["beta_bar"])),
        "A":          jnp.array(np.array(prev_params["A"])),
        "sigma2":     jnp.array(float(np.array(prev_params["sigma2"]).ravel()[0])),
        "omega_load": jnp.array(np.array(prev_params["omega_load"])),
        "eta":        jnp.array(np.array(prev_params["eta"])),
        "alpha":      jnp.array([float(np.array(prev_params["alpha"]).ravel()[0])]),
        "C":          jnp.array(np.array(prev_params["C"])),
        "nu":         jnp.array(float(np.array(prev_params["nu"]).ravel()[0])),
    }


def serialize_params(fit_output):
    serializable = {}
    for key, value in fit_output.items():
        if hasattr(value, "tolist"):
            serializable[key] = value.tolist()
        elif hasattr(value, "item"):
            serializable[key] = value.item()
        else:
            serializable[key] = value
    return serializable


def load_params(path):
    with open(path) as f:
        raw = json.load(f)
    return {k: np.array(v) if isinstance(v, list) else v for k, v in raw.items()}


def main():
    print("Loading data...")
    y_cube, Z_cube, n_buckets, bucket_cols, dates = load_and_reshape(PARQUET_PATH)
    T, max_n = y_cube.shape
    print(f"  T={T}, max_n={max_n}, n_buckets={n_buckets}")

    beta_ols = pooled_ols_beta(y_cube, Z_cube)
    sigma2   = residual_variance(y_cube, Z_cube, beta_ols)

    prev_params = None

    for k in K_VALUES:
        opath = output_path(k)

        print(f"\n--- K={k} ---")

        if os.path.exists(opath) and os.path.getsize(opath) > 0:
            print(f"  Found existing params, loading from {opath}")
            fit_output = load_params(opath)
            ws, rhos = _solve_weights_ff(jnp.array(fit_output["eta"]), jnp.array(fit_output["alpha"]), k)
            fit_output["ws"] = np.array(ws)
            fit_output["rhos"] = np.array(rhos)
        else:
            if prev_params is None:
                print(f"  Cold start")
                initial_guess = build_initial_guess(beta_ols, sigma2, n_buckets)
            else:
                print(f"  Warm start from K={K_VALUES[K_VALUES.index(k) - 1]}")
                initial_guess = warm_start_guess(prev_params)

            fit_output = fit(
                data=jnp.array(y_cube),
                covariates=jnp.array(Z_cube),
                initial_guess=initial_guess,
                K=k,
                score_power=1.0,
                opt_options={"learning_rate": 1e-3, "tol": 1e-6},
                maxiter=20_000,
            )
            ws, rhos = _solve_weights_ff(fit_output["eta"], fit_output["alpha"], k)
            fit_output["ws"] = np.array(ws)
            fit_output["rhos"] = np.array(rhos)
            os.makedirs(os.path.dirname(opath), exist_ok=True)
            with open(opath, "w") as f:
                json.dump(serialize_params(fit_output), f, indent=2)
            print(f"  Saved to {opath}")

        nu_fit    = float(np.array(fit_output["nu"]).ravel()[0])
        sigma2_fit = float(np.array(fit_output["sigma2"]).ravel()[0])
        eta_fit   = np.array(fit_output["eta"])
        alpha_fit = np.array(fit_output["alpha"])
        print(f"  nu={nu_fit:.2f}, sigma2={sigma2_fit:.4f}")
        print(f"  eta={eta_fit.round(4)}")
        print(f"  alpha={float(alpha_fit.ravel()[0]):.4f}")
        print(f"  log_lik={float(np.array(fit_output['log_likelihood']).ravel()[0]):.2f}")
        print(f"  rhos (factor 0)={fit_output['rhos'][:, 0].round(4)}")
        print(f"  ws (factor 0)={fit_output['ws'][:, 0].round(4)}")

        prev_params = fit_output

    print("\nDone.")


if __name__ == "__main__":
    main()
