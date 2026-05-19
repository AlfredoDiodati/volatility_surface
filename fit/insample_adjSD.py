import json
import os
import numpy as np
import jax.numpy as jnp
import polars as pl

from models.adjSD import fit

PARQUET_PATH = "data/SPX/otm/bucket.parquet"
OUTPUT_PATH  = "out/SPX/otm/params_adjSD.json"

FACTOR_LOADING_COLS = ["level", "moneyness", "maturity"]
P_BASE = len(FACTOR_LOADING_COLS)
P = P_BASE + 1

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
    T = len(dates)
    max_n = int(raw.group_by("DATE").len()["len"].max())

    y_cube = np.full((T, max_n), np.nan, dtype=np.float64)
    Z_cube = np.zeros((T, max_n, P_BASE + 1), dtype=np.float64)

    for t, date in enumerate(dates):
        slice_t = raw.filter(pl.col("DATE") == date).sort(FACTOR_LOADING_COLS)
        n_t = len(slice_t)
        y_cube[t, :n_t] = slice_t["logIV"].to_numpy()
        Z_cube[t, :n_t, :P_BASE] = slice_t[FACTOR_LOADING_COLS].to_numpy()
        Z_cube[t, :n_t, P_BASE] = slice_t["bucket_idx"].to_numpy().astype(float)

    return y_cube, Z_cube, n_buckets, bucket_cols, dates

def pooled_ols_beta(y_cube, Z_cube):
    T, max_n, _ = Z_cube.shape
    X = Z_cube[:, :, :P_BASE].reshape(T * max_n, P_BASE)
    y = y_cube.reshape(T * max_n)
    mask = ~np.isnan(y)
    beta_ols, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
    return beta_ols

def residual_variance(y_cube, Z_cube, beta):
    X = Z_cube[:, :, :P_BASE].reshape(-1, P_BASE)
    y = y_cube.reshape(-1)
    mask = ~np.isnan(y)
    return float(np.var(y[mask] - X[mask] @ beta))

def build_initial_guess(beta_ols, sigma2, n_buckets):
    omega = np.zeros(n_buckets)
    omega[1:] = 1e-2
    return {
        "beta_bar": jnp.array(np.append(beta_ols, 0.0)),
        "B": jnp.array(0.95 * np.eye(P, dtype=np.float64)),
        "A": jnp.array(0.05 * np.eye(P, dtype=np.float64)),
        "sigma2": jnp.array(sigma2),
        "omega": jnp.array(omega),
        "C": jnp.array(1e-3 * np.ones(P, dtype=np.float64)),
        "nu": jnp.array(10.0),
    }

def serialize_params(fit_output):
    serializable = {}
    for key, value in fit_output.items():
        if hasattr(value, "tolist"):
            serializable[key] = value.tolist()
        elif hasattr(value, "item"):
            serializable[key] = value.item()
        else: serializable[key] = value
    return serializable

def main():
    print("Loading data...")
    y_cube, Z_cube, n_buckets, bucket_cols, dates = load_and_reshape(PARQUET_PATH)
    T, max_n = y_cube.shape
    print(f"  T={T}, max_n={max_n}, n_buckets={n_buckets}")

    if os.path.exists(OUTPUT_PATH):
        print("Found existing params, loading...")
        with open(OUTPUT_PATH) as f:
            raw_params = json.load(f)
        fit_output = {k: np.array(v) if isinstance(v, list) else v
            for k, v in raw_params.items()}
    else:
        print("No params found, fitting model...")
        beta_ols = pooled_ols_beta(y_cube, Z_cube)
        sigma2   = residual_variance(y_cube, Z_cube, beta_ols)

        initial_guess = build_initial_guess(beta_ols, sigma2, n_buckets)

        fit_output = fit(
            data=jnp.array(y_cube),
            covariates=jnp.array(Z_cube),
            initial_guess=initial_guess,
            opt_options={"learning_rate": 1e-3, "tol": 1e-6},
            maxiter=20_000,
        )
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        with open(OUTPUT_PATH, "w") as f:
            json.dump(serialize_params(fit_output), f, indent=2)
        print(f"  Saved to {OUTPUT_PATH}")

    sigma2 = float(np.array(fit_output["sigma2"]).ravel()[0])
    nu = float(np.array(fit_output["nu"]).ravel()[0])
    print(f" nu={nu:.2f}, sigma2={sigma2:.4f}")
    print(f" ={float(np.array(fit_output['log_likelihood']).ravel()[0]):.2f}")
    print("Done.")

if __name__ == "__main__": main()
