import json
import os
os.environ["JAX_PLATFORMS"] = "cpu"
import numpy as np
import jax.numpy as jnp
import polars as pl

from models.MSMSD import fit

PARQUET_PATH = "data/SPX/otm/bucket.parquet"
OUTPUT_PATH  = "out/SPX/otm/params_msmSD.json"
K            = 3

FACTOR_LOADING_COLS = ["level", "moneyness", "maturity"]
P_BASE = len(FACTOR_LOADING_COLS)
P_FULL = P_BASE + 1


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

    return y_cube, Z_cube, n_buckets, dates


def pooled_ols_beta(y_cube, Z_cube):
    X    = Z_cube[:, :, :P_BASE].reshape(-1, P_BASE)
    y    = y_cube.reshape(-1)
    mask = ~np.isnan(y)
    beta_ols, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
    return beta_ols


def residual_variance(y_cube, Z_cube, beta):
    X    = Z_cube[:, :, :P_BASE].reshape(-1, P_BASE)
    y    = y_cube.reshape(-1)
    mask = ~np.isnan(y)
    return float(np.var(y[mask] - X[mask] @ beta))


def build_initial_guess(beta_ols, sigma2, n_buckets):
    return {
        "beta_bar":   jnp.array(beta_ols),
        "B":          jnp.array(0.95 * np.eye(P_BASE, dtype=np.float64)),
        "A":          jnp.array(0.05 * np.eye(P_BASE, dtype=np.float64)),
        "sigma2":     jnp.array(sigma2),
        "sigma_0":    jnp.array(0.1),
        "omega_load": jnp.array(np.concatenate([np.zeros(1), np.full(n_buckets - 1, 1e-2)])),
        "C":          jnp.array(1e-3 * np.eye(P_FULL, dtype=np.float64)),
        "nu":         jnp.array(10.0),
        "m0":         jnp.array(1.5),
        "gamma_K":    jnp.array(0.5),
        "b":          jnp.array(2.0),
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


def main():
    print("Loading data...")
    y_cube, Z_cube, n_buckets, dates = load_and_reshape(PARQUET_PATH)
    T, max_n = y_cube.shape
    print(f"  T={T}, max_n={max_n}, n_buckets={n_buckets}, K={K}")

    if os.path.exists(OUTPUT_PATH):
        print(f"Found existing params at {OUTPUT_PATH}, done.")
        with open(OUTPUT_PATH) as f:
            fit_output = json.load(f)
        print(f"  log_lik={float(np.array(fit_output['log_likelihood']).ravel()[0]):.2f}")
        print(f"  nu={float(np.array(fit_output['nu']).ravel()[0]):.2f}")
        print(f"  m0={float(np.array(fit_output['m0']).ravel()[0]):.4f}")
        print(f"  gamma_K={float(np.array(fit_output['gamma_K']).ravel()[0]):.4f}")
        print(f"  b={float(np.array(fit_output['b']).ravel()[0]):.4f}")
        return

    print("Fitting msmSD model...")
    beta_ols = pooled_ols_beta(y_cube, Z_cube)
    sigma2   = residual_variance(y_cube, Z_cube, beta_ols)
    ig       = build_initial_guess(beta_ols, sigma2, n_buckets)

    fit_output = fit(
        data=jnp.array(y_cube),
        covariates=jnp.array(Z_cube),
        initial_guess=ig,
        K=K,
        score_power=1.0,
        opt_options={"learning_rate": 1e-3, "tol": 1e-6},
        maxiter=20_000,
    )

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(serialize_params(fit_output), f, indent=2)
    print(f"  Saved to {OUTPUT_PATH}")
    print(f"  log_lik={float(jnp.array(fit_output['log_likelihood']).ravel()[0]):.2f}")
    print(f"  converged={bool(fit_output['is_converged'])}, niter={int(fit_output['niter'])}")
    print(f"  nu={float(jnp.array(fit_output['nu']).ravel()[0]):.2f}")
    print(f"  m0={float(jnp.array(fit_output['m0']).ravel()[0]):.4f}")
    print(f"  gamma_K={float(jnp.array(fit_output['gamma_K']).ravel()[0]):.4f}")
    print(f"  b={float(jnp.array(fit_output['b']).ravel()[0]):.4f}")


if __name__ == "__main__":
    main()
