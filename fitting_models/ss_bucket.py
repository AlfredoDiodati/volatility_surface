import json
import numpy as np
import jax.numpy as jnp
import polars as pl

from models.ss import fit_collapsed

PARQUET_PATH = "data/SPX/put/bucket.parquet"
OUTPUT_PATH  = "out/SPX/put/params.json"

FACTOR_LOADING_COLS = ["level", "moneyness", "maturity"]
P_BASE = len(FACTOR_LOADING_COLS)
P      = P_BASE + 1


def load_and_reshape(path: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (log_iv_matrix, covariates, n_buckets).

    covariates has shape (N, P_BASE + 1) — time-invariant, stored once:
      - columns 0..P_BASE-1 : continuous factor loadings
      - column  P_BASE       : integer bucket index (float-encoded for JAX)

    Bucket index encoding:
      0            = reference category (no flag set)
      1 .. n_named = named bucket columns in sorted order
    """
    raw = (
        pl.read_parquet(path)
        .with_columns(pl.col("DATE").cast(pl.Utf8))
    )

    bucket_cols = sorted([c for c in raw.columns if c.startswith("bucket_")])
    n_buckets = len(bucket_cols) + 1          # +1 for reference category

    raw = raw.with_columns(
        pl.max_horizontal(
            [pl.when(pl.col(c)).then(i + 1).otherwise(0) for i, c in enumerate(bucket_cols)]
        ).alias("bucket_idx")
    ).sort(["DATE", *FACTOR_LOADING_COLS])

    dates = raw["DATE"].unique(maintain_order=True).sort().to_list()
    T = len(dates)

    # One bucket_idx per unique option spec (static across time)
    full_grid = (
        raw.select([*FACTOR_LOADING_COLS, "bucket_idx"])
        .unique(subset=FACTOR_LOADING_COLS, keep="first")
        .sort(FACTOR_LOADING_COLS)
    )
    N = full_grid.height

    base_vals   = full_grid[FACTOR_LOADING_COLS].to_numpy()          # (N, P_BASE)
    bucket_vals = full_grid["bucket_idx"].to_numpy().astype(float)   # (N,)

    # Single (N, P_BASE+1) array — no T copies
    covariates = np.empty((N, P_BASE + 1), dtype=np.float64)
    covariates[:, :P_BASE] = base_vals
    covariates[:, P_BASE]  = bucket_vals

    log_iv_matrix = np.full((T, N), np.nan, dtype=np.float64)
    for t, date in enumerate(dates):
        slice_t = raw.filter(pl.col("DATE") == date)
        joined  = full_grid.join(slice_t, on=FACTOR_LOADING_COLS, how="left")
        log_iv_matrix[t] = joined["logIV"].to_numpy()

    return log_iv_matrix, covariates, n_buckets


def pooled_ols_beta(
    log_iv_matrix: np.ndarray,
    covariates: np.ndarray) -> np.ndarray:
    """OLS beta using time-invariant (N, P_BASE+1) covariates."""
    Xbase = covariates[:, :P_BASE]  # (N, P_BASE)
    mask  = ~np.isnan(log_iv_matrix)           # (T, N)
    count_n = mask.sum(axis=0)                 # (N,) — valid obs per option
    XtX  = (Xbase * count_n[:, None]).T @ Xbase          # (P_BASE, P_BASE)
    Xty  = Xbase.T @ np.where(mask, log_iv_matrix, 0.0).sum(axis=0)  # (P_BASE,)
    beta_ols, *_ = np.linalg.lstsq(XtX, Xty, rcond=None)
    return beta_ols


def residual_variance(
    log_iv_matrix: np.ndarray,
    covariates: np.ndarray,
    beta: np.ndarray) -> float:
    fitted    = covariates[:, :P_BASE] @ beta  # (N,) — time-invariant
    residuals = log_iv_matrix - fitted[None, :]  # (T, N), broadcasting
    return float(np.nanvar(residuals))


def build_initial_guess(
    beta_ols: np.ndarray,
    sigma2: float,
    N: int,
    n_buckets: int) -> dict:
    B        = 0.95 * np.eye(P, dtype=np.float64)
    bar_beta = np.append(beta_ols, 0.0)                 # (P,)
    Q_param  = np.diag(np.full(P, 1e-3, dtype=np.float64))
    H_param  = np.array([[sigma2]], dtype=np.float64)   # (1, 1) — scalar sigma2
    omega    = np.zeros(n_buckets, dtype=np.float64)    # (n_buckets,)
    return {"Q_param": Q_param, "H_param": H_param, "B": B, "bar_beta": bar_beta, "omega": omega, "N_obs": N}


def build_jax_initial_guess(initial_guess: dict) -> dict:
    return {k: (int(v) if k == "N_obs" else jnp.array(v)) for k, v in initial_guess.items()}


def build_initialization(
    beta_ols: np.ndarray,
    sigma2: float) -> tuple:
    """8-tuple (a1, P1, Z0, T0, H0, R0, Q0, idx0) expected by fit_collapsed."""
    a1   = jnp.array(np.append(beta_ols, 0.0), dtype=float)  # (P,)
    P1   = 10.0 * jnp.eye(P, dtype=float)                    # (P, P)
    Z0   = jnp.zeros((P, P), dtype=float)                    # (P, P)
    T0   = 0.95 * jnp.eye(P, dtype=float)                    # (P, P)
    H0   = sigma2 * jnp.eye(P, dtype=float)                  # (P, P) — overwritten by collapsed dynamics
    R0   = jnp.eye(P, dtype=float)                           # (P, P)
    Q0   = jnp.diag(jnp.full(P, 1e-3, dtype=float))         # (P, P)
    idx0 = jnp.asarray(0, dtype=jnp.int32)
    return (a1, P1, Z0, T0, H0, R0, Q0, idx0)


def serialize_params(fit_output: dict) -> dict:
    serializable = {}
    for key, value in fit_output.items():
        if hasattr(value, "tolist"): serializable[key] = value.tolist()
        else: serializable[key] = value
    return serializable


def main():
    print("Loading data...")
    log_iv_matrix, covariates, n_buckets = load_and_reshape(PARQUET_PATH)
    T, N = log_iv_matrix.shape
    print(f"  T={T} dates, N={N} obs per date, P_base={P_BASE} factors, n_buckets={n_buckets}")

    print("Computing OLS initialisation...")
    beta_ols = pooled_ols_beta(log_iv_matrix, covariates)
    sigma2   = residual_variance(log_iv_matrix, covariates, beta_ols)
    print(f"  bar_beta (OLS) = {np.round(beta_ols, 4)}")
    print(f"  sigma2 (OLS residual) = {sigma2:.6f}")

    initial_guess  = build_jax_initial_guess(
        build_initial_guess(beta_ols, sigma2, N, n_buckets)
    )
    initialization = build_initialization(beta_ols, sigma2)

    print("Fitting model...")
    fit_output = fit_collapsed(
        data=jnp.array(log_iv_matrix),
        covariates=jnp.array(covariates),     # (N, P_BASE+1) — single copy
        initial_guess=initial_guess,
        initialization=initialization,
    )

    print("Saving parameter estimates...")
    with open(OUTPUT_PATH, "w") as f:
        json.dump(serialize_params(fit_output), f, indent=2)
    print(f"  Saved to {OUTPUT_PATH}")

    print("\nParameter summary:")
    for key, value in fit_output.items():
        if hasattr(value, "shape"): print(f"  {key}: shape={value.shape}")
        else: print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
