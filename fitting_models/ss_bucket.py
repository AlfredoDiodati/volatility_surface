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
    """Return (y_cube, Z_cube, n_buckets).

    y_cube: (T, max_n)         — logIV, NaN-padded for dates with fewer obs
    Z_cube: (T, max_n, P_BASE+1) — per-date loading matrices, zero-padded
      - columns 0..P_BASE-1 : continuous factor loadings
      - column  P_BASE       : integer bucket index (float-encoded)

    max_n = maximum observations per date (16).  Only the first n_t rows
    of y_cube[t] / Z_cube[t] are real; the rest are NaN / zero padding.
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
    T     = len(dates)
    max_n = int(raw.group_by("DATE").len()["len"].max())

    # Per-date aligned arrays — tiny: (T, max_n) instead of (T, N_unique)
    y_cube = np.full((T, max_n), np.nan, dtype=np.float64)
    Z_cube = np.zeros((T, max_n, P_BASE + 1), dtype=np.float64)

    for t, date in enumerate(dates):
        slice_t = raw.filter(pl.col("DATE") == date).sort(FACTOR_LOADING_COLS)
        n_t = len(slice_t)
        y_cube[t, :n_t]            = slice_t["logIV"].to_numpy()
        Z_cube[t, :n_t, :P_BASE]   = slice_t[FACTOR_LOADING_COLS].to_numpy()
        Z_cube[t, :n_t, P_BASE]    = slice_t["bucket_idx"].to_numpy().astype(float)

    return y_cube, Z_cube, n_buckets


def pooled_ols_beta(
    y_cube: np.ndarray,
    Z_cube: np.ndarray) -> np.ndarray:
    T, max_n, _ = Z_cube.shape
    X    = Z_cube[:, :, :P_BASE].reshape(T * max_n, P_BASE)
    y    = y_cube.reshape(T * max_n)
    mask = ~np.isnan(y)
    beta_ols, *_ = np.linalg.lstsq(X[mask], y[mask], rcond=None)
    return beta_ols


def residual_variance(
    y_cube: np.ndarray,
    Z_cube: np.ndarray,
    beta: np.ndarray) -> float:
    X    = Z_cube[:, :, :P_BASE].reshape(-1, P_BASE)
    y    = y_cube.reshape(-1)
    mask = ~np.isnan(y)
    return float(np.var(y[mask] - X[mask] @ beta))


def build_initial_guess(
    beta_ols: np.ndarray,
    sigma2: float,
    n_buckets: int) -> dict:
    B        = 0.95 * np.eye(P, dtype=np.float64)
    bar_beta = np.append(beta_ols, 0.0)
    Q_param  = np.diag(np.full(P, 1e-3, dtype=np.float64))
    H_param  = np.array([[sigma2]], dtype=np.float64)  # (1,1) — scalar sigma2
    omega    = np.zeros(n_buckets, dtype=np.float64)
    omega[1:] = 1e-2    # non-zero to avoid singular Zt'Zt (omega=0 ⇒ last column of Z is zero)
    return {"Q_param": Q_param, "H_param": H_param, "B": B, "bar_beta": bar_beta, "omega": omega}


def build_jax_initial_guess(initial_guess: dict) -> dict:
    return {k: jnp.array(v) for k, v in initial_guess.items()}


def build_initialization(
    beta_ols: np.ndarray,
    sigma2: float) -> tuple:
    """8-tuple (a1, P1, Z0, T0, H0, R0, Q0, idx0) — shapes for the collapsed filter."""
    a1   = jnp.array(np.append(beta_ols, 0.0), dtype=float)  # (P,)
    P1   = 10.0 * jnp.eye(P, dtype=float)                    # (P, P)
    Z0   = jnp.zeros((P, P), dtype=float)                    # (P, P)
    T0   = 0.95 * jnp.eye(P, dtype=float)                    # (P, P)
    H0   = sigma2 * jnp.eye(P, dtype=float)                  # (P, P) — overwritten in carry
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
    y_cube, Z_cube, n_buckets = load_and_reshape(PARQUET_PATH)
    T, max_n = y_cube.shape
    print(f"  T={T} dates, max_n={max_n} obs/date, P_base={P_BASE}, n_buckets={n_buckets}")
    print(f"  y_cube: {y_cube.nbytes/1e6:.1f} MB  Z_cube: {Z_cube.nbytes/1e6:.1f} MB")

    print("Computing OLS initialisation...")
    beta_ols = pooled_ols_beta(y_cube, Z_cube)
    sigma2   = residual_variance(y_cube, Z_cube, beta_ols)
    print(f"  bar_beta (OLS) = {np.round(beta_ols, 4)}")
    print(f"  sigma2  (OLS residual) = {sigma2:.6f}")

    initial_guess  = build_jax_initial_guess(build_initial_guess(beta_ols, sigma2, n_buckets))
    initialization = build_initialization(beta_ols, sigma2)

    print("Fitting model...")
    fit_output = fit_collapsed(
        data=jnp.array(y_cube),
        covariates=jnp.array(Z_cube),    # (T, max_n, P_BASE+1)
        initial_guess=initial_guess,
        initialization=initialization,
        opt_options={"learning_rate": 1e-3, "tol": 1e-6},
        maxiter=20_000,
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
