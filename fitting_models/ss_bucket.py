import json
import numpy as np
import jax.numpy as jnp
import polars as pl

from models.ss import fit

PARQUET_PATH = "data/SPX/put/bucket.parquet"
OUTPUT_PATH  = "out/SPX/put/params.json"

FACTOR_LOADING_COLS = ["level", "moneyness", "moneyness2", "maturity", "interaction"]
P = len(FACTOR_LOADING_COLS)

def load_and_reshape(path: str) -> tuple[np.ndarray, np.ndarray]:
    raw = (
        pl.read_parquet(path)
        .with_columns(pl.col("DATE").cast(pl.Utf8))
        .sort(["DATE", *FACTOR_LOADING_COLS])
    )

    dates = raw["DATE"].unique(maintain_order=True).sort().to_list()
    T = len(dates)

    full_grid = raw.select(FACTOR_LOADING_COLS).unique().sort(FACTOR_LOADING_COLS)
    N = full_grid.height

    log_iv_matrix = np.full((T, N), np.nan, dtype=np.float64)
    covariates_cube = np.full((T, N, P), np.nan, dtype=np.float64)

    for t, date in enumerate(dates):
        slice_t = raw.filter(pl.col("DATE") == date)
        joined = full_grid.join(slice_t, on=FACTOR_LOADING_COLS, how="left")
        log_iv_matrix[t] = joined["logIV"].to_numpy()
        covariates_cube[t] = joined[FACTOR_LOADING_COLS].to_numpy()

    return log_iv_matrix, covariates_cube

def pooled_ols_beta(
    log_iv_matrix: np.ndarray,
    covariates_cube: np.ndarray) -> np.ndarray:
    T, N, _ = covariates_cube.shape
    X = covariates_cube.reshape(T * N, P)
    y = log_iv_matrix.reshape(T * N)
    beta_ols, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta_ols

def residual_variance(
    log_iv_matrix: np.ndarray,
    covariates_cube: np.ndarray,
    beta: np.ndarray) -> float:
    T, N, _ = covariates_cube.shape
    fitted   = (covariates_cube.reshape(T * N, P) @ beta)
    residuals = log_iv_matrix.reshape(T * N) - fitted
    return float(np.var(residuals))

def build_initial_guess(beta_ols: np.ndarray, sigma2: float) -> dict:
    B = 0.95 * np.eye(P, dtype=np.float64)
    bar_beta = beta_ols.copy()
    ct = (np.eye(P) - B) @ bar_beta
    Q_param = np.diag(np.full(P, 1e-3, dtype=np.float64))
    H_param = np.eye(P) * sigma2
    return {"Q_param": Q_param,"H_param": H_param,
        "B": B,"bar_beta": bar_beta, "ct": ct}

def build_jax_initial_guess(initial_guess: dict) -> dict:
    return {k: jnp.array(v) for k, v in initial_guess.items()}

def diffuse_kalman_initialization(beta_ols: np.ndarray) -> tuple:
    a0 = jnp.array(beta_ols, dtype=float)
    P0 = jnp.eye(P, dtype=float) * 10.0
    return (a0, P0)

def serialize_params(fit_output: dict) -> dict:
    serializable = {}
    for key, value in fit_output.items():
        if hasattr(value, "tolist"):serializable[key] = value.tolist()
        else: serializable[key] = value
    return serializable

def main():
    print("Loading data...")
    log_iv_matrix, covariates_cube = load_and_reshape(PARQUET_PATH)
    T, N = log_iv_matrix.shape
    print(f"  T={T} dates, N={N} obs per date, p={P} factors")

    print("Computing OLS initialisation...")
    beta_ols = pooled_ols_beta(log_iv_matrix, covariates_cube)
    sigma2   = residual_variance(log_iv_matrix, covariates_cube, beta_ols)
    print(f"  bar_beta (OLS) = {np.round(beta_ols, 4)}")
    print(f"  sigma2 (OLS residual) = {sigma2:.6f}")

    initial_guess = build_jax_initial_guess(
        build_initial_guess(beta_ols, sigma2)
    )
    initialization = diffuse_kalman_initialization(beta_ols)

    print("Fitting model...")
    fit_output = fit(
        data=jnp.array(log_iv_matrix),
        covariates=jnp.array(covariates_cube),
        initial_guess=initial_guess,
        initialization=initialization,)

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