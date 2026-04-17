"""
Generalization of Diebold-Mariano test statistics for multiple model comparisons. Reference:
Hansen, P. R., A. Lunde, and J. M. Nason (2011). The model confidence set. Econometrica 79 (2), 453-497.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

def loss_differentials(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a set of labeled losses at multiple horizon forecasts,
    returns all pairwise loss differentials between models.
    The output is a (dataframe) 3-D tensor in long tidy form.
    """
    L = df.to_numpy()
    _, M = L.shape
    diff = L[:, :, None] - L[:, None, :]
    h_idx, i_idx, j_idx = np.where(np.triu(np.ones((M, M)), k=1))
    return pd.DataFrame({
        "horizon": df.index.values[h_idx],
        "model_i": df.columns.values[i_idx],
        "model_j": df.columns.values[j_idx],
        "loss_diff": diff[h_idx, i_idx, j_idx]
    })

def _block_bootstrap_indices(T: int, B: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """
    Moving block bootstrap indices: shape (B, T).
    """
    block_length = int(block_length)
    if block_length <= 0: raise ValueError("block_length must be positive")
    n_blocks = int(np.ceil(T / block_length))
    max_start = T - block_length
    if max_start < 0: raise ValueError("block_length cannot exceed T")
    starts = rng.integers(0, max_start + 1, size=(B, n_blocks), endpoint=False)
    offsets = np.arange(block_length, dtype=int)[None, None, :]
    idx = (starts[:, :, None] + offsets).reshape(B, n_blocks * block_length)[:, :T]
    return idx

def _hac_var(x: np.ndarray, L: int) -> np.ndarray:
    """
    Bartlett HAC variance of the sample mean, vectorized.
    x: (..., T)
    returns: (...) array of Var( mean(x) ) estimates.
    Matches the common Newey-West style:
    s = gamma0 + 2*sum_{l=1..L} w_l * gamma_l, with gamma_l = (1/T)*sum x_t x_{t-l}
    Var(mean) = s / T
    """
    x = np.asarray(x, float)
    T = x.shape[-1]
    if T < 2:return np.full(x.shape[:-1], np.nan, float)
    L = int(L)
    if L < 0: raise ValueError("L must be >= 0")
    if L >= T: L = T - 1
    xc = x - x.mean(axis=-1, keepdims=True)
    n_fft = 1 << (2 * T - 1).bit_length()
    fx = np.fft.rfft(xc, n=n_fft, axis=-1)
    acov_full = np.fft.irfft(fx * np.conj(fx), n=n_fft, axis=-1)[..., : (L + 1)]
    gamma = acov_full / T
    w = 1.0 - (np.arange(L + 1, dtype=float) / (L + 1.0))
    s = gamma[..., 0] + 2.0 * np.sum((w[1:][None, ...] if gamma.ndim == 2 else w[1:]) * gamma[..., 1:], axis=-1)
    return s / T

def _pairwise_diffs(L: np.ndarray) -> np.ndarray:
    """
    L: (T, M)
    d: (T, M, M) with d[:, m, mp] = L[:, m] - L[:, mp]
    """
    return L[:, :, None] - L[:, None, :]

def _t_stats_pairwise(d: np.ndarray, L_hac: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    d: (T, M, M)
    returns:
      dbar: (M, M)
      var_mean: (M, M)
      t_mm: (M, M)
    """
    dbar = d.mean(axis=0)
    d_flat = np.swapaxes(d, 0, 2)
    var_mean = _hac_var(d_flat, L=L_hac)
    t_mm = dbar / np.sqrt(var_mean)
    np.fill_diagonal(t_mm, np.nan)
    np.fill_diagonal(dbar, np.nan)
    np.fill_diagonal(var_mean, np.nan)
    return dbar, var_mean, t_mm

def _t_stats_dot(L: np.ndarray, L_hac: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    L: (T, M)
    returns:
      dbar_dot: (M,)
      var_mean_dot: (M,)
      t_mdot: (M,)
    where d_{m,.}(t) = L_{t,m} - avg_{mp!=m} L_{t,mp}
    """
    L = np.asarray(L, float)
    T, M = L.shape
    mean_all = L.mean(axis=1, keepdims=True)
    mean_others = (M * mean_all - L) / (M - 1.0)
    d_dot = L - mean_others
    dbar_dot = d_dot.mean(axis=0)
    var_mean_dot = _hac_var(d_dot.T, L=L_hac)
    t_mdot = dbar_dot / np.sqrt(var_mean_dot)
    return dbar_dot, var_mean_dot, t_mdot

def _T_emp(L: np.ndarray, stat: str, L_hac: int) -> tuple[float, dict]:
    """
    Compute empirical test statistic for the current model set.
    """
    stat = stat.upper()
    if stat == "TR":
        d = _pairwise_diffs(L)
        dbar, var_mean, t_mm = _t_stats_pairwise(d, L_hac=L_hac)
        T_emp = np.nanmax(np.abs(t_mm))
        cache = {"dbar": dbar, "var_mean": var_mean, "t_mm": t_mm}
        return float(T_emp), cache
    if stat == "TMAX":
        dbar_dot, var_mean_dot, t_mdot = _t_stats_dot(L, L_hac=L_hac)
        T_emp = np.nanmax(np.abs(t_mdot))
        cache = {"dbar_dot": dbar_dot, "var_mean_dot": var_mean_dot, "t_mdot": t_mdot}
        return float(T_emp), cache
    raise ValueError("stat must be 'TR' or 'Tmax'")

def _T_star(L: np.ndarray, stat: str,
    L_hac: int, B: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    """
    Bootstrap distribution of the test statistic under H0 via re-centering.
    Returns: (B,) array of T*.
    """
    L = np.asarray(L, float)
    T, M = L.shape
    idx = _block_bootstrap_indices(T=T, B=B, block_length=block_length, rng=rng)
    stat = stat.upper()

    if stat == "TR":
        d = _pairwise_diffs(L)
        dbar = d.mean(axis=0)

        T_star = np.empty(B, dtype=float)
        diag = np.arange(M)

        for b in range(B):
            idx_b = idx[b]
            d_star_b = d[idx_b, :, :]
            dbar_star_b = d_star_b.mean(axis=0)

            d_star_for_hac_b = np.swapaxes(d_star_b, 0, 2)
            var_mean_star_b = _hac_var(
                d_star_for_hac_b.reshape(M * M, T), L=L_hac
            ).reshape(M, M)

            t_star_b = (dbar_star_b - dbar) / np.sqrt(np.maximum(var_mean_star_b, 1e-12))
            t_star_b[diag, diag] = np.nan

            T_star[b] = np.nanmax(np.abs(t_star_b))

        return T_star.astype(float)

    if stat == "TMAX":
        mean_all = L.mean(axis=1, keepdims=True)
        mean_others = (M * mean_all - L) / (M - 1.0)
        d_dot = L - mean_others
        dbar_dot = d_dot.mean(axis=0)
        d_dot_star = d_dot[idx, :]
        dbar_dot_star = d_dot_star.mean(axis=1)
        var_mean_dot_star = _hac_var(np.swapaxes(d_dot_star, 1, 2).reshape(B * M, T), L=L_hac).reshape(B, M)
        t_mdot_star = (dbar_dot_star - dbar_dot[None, :]) / np.sqrt(var_mean_dot_star)
        return np.nanmax(np.abs(t_mdot_star), axis=1).astype(float)

    raise ValueError("stat must be 'TR' or 'Tmax'")

def _elimination_rule(L: np.ndarray) -> int:
    """
    'Worst' model index based on average loss differential:
    e_m = mean_{mp!=m} \bar d_{m,mp} = \bar L_m - mean_{mp!=m} \bar L_mp
    Remove argmax e_m.
    """
    L = np.asarray(L, float)
    mean_loss = L.mean(axis=0)  # (M,)
    M = mean_loss.shape[0]
    mean_others = (M * mean_loss.mean() - mean_loss) / (M - 1.0)
    e = mean_loss - mean_others
    return int(np.nanargmax(e))

def mcs(
    losses: pd.DataFrame | np.ndarray,
    alpha: float = 0.05,
    B: int = 2000,
    block_length: int = 10,
    L_hac: int | None = None,
    stat: str = "Tmax",
    rng: np.random.Generator = np.random.default_rng(123)
    ) -> dict:
    """
    Model Confidence Set (MCS) main routine.
    losses:
      - pd.DataFrame with columns = model names, rows = time
      - or np.ndarray (T, M)
    Returns dict with:
      - "mcs_models": list[str] or list[int]
      - "pvalues": pd.Series or np.ndarray (assigned along elimination path)
      - "elimination_order": list[str] or list[int] (worst to best)
      - "final_pvalue": float (p-hat when stopping)
    """
    if isinstance(losses, pd.DataFrame):
        model_names = losses.columns.to_numpy()
        L_full = losses.to_numpy(dtype=float)
    else:
        model_names = None
        L_full = np.asarray(losses, float)
    if L_full.ndim != 2:
        raise ValueError("losses must be (T, M)")
    T, M0 = L_full.shape
    if M0 < 2: raise ValueError("need at least 2 models")

    if L_hac is None: L_hac = max(int(block_length) - 1, 0)
    active = np.arange(M0, dtype=int)
    elimination_order = []
    pvals = np.full(M0, np.nan, float)
    best_p_so_far = 0.0
    final_pvalue = np.nan

    while active.size >= 2:
        L = L_full[:, active]
        T_emp, _ = _T_emp(L, stat=stat, L_hac=L_hac)
        T_star = _T_star(L, stat=stat, L_hac=L_hac, B=B, block_length=block_length, rng=rng)
        p_hat = float(np.mean(T_star > T_emp))

        best_p_so_far = max(best_p_so_far, p_hat)
        if p_hat > alpha:
            final_pvalue = p_hat
            break
        worst_local = _elimination_rule(L)
        worst_global = int(active[worst_local])
        pvals[worst_global] = best_p_so_far
        elimination_order.append(worst_global)
        active = np.delete(active, worst_local)
    if active.size > 0: pvals[active] = 1.0

    if model_names is None:
        return {
            "mcs_models": active.tolist(),
            "pvalues": pvals,
            "elimination_order": elimination_order,
            "final_pvalue": final_pvalue,
        }
    return {
        "mcs_models": model_names[active].tolist(),
        "pvalues": pd.Series(pvals, index=model_names, name="pvalue"),
        "elimination_order": model_names[np.array(elimination_order, dtype=int)].tolist(),
        "final_pvalue": final_pvalue,
    }

def calculate_loss(target: np.ndarray, forecast: np.ndarray, method: str = "MSE") -> np.ndarray:
    """Computes vector of losses for a given target and forecast series. Returns: np.ndarray: vector of losses at time t."""

    target = np.asarray(target, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    method = method.upper()

    if method == "MSE":
        # Standard quadratic loss:
        return (target - forecast) ** 2

    elif method == "MAE":
        # Absolute error:
        return np.abs(target - forecast)

    elif method == "QLIKE":
        # Quasi Likelihood Loss for volatility
        # L = log(h_hat) + taret / h_hat
        # Target here corresponds to the realized variance proxy (y^2 or RK) not raw return

        forecast = np.maximum(forecast, 1e-12)
        return np.log(forecast) + (target / forecast)

    else:
        print(f"Unknown loss method: {method}, method must be 'MSE', 'MAE', 'QLIKE'")


def diebold_mariano_test(loss1: np.ndarray, loss2: np.ndarray, h: int = 1, L_hac: int | None = None) -> dict:
    """Performs DM test for predictive accuracy. """
    loss1 = np.asarray(loss1, dtype=float)
    loss2 = np.asarray(loss2, dtype=float)

    # First we comput the loss differential
    d = loss1 - loss2
    mean_diff = np.mean(d)

    #Compute variance of the mean (This is the HAC)
    if L_hac is None:
        L_hac = max(0, h-1)
    var_mean = _hac_var(d, L=L_hac)

    # Compute the DM test statistic
    hac_std_error = np.sqrt(var_mean)

    if hac_std_error < 1e-14:
        return {"dm_stat": 0.0, "p_value": 1.0, "mean_diff": mean_diff, "hac_std_error": 0.0}

    dm_stat = mean_diff / hac_std_error

    # Now we compute the P-value (two sided normal approximation)
    p_value = 2.0 * (1.0 - norm.cdf(np.abs(dm_stat)))

    return {"dm_stat": float(dm_stat), "p_value": float(p_value), "mean_diff": float(mean_diff), "hac_std_error": float(hac_std_error)}