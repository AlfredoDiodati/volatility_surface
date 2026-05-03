import jax
import jax.numpy as jnp
import polars as pl

def _block_bootstrap_indices(T: int, B: int, block_length: int, key: jax.Array) -> jnp.ndarray:
    n_blocks = int(jnp.ceil(T / block_length))
    max_start = T - block_length
    starts = jax.random.randint(key, shape=(B, n_blocks), minval=0, maxval=max_start + 1)
    offsets = jnp.arange(block_length)[None, None, :]
    idx = (starts[:, :, None] + offsets).reshape(B, n_blocks * block_length)[:, :T]
    return idx

def _hac_var(x: jnp.ndarray, L: int) -> jnp.ndarray:
    T = x.shape[-1]
    L = min(int(L), T - 1)
    xc = x - x.mean(axis=-1, keepdims=True)
    n_fft = 1 << (2 * T - 1).bit_length()
    fx = jnp.fft.rfft(xc, n=n_fft, axis=-1)
    acov_full = jnp.fft.irfft(fx * jnp.conj(fx), n=n_fft, axis=-1)[..., : L + 1]
    gamma = acov_full / T
    w = 1.0 - jnp.arange(L + 1, dtype=float) / (L + 1.0)
    s = gamma[..., 0] + 2.0 * jnp.sum(w[1:] * gamma[..., 1:], axis=-1)
    return s / T

def _pairwise_diffs(L: jnp.ndarray) -> jnp.ndarray: return L[:, :, None] - L[:, None, :]

def _t_stats_pairwise(d: jnp.ndarray, L_hac: int) -> tuple:
    T, M, _ = d.shape
    dbar = d.mean(axis=0)
    d_hac = jnp.moveaxis(d, 0, -1).reshape(M * M, T)
    var_mean = _hac_var(d_hac, L=L_hac).reshape(M, M)
    t_mm = dbar / jnp.sqrt(var_mean)
    diag = jnp.eye(M, dtype=bool)
    t_mm = jnp.where(diag, jnp.nan, t_mm)
    dbar = jnp.where(diag, jnp.nan, dbar)
    var_mean = jnp.where(diag, jnp.nan, var_mean)
    return dbar, var_mean, t_mm

def _t_stats_dot(L: jnp.ndarray, L_hac: int) -> tuple:
    T, M = L.shape
    mean_all = L.mean(axis=1, keepdims=True)
    mean_others = (M * mean_all - L) / (M - 1.0)
    d_dot = L - mean_others
    dbar_dot = d_dot.mean(axis=0)
    var_mean_dot = _hac_var(d_dot.T, L=L_hac)
    t_mdot = dbar_dot / jnp.sqrt(var_mean_dot)
    return dbar_dot, var_mean_dot, t_mdot

def _T_emp(L: jnp.ndarray, stat: str, L_hac: int) -> tuple:
    stat = stat.upper()
    if stat == "TR":
        d = _pairwise_diffs(L)
        dbar, var_mean, t_mm = _t_stats_pairwise(d, L_hac=L_hac)
        t_abs = jnp.where(jnp.isnan(t_mm), -jnp.inf, jnp.abs(t_mm))
        return float(t_abs.max()), {"dbar": dbar, "var_mean": var_mean, "t_mm": t_mm}
    if stat == "TMAX":
        dbar_dot, var_mean_dot, t_mdot = _t_stats_dot(L, L_hac=L_hac)
        return float(jnp.abs(t_mdot).max()), {"dbar_dot": dbar_dot, "var_mean_dot": var_mean_dot, "t_mdot": t_mdot}
    raise ValueError("stat must be 'TR' or 'Tmax'")

def _T_star(L: jnp.ndarray, stat: str, L_hac: int, B: int, block_length: int, key: jax.Array) -> jnp.ndarray:
    T, M = L.shape
    idx = _block_bootstrap_indices(T=T, B=B, block_length=block_length, key=key)
    stat = stat.upper()

    if stat == "TR":
        d = _pairwise_diffs(L)
        dbar = d.mean(axis=0)
        d_star = d[idx]
        dbar_star = d_star.mean(axis=1)
        var_mean_star = _hac_var(
            jnp.moveaxis(d_star, 1, -1).reshape(B * M * M, T), L=L_hac
        ).reshape(B, M, M)
        t_star = (dbar_star - dbar[None]) / jnp.sqrt(jnp.maximum(var_mean_star, 1e-12))
        diag = jnp.eye(M, dtype=bool)[None]
        t_abs = jnp.where(diag, -jnp.inf, jnp.abs(t_star))
        return t_abs.reshape(B, M * M).max(axis=1)

    if stat == "TMAX":
        mean_all = L.mean(axis=1, keepdims=True)
        mean_others = (M * mean_all - L) / (M - 1.0)
        d_dot = L - mean_others
        dbar_dot = d_dot.mean(axis=0)
        d_dot_star = d_dot[idx]
        dbar_dot_star = d_dot_star.mean(axis=1)
        var_mean_dot_star = _hac_var(
            jnp.moveaxis(d_dot_star, 1, -1).reshape(B * M, T), L=L_hac
        ).reshape(B, M)
        t_mdot_star = (dbar_dot_star - dbar_dot[None]) / jnp.sqrt(var_mean_dot_star)
        return jnp.abs(t_mdot_star).max(axis=1)

    raise ValueError("stat must be 'TR' or 'Tmax'")

def _elimination_rule(L: jnp.ndarray) -> int:
    mean_loss = L.mean(axis=0)
    M = mean_loss.shape[0]
    mean_others = (M * mean_loss.mean() - mean_loss) / (M - 1.0)
    return int(jnp.argmax(mean_loss - mean_others))

def mcs(
    losses: pl.DataFrame | jnp.ndarray,
    alpha: float = 0.05,
    B: int = 2000,
    block_length: int = 10,
    L_hac: int | None = None,
    stat: str = "Tmax",
    key: jax.Array = jax.random.PRNGKey(123),
) -> dict:
    if isinstance(losses, pl.DataFrame):
        model_names = losses.columns
        L_full = jnp.array(losses.to_numpy())
    else:
        model_names = None
        L_full = jnp.asarray(losses, float)

    T, M0 = L_full.shape
    if L_hac is None:
        L_hac = max(int(block_length) - 1, 0)

    active = list(range(M0))
    elimination_order = []
    pvals = [float("nan")] * M0
    best_p_so_far = 0.0
    final_pvalue = float("nan")

    chunk_size = 200
    while len(active) >= 2:
        L = L_full[:, jnp.array(active)]
        T_emp, _ = _T_emp(L, stat=stat, L_hac=L_hac)
        exceedances = 0
        for b_start in range(0, B, chunk_size):
            b_chunk = min(chunk_size, B - b_start)
            key, subkey = jax.random.split(key)
            T_star_chunk = _T_star(L, stat=stat, L_hac=L_hac, B=b_chunk,
                                   block_length=block_length, key=subkey)
            exceedances += int((T_star_chunk > T_emp).sum())
            jax.effects_barrier()
        p_hat = exceedances / B
        best_p_so_far = max(best_p_so_far, p_hat)

        if p_hat > alpha:
            final_pvalue = p_hat
            break

        worst_local = _elimination_rule(L)
        worst_global = active[worst_local]
        pvals[worst_global] = best_p_so_far
        elimination_order.append(worst_global)
        active.pop(worst_local)

    for i in active: pvals[i] = 1.0

    if model_names is None:
        return {
            "mcs_models": active,
            "pvalues": pvals,
            "elimination_order": elimination_order,
            "final_pvalue": final_pvalue,
        }
    return {
        "mcs_models": [model_names[i] for i in active],
        "pvalues": {model_names[i]: pvals[i] for i in range(M0)},
        "elimination_order": [model_names[i] for i in elimination_order],
        "final_pvalue": final_pvalue,
    }


def loss_differentials(df: pl.DataFrame) -> pl.DataFrame:
    model_names = df.columns
    L = jnp.array(df.to_numpy())
    T, M = L.shape
    d = L[:, :, None] - L[:, None, :]
    rows = []
    for i in range(M):
        for j in range(i + 1, M):
            rows.append(pl.DataFrame({
                "model_i": pl.Series([model_names[i]] * T),
                "model_j": pl.Series([model_names[j]] * T),
                "loss_diff": d[:, i, j].tolist(),
            }))
    return pl.concat(rows)


def calculate_loss(target: jnp.ndarray, forecast: jnp.ndarray, method: str = "MSE") -> jnp.ndarray:
    target = jnp.asarray(target, float)
    forecast = jnp.asarray(forecast, float)
    method = method.upper()
    if method == "MSE":
        return (target - forecast) ** 2
    if method == "MAE":
        return jnp.abs(target - forecast)
    if method == "QLIKE":
        return jnp.log(forecast) + target / forecast
    raise ValueError(f"Unknown loss method: {method}, must be 'MSE', 'MAE', or 'QLIKE'")

def diebold_mariano_test(loss1: jnp.ndarray, loss2: jnp.ndarray, h: int = 1, L_hac: int | None = None) -> dict:
    loss1 = jnp.asarray(loss1, float)
    loss2 = jnp.asarray(loss2, float)
    d = loss1 - loss2
    mean_diff = float(d.mean())
    if L_hac is None:
        L_hac = max(0, h - 1)
    var_mean = float(_hac_var(d, L=L_hac))
    hac_std_error = float(jnp.sqrt(var_mean))
    if hac_std_error < 1e-14:
        return {"dm_stat": 0.0, "p_value": 1.0, "mean_diff": mean_diff, "hac_std_error": 0.0}
    dm_stat = mean_diff / hac_std_error
    p_value = float(2.0 * (1.0 - jax.scipy.stats.norm.cdf(jnp.abs(dm_stat))))
    return {"dm_stat": dm_stat, "p_value": p_value, "mean_diff": mean_diff, "hac_std_error": hac_std_error}
