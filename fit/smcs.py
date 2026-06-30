import jax
import jax.numpy as jnp
import polars as pl


def _psi_e(c: jnp.ndarray, lam: jnp.ndarray) -> jnp.ndarray:
    cl = c * lam
    return (-jnp.log1p(-cl) - cl) / (c ** 2)


def _pairwise_diffs(L: jnp.ndarray) -> jnp.ndarray:
    return L[:, :, None] - L[:, None, :]


def _log_e_processes_strong(d: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    c = 2.0 * jnp.max(jnp.abs(d), axis=0) + eps
    lam = 1.0 / (2.0 * c)
    increments = jnp.maximum(1.0 + lam[None] * d, eps)
    return jnp.cumsum(jnp.log(increments), axis=0)


def _log_e_processes_uw(d: jnp.ndarray, c: float | None, eps: float = 1e-8) -> jnp.ndarray:
    T, M, _ = d.shape
    c_mat = jnp.full((M, M), float(c)) if c is not None else 2.0 * jnp.max(jnp.abs(d), axis=0) + eps
    lam = 1.0 / (2.0 * c_mat)
    psi = _psi_e(c_mat, lam)
    t_arr = jnp.arange(1, T + 1, dtype=float)[:, None, None]
    delta_hat = jnp.cumsum(d, axis=0) / t_arr
    V = jnp.cumsum(d ** 2, axis=0)
    return lam[None] * t_arr * delta_hat - psi[None] * V


def _log_e_adjust_single(log_e: jnp.ndarray) -> jnp.ndarray:
    m = log_e.shape[0]
    order = jnp.argsort(log_e)
    log_e_sorted = log_e[order]
    log_cumsum = jax.lax.associative_scan(jnp.logaddexp, log_e_sorted)
    log_S = jnp.concatenate([jnp.array([-jnp.inf]), log_cumsum[:-1]])
    k_idx = jnp.arange(m)[:, None]
    i_idx = jnp.arange(m)[None, :]
    log_ratio = jnp.logaddexp(log_e_sorted[:, None], log_S[None, :]) - jnp.log(i_idx + 1.0)
    log_ratio_masked = jnp.where(i_idx <= k_idx, log_ratio, jnp.inf)
    log_e_star_sorted = log_ratio_masked.min(axis=1)
    return jnp.full(m, jnp.inf).at[order].set(log_e_star_sorted)


def smcs(
    losses: pl.DataFrame | jnp.ndarray,
    alpha: float = 0.05,
    hypothesis: str = "strong",
    c: float | None = None,
    running_intersection: bool = False,
) -> dict:
    if isinstance(losses, pl.DataFrame):
        model_names = losses.columns
        L = jnp.array(losses.to_numpy())
    else:
        model_names = None
        L = jnp.asarray(losses, float)

    T, M = L.shape
    d = _pairwise_diffs(L)
    hypothesis = hypothesis.lower()

    if hypothesis == "strong":
        log_e_procs = _log_e_processes_strong(d)
    elif hypothesis in ("uniformly_weak", "weak"):
        log_e_procs = _log_e_processes_uw(d, c=c)
    else:
        raise ValueError("hypothesis must be 'strong', 'uniformly_weak', or 'weak'")

    diag_mask = jnp.eye(M, dtype=bool)[None]

    log_e_adjusted = None
    if hypothesis in ("strong", "uniformly_weak"):
        log_e_no_diag = jnp.where(diag_mask, -jnp.inf, log_e_procs)
        log_e_merged = jax.scipy.special.logsumexp(log_e_no_diag, axis=2) - jnp.log(M - 1.0)
        log_e_adjusted = jax.vmap(_log_e_adjust_single)(log_e_merged)
        in_smcs = log_e_adjusted < -jnp.log(alpha)
        if running_intersection:
            in_smcs = jax.lax.associative_scan(jnp.logical_and, in_smcs, axis=0)
    else:
        log_threshold = jnp.log(M * (M - 1.0) / alpha)
        log_e_safe = jnp.where(diag_mask, -jnp.inf, log_e_procs)
        in_smcs = (log_e_safe <= log_threshold).all(axis=2)
        log_e_adjusted = jnp.max(log_e_safe, axis=2) - jnp.log(M * (M - 1.0))

    final_idx = [int(i) for i in jnp.where(in_smcs[-1])[0]]

    if model_names is None:
        return {
            "smcs_models": final_idx,
            "smcs_history": in_smcs,
            "log_e_processes": log_e_procs,
            "log_e_adjusted": log_e_adjusted,
        }
    return {
        "smcs_models": [model_names[i] for i in final_idx],
        "smcs_history": {model_names[i]: in_smcs[:, i].tolist() for i in range(M)},
        "log_e_processes": {
            (model_names[i], model_names[j]): log_e_procs[:, i, j].tolist()
            for i in range(M) for j in range(M) if i != j
        },
    }