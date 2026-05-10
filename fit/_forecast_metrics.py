import jax.numpy as jnp

def per_step_mse(log_iv: jnp.ndarray, log_iv_forecast: jnp.ndarray, mask: jnp.ndarray) -> list:
    per_t = jnp.where(mask, (log_iv - log_iv_forecast) ** 2, 0.0).sum(axis=1) / mask.sum(axis=1)
    return per_t.tolist()

def per_step_mae(log_iv: jnp.ndarray, log_iv_forecast: jnp.ndarray, mask: jnp.ndarray) -> list:
    per_t = jnp.where(mask, jnp.abs(log_iv - log_iv_forecast), 0.0).sum(axis=1) / mask.sum(axis=1)
    return per_t.tolist()

def compute_mse(log_iv: jnp.ndarray, log_iv_forecast: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """
    log_iv:          (T_star, N_max) observed log implied volatilities
    log_iv_forecast: (T_star, N_max) one-step-ahead forecasts
    mask:            (T_star, N_max) boolean, True where observations exist
    MSE = (1 / T_star) * sum_t [ (1 / N_t) * sum_i mask_{i,t} * (log_iv - log_iv_forecast)^2 ]
    """
    squared_errors = jnp.where(mask, (log_iv - log_iv_forecast) ** 2, 0.0)
    counts_per_t = mask.sum(axis=1)
    per_t_mean = squared_errors.sum(axis=1) / counts_per_t
    return per_t_mean.mean()

def compute_mae(log_iv: jnp.ndarray, log_iv_forecast: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """
    log_iv: (T_star, N_max) observed log implied volatilities
    log_iv_forecast: (T_star, N_max) one-step-ahead forecasts
    mask: (T_star, N_max) boolean, True where observations exist

    MAE = (1 / T_star) * sum_t [ (1 / N_t) * sum_i mask_{i,t} * |log_iv - log_iv_forecast| ]
    """
    absolute_errors = jnp.where(mask, jnp.abs(log_iv - log_iv_forecast), 0.0)
    counts_per_t = mask.sum(axis=1)
    per_t_mean = absolute_errors.sum(axis=1) / counts_per_t
    return per_t_mean.mean()

def per_step_aic(neg_loglik_series, n_params: int, n_steps: int) -> list:
    penalty = 2.0 * n_params / n_steps
    return (2.0 * jnp.array(neg_loglik_series) + penalty).tolist()

def per_step_bic(neg_loglik_series, n_params: int, n_obs: int, n_steps: int) -> list:
    penalty = n_params * jnp.log(n_obs) / n_steps
    return (2.0 * jnp.array(neg_loglik_series) + penalty).tolist()

def compute_aic(loglik: jnp.ndarray, num_params: int) -> jnp.ndarray:
    """
    loglik:     scalar total log-likelihood (out of sample)
    num_params: number of estimated static parameters k

    AIC = -2 * loglik + 2 * k
    """
    return -2.0 * loglik + 2.0 * num_params

def compute_bic(loglik: jnp.ndarray, num_params: int, num_obs: int) -> jnp.ndarray:
    """
    loglik:     scalar total log-likelihood (out of sample)
    num_params: number of estimated static parameters k
    num_obs:    total number of scalar observations n

    BIC = -2 * loglik + k * log(n)
    """
    return -2.0 * loglik + num_params * jnp.log(num_obs)