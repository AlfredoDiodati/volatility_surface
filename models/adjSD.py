import jax
import jax.numpy as np
from jax import lax
from jax.scipy.special import gammaln

def _filter(y_masked, base_covariates, bucket_indices, mask_f, params, state0=None):
    B = params["B"]
    A = params["A"]
    sigma2 = params["sigma2"]
    omega = params["omega"]
    C = params["C"]
    nu = params["nu"]
    beta_bar = params["beta_bar"]
    p = beta_bar.shape[0]
    h_inv = 1.0 / sigma2
    IminusB = np.eye(p) - B
    C_diag = C
    C_inv = 1.0 / C_diag

    def _step(beta_t, inputs):
        y_t, base_t, bidx_t, mask_t = inputs
        omega_col = omega[bidx_t]
        Z_t = np.concatenate([base_t, omega_col[:, None]], axis=-1)
        Z_mask = Z_t * mask_t[:, None]
        eps_t = (y_t - Z_t @ beta_t) * mask_t
        N_t = np.sum(mask_t)
        ZtZ = Z_mask.T @ Z_mask
        Zte = Z_mask.T @ eps_t
        ZtHinvZ = h_inv * ZtZ
        gls_step = np.linalg.solve(ZtHinvZ + 1e-8 * np.eye(p), h_inv * Zte)
        mahal_H = h_inv * (eps_t @ eps_t)
        weight = (1.0 + (N_t + 2.0) / nu) / (1.0 + mahal_H / (nu - 2.0))
        xi = A @ (weight * gls_step)
        S_diag = C_inv + h_inv * np.diag(ZtZ)
        log_det_F = N_t * np.log(sigma2) + np.sum(np.log(C_diag)) + np.sum(np.log(S_diag))
        v = Zte / S_diag
        quad = h_inv * (eps_t @ eps_t) - h_inv**2 * (v @ v)
        ll_t = (
            gammaln((nu + N_t) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * N_t * np.log((nu - 2.0) * np.pi)
            - 0.5 * log_det_F
            - 0.5 * (nu + N_t) * np.log(1.0 + quad / (nu - 2.0))
        )
        beta_next = IminusB @ beta_bar + B @ beta_t + xi
        return beta_next, (ll_t, beta_t)

    init = beta_bar if state0 is None else state0

    beta_T, (lls, betas_prev) = lax.scan(
        _step, init,
        (y_masked, base_covariates, bucket_indices, mask_f),
    )
    betas = np.concatenate([betas_prev, beta_T[None]], axis=0)
    return betas, lls, beta_T

def fit(
    data: np.ndarray,
    covariates: np.ndarray,
    initial_guess: dict,
    opt_options: dict | None = None,
    maxiter: int = 5000,
):
    data = np.asarray(data, dtype=float)
    covariates = np.asarray(covariates, dtype=float)
    p = initial_guess["beta_bar"].shape[0]
    n_buckets = initial_guess["omega"].shape[0]
    maxiter = int(maxiter)
    opt_options = opt_options or {}
    lr = opt_options.get("learning_rate", 1e-2)
    tol = opt_options.get("tol", 1e-6)
    b1 = opt_options.get("beta1", 0.9)
    b2 = opt_options.get("beta2", 0.999)
    eps = opt_options.get("eps", 1e-8)
    mask_bool = ~np.isnan(data)
    y_masked = np.where(mask_bool, data, 0.0)
    mask_f = mask_bool.astype(float)
    base_covariates = covariates[:, :, :-1]
    bucket_indices = covariates[:, :, -1].astype(np.int32)

    def _link(theta):
        idx = 0
        beta_bar = theta[idx:idx + p]
        idx += p
        B_diag = np.tanh(theta[idx:idx + p])
        B = np.diag(B_diag)
        idx += p
        A = np.diag(theta[idx:idx + p])
        idx += p
        sigma2 = np.exp(theta[idx])
        idx += 1
        omega = np.concatenate([np.zeros(1), theta[idx:idx + n_buckets - 1]])
        idx += n_buckets - 1
        C_diag = theta[idx:idx + p]
        idx += p
        nu = np.exp(theta[idx]) + 2.0
        return {"beta_bar": beta_bar, "B": B, "A": A, "sigma2": sigma2,
                "omega": omega, "C": C_diag, "nu": nu}

    def _invlink(params):
        B_diag = np.diag(params["B"])
        unc_B = np.arctanh(B_diag)
        unc_A = np.diag(params["A"])
        unc_s2 = np.log(params["sigma2"])
        unc_omega = params["omega"][1:]
        unc_C = params["C"]
        unc_nu = np.log(params["nu"] - 2.0)
        return np.concatenate([
            params["beta_bar"], unc_B, unc_A,
            np.array([unc_s2]), unc_omega,
            unc_C, np.array([unc_nu]),
        ])

    def _criterion(theta):
        params = _link(theta)
        _, lls, _ = _filter(y_masked, base_covariates, bucket_indices, mask_f, params)
        return -np.sum(lls)

    value_and_grad = jax.value_and_grad(_criterion)

    def _adam_step(state):
        theta, m, v, i, prev_loss, converged = state
        loss, g = value_and_grad(theta)
        m_new = b1 * m + (1.0 - b1) * g
        v_new = b2 * v + (1.0 - b2) * g * g
        i1 = i + 1
        mhat = m_new / (1.0 - b1 ** i1)
        vhat = v_new / (1.0 - b2 ** i1)
        theta_new = theta - lr * mhat / (np.sqrt(vhat) + eps)
        converged_new = np.abs(loss - prev_loss) / (np.abs(prev_loss) + 1.0) < tol
        return (theta_new, m_new, v_new, i1, loss, converged_new)

    def _not_converged(state):
        _, _, _, i, _, converged = state
        return (i < maxiter) & ~converged

    theta0 = np.asarray(_invlink(initial_guess))
    state0 = (
        theta0,
        np.zeros_like(theta0),
        np.zeros_like(theta0),
        np.asarray(0, dtype=np.int32),
        np.asarray(np.inf),
        np.asarray(False),
    )
    theta_opt, _, _, niter, final_loss, is_converged = lax.while_loop(
        _not_converged, _adam_step, state0
    )
    params_opt = _link(theta_opt)
    betas, _, beta_T = _filter(y_masked, base_covariates, bucket_indices, mask_f, params_opt)
    return params_opt | {
        "betas": betas,
        "beta_T": beta_T,
        "log_likelihood": -final_loss,
        "niter": niter,
        "is_converged": is_converged,
    }

fit = jax.jit(fit, static_argnames=("maxiter",))

def forecast(fit_result, covariates, q_alpha):
    state0 = fit_result["beta_T"]

    covariates = np.asarray(covariates, dtype=float)
    H = covariates.shape[0]
    n_obs = covariates.shape[1]
    base_covariates = covariates[:, :, :-1]
    bucket_indices = covariates[:, :, -1].astype(np.int32)

    y_zeros = np.zeros((H, n_obs))
    mask_zeros = np.zeros((H, n_obs))

    betas, _, _ = _filter(y_zeros, base_covariates, bucket_indices, mask_zeros, fit_result, state0=state0)

    omega_cols = fit_result["omega"][bucket_indices]
    Z = np.concatenate([base_covariates, omega_cols[:, :, None]], axis=-1)
    predictions = np.einsum("hni,hi->hn", Z, betas[:H])

    P = predictions.mean(axis=1)

    z_sum = Z.sum(axis=1)
    F_sum = n_obs * fit_result["sigma2"] + (z_sum ** 2 * fit_result["C"]).sum(axis=1)
    VaR = P + q_alpha / n_obs * np.sqrt(F_sum)

    return predictions, betas, P, VaR

forecast = jax.jit(forecast)