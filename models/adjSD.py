import jax
import jax.numpy as np
from jax import lax
from jax.scipy.special import gammaln
from jax.scipy.linalg import solve_triangular
from models.ss import _link_stable_matrix, _invlink_stable_matrix


def _filter(y_masked, base_covariates, bucket_indices, mask_f, params):
    B        = params["B"]
    A        = params["A"]
    sigma2   = params["sigma2"]
    omega    = params["omega"]
    C        = params["C"]
    nu       = params["nu"]
    beta_bar = params["beta_bar"]
    p        = beta_bar.shape[0]

    h_inv   = 1.0 / sigma2
    IminusB = np.eye(p) - B

    C_reg     = C + 1e-8 * np.eye(p)
    L_C       = np.linalg.cholesky(C_reg)
    log_det_C = 2.0 * np.sum(np.log(np.diag(L_C)))
    C_inv     = np.linalg.inv(C_reg)

    def _step(beta_t, inputs):
        y_t, base_t, bidx_t, mask_t = inputs

        omega_col = omega[bidx_t]
        Z_t    = np.concatenate([base_t, omega_col[:, None]], axis=-1)
        Z_mask = Z_t * mask_t[:, None]
        eps_t  = (y_t - Z_t @ beta_t) * mask_t
        N_t    = np.sum(mask_t)

        ZtZ      = Z_mask.T @ Z_mask
        Zte      = Z_mask.T @ eps_t
        ZtHinvZ  = h_inv * ZtZ
        gls_step = np.linalg.solve(ZtHinvZ + 1e-8 * np.eye(p), h_inv * Zte)
        mahal_H  = h_inv * (eps_t @ eps_t)
        weight   = (1.0 + (N_t + 2.0) / nu) / (1.0 + mahal_H / (nu - 2.0))
        xi       = A @ (weight * gls_step)

        S     = C_inv + h_inv * ZtZ
        L_S   = np.linalg.cholesky(S + 1e-8 * np.eye(p))
        log_det_F = (N_t * np.log(sigma2)
                     + log_det_C
                     + 2.0 * np.sum(np.log(np.diag(L_S))))

        v    = solve_triangular(L_S, Zte, lower=True)
        quad = h_inv * (eps_t @ eps_t) - h_inv**2 * (v @ v)

        ll_t = (gammaln((nu + N_t) / 2.0)
                - gammaln(nu / 2.0)
                - 0.5 * N_t * np.log((nu - 2.0) * np.pi)
                - 0.5 * log_det_F
                - 0.5 * (nu + N_t) * np.log(1.0 + quad / (nu - 2.0)))

        beta_next = IminusB @ beta_bar + B @ beta_t + xi
        return beta_next, (ll_t, beta_t)

    beta_T, (lls, betas_prev) = lax.scan(
        _step, beta_bar,
        (y_masked, base_covariates, bucket_indices, mask_f),
    )
    betas = np.concatenate([betas_prev, beta_T[None]], axis=0)
    return betas, lls


def fit(
    data: np.ndarray,
    covariates: np.ndarray,
    initial_guess: dict,
    opt_options: dict | None = None,
    maxiter: int = 5000,
):
    data       = np.asarray(data,       dtype=float)
    covariates = np.asarray(covariates, dtype=float)

    p         = initial_guess["beta_bar"].shape[0]
    n_buckets = initial_guess["omega"].shape[0]

    maxiter     = int(maxiter)
    opt_options = opt_options or {}
    lr  = opt_options.get("learning_rate", 1e-2)
    tol = opt_options.get("tol",           1e-6)
    b1  = opt_options.get("beta1",         0.9)
    b2  = opt_options.get("beta2",         0.999)
    eps = opt_options.get("eps",           1e-8)

    mask_bool       = ~np.isnan(data)
    y_masked        = np.where(mask_bool, data, 0.0)
    mask_f          = mask_bool.astype(float)
    base_covariates = covariates[:, :, :-1]
    bucket_indices  = covariates[:, :, -1].astype(np.int32)

    tril_r, tril_c = np.tril_indices(p)
    len_uncB = (p * (p - 1) // 2) * 2 + p
    n_chol   = p * (p + 1) // 2

    def _link(theta):
        idx      = 0
        beta_bar = theta[idx:idx + p];                                  idx += p
        B        = _link_stable_matrix(theta[idx:idx + len_uncB], p);  idx += len_uncB
        A        = np.diag(np.exp(theta[idx:idx + p]));                 idx += p
        sigma2   = np.exp(theta[idx]);                                  idx += 1
        omega    = np.concatenate(
                       [np.zeros(1), theta[idx:idx + n_buckets - 1]]); idx += n_buckets - 1
        L        = np.zeros((p, p)).at[tril_r, tril_c].set(
                       theta[idx:idx + n_chol]);                        idx += n_chol
        C        = L @ L.T
        nu       = np.exp(theta[idx]) + 2.0
        return {"beta_bar": beta_bar, "B": B, "A": A,
                "sigma2": sigma2, "omega": omega, "C": C, "nu": nu}

    def _invlink(params):
        unc_B     = _invlink_stable_matrix(params["B"], p)
        unc_A     = np.log(np.diag(params["A"]))
        unc_s2    = np.log(params["sigma2"])
        unc_omega = params["omega"][1:]
        L_C       = np.linalg.cholesky(params["C"] + 1e-8 * np.eye(p))
        unc_C     = L_C[tril_r, tril_c]
        unc_nu    = np.log(params["nu"] - 2.0)
        return np.concatenate([params["beta_bar"], unc_B, unc_A,
                                np.array([unc_s2]), unc_omega,
                                unc_C, np.array([unc_nu])])

    def _criterion(theta):
        params = _link(theta)
        _, lls = _filter(y_masked, base_covariates, bucket_indices, mask_f, params)
        return -np.sum(lls)

    value_and_grad = jax.value_and_grad(_criterion)

    def _adam_step(state):
        theta, m, v, i, prev_loss, converged = state
        loss, g = value_and_grad(theta)
        m_new = b1 * m + (1.0 - b1) * g
        v_new = b2 * v + (1.0 - b2) * g * g
        i1    = i + 1
        mhat  = m_new / (1.0 - b1 ** i1)
        vhat  = v_new / (1.0 - b2 ** i1)
        theta_new     = theta - lr * mhat / (np.sqrt(vhat) + eps)
        converged_new = np.abs(loss - prev_loss) / (np.abs(prev_loss) + 1.0) < tol
        return (theta_new, m_new, v_new, i1, loss, converged_new)

    def _not_converged(state):
        _, _, _, i, _, converged = state
        return (i < maxiter) & ~converged

    theta0 = np.asarray(_invlink(initial_guess))
    state0 = (theta0,
              np.zeros_like(theta0),
              np.zeros_like(theta0),
              np.asarray(0, dtype=np.int32),
              np.asarray(np.inf),
              np.asarray(False))

    theta_opt, _, _, niter, final_loss, is_converged = lax.while_loop(
        _not_converged, _adam_step, state0
    )

    params_opt = _link(theta_opt)
    betas, _   = _filter(y_masked, base_covariates, bucket_indices, mask_f, params_opt)

    return params_opt | {
        "betas":          betas,
        "log_likelihood": -final_loss,
        "niter":          niter,
        "is_converged":   is_converged,
    }


fit = jax.jit(fit, static_argnames=("maxiter",))
