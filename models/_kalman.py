"""Implementation of the Kalman filter.
Implementation and notation based on:
Durbin, J. and Siem Jan Koopman (2012). Time Series Analysis by State Space Methods. OUP Oxford.
"""

import jax
import jax.numpy as np
from jax import lax
from jax.scipy.linalg import solve_triangular

def _filter(data: np.ndarray, dynamics: callable, params: dict, carry0: tuple) -> dict:
    """Kalman Filter implementation

    Args:
        data (ndarray): data in compatible ndarray
        dynamics (callable): function that specifies Zt, Tt, Rt and Qt
        params (dict): parameters of the model
        carry0 (tuple[float, float]): initial state prediction and variance
    """

    def _step(carry, yt):
        at, Pt, Zt, Tt, Ht, Rt, Qt, idx = carry

        missing = np.isnan(yt)
        Zt, Tt, Ht, Rt, Qt, dt, ct = dynamics(yt, at, Pt, params, Zt, Tt, Ht, Rt, Qt, idx)
        yt = np.where(missing, 0.0, yt)
        vt = yt - Zt @ at - dt
        ZP = Zt @ Pt
        Ft = ZP @ Zt.T + Ht
        L = np.linalg.cholesky(Ft)
        PtZtT = Pt @ Zt.T
        x = solve_triangular(L, PtZtT.T, lower=True)
        Kt = solve_triangular(L.T, x, lower=False).T
        att = at + Kt @ vt
        atp1 = Tt @ att + ct
        Ptt = Pt - Kt @ Zt @ Pt
        Ptp1 = Tt @ Ptt @ Tt.T + Rt @ Qt @ Rt.T
        Linv_v = solve_triangular(L, vt, lower=True)
        quad_t = Linv_v.T @ Linv_v
        logdet_t = 2.0 * np.sum(np.log(np.diag(L)))
        idx = idx + 1
        new_carry = (atp1, Ptp1, Zt, Tt, Ht, Rt, Qt, idx)
        return new_carry, (logdet_t, quad_t, at, Pt, att, Ptt, vt, Ft, Kt)

    _, (logdetF, quad, at, Pt, att, Ptt, vt, Ft, Kt) = lax.scan(_step, carry0, data)
    return {
        "logdetF": logdetF,"quad": quad,"a": at,
        "P": Pt, "att": att, "Ptt": Ptt, "v": vt,
        "F": Ft, "K": Kt,
    }

def _loglikelihood(filter_output: dict):
    """Without constant term"""
    return -0.5 * np.sum(filter_output["logdetF"] + filter_output["quad"])

def _fit(
    data: np.ndarray,
    initial_guess: dict,
    covariates: np.ndarray | None,
    carry_initial: tuple,
    _dynamics: callable,
    _link: callable = lambda x: x,
    _invlink: callable = lambda x: x,
    opt_options: dict = {}) -> dict:
    """
    Args:
        data (np.ndarray)
        params (dict)
        covariates (np.ndarray): _description_
        _dynamics (callable): _description_
        _link (callable | None, optional): function that maps uncostrained, ndarray parameters to
        constrained space and returns them in a dictionary. Defaults to None.
        _invlink (callable | None, optional): inverse of _link. Defaults to None.
    """
    maxiter = opt_options.get("maxiter", 500)
    learning_rate = opt_options.get("learning_rate", 1e-2)
    tol = opt_options.get("tol", 1e-6)
    beta1 = opt_options.get("beta1", 0.9)
    beta2 = opt_options.get("beta2", 0.999)
    eps = opt_options.get("eps", 1e-8)

    initial_guess = dict(initial_guess)
    initial_guess["covariates"] = covariates
    unc_params0 = _invlink(initial_guess)

    def _criterion(params):
        constr_params = _link(params)
        kf = _filter(data, _dynamics, constr_params | {"covariates": covariates}, carry_initial)
        return -_loglikelihood({"logdetF": kf["logdetF"], "quad": kf["quad"]})

    value_and_grad = jax.value_and_grad(_criterion)

    def _adam_step(state, _):
        params, m, v, i, prev_loss = state
        loss, g = value_and_grad(params)
        m = beta1 * m + (1.0 - beta1) * g
        v = beta2 * v + (1.0 - beta2) * (g * g)
        i1 = i + 1
        mhat = m / (1.0 - beta1**i1)
        vhat = v / (1.0 - beta2**i1)
        params = params - learning_rate * mhat / (np.sqrt(vhat) + eps)
        return (params, m, v, i1, loss), (loss, prev_loss)

    unc_params0 = np.asarray(unc_params0)
    m0 = np.zeros_like(unc_params0)
    v0 = np.zeros_like(unc_params0)

    loss0 = _criterion(unc_params0)
    state0 = (unc_params0, m0, v0, np.asarray(0, dtype=np.int32), loss0)

    stateT, losses = lax.scan(_adam_step, state0, xs=None, length=maxiter)
    unc_params = stateT[0]
    final_loss = stateT[4]

    params = _link(unc_params)
    params = dict(params)
    params["covariates"] = covariates
    kf = _filter(data, _dynamics, params, carry_initial)

    out = {
        "loglikelihood": -final_loss,
        "niter": maxiter,
        "is_converged": np.asarray(np.abs(losses[0][-1] - losses[1][-1]) < tol),
    }
    return params | kf | out

_fit = jax.jit( _fit, static_argnames=("_dynamics", "_link", "_invlink"))

def _simulation(fit_output: dict, nsim: int, dynamics: callable, npaths: int, key: jax.Array):
    Qt, Ht = fit_output["Q"][-1], fit_output["H"][-1]
    at, Pt = fit_output["a"][-1], fit_output["P"][-1]
    Zt, Tt = fit_output["Z"][-1], fit_output["T"][-1]
    Rt = fit_output["R"][-1]
    q = Qt.shape[0]
    p = Ht.shape[0]
    k1, k2 = jax.random.split(key, 2)
    eta_draws = jax.random.multivariate_normal(k1, np.zeros(q), Qt, shape=(nsim, npaths))
    eps_draws = jax.random.multivariate_normal(k2, np.zeros(p), Ht, shape=(nsim, npaths))
    at = np.broadcast_to(at, (npaths,) + at.shape)
    carry0 = (at, Pt, Zt, Tt, Ht, Rt, Qt, 0)
    def _sim_step(carry, noise_t):
        at, Pt, Zt, Tt, Ht, Rt, Qt, idx = carry
        eta_t, eps_t = noise_t

        y_dummy = np.zeros((p,), dtype=float)
        Zt, Tt, Ht, Rt, Qt, dt, ct = dynamics(y_dummy, at[0], Pt, fit_output, Zt, Tt, Ht, Rt, Qt, idx)

        dt = np.asarray(dt, dtype=float)
        ct = np.asarray(ct, dtype=float)

        y_t = np.einsum("ij,pj->pi", Zt, at) + dt + eps_t
        atp1 = np.einsum("ij,pj->pi", Tt, at) + ct + np.einsum("ij,pj->pi", Rt, eta_t)

        Ptp1 = Tt @ Pt @ Tt.T + Rt @ Qt @ Rt.T
        idx = idx + 1

        return (atp1, Ptp1, Zt, Tt, Ht, Rt, Qt, idx), y_t

    _, y_sim = lax.scan(_sim_step, carry0, (eta_draws, eps_draws), length=nsim)

    return {"y": y_sim, "eta": eta_draws, "eps": eps_draws}

_simulation = jax.jit(_simulation, static_argnames=("nsim", "npaths", "dynamics"))