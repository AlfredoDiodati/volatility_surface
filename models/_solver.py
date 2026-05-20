import jax
import jax.numpy as jnp
from jax import lax

def adam(criterion, theta0, opt_options=None, maxiter=5000):
    opt_options = opt_options or {}
    lr = opt_options.get("learning_rate", 1e-2)
    tol = opt_options.get("tol", 1e-6)
    b1 = opt_options.get("beta1", 0.9)
    b2 = opt_options.get("beta2", 0.999)
    eps = opt_options.get("eps", 1e-8)
    maxiter = int(maxiter)

    theta0 = jnp.asarray(theta0)
    if not jnp.issubdtype(theta0.dtype, jnp.floating):
        theta0 = theta0.astype(jnp.float64)
    float_info = jnp.finfo(theta0.dtype)
    value_and_grad = jax.value_and_grad(criterion)

    def _step(state):
        theta, m, v, b1t, b2t, i, loss_finite, best_theta, best_loss, converged = state
        loss, g = value_and_grad(theta)

        m_new = b1 * m + (1.0 - b1) * g
        v_new = b2 * v + (1.0 - b2) * g * g
        b1t_new = b1t * b1
        b2t_new = b2t * b2

        mhat = m_new / (1.0 - b1t_new)
        vhat = v_new / (1.0 - b2t_new)

        i1 = i + 1
        lr_t = lr * 0.5 * (1.0 + jnp.cos(jnp.pi * i1 / maxiter))
        theta_new = theta - lr_t * mhat / (jnp.sqrt(vhat) + eps)

        best_theta_new = jnp.where(loss < best_loss, theta, best_theta)
        best_loss_new = jnp.minimum(loss, best_loss)
        converged_new = jnp.linalg.norm(g) / jnp.sqrt(g.size) < tol

        return (theta_new, m_new, v_new, b1t_new, b2t_new, i1, jnp.isfinite(loss), best_theta_new, best_loss_new, converged_new)

    def _not_converged(state):
        _, _, _, _, _, i, loss_finite, _, _, converged = state
        return (i < maxiter) & ~converged & loss_finite

    state0 = (
        theta0,
        jnp.zeros_like(theta0),
        jnp.zeros_like(theta0),
        jnp.asarray(1.0, dtype=theta0.dtype),
        jnp.asarray(1.0, dtype=theta0.dtype),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(True),
        theta0,
        jnp.asarray(float_info.max, dtype=theta0.dtype),
        jnp.asarray(False),
    )

    _, _, _, _, _, niter, _, best_theta, best_loss, is_converged = lax.while_loop(
        _not_converged, _step, state0
    )

    return best_theta, niter, best_loss, is_converged