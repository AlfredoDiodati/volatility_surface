import time
import jax
import jax.numpy as jnp
import jaxopt
import optax
import numpy as np

from models._solver import lbfgs as our_lbfgs

jax.config.update("jax_platform_name", "gpu")

KEY = jax.random.PRNGKey(0)
DIM = 40
N_RUNS = 5
MAXITER = 2000
TOL = 1e-4


def make_criterion(dim, key):
    A = jax.random.normal(key, (dim, dim))
    H = A.T @ A / dim + 0.1 * jnp.eye(dim)
    b = jax.random.normal(key, (dim,))

    def criterion(theta):
        r = H @ theta - b
        return 0.5 * jnp.dot(r, r) + 0.01 * jnp.sum(jnp.log(jnp.cosh(theta)))

    return criterion


def bench(name, run_fn, n_runs=N_RUNS):
    run_fn()  # warmup / JIT compile
    jax.effects_barrier()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = run_fn()
        jax.effects_barrier()
        times.append(time.perf_counter() - t0)
    t_mean = np.mean(times) * 1e3
    t_std = np.std(times) * 1e3
    return t_mean, t_std, result


criterion = make_criterion(DIM, KEY)
theta0 = jnp.zeros(DIM)

# ── ours ──────────────────────────────────────────────────────────────────────
our_fn = jax.jit(lambda: our_lbfgs(criterion, theta0,
                                    opt_options={"tol": TOL}, maxiter=MAXITER))
t_our, s_our, (our_theta, our_niter, our_loss, our_conv) = bench("ours", our_fn)

# ── jaxopt ────────────────────────────────────────────────────────────────────
solver_jaxopt = jaxopt.LBFGS(fun=criterion, maxiter=MAXITER, tol=TOL,
                               history_size=10, implicit_diff=False)
jaxopt_fn = jax.jit(lambda: solver_jaxopt.run(theta0))
t_jaxopt, s_jaxopt, jaxopt_res = bench("jaxopt", jaxopt_fn)
jaxopt_niter = int(jaxopt_res.state.iter_num)
jaxopt_conv = bool(jaxopt_res.state.error < TOL)
jaxopt_loss = float(criterion(jaxopt_res.params))

# ── optax ─────────────────────────────────────────────────────────────────────
optax_solver = optax.lbfgs(memory_size=10)

@jax.jit
def run_optax():
    value_and_grad = jax.value_and_grad(criterion)

    def step(carry, _):
        params, state = carry
        value, grad = value_and_grad(params)
        updates, new_state = optax_solver.update(grad, state, params,
                                                  value=value,
                                                  grad=grad,
                                                  value_fn=criterion)
        new_params = optax.apply_updates(params, updates)
        return (new_params, new_state), value

    init_state = optax_solver.init(theta0)
    (final_params, final_state), losses = jax.lax.scan(
        step, (theta0, init_state), None, length=MAXITER)
    return final_params, losses

t_optax, s_optax, (optax_theta, optax_losses) = bench("optax", run_optax)
optax_loss = float(criterion(optax_theta))

print(f"\n{'Solver':<12} {'Time (ms)':>12} {'±':>6} {'Iters':>8} {'Conv':>6} {'Loss':>14}")
print("-" * 62)
print(f"{'ours':<12} {t_our:>12.2f} {s_our:>6.2f} {int(our_niter):>8} {str(bool(our_conv)):>6} {float(our_loss):>14.6f}")
print(f"{'jaxopt':<12} {t_jaxopt:>12.2f} {s_jaxopt:>6.2f} {jaxopt_niter:>8} {str(jaxopt_conv):>6} {jaxopt_loss:>14.6f}")
print(f"{'optax':<12} {t_optax:>12.2f} {s_optax:>6.2f} {'?':>8} {'?':>6} {optax_loss:>14.6f}")
