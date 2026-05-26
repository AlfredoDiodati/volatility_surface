import os, sys, time
import shutil
if shutil.which("nvidia-smi") is not None:
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import numpy as np
import jax
import jax.numpy as jnp
import jaxopt

from models._solver import lbfgs as our_lbfgs
from models.ff_SD import _filter, _solve_weights_ff
from models.lmSD import simulate
from fit.MC_lmSD import load_params, make_Z_fixed, cold_ffSD, PARAMS_PATH

MAXITER = 2000
TOL     = 1e-4
K       = 1
T       = 400
N_RUNS  = 5

# ── build data ────────────────────────────────────────────────────────────────
params_true = load_params(PARAMS_PATH)
Z_fixed     = make_Z_fixed()

_sim_jit = jax.jit(simulate, static_argnames=("horizon", "score_buf_size"))
key       = jax.random.PRNGKey(42)
y_sim, _  = _sim_jit(params_true, Z_fixed, horizon=T, key=key, score_buf_size=T)
jax.effects_barrier()

y_train = np.asarray(y_sim)
Z_np    = np.asarray(Z_fixed)

# ── replicate ff_SD.fit internals ─────────────────────────────────────────────
p          = 4
n_buckets  = int(Z_fixed[:, -1].max()) + 1

mask_bool       = ~np.isnan(y_train)
y_masked        = jnp.where(mask_bool, y_train, 0.0)
mask_f          = mask_bool.astype(float)
base_covariates = jnp.asarray(Z_np[:, :-1])[None].repeat(T, axis=0)
bucket_indices  = jnp.asarray(Z_np[:, -1].astype(int))[None].repeat(T, axis=0)

ig      = cold_ffSD(jnp.asarray(y_train), Z_fixed)

def _link(theta):
    idx = 0
    beta_bar   = theta[idx:idx+p];  idx += p
    A          = jnp.diag(theta[idx:idx+p]);  idx += p
    sigma2     = jnp.exp(theta[idx]);  idx += 1
    omega_load = jnp.concatenate([jnp.zeros(1), theta[idx:idx+n_buckets-1]]);  idx += n_buckets-1
    eta        = jnp.exp(theta[idx:idx+p]);  idx += p
    alpha      = jnp.full(p, jax.nn.softplus(theta[idx]) + 1.0);  idx += 1
    C          = jnp.diag(jnp.exp(theta[idx:idx+p]));  idx += p
    nu         = jnp.exp(theta[idx]) + 2.0
    return {"beta_bar": beta_bar, "A": A, "sigma2": sigma2,
            "omega_load": omega_load, "eta": eta, "alpha": alpha, "C": C, "nu": nu}

def _invlink(params):
    unc_alpha = jnp.log(jnp.exp(params["alpha"][0] - 1.0) - 1.0)
    return jnp.concatenate([
        params["beta_bar"],
        jnp.diag(params["A"]),
        jnp.array([jnp.log(params["sigma2"])]),
        params["omega_load"][1:],
        jnp.log(params["eta"]),
        jnp.array([unc_alpha]),
        jnp.log(jnp.diag(params["C"])),
        jnp.array([jnp.log(params["nu"] - 2.0)]),
    ])

def criterion(theta):
    params = _link(theta)
    _, lls, _ = _filter(y_masked, base_covariates, bucket_indices, mask_f,
                        params, K, jnp.zeros((K+1, p)))
    return -jnp.sum(lls)

theta0 = jnp.asarray(_invlink(ig))

# ── benchmark helper ──────────────────────────────────────────────────────────
def bench(name, fn, n_runs=N_RUNS):
    fn()
    jax.effects_barrier()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = fn()
        jax.effects_barrier()
        times.append(time.perf_counter() - t0)
    t_mean = np.mean(times) * 1e3
    t_std  = np.std(times)  * 1e3
    return t_mean, t_std, result

# ── ours ──────────────────────────────────────────────────────────────────────
our_fn = jax.jit(lambda: our_lbfgs(criterion, theta0,
                                    opt_options={"tol": TOL}, maxiter=MAXITER))
t_our, s_our, (our_theta, our_niter, our_loss, our_conv) = bench("ours", our_fn)

# ── jaxopt (backtracking = same line search as ours) ─────────────────────────
solver_bt = jaxopt.LBFGS(fun=criterion, maxiter=MAXITER, tol=TOL,
                          history_size=10, linesearch="backtracking",
                          implicit_diff=False)
jaxopt_bt_fn = jax.jit(lambda: solver_bt.run(theta0))
t_jbt, s_jbt, res_jbt = bench("jaxopt-bt", jaxopt_bt_fn)
jbt_loss = float(criterion(res_jbt.params))

# ── jaxopt (zoom = Wolfe, their default) ─────────────────────────────────────
solver_zoom = jaxopt.LBFGS(fun=criterion, maxiter=MAXITER, tol=TOL,
                            history_size=10, linesearch="zoom",
                            implicit_diff=False)
jaxopt_zoom_fn = jax.jit(lambda: solver_zoom.run(theta0))
t_jzoom, s_jzoom, res_jzoom = bench("jaxopt-zoom", jaxopt_zoom_fn)
jzoom_loss = float(criterion(res_jzoom.params))

# ── results ───────────────────────────────────────────────────────────────────
print(f"\nT={T}  K={K}  maxiter={MAXITER}  tol={TOL}  n_runs={N_RUNS}\n")
print(f"{'Solver':<16} {'Time (ms)':>12} {'±':>8} {'Conv':>6} {'Loss':>14}")
print("-" * 62)
print(f"{'ours':<16} {t_our:>12.1f} {s_our:>8.1f} {str(bool(our_conv)):>6} {float(our_loss):>14.4f}")
print(f"{'jaxopt-bt':<16} {t_jbt:>12.1f} {s_jbt:>8.1f} {'?':>6} {jbt_loss:>14.4f}")
print(f"{'jaxopt-zoom':<16} {t_jzoom:>12.1f} {s_jzoom:>8.1f} {'?':>6} {jzoom_loss:>14.4f}")
