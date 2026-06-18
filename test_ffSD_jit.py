import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import jax
import jax.numpy as jnp

jax.config.update("jax_default_device", jax.devices("cpu")[0])
jax.config.update("jax_enable_x64", True)

from models.ff_SD import fit as ffSD_fit, forecast as ffSD_forecast, forecast_rolling_h as ffSD_forecast_rolling_h

T_TRAIN = 200
T_TEST  = 100
N       = 20
P_BASE  = 3
P       = P_BASE + 1
N_BUCK  = 5
K       = 3
FCST_H  = (5, 22)
ALPHA   = 0.05
OPT     = {"learning_rate": 1.0, "tol": 1e-2}
MAXITER = 1

key = jax.random.PRNGKey(0)
k1, k2, k3 = jax.random.split(key, 3)

y_train = jax.random.normal(k1, (T_TRAIN, N))
y_test  = jax.random.normal(k2, (T_TEST,  N))
Z_base  = jax.random.normal(k3, (T_TRAIN + T_TEST + max(FCST_H), N, P_BASE))
buck    = jnp.tile(jnp.arange(N_BUCK, dtype=jnp.float64), N // N_BUCK + 1)[:N]
buck_col = jnp.broadcast_to(buck[None, :, None], (T_TRAIN + T_TEST + max(FCST_H), N, 1))
Z_full  = jnp.concatenate([Z_base, buck_col], axis=-1)

Z_train    = Z_full[:T_TRAIN]
Z_test     = Z_full[T_TRAIN:T_TRAIN + T_TEST]
Z_test_ext = Z_full[T_TRAIN:]

ig = {
    "beta_bar": jnp.zeros(P),
    "A": 0.05 * jnp.eye(P),
    "sigma2": 1.0,
    "omega_load": jnp.zeros(N_BUCK),
    "eta": jnp.full(P, 0.4),
    "phi": jnp.full(P, 8.0),
    "C": 1e-3 * jnp.eye(P),
    "nu": jnp.array(10.0),
}


def time_jit(label, fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    compile_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    run_ms = (time.perf_counter() - t0) * 1000

    print(f"  {label:35s}  compile+run={compile_ms:7.0f}ms  run={run_ms:5.0f}ms")
    return out


print("=== Individual JITs ===")
fit_jit   = jax.jit(lambda y, Z, ig: ffSD_fit(y, Z, ig, K, opt_options=OPT, maxiter=MAXITER))
fcst_jit  = jax.jit(lambda r, Z, y: ffSD_forecast(r, Z, y, K, ALPHA))
rh_jit    = jax.jit(lambda r, Z, y: ffSD_forecast_rolling_h(r, Z, y, K, FCST_H))

r = time_jit("fit", fit_jit, y_train, Z_train, ig)
time_jit("forecast", fcst_jit, r, Z_test, y_test)
time_jit("forecast_rolling_h", rh_jit, r, Z_test_ext, y_test)

print()
print("=== Combined JIT ===")

def _run_ffSD(y_tr, Z_tr, ig, Z_te, y_te, Z_te_ext):
    r = ffSD_fit(y_tr, Z_tr, ig, K, opt_options=OPT, maxiter=MAXITER)
    y_hat, P_mean, VaR, oos_ll = ffSD_forecast(r, Z_te, y_te, K, ALPHA)
    preds_h = ffSD_forecast_rolling_h(r, Z_te_ext, y_te, K, FCST_H)
    return r, y_hat, P_mean, VaR, oos_ll, preds_h

combined_jit = jax.jit(_run_ffSD)
time_jit("fit+forecast+rolling_h", combined_jit, y_train, Z_train, ig, Z_test, y_test, Z_test_ext)
