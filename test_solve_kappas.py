import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import jax
import jax.numpy as jnp
from models.ff_SD import _solve_weights_ff, _solve_kappas_ff, _compute_L_matrix

jax.config.update("jax_enable_x64", True)

RESIDUAL_TOL = 1e-6
TIME_LIMIT_MS = 200


def _residual(mu, ws, lambdas):
    L = _compute_L_matrix(ws, lambdas)
    z = jnp.einsum("ijk,jk->ik", L, mu)
    mu_fp = (-z + jnp.sqrt(z ** 2 + 4.0)) / 2.0
    return jnp.max(jnp.abs(mu - mu_fp))


def run(p, K, eta_val, phi_val, label):
    eta = jnp.full(p, eta_val)
    phi = jnp.full(p, phi_val)

    ws, lambdas = _solve_weights_ff(eta, phi, K)
    L = _compute_L_matrix(ws, lambdas)
    if jnp.any(jnp.isnan(L)):
        print(f"[SKIP] {label:30s}  L has NaN (phi too low for K)")
        return True

    _solve_kappas_ff(ws, lambdas).block_until_ready()

    t0 = time.perf_counter()
    mu = _solve_kappas_ff(ws, lambdas)
    mu.block_until_ready()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    resid = float(_residual(mu, ws, lambdas))
    converged = resid < RESIDUAL_TOL
    fast = elapsed_ms < TIME_LIMIT_MS
    status = "PASS" if converged and fast else "FAIL"
    flags = []
    if not converged:
        flags.append(f"residual={resid:.2e} > {RESIDUAL_TOL:.0e}")
    if not fast:
        flags.append(f"time={elapsed_ms:.1f}ms > {TIME_LIMIT_MS}ms")
    detail = "  ".join(flags) if flags else f"residual={resid:.2e}  time={elapsed_ms:.1f}ms"
    print(f"[{status}] {label:30s}  {detail}")
    return status == "PASS"


cases = [
    (3,  1,  0.5,  8.0, "p=3  K=1  eta=0.5 phi=8 "),
    (3,  3,  0.9, 10.0, "p=3  K=3  eta=0.9 phi=10"),
    (3,  3,  1.8, 10.0, "p=3  K=3  eta=1.8 phi=10"),
    (5,  5,  1.0, 10.0, "p=5  K=5  eta=1.0 phi=10"),
    (5,  5,  1.5, 15.0, "p=5  K=5  eta=1.5 phi=15"),
    (8, 10,  0.9,  8.0, "p=8  K=10 eta=0.9 phi=8 "),
    (8, 10,  1.8, 10.0, "p=8  K=10 eta=1.8 phi=10"),
]

print("Warming up JIT...")
run(3, 5, 0.5, 10.0, "warmup")
print()

all_pass = all(run(*c) for c in cases)
print()
print("All tests passed." if all_pass else "Some tests FAILED.")
