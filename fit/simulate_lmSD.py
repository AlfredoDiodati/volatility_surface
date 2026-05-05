import os
import shutil

if shutil.which("nvidia-smi") is not None:
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

import json
import jax
import jax.numpy as jnp

from models.lmSD import simulate

print(f"Running on: {jax.devices()}")

PARAMS_PATH = "out/SPX/put/params_lmSD.json"

MONEYNESS = jnp.array([0.9, 0.98, 1.05, 1.15, 1.3, 1.5])
MATURITY  = jnp.array([10, 50, 100, 180]) / 255.0

HORIZONS      = [200, 1000, 3500]
SIGMA2_SCALES = [1.0, 10.0, 0.1]


def load_params(path):
    with open(path) as f:
        raw = json.load(f)
    return {
        "beta_bar": jnp.array(raw["beta_bar"]),
        "B":        jnp.array(raw["B"]),
        "A":        jnp.array(raw["A"]),
        "d":        jnp.array(raw["d"]),
        "sigma2":   jnp.array(raw["sigma2"]),
        "omega":    jnp.array(raw["omega"]),
        "C":        jnp.array(raw["C"]),
        "nu":       jnp.array(raw["nu"]),
    }


def make_Z_fixed(moneyness, maturity):
    mon_g, mat_g = jnp.meshgrid(moneyness, maturity, indexing="ij")
    mon_f = mon_g.ravel()
    mat_f = mat_g.ravel()
    N = mon_f.shape[0]
    level  = jnp.ones(N)
    bidx   = jnp.zeros(N)
    return jnp.stack([level, mon_f, mat_f, bidx], axis=1)


def main():
    params_base = load_params(PARAMS_PATH)
    Z_fixed     = make_Z_fixed(MONEYNESS, MATURITY)
    sigma2_base = params_base["sigma2"]

    sim_jit = jax.jit(simulate, static_argnames=("horizon", "score_buf_size"))

    key = jax.random.PRNGKey(42)
    results = {}

    for T in HORIZONS:
        for scale in SIGMA2_SCALES:
            params = params_base | {"sigma2": sigma2_base * scale}
            key, subkey = jax.random.split(key)
            y_sim, beta_sim = sim_jit(
                params, Z_fixed, horizon=T, key=subkey, score_buf_size=T
            )
            jax.effects_barrier()
            results[(T, scale)] = (y_sim, beta_sim)
            print(f"T={T:5d}  sigma2_scale={scale:.1f}  y_sim={y_sim.shape}  beta_sim={beta_sim.shape}")

    return results


if __name__ == "__main__":
    main()
