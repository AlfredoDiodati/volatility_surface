"""
Run once to determine the fastest JAX compute device for this project.
The result is written to device.json and read at startup by all
entry-point scripts via _device.py.

The benchmark kernel is a lax.scan over a score-driven filter inner loop
(p x p linear solve + score update at each of T steps), representative of
the sequential structure of the model filters in models/.

VRAM threshold
--------------
At least VRAM_MIN_MiB of *free* GPU memory is required before the GPU
benchmark is attempted. This is set to 4000 MiB so that GPUs with 4 GiB
of total VRAM are excluded: OS/driver/display overhead prevents such cards
from ever having 4000 MiB free at the same time.

The NVIDIA GeForce GTX 1050 Ti (4096 MiB total) on the development machine
is a known example of a card that reliably runs out of memory during the
filter scans in this project despite appearing to have enough free VRAM at
process start. It is correctly excluded by this threshold.
"""

import json
import os
import subprocess
import time
from pathlib import Path

VRAM_MIN_MiB = 4000
OUTPUT = Path(__file__).parent / "device.json"

BENCH_T = 2000
BENCH_N = 30
BENCH_P = 4
BENCH_REPS = 3


def _free_vram_mib():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
        values = [int(x.strip()) for x in out.strip().splitlines() if x.strip()]
        return max(values) if values else 0
    except Exception:
        return 0


def _bench(device):
    import jax
    import jax.numpy as jnp
    from jax import lax

    with jax.default_device(device):
        key = jax.random.PRNGKey(0)
        y = jax.random.normal(key, (BENCH_T, BENCH_N))
        Z = jax.random.normal(key, (BENCH_T, BENCH_N, BENCH_P))
        mask = jnp.ones((BENCH_T, BENCH_N))
        C_inv = jnp.ones(BENCH_P)
        h_inv = 1.0
        A = jnp.eye(BENCH_P) * 0.1

        def step(beta, inputs):
            y_t, Z_t, mask_t = inputs
            Z_m = Z_t * mask_t[:, None]
            eps = (y_t - Z_t @ beta) * mask_t
            G = h_inv * (Z_m.T @ eps)
            V = h_inv * (Z_m.T @ Z_m)
            S = jnp.diag(C_inv) + V
            s = jnp.linalg.solve(S, G)
            return beta + A @ s, None

        lax.scan(step, jnp.zeros(BENCH_P), (y, Z, mask))[0].block_until_ready()

        t0 = time.perf_counter()
        for _ in range(BENCH_REPS):
            lax.scan(step, jnp.zeros(BENCH_P), (y, Z, mask))[0].block_until_ready()
        return (time.perf_counter() - t0) / BENCH_REPS


def main():
    if subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0:
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

    import jax

    try:
        gpus = jax.devices("gpu")
    except RuntimeError:
        gpus = []

    if not gpus:
        print("No GPU detected. Selecting CPU.")
        OUTPUT.write_text(json.dumps({"device": "cpu"}))
        return

    free_vram = _free_vram_mib()
    print(f"GPU detected: {gpus[0]}. Free VRAM: {free_vram} MiB (threshold: {VRAM_MIN_MiB} MiB)")

    if free_vram < VRAM_MIN_MiB:
        print("Insufficient free VRAM. Selecting CPU.")
        OUTPUT.write_text(json.dumps({"device": "cpu"}))
        return

    cpu = jax.devices("cpu")[0]
    gpu = gpus[0]

    print(f"Benchmarking CPU ({cpu}) ...")
    cpu_time = _bench(cpu)
    print(f"  {cpu_time * 1000:.1f} ms/call")

    print(f"Benchmarking GPU ({gpu}) ...")
    try:
        gpu_time = _bench(gpu)
        print(f"  {gpu_time * 1000:.1f} ms/call")
    except Exception as e:
        print(f"  GPU benchmark failed: {e}. Selecting CPU.")
        OUTPUT.write_text(json.dumps({"device": "cpu"}))
        return

    if gpu_time < cpu_time:
        print(f"GPU wins ({gpu_time * 1000:.1f} ms vs {cpu_time * 1000:.1f} ms). Selecting GPU.")
        OUTPUT.write_text(json.dumps({"device": "gpu"}))
    else:
        print(f"CPU wins ({cpu_time * 1000:.1f} ms vs {gpu_time * 1000:.1f} ms). Selecting CPU.")
        OUTPUT.write_text(json.dumps({"device": "cpu"}))

    print(f"Result written to {OUTPUT}")


if __name__ == "__main__":
    main()
