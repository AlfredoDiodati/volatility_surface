import json
from pathlib import Path

import jax

_CONFIG = Path(__file__).parent / "device.json"


def get_device():
    if not _CONFIG.exists():
        print(
            "[device] device.json not found — run benchmark_device.py first. "
            "Defaulting to CPU."
        )
        return jax.devices("cpu")[0]

    name = json.loads(_CONFIG.read_text()).get("device", "cpu")

    if name == "gpu":
        try:
            return jax.devices("gpu")[0]
        except RuntimeError:
            print("[device] GPU selected in device.json but no GPU is available. Defaulting to CPU.")
            return jax.devices("cpu")[0]

    return jax.devices("cpu")[0]
