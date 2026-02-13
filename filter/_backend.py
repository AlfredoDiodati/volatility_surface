"""Generates Backend classes to make the code work for both numpy and JAX"""

import numpy as np
from typing import Protocol

class ndarray(Protocol):
    shape: tuple
    ndim: int

import numpy as np

def numpy_scan(f, carry, xs):
    """
    NumPy scan that matches jax.scan output structure:
    - y can be a nested structure (tuple/list/dict) of leaves
    - each leaf must keep the same shape/dtype across iterations
    - different leaves may have different shapes/dtypes
    - returns ys with a leading time dimension stacked per-leaf
    """
    T = xs.shape[0]
    if T == 0: return carry, None
    def _alloc_like(y0):
        if isinstance(y0, dict): return {k: _alloc_like(v) for k, v in y0.items()}
        if isinstance(y0, tuple): return tuple(_alloc_like(v) for v in y0)
        if isinstance(y0, list): return [_alloc_like(v) for v in y0]
        a0 = y0 if isinstance(y0, np.ndarray) else np.asarray(y0)
        out_shape = (T,) if a0.ndim == 0 else (T,) + a0.shape
        return np.empty(out_shape, dtype=a0.dtype)

    def _write(out, i, y):
        t = type(out)
        if t is dict:
            for k in out: _write(out[k], i, y[k])
            return
        if t is tuple:
            for o, v in zip(out, y): _write(o, i, v)
            return
        if t is list:
            for o, v in zip(out, y): _write(o, i, v)
            return
        if np.isscalar(y) or isinstance(y, np.generic):
            out[i] = y
        else: out[i, ...] = y
    it = iter(xs)
    carry, y0 = f(carry, next(it))
    ys = _alloc_like(y0)
    _write(ys, 0, y0)
    for i, x in enumerate(it, start=1):
        carry, y = f(carry, x)
        _write(ys, i, y)
    return carry, ys

class Backend(Protocol):
    xp = ...
    scan = ...

class NumpyBackend(Backend):
    xp = np
    scan = staticmethod(numpy_scan)

class JaxBackend(Backend):
    import jax.numpy as xp
    from jax.lax import scan

def _set_backend(backend:Backend) -> None:
    """Sets the linear algebra backend for the code, if nothing is selected it defaults to numpy.

    Args:
        backend (ModuleType): linear algebra module with numpy syntax.

    Possible alternatives are numpy and jax. CUPy may also work but would require further implementation
    """
    global xp
    global scan
    xp = backend.xp
    scan = backend.scan