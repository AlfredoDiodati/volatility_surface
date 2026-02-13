"""Generates Backend classes to make the code work for both numpy and JAX"""

import numpy as np

def numpy_scan(f, carry, xs):
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, np.stack(ys)

class Backend:
    xp = ...
    scan = ...

class NumpyBackend(Backend):
    xp = np
    scan = staticmethod(numpy_scan)

class JaxBackend(Backend):
    import jax.numpy as xp
    from jax.lax import scan

def _set_backend(backend:Backend):
    """Sets the linear algebra backend for the code, if nothing is selected it defaults to numpy.

    Args:
        backend (ModuleType): linear algebra module with numpy syntax.

    Possible alternatives are numpy and jax. CUPy may also work but would require further implementation
    """
    global xp
    xp = backend