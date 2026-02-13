"""Generates Backend classes to make the code work for both numpy and JAX"""

import numpy as np
from typing import Protocol

class ndarray(Protocol):
    shape: tuple
    ndim: int

def numpy_scan(f, carry, xs):
    """
    NumPy-based implementation of a scan (cumulative fold) operation.

    Iteratively applies a function `f` over a sequence `xs`, carrying forward
    a state variable and collecting intermediate outputs.

    Parameters
    ----------
    f : callable
        Function of the form `f(carry, x) -> (new_carry, y)`, where
        `carry` is the current state and `x` is the current element of `xs`.
    carry : any
        Initial state passed to the first call of `f`.
    xs : iterable
        Sequence of inputs to iterate over.

    Returns
    -------
    carry : any
        Final state after processing all elements of `xs`.
    ys : np.ndarray
        Stacked array of outputs `y` returned at each iteration.
    """
    ys = []
    for x in xs:
        carry, y = f(carry, x)
        ys.append(y)
    return carry, np.stack(ys)

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
    xp = backend