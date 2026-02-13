from filter._backend import _set_backend, NumpyBackend, JaxBackend
from filter._simple import filter

backend = _set_backend(NumpyBackend)
xp = backend.xp

def _dynamics()->dict:
    Z = ...

    return  

def fit(): ...