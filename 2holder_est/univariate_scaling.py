import numpy as np

def _partition(x: np.ndarray, delta_t: int) -> np.ndarray:
    """Given an array and a partition length size, reshapes the array in a 
    matrix such that each row is a partition of such length. 
    If the array is not reshapeable in such size, additional 0 entries are added. """
    T = int(x.shape[0])
    delta_t = int(delta_t)
    N = (T + delta_t - 1) // delta_t
    pad = N * delta_t - T
    x_pad = np.pad(x, (0, pad), mode="constant", constant_values=0)
    return x_pad.reshape((N, delta_t))

def _expected_power_variation(X_spaced:np.ndarray, q:float) -> float:
    """
        X: time series geometrically spaced
        q: moment
    """
    x_diff = np.diff(X_spaced, axis=0)
    return np.nansum(np.abs(x_diff) ** q, axis=1).mean()

def _make_time_lags(minf, maxf, factor = 1.1)->np.ndarray:
    n = np.log(maxf)-np.log(minf)-np.log(factor)
    return np.unique(np.round(minf * factor ** np.arange(n + 1)).astype(int))

def moment_scaling(x, minf, maxf, qs:np.ndarray, factor = 1.1)->dict:
    out = {}
    delta_ts = _make_time_lags(minf, maxf, factor=factor)
    log_t = np.log(delta_ts)
    for q in qs:
        power_var = np.empty_like(delta_ts, dtype=float)
        for i,dt in enumerate(delta_ts):
            x_part = _partition(x,dt)
            power_var[i] = _expected_power_variation(x_part, q)
        log_power_var = np.log(power_var)
        bad = ~np.isfinite(power_var)
        if np.any(bad):
            print("bad power_var at q=", q, "delta_ts=", delta_ts[bad], "raw=", power_var[bad])

        holder = np.sum(log_t*log_power_var)/np.sum(log_t**2)
        intercept = np.mean(log_power_var - log_t * holder)
        out[q] = {
                "log_power_var": log_power_var,
                "shifter_power_var": log_power_var - intercept,
                "intercept": intercept,
                "holder_exp": holder}
    out["delta_ts"] = delta_ts
    return out