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

def _expected_power_variation(x_spaced: np.ndarray, q: float) -> float:
    coarse_inc = x_spaced[1:, 0] - x_spaced[:-1, 0]
    return np.nansum(np.abs(coarse_inc) ** q)

def _make_time_lags(minf, maxf, factor=1.1) -> np.ndarray:
    n = (np.log(maxf) - np.log(minf)) / np.log(factor)
    return np.unique(
        np.round(minf * factor ** np.arange(int(np.floor(n)) + 1)).astype(int))

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

        lt = log_t - log_t.mean()
        lp = log_power_var - log_power_var.mean()
        holder = np.sum(lt * lp) / np.sum(lt**2)
        intercept = log_power_var.mean() - holder * log_t.mean()

        out[q] = {
                "log_power_var": log_power_var,
                "shifted_power_var": log_power_var - intercept,
                "intercept": intercept,
                "holder": holder}
    out["delta_ts"] = delta_ts
    out["log_t"] = log_t
    return out