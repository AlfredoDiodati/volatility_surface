from univariate_scaling import moment_scaling
import pandas as pd
import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

def _brownian_scale(q:np.ndarray)->np.ndarray:
    return (2.0**q * gamma((q + 1.0) / 2.0) / np.pi)

def _brownian_logS(q:float, log_t:np.ndarray)->np.ndarray:
    return (q/2 -1.0)*(log_t - log_t[0])

def main():
    """
    Computes realized power variation and scaling
    exponent from daily observations to 6 months
    """

    subfolder = "SPY"
    data = pd.read_parquet("data/" + subfolder + "/put/bucket_matrix.parquet")
    print(f"nans in dataset {np.sum(data.isna().to_numpy())}")

    moments = np.arange(1, 9) / 2
    tick_days = np.array([1, 5, 21, 63, 116])
    tick_labels = ["", "1 week", "1 month", "3 months", "6 months"]

    test = data["mat1_mon1"]
    scaling_dict = moment_scaling(test, 1.0, 126.0, moments)
    dt = scaling_dict["delta_ts"]
    log_t = scaling_dict["log_t"]
    for x in tick_days:
        plt.axvline(x, linestyle="--", linewidth=0.8, color="black")

    ax = plt.gca()
    ax.xaxis.set_major_locator(FixedLocator(tick_days))
    ax.xaxis.set_major_formatter(FixedFormatter(tick_labels))
    holder = []
    for q in moments:
        brownian_scale = _brownian_logS(q, log_t)
        holder.append(scaling_dict[q]["holder"])
        y = scaling_dict[q]["shifted_power_var"]
        line, = plt.plot(dt, y)
        #plt.plot(dt, brownian_scale, color="black", linestyle="--")
        plt.text(
            dt[-1]*1.01,
            y[-1],
            f"q={q}",
            color=line.get_color(),
            va="center",
            fontsize=9)
        
    plt.ylabel(r"$S(q, \Delta t)$")
    plt.xlabel(r"$\Delta t$")
    plt.xlim((dt[0], dt[-1]+15))
    plt.xticks(rotation=0, ha="right")
    plt.savefig("plot/" + subfolder + "/put/_scaling" + "TEST" + ".pdf")
    plt.close()

    holder = np.array(holder)
    bolder_bm = _brownian_scale(q=moments)
    plt.figure()
    plt.plot(moments, holder)
    #plt.plot(moments, bolder_bm, color="black", linestyle="--")
    plt.savefig("plot/" + subfolder + "/put/_moments" + "TEST"+".pdf")
    plt.close()

if __name__ == "__main__":
    main()