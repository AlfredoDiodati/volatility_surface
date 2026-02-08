from univariate_scaling import moment_scaling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

def main():
    """Computes realized power variation and scaling
    exponent from daily observations to 6 months"""

    subfolder = "SPY"
    data = pd.read_parquet("data/" + subfolder + "/put/bucket_matrix.parquet")
    print(f"nans in dataset {np.sum(data.isna().to_numpy())}")

    moments = np.arange(1, 9) / 2
    tick_days = np.array([1, 5, 21, 63, 116])
    tick_labels = ["", "1 week", "1 month", "3 months", "6 months"]

    test = data["mat1_mon1"]
    scaling_dict = moment_scaling(test, 1.0, 126.0, moments)
    dt = scaling_dict["delta_ts"]
    
    for x in tick_days:
        plt.axvline(x, linestyle="--", linewidth=0.8, color="black")

    ax = plt.gca()
    ax.xaxis.set_major_locator(FixedLocator(tick_days))
    ax.xaxis.set_major_formatter(FixedFormatter(tick_labels))
    holder = []
    for i,q in enumerate(moments):
        holder.append(scaling_dict[q]["holder"])
        y = scaling_dict[q]["shifted_power_var"]
        line, = plt.plot(dt, y)
        plt.text(
        dt[-1]*1.01,
        y[-1],
        f"q={q}",
        color=line.get_color(),
        va="center",
        fontsize=9
    )
    plt.ylabel(r"$S(q, \Delta t)$")
    plt.xlabel(r"$\Delta t$")
    plt.xlim((dt[0], dt[-1]+15))
    plt.xticks(rotation=0, ha="right")
    plt.savefig("plot/" + subfolder + "/put/_scaling" + "TEST" + ".pdf")
    plt.close()

    plt.figure()
    holder = np.array(holder)
    plt.plot(moments, holder)
    plt.savefig("plot/" + subfolder + "/put/_moments" + "TEST"+".pdf")
    plt.close()

if __name__ == "__main__":
    main()