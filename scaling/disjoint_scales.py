from scaling_reg import moment_scaling
import pandas as pd
import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

def main():
    """
    Computes realized power variation and scaling
    exponent from daily observations to 6 months
    """

    subfolder = "SPX"
    data = pd.read_parquet("data/" + subfolder + "/put/bucket_matrix.parquet").set_index("DATE")
    print(f"nans in dataset {np.sum(data.isna().to_numpy())}")

    moments = np.arange(1, 9) / 2
    tick_days = np.array([1, 5, 21, 63, 116])
    tick_labels = ["", "1 week", "1 month", "3 months", "6 months"]
    
    scalings = {}

    for label in data.columns:
        col = data[label]
        print(f"{label}: {col.notna().sum()} valid out of {len(col)}")

    for label in data.columns:
        column = data[label].replace(0, np.nan)
        scaling_dict = moment_scaling(column, 1.0, 126.0, moments)
        dt = scaling_dict["delta_ts"]

        for x in tick_days:
            plt.axvline(x, linestyle="--", linewidth=0.8, color="black")

        ax = plt.gca()
        ax.xaxis.set_major_locator(FixedLocator(tick_days))
        ax.xaxis.set_major_formatter(FixedFormatter(tick_labels))
        holder = []
        for q in moments:
            holder.append(scaling_dict[q]["holder"])
            y_adjusted = scaling_dict[q]["shifted_power_var"]
            y_raw = scaling_dict[q]["log_power_var"]
            line, = plt.plot(dt, y_adjusted)
            plt.plot(dt, y_raw, linestyle=":", color=line.get_color(), linewidth=0.8)
            plt.text(
                dt[-1]*1.01,
                y_adjusted[-1],
                f"q={q}",
                color=line.get_color(),
                va="center",
                fontsize=9)
            
        plt.ylabel(r"$S(q, \Delta t)$")
        plt.xlabel(r"$\Delta t$")
        plt.xlim((dt[0], dt[-1]+15))
        plt.xticks(rotation=0, ha="right")
        plt.savefig("plot/" + subfolder + "/put/scaling/scaling_" + label + ".pdf")
        plt.close()

        scalings[label] = np.array(holder)
    
    holder_bm = moments / 2.0 - 1.0
    plt.figure()
    plt.plot(moments, holder_bm, color="black", linestyle="--")
    for label in scalings.keys():
        plt.plot(moments, scalings[label], label = label)
    plt.ylabel(r"$\tau(q)$")
    plt.xlabel(r"$q$")
    plt.xlim((moments[0], moments[-1]))
    plt.ylim((None, 1.0))
    plt.legend()
    plt.savefig("plot/" + subfolder + "/put/scaling/moments_scaling_legend.pdf")
    plt.close()

    plt.figure()
    plt.plot(moments, holder_bm, color="black", linestyle="--")
    for label in scalings.keys():
        plt.plot(moments, scalings[label], label = label)
    plt.ylabel(r"$\tau(q)$")
    plt.xlabel(r"$q$")
    plt.xlim((moments[0], moments[-1]))
    plt.ylim((None, 1.0))
    plt.savefig("plot/" + subfolder + "/put/scaling/moments_scaling.pdf")
    plt.close()

if __name__ == "__main__":
    main()