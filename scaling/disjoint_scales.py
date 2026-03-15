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

        plt.figure()
        for x in tick_days:
            plt.axvline(x, linestyle="--", linewidth=0.8, color="black")

        ax = plt.gca()
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.xaxis.set_major_locator(FixedLocator(tick_days))
        ax.xaxis.set_major_formatter(FixedFormatter(tick_labels))
        holder = []
        for q in moments:
            holder.append(scaling_dict[q]["holder"])
            y_adjusted = scaling_dict[q]["shifted_power_var"]
            y_adjusted_exp = np.exp(y_adjusted)
            line, = plt.loglog(dt, y_adjusted_exp)
            plt.text(
                dt[-1]*1.01,
                y_adjusted_exp[-1],
                f"q={q}",
                color=line.get_color(),
                va="center",
                fontsize=9)
            
        plt.ylabel(r"$S(q, \Delta t)$")
        plt.xlabel(r"$\Delta t$")
        plt.xticks(rotation=0, ha="right")
        plt.savefig("plot/" + subfolder + "/put/scaling/scaling" + label + ".pdf")
        plt.close()

        plt.figure()
        for x in tick_days:
            plt.axvline(x, linestyle="--", linewidth=0.8, color="black")

        ax = plt.gca()
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.xaxis.set_major_locator(FixedLocator(tick_days))
        ax.xaxis.set_major_formatter(FixedFormatter(tick_labels))
        for q in moments:
            y_raw = scaling_dict[q]["log_power_var"]
            y_raw_exp = np.exp(y_raw)
            line, = plt.loglog(dt, y_raw_exp)
            plt.text(
                dt[-1]*1.01,
                y_raw_exp[-1],
                f"q={q}",
                color=line.get_color(),
                va="center",
                fontsize=9)
            
        plt.ylabel(r"$S(q, \Delta t)$")
        plt.xlabel(r"$\Delta t$")
        plt.xticks(rotation=0, ha="right")
        plt.savefig("plot/" + subfolder + "/put/scaling/scaling_raw" + label + ".pdf")
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

    underlying_series = pd.read_parquet("data/" + subfolder + "/put/underlying.parquet")
    log_price = np.log(underlying_series["UNDERLYING_LAST"].to_numpy())
    
    scaling_dict_underlying = moment_scaling(log_price, 1.0, 126.0, moments)
    underlying_tau = np.array([scaling_dict_underlying[q]["holder"] for q in moments])
    
    bucket_labels = sorted([label for label in data.columns if label != 'DATE'])
    moneyness_groups = sorted(set([label.split('_')[1] for label in bucket_labels]))
    maturity_groups = sorted(set([label.split('_')[0] for label in bucket_labels]))
    
    fig, axes = plt.subplots(1, len(moneyness_groups), figsize=(18, 4))
    if len(moneyness_groups) == 1:
        axes = [axes]
    
    fig.suptitle(r'$\tau(q)$ vs $q$ by Moneyness (Maturities in Blue)', fontsize=12)
    
    holder_bm = moments / 2.0 - 1.0
    blue_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(maturity_groups)))
    
    for mon_idx, moneyness in enumerate(moneyness_groups):
        ax = axes[mon_idx]
        
        ax.plot(moments, holder_bm, color='black', linestyle='--', linewidth=2.0, label='Brownian Motion', zorder=5)
        ax.plot(moments, underlying_tau, color='red', linewidth=2.0, label='Underlying', zorder=5)
        
        for mat_idx, maturity in enumerate(maturity_groups):
            label = f"{maturity}_{moneyness}"
            if label in data.columns:
                column = data[label].replace(0, np.nan)
                scaling_dict_iv = moment_scaling(column, 1.0, 126.0, moments)
                iv_tau = np.array([scaling_dict_iv[q]["holder"] for q in moments])
                ax.plot(moments, iv_tau, color=blue_colors[mat_idx], linewidth=1.5, label=maturity, alpha=0.8)
        
        ax.set_xlabel(r'$q$', fontsize=10)
        ax.set_ylabel(r'$\tau(q)$', fontsize=10)
        ax.set_title(f'Moneyness: {moneyness}', fontsize=11)
        ax.set_xlim((moments[0], moments[-1]))
        ax.set_ylim((-0.8, 1.0))
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("plot/" + subfolder + "/put/scaling/panel_tau_by_moneyness.pdf", dpi=100)
    plt.close()
    
    fig, axes = plt.subplots(1, len(maturity_groups), figsize=(18, 4))
    if len(maturity_groups) == 1:
        axes = [axes]
    
    fig.suptitle(r'$\tau(q)$ vs $q$ by Maturity (Moneyness in Blue)', fontsize=12)
    
    blue_colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(moneyness_groups)))
    
    for mat_idx, maturity in enumerate(maturity_groups):
        ax = axes[mat_idx]
        
        ax.plot(moments, holder_bm, color='black', linestyle='--', linewidth=2.0, label='Brownian Motion', zorder=5)
        ax.plot(moments, underlying_tau, color='red', linewidth=2.0, label='Underlying', zorder=5)
        
        for mon_idx, moneyness in enumerate(moneyness_groups):
            label = f"{maturity}_{moneyness}"
            if label in data.columns:
                column = data[label].replace(0, np.nan)
                scaling_dict_iv = moment_scaling(column, 1.0, 126.0, moments)
                iv_tau = np.array([scaling_dict_iv[q]["holder"] for q in moments])
                ax.plot(moments, iv_tau, color=blue_colors[mon_idx], linewidth=1.5, label=moneyness, alpha=0.8)
        
        ax.set_xlabel(r'$q$', fontsize=10)
        ax.set_ylabel(r'$\tau(q)$', fontsize=10)
        ax.set_title(f'Maturity: {maturity}', fontsize=11)
        ax.set_xlim((moments[0], moments[-1]))
        ax.set_ylim((-0.8, 1.0))
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("plot/" + subfolder + "/put/scaling/panel_tau_by_maturity.pdf", dpi=100)
    plt.close()

if __name__ == "__main__":
    main()