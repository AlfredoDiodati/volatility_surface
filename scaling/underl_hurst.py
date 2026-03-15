import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import io
from contextlib import redirect_stdout

from hurst import estimate_hurst_from_volatility, plot_estimation_results

subfolder = "SPX"
plot_dir = Path("plot") / subfolder / "underlying"

def setup_directories():
    plot_dir.mkdir(parents=True, exist_ok=True)

def load_underlying_series(parquet_path):
    underlying_df = pl.read_parquet(parquet_path)
    underlying_df = underlying_df.sort("DATE")
    return underlying_df

def compute_log_returns(underlying_price_series):
    price_array = underlying_price_series.to_numpy()
    log_returns = np.diff(np.log(price_array))
    return log_returns

def compute_log_volatility_rolling(log_returns_series, window_size=20):
    log_volatility = np.zeros(len(log_returns_series) - window_size + 1)
    for index in range(len(log_volatility)):
        window_returns = log_returns_series[index:index + window_size]
        realized_variance = np.sum(window_returns ** 2)
        log_volatility[index] = 0.5 * np.log(realized_variance)
    return log_volatility

def plot_underlying_price_series(dates_array, price_series, output_path):
    figure, axes = plt.subplots(2, 1, figsize=(14, 9))
    
    axes[0].plot(dates_array, price_series, linewidth=1.2, color='darkblue')
    axes[0].set_title('SPX Underlying Price', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Price')
    axes[0].grid(True, alpha=0.3)
    
    log_returns = np.diff(np.log(price_series))
    axes[1].plot(dates_array[1:], log_returns, linewidth=0.8, color='darkred', alpha=0.7)
    axes[1].set_title('Log Returns', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('log R_t')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', format='pdf')
    plt.close(figure)
    return output_path

def plot_volatility_series(dates_array, log_volatility_series, output_path):
    figure, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(dates_array, log_volatility_series, linewidth=1.0, color='darkgreen')
    ax.set_title('Log Realized Volatility (20-day rolling window)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('log sqrt(RV)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', format='pdf')
    plt.close(figure)
    return output_path

def save_hurst_estimation_plots(hurst_results, log_volatility_series, output_path):
    estimation_figure = plot_estimation_results(
        log_volatility_series,
        hurst_results['hurst_monofractal'],
        hurst_results['zeta_scaling'],
        hurst_results['moment_orders'],
        hurst_results['hurst_acf'],
        hurst_results['acf_values'],
        hurst_results['acf_lags'],
        hurst_results['hurst_fukasawa'],
        hurst_results['cov_values'],
        hurst_results['cov_lags']
    )
    
    estimation_figure.savefig(output_path, dpi=150, bbox_inches='tight', format='pdf')
    plt.close(estimation_figure)
    return output_path

def save_summary_report(hurst_results, log_returns_stats, output_path):
    with open(output_path, 'w') as report_file:
        report_file.write("="*75 + "\n")
        report_file.write("HURST PARAMETER ANALYSIS - SPX UNDERLYING SERIES\n")
        report_file.write("="*75 + "\n\n")
        
        report_file.write("LOG RETURNS STATISTICS\n")
        report_file.write("-"*75 + "\n")
        report_file.write(f"Mean:                {log_returns_stats['mean']:.8f}\n")
        report_file.write(f"Std Dev:             {log_returns_stats['std']:.8f}\n")
        report_file.write(f"Min:                 {log_returns_stats['min']:.8f}\n")
        report_file.write(f"Max:                 {log_returns_stats['max']:.8f}\n")
        report_file.write(f"Skewness:            {log_returns_stats['skewness']:.6f}\n")
        report_file.write(f"Kurtosis:            {log_returns_stats['kurtosis']:.6f}\n")
        report_file.write(f"Number of obs:       {log_returns_stats['nobs']}\n\n")
        
        report_file.write("HURST PARAMETER ESTIMATES\n")
        report_file.write("-"*75 + "\n")
        report_file.write(f"Monofractal Scaling: {hurst_results['hurst_monofractal']:.6f}\n")
        report_file.write(f"ACF Method:          {hurst_results['hurst_acf']:.6f}\n")
        if hurst_results['hurst_fukasawa'] is not None:
            report_file.write(f"Fukasawa Robust:     {hurst_results['hurst_fukasawa']:.6f}\n")
        report_file.write("\n")
        
        report_file.write("INTERPRETATION\n")
        report_file.write("-"*75 + "\n")
        h_average = np.nanmean([hurst_results['hurst_monofractal'], hurst_results['hurst_acf']])
        report_file.write(f"Average H (excluding Fukasawa): {h_average:.6f}\n\n")
        
        if h_average < 0.5:
            report_file.write("Result: ROUGH VOLATILITY DETECTED\n")
            report_file.write("Interpretation: Volatility exhibits mean-reversion due to negative\n")
            report_file.write("autocorrelation at high frequencies. H < 0.5 is consistent with\n")
            report_file.write("fractional Brownian motion with rough paths.\n")
        else:
            report_file.write("Result: PERSISTENT VOLATILITY\n")
            report_file.write("Interpretation: Volatility exhibits trending behavior with positive\n")
            report_file.write("autocorrelation. H > 0.5 suggests mean-aversion.\n")
        
        report_file.write("\n" + "="*75 + "\n")

def write_log_file(log_output, output_path):
    with open(output_path, 'w') as log_file:
        log_file.write(log_output)

def main():
    setup_directories()
    
    log_buffer = io.StringIO()
    
    log_buffer.write("\n" + "="*75 + "\n")
    log_buffer.write("SPX UNDERLYING SERIES - HURST PARAMETER ANALYSIS\n")
    log_buffer.write("="*75 + "\n\n")
    
    parquet_file = "data/SPX/put/underlying.parquet"
    
    log_buffer.write(f"Loading underlying series from: {parquet_file}\n")
    underlying_data = load_underlying_series(parquet_file)
    
    dates_series = underlying_data["DATE"].to_list()
    price_series = underlying_data["UNDERLYING_LAST"]
    
    log_buffer.write(f"Loaded {len(price_series)} observations from {dates_series[0]} to {dates_series[-1]}\n")
    log_buffer.write(f"Price range: [{price_series.min():.2f}, {price_series.max():.2f}]\n\n")
    
    log_returns_series = compute_log_returns(price_series)
    
    log_returns_stats = {
        'mean': np.mean(log_returns_series),
        'std': np.std(log_returns_series),
        'min': np.min(log_returns_series),
        'max': np.max(log_returns_series),
        'skewness': (np.mean((log_returns_series - np.mean(log_returns_series))**3) / 
                    (np.std(log_returns_series)**3)),
        'kurtosis': (np.mean((log_returns_series - np.mean(log_returns_series))**4) / 
                    (np.std(log_returns_series)**4)) - 3,
        'nobs': len(log_returns_series)
    }
    
    log_buffer.write("LOG RETURNS STATISTICS:\n")
    log_buffer.write(f"  Mean: {log_returns_stats['mean']:.8f}\n")
    log_buffer.write(f"  Std:  {log_returns_stats['std']:.8f}\n")
    log_buffer.write(f"  Skew: {log_returns_stats['skewness']:.6f}\n")
    log_buffer.write(f"  Kurt: {log_returns_stats['kurtosis']:.6f}\n\n")
    
    rolling_window_size = 20
    log_volatility_series = compute_log_volatility_rolling(log_returns_series, rolling_window_size)
    
    dates_for_volatility = dates_series[rolling_window_size:]
    
    log_buffer.write(f"Computing Hurst estimates on log realized volatility (window={rolling_window_size} days)...\n")
    log_buffer.write(f"Log volatility series length: {len(log_volatility_series)}\n\n")
    
    hurst_output_buffer = io.StringIO()
    with redirect_stdout(hurst_output_buffer):
        hurst_results = estimate_hurst_from_volatility(
            log_volatility_series,
            verbose=True
        )
    
    log_buffer.write(hurst_output_buffer.getvalue())
    
    log_buffer.write("\nGenerating plots...\n")
    
    price_plot_path = plot_dir / "underlying_price.pdf"
    plot_underlying_price_series(dates_series, price_series, price_plot_path)
    log_buffer.write(f"Saved: {price_plot_path}\n")
    
    volatility_plot_path = plot_dir / "log_volatility_rolling.pdf"
    plot_volatility_series(dates_for_volatility, log_volatility_series, volatility_plot_path)
    log_buffer.write(f"Saved: {volatility_plot_path}\n")
    
    estimation_plot_path = plot_dir / "hurst_estimation_methods.pdf"
    save_hurst_estimation_plots(hurst_results, log_volatility_series, estimation_plot_path)
    log_buffer.write(f"Saved: {estimation_plot_path}\n")
    
    log_buffer.write("\n" + "="*75 + "\n")
    log_buffer.write("ANALYSIS COMPLETE\n")
    log_buffer.write("="*75 + "\n\n")
    
    log_buffer.write("Generated files:\n")
    log_buffer.write(f"  - {price_plot_path}\n")
    log_buffer.write(f"  - {volatility_plot_path}\n")
    log_buffer.write(f"  - {estimation_plot_path}\n")
    log_buffer.write(f"  - {plot_dir / 'hurst_analysis_summary.txt'}\n")
    log_buffer.write(f"  - {plot_dir / 'log.txt'}\n\n")
    
    log_file_path = plot_dir / "log.txt"
    write_log_file(log_buffer.getvalue(), log_file_path)
    
    summary_report_path = plot_dir / "hurst_analysis_summary.txt"
    save_summary_report(hurst_results, log_returns_stats, summary_report_path)
    
    print(log_buffer.getvalue())
    
    return hurst_results, log_returns_stats

if __name__ == "__main__":
    main()