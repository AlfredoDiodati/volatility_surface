import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from pathlib import Path

def rolling_variance(series, window):
    return series.rolling(window=window).var()

def autocorrelation_variance(variance_series, max_lag):
    var_mean = variance_series.mean()
    var_centered = variance_series - var_mean
    
    acf_values = []
    for lag in range(1, max_lag + 1):
        numerator = (var_centered[:-lag] * var_centered[lag:]).mean()
        denominator = var_centered.var()
        acf_values.append(numerator / denominator)
    
    return np.array(acf_values)

def power_law_model(lag, amplitude, exponent):
    return amplitude * (lag ** (-exponent))

def estimate_nu_from_bucket(iv_values, variance_window=20, fit_start_lag=10, fit_end_lag=100, max_lag=150):
    
    if isinstance(iv_values, (list, np.ndarray)):
        iv_series = pd.Series(iv_values)
    else:
        iv_series = iv_values.copy()
    
    iv_increments = iv_series.diff().dropna()
    realized_var = rolling_variance(iv_increments, variance_window).dropna()
    
    if len(realized_var) < max_lag + 1:
        max_lag = len(realized_var) // 2
    
    acf_var = autocorrelation_variance(realized_var, max_lag)
    
    lags_fit = np.arange(fit_start_lag, min(fit_end_lag, len(acf_var)))
    acf_fit = acf_var[fit_start_lag:min(fit_end_lag, len(acf_var))]
    
    valid_acf = acf_fit > 0
    lags_fit = lags_fit[valid_acf]
    acf_fit = acf_fit[valid_acf]
    
    if len(lags_fit) < 3:
        return None
    
    try:
        popt, _ = curve_fit(power_law_model, lags_fit, acf_fit, p0=[1.0, 0.3], maxfev=10000)
        nu_estimate = popt[1]
        amplitude_estimate = popt[0]
    except RuntimeError:
        return None
    
    return {
        'nu': nu_estimate,
        'amplitude': amplitude_estimate,
        'lags': np.arange(1, max_lag + 1),
        'acf': acf_var,
        'lags_fit': lags_fit,
        'acf_fit': acf_fit
    }

def save_result(result, bucket_name, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    txt_file = output_path / f"{bucket_name}_nu_estimate.txt"
    with open(txt_file, 'w') as f:
        f.write(f"Bucket: {bucket_name}\n")
        f.write(f"Estimated nu: {result['nu']:.6f}\n")
        f.write(f"Amplitude: {result['amplitude']:.6f}\n")
        f.write(f"Fit range: lags {result['lags_fit'][0]} to {result['lags_fit'][-1]}\n")
    
    acf_file = output_path / f"{bucket_name}_acf_values.txt"
    np.savetxt(acf_file, result['acf'], fmt='%.8f')
    
    fit_file = output_path / f"{bucket_name}_fit_lags_and_values.txt"
    fit_data = np.column_stack([result['lags_fit'], result['acf_fit']])
    np.savetxt(fit_file, fit_data, fmt='%.8f', header='lag acf_value')

def plot_nu_estimation(result, bucket_name, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    lags = result['lags']
    acf = result['acf']
    lags_fit = result['lags_fit']
    acf_fit = result['acf_fit']
    
    ax1.loglog(lags, acf, 'o', alpha=0.6, label='ACF')
    ax1.loglog(lags_fit, power_law_model(lags_fit, result['amplitude'], result['nu']), 
               'r-', linewidth=2, label=f"Power law: lag^(-{result['nu']:.3f})")
    ax1.set_xlabel('Lag')
    ax1.set_ylabel('ACF of Variance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'Variance ACF {bucket_name}')
    
    residuals = np.log(acf_fit) - np.log(power_law_model(lags_fit, result['amplitude'], result['nu']))
    ax2.plot(lags_fit, residuals, 'o-', alpha=0.6)
    ax2.axhline(0, color='r', linestyle='--')
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('Log Residuals')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Fit Quality')
    
    plt.tight_layout()
    
    plot_file = output_path / f"{bucket_name}_nu_plot.pdf"
    plt.savefig(plot_file, bbox_inches='tight')
    plt.close()

def process_bucket(iv_values, bucket_name, output_dir, variance_window=20, fit_start_lag=10, fit_end_lag=100):
    result = estimate_nu_from_bucket(iv_values, variance_window=variance_window, 
                                     fit_start_lag=fit_start_lag, fit_end_lag=fit_end_lag)
    
    if result is None:
        return None
    
    save_result(result, bucket_name, output_dir)
    plot_nu_estimation(result, bucket_name, output_dir)
    
    return result