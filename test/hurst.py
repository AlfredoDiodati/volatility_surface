"""
Examples of using the Hurst parameter estimation module

Three estimation methods implemented from Gatheral (2024) "Volatility is rough":
1. Monofractal scaling: m(q, Δ) ∝ Δ^{ζq} with ζq = qH
2. ACF method: log(1 - p(Δ)) = a + 2H log Δ
3. Fukasawa's robust: cov(Σ log R, ΔX) ∝ (kΔ)^{H+3/2}
"""

import numpy as np
import pandas as pd
from scaling.hurst import (
    estimate_hurst_from_volatility,
    plot_estimation_results,
    generate_fbm_mandelbrot_van_ness,
    create_log_volatility_from_fbm,
    load_from_csv
)

def example_1_synthetic_fbm():
    """
    Example 1: Synthetic fractional Brownian motion
    Generate data with known Hurst parameter and estimate it
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Synthetic fBM with Known Hurst Parameter")
    print("="*70)
    
    true_h = 0.18
    n_samples = 3000
    
    fbm = generate_fbm_mandelbrot_van_ness(n_samples, true_h)
    log_vol = create_log_volatility_from_fbm(fbm, volatility_scale=0.3)
    
    results = estimate_hurst_from_volatility(log_vol, verbose=True)
    
    print(f"TRUE H: {true_h:.4f}")
    print(f"Monofractal estimate:  {results['hurst_monofractal']:.4f} (error: {abs(results['hurst_monofractal'] - true_h):.4f})")
    print(f"ACF estimate:          {results['hurst_acf']:.4f} (error: {abs(results['hurst_acf'] - true_h):.4f})")
    
    return results

def example_2_with_price_and_rv():
    """
    Example 2: All three methods using price series and realized variance
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: All Three Methods (Monofractal, ACF, Fukasawa)")
    print("="*70)
    
    true_h = 0.20
    n_samples = 4000
    
    np.random.seed(123)
    fbm = generate_fbm_mandelbrot_van_ness(n_samples, true_h)
    log_vol = create_log_volatility_from_fbm(fbm, volatility_scale=0.28)
    
    price = np.exp(np.cumsum(0.007 * np.random.randn(n_samples)))
    realized_var = np.abs(np.diff(log_vol)) + 0.004
    
    results = estimate_hurst_from_volatility(
        log_vol,
        price_input=price,
        realized_variance_input=realized_var,
        verbose=True
    )
    
    print(f"\nTRUE H: {true_h:.4f}")
    
    return results, log_vol, price, realized_var

def example_3_from_csv():
    """
    Example 3: Load data from CSV file
    
    CSV file should have structure:
    timestamp, log_volatility [, price, realized_variance]
    
    Or modify column_name parameter to select specific columns
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: Loading Data from CSV")
    print("="*70)
    
    # This creates a sample CSV for demonstration
    np.random.seed(99)
    n = 2000
    true_h_demo = 0.17
    fbm_demo = generate_fbm_mandelbrot_van_ness(n, true_h_demo)
    log_vol_demo = create_log_volatility_from_fbm(fbm_demo)
    
    df = pd.DataFrame({
        'timestamp': pd.date_range('2020-01-01', periods=n, freq='D'),
        'log_volatility': log_vol_demo,
        'price': np.exp(np.cumsum(0.006 * np.random.randn(n))),
        'realized_variance': np.abs(np.diff(log_vol_demo, prepend=0)) + 0.003
    })
    
    csv_filename = '/tmp/volatility_data_sample.csv'
    df.to_csv(csv_filename, index=False)
    print(f"Sample CSV created at: {csv_filename}")
    
    log_vol_from_csv = load_from_csv(csv_filename, column_name='log_volatility')
    price_from_csv = load_from_csv(csv_filename, column_name='price')
    rv_from_csv = load_from_csv(csv_filename, column_name='realized_variance')
    
    results = estimate_hurst_from_volatility(
        log_vol_from_csv,
        price_input=price_from_csv,
        realized_variance_input=rv_from_csv,
        verbose=True
    )
    
    return results

def example_4_rolling_window_estimation():
    """
    Example 4: Rolling window estimation to detect time-varying H
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Time-Varying Hurst Parameter (Rolling Windows)")
    print("="*70)
    
    n_samples = 5000
    window_size = 600
    step_size = 100
    
    np.random.seed(45)
    fbm_process = generate_fbm_mandelbrot_van_ness(n_samples, 0.18)
    log_vol_process = create_log_volatility_from_fbm(fbm_process)
    
    rolling_h_monofractal = []
    rolling_h_acf = []
    window_centers = []
    
    for start_idx in range(0, len(log_vol_process) - window_size, step_size):
        end_idx = start_idx + window_size
        window_data = log_vol_process[start_idx:end_idx]
        
        window_results = estimate_hurst_from_volatility(
            window_data,
            verbose=False
        )
        
        rolling_h_monofractal.append(window_results['hurst_monofractal'])
        rolling_h_acf.append(window_results['hurst_acf'])
        window_centers.append((start_idx + end_idx) // 2)
    
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(window_centers, rolling_h_monofractal, 'o-', label='Monofractal', linewidth=2, markersize=5)
    ax.plot(window_centers, rolling_h_acf, 's-', label='ACF', linewidth=2, markersize=5)
    ax.axhline(y=0.18, color='k', linestyle='--', alpha=0.5, label='True H')
    ax.set_xlabel('Window center (time index)', fontsize=11)
    ax.set_ylabel('Estimated H', fontsize=11)
    ax.set_title('Time-Varying Hurst Parameter Estimation (Rolling Windows)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"Monofractal - Mean H: {np.mean(rolling_h_monofractal):.4f}, Std: {np.std(rolling_h_monofractal):.4f}")
    print(f"ACF - Mean H: {np.mean(rolling_h_acf):.4f}, Std: {np.std(rolling_h_acf):.4f}")

def example_5_comparison_different_true_h():
    """
    Example 5: Estimate multiple series with different true H values
    to assess estimator accuracy
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Accuracy Assessment Across Different H Values")
    print("="*70)
    
    true_h_values = np.array([0.10, 0.15, 0.20, 0.25, 0.30])
    n_per_series = 3000
    
    results_table = {
        'True H': true_h_values,
        'Monofractal': [],
        'ACF': [],
        'Monofractal Error': [],
        'ACF Error': []
    }
    
    for true_h in true_h_values:
        np.random.seed(int(true_h * 1000))
        fbm = generate_fbm_mandelbrot_van_ness(n_per_series, true_h)
        log_vol = create_log_volatility_from_fbm(fbm)
        
        est = estimate_hurst_from_volatility(log_vol, verbose=False)
        
        results_table['Monofractal'].append(est['hurst_monofractal'])
        results_table['ACF'].append(est['hurst_acf'])
        results_table['Monofractal Error'].append(abs(est['hurst_monofractal'] - true_h))
        results_table['ACF Error'].append(abs(est['hurst_acf'] - true_h))
    
    df_results = pd.DataFrame(results_table)
    print("\n", df_results.to_string(index=False))
    
    print(f"\nMonofractal - Mean Absolute Error: {np.mean(results_table['Monofractal Error']):.4f}")
    print(f"ACF - Mean Absolute Error: {np.mean(results_table['ACF Error']):.4f}")
    
    return df_results

if __name__ == '__main__':
    
    print("\n" + "█"*70)
    print("HURST PARAMETER ESTIMATION - USAGE EXAMPLES")
    print("█"*70)
    
    results_ex1 = example_1_synthetic_fbm()
    
    results_ex2, log_vol_ex2, price_ex2, rv_ex2 = example_2_with_price_and_rv()
    
    fig_ex2 = plot_estimation_results(
        log_vol_ex2,
        results_ex2['hurst_monofractal'],
        results_ex2['zeta_scaling'],
        results_ex2['moment_orders'],
        results_ex2['hurst_acf'],
        results_ex2['acf_values'],
        results_ex2['acf_lags'],
        results_ex2['hurst_fukasawa'],
        results_ex2['cov_values'],
        results_ex2['cov_lags']
    )
    
    results_ex3 = example_3_from_csv()
    
    example_4_rolling_window_estimation()
    
    df_ex5 = example_5_comparison_different_true_h()
    
    print("\n" + "█"*70)
    print("ALL EXAMPLES COMPLETED")
    print("█"*70 + "\n")