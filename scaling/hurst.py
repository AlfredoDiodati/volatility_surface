import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

def generate_fbm_mandelbrot_van_ness(n_samples, hurst_param, dt=1.0):
    time_indices = np.arange(n_samples, dtype=np.float64)
    
    covariance_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            covariance_matrix[i, j] = 0.5 * (
                (time_indices[i] ** (2 * hurst_param)) +
                (time_indices[j] ** (2 * hurst_param)) -
                (np.abs(time_indices[i] - time_indices[j]) ** (2 * hurst_param))
            )
    
    covariance_matrix = np.maximum(covariance_matrix, 1e-12 * np.eye(n_samples))
    
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    eigenvalues = np.maximum(eigenvalues, 0)
    
    cholesky_matrix = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    
    standard_normal_samples = np.random.randn(n_samples)
    fbm_path = cholesky_matrix @ standard_normal_samples
    
    return fbm_path / (np.std(fbm_path) + 1e-10)

def create_log_volatility_from_fbm(fbm_series, volatility_scale=0.3):
    return volatility_scale * fbm_series

def estimate_hurst_monofractal_scaling(log_volatility_series, moment_orders=None, lag_range=None):
    
    if moment_orders is None:
        moment_orders = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
    
    if lag_range is None:
        max_lag = max(30, len(log_volatility_series) // 15)
        lag_range = np.logspace(0, np.log10(max_lag), 18, dtype=int)
    
    scaling_exponents_zeta = []
    
    for q in moment_orders:
        moment_at_each_lag = []
        valid_lags = []
        
        for lag in lag_range:
            if lag >= len(log_volatility_series):
                continue
            
            increments = log_volatility_series[lag:] - log_volatility_series[:-lag]
            absolute_increments = np.abs(increments)
            
            if len(absolute_increments) < 5:
                continue
            
            moment_q = np.mean(absolute_increments ** q)
            
            if moment_q > 0 and not np.isnan(moment_q) and not np.isinf(moment_q):
                moment_at_each_lag.append(moment_q)
                valid_lags.append(lag)
        
        if len(moment_at_each_lag) < 3:
            scaling_exponents_zeta.append(np.nan)
            continue
        
        log_lags = np.log(np.array(valid_lags, dtype=float))
        log_moments = np.log(np.array(moment_at_each_lag))
        
        slope_zeta, intercept, r_squared, p_val, std_err = stats.linregress(log_lags, log_moments)
        
        if not np.isnan(slope_zeta) and not np.isinf(slope_zeta) and slope_zeta > -2 and slope_zeta < 3:
            scaling_exponents_zeta.append(slope_zeta)
        else:
            scaling_exponents_zeta.append(np.nan)
    
    scaling_exponents_zeta = np.array(scaling_exponents_zeta)
    valid_indices = ~np.isnan(scaling_exponents_zeta)
    
    if np.sum(valid_indices) < 2:
        return np.nan, scaling_exponents_zeta, moment_orders
    
    hurst_estimate, intercept_h, r_squared_h, p_val_h, std_err_h = stats.linregress(
        moment_orders[valid_indices],
        scaling_exponents_zeta[valid_indices]
    )
    
    return hurst_estimate, scaling_exponents_zeta, moment_orders

def estimate_hurst_acf_method(log_volatility_series, lag_range=None):
    
    if lag_range is None:
        max_lag = max(25, len(log_volatility_series) // 4)
        lag_range = np.logspace(0, np.log10(max(40, max_lag)), 20, dtype=int)
    
    autocorrelation_at_lag = []
    valid_lags = []
    
    for lag in lag_range:
        if lag >= len(log_volatility_series) - 5:
            continue
        
        series_early = log_volatility_series[:-lag]
        series_late = log_volatility_series[lag:]
        
        if len(series_early) < 10:
            continue
        
        early_centered = series_early - np.mean(series_early)
        late_centered = series_late - np.mean(series_late)
        
        early_std = np.std(early_centered)
        late_std = np.std(late_centered)
        
        if early_std < 1e-10 or late_std < 1e-10:
            continue
        
        corr_coeff = np.mean(early_centered * late_centered) / (early_std * late_std)
        corr_coeff = np.clip(corr_coeff, -0.999, 0.999)
        
        autocorrelation_at_lag.append(corr_coeff)
        valid_lags.append(lag)
    
    if len(valid_lags) < 3:
        return np.nan, np.array([]), np.array([])
    
    autocorrelation_at_lag = np.array(autocorrelation_at_lag)
    lags_used = np.array(valid_lags, dtype=float)
    
    one_minus_rho = 1.0 - autocorrelation_at_lag
    one_minus_rho = np.clip(one_minus_rho, 1e-8, 2.0)
    
    log_one_minus_rho = np.log(one_minus_rho)
    log_lags = np.log(lags_used)
    
    slope_2h, intercept_a, r_squared_acf, p_val_acf, std_err_acf = stats.linregress(
        log_lags, log_one_minus_rho
    )
    
    hurst_from_acf = slope_2h / 2.0
    hurst_from_acf = np.clip(hurst_from_acf, -0.5, 0.95)
    
    return hurst_from_acf, autocorrelation_at_lag, lags_used

def estimate_hurst_fukasawa_robust(price_series, realized_variance_series, lag_range=None):
    
    if lag_range is None:
        max_lag = max(30, len(price_series) // 8)
        lag_range = np.logspace(0, np.log10(max(40, max_lag)), 18, dtype=int)
    
    log_price_returns = np.diff(np.log(price_series))
    log_realized_variance = np.log(np.clip(realized_variance_series, 1e-10, None))
    
    covariance_at_lag = []
    valid_lags = []
    
    for k in lag_range:
        k_int = int(np.round(k))
        
        if k_int >= len(log_price_returns) or k_int >= len(log_realized_variance):
            continue
        
        cumsum_log_rv = np.cumsum(log_realized_variance)
        
        cumsum_truncated = cumsum_log_rv[k_int:]
        returns_truncated = log_price_returns[:(len(cumsum_truncated))]
        
        if len(cumsum_truncated) < 15 or len(returns_truncated) < 15:
            continue
        
        cumsum_centered = cumsum_truncated - np.mean(cumsum_truncated)
        returns_centered = returns_truncated - np.mean(returns_truncated)
        
        cumsum_std = np.std(cumsum_centered)
        returns_std = np.std(returns_centered)
        
        if cumsum_std < 1e-10 or returns_std < 1e-10:
            continue
        
        cov_value = np.mean(cumsum_centered * returns_centered)
        cov_value = np.abs(cov_value)
        
        if cov_value > 0:
            covariance_at_lag.append(cov_value)
            valid_lags.append(float(k_int))
    
    if len(valid_lags) < 3:
        return np.nan, np.array([]), np.array([])
    
    covariance_at_lag = np.array(covariance_at_lag)
    lags_used = np.array(valid_lags)
    
    covariance_at_lag = np.clip(covariance_at_lag, 1e-10, None)
    
    log_covariance = np.log(covariance_at_lag)
    log_lag_index = np.log(lags_used)
    
    slope_h_plus_32, intercept_fuk, r_squared_fuk, p_val_fuk, std_err_fuk = stats.linregress(
        log_lag_index, log_covariance
    )
    
    hurst_from_fukasawa = slope_h_plus_32 - 1.5
    hurst_from_fukasawa = np.clip(hurst_from_fukasawa, -0.5, 0.95)
    
    return hurst_from_fukasawa, covariance_at_lag, lags_used

def plot_estimation_results(log_vol_input, hurst_mono, zeta_estimates, moment_orders_used,
                            hurst_acf, acf_values, acf_lags,
                            hurst_fuk, cov_values, cov_lags):
    
    figure, subplots = plt.subplots(2, 2, figsize=(14, 10))
    
    subplots[0, 0].plot(log_vol_input, linewidth=0.7, color='darkblue')
    subplots[0, 0].set_title('Log Volatility Input Series', fontsize=12, fontweight='bold')
    subplots[0, 0].set_xlabel('Time index')
    subplots[0, 0].set_ylabel('log σ_t')
    subplots[0, 0].grid(True, alpha=0.3)
    
    valid_zeta = zeta_estimates[~np.isnan(zeta_estimates)]
    valid_moments = moment_orders_used[~np.isnan(zeta_estimates)]
    
    if len(valid_zeta) > 1:
        subplots[0, 1].scatter(valid_moments, valid_zeta, s=70, color='darkred', alpha=0.75, label='Measured ζ_q')
        fitted_line_zeta = hurst_mono * valid_moments
        subplots[0, 1].plot(valid_moments, fitted_line_zeta, 'r-', linewidth=2.5, label=f'Fit: H = {hurst_mono:.4f}')
    else:
        subplots[0, 1].text(0.5, 0.5, 'Insufficient valid data', ha='center', va='center', transform=subplots[0, 1].transAxes)
    
    subplots[0, 1].set_title('Monofractal Scaling: ζ_q = qH', fontsize=12, fontweight='bold')
    subplots[0, 1].set_xlabel('Moment order q')
    subplots[0, 1].set_ylabel('Scaling exponent ζ_q')
    subplots[0, 1].legend(fontsize=10)
    subplots[0, 1].grid(True, alpha=0.3)
    
    if acf_values is not None and acf_lags is not None and len(acf_lags) > 1:
        subplots[1, 0].loglog(acf_lags, 1.0 - acf_values, 'o', color='darkgreen', markersize=6, alpha=0.7, label='1 - ρ(Δ)')
        power_law_fit_acf = (1.0 - acf_values[0]) * (acf_lags / acf_lags[0]) ** (2 * hurst_acf)
        subplots[1, 0].loglog(acf_lags, power_law_fit_acf, 'g-', linewidth=2.5, label=f'Fit: H = {hurst_acf:.4f}')
        subplots[1, 0].legend(fontsize=10)
    
    subplots[1, 0].set_title('ACF Estimator: log(1 - ρ) = a + 2H log Δ', fontsize=12, fontweight='bold')
    subplots[1, 0].set_xlabel('Lag Δ')
    subplots[1, 0].set_ylabel('1 - ρ(Δ)')
    subplots[1, 0].grid(True, alpha=0.3, which='both')
    
    if cov_values is not None and cov_lags is not None and len(cov_lags) > 1:
        subplots[1, 1].loglog(cov_lags, cov_values, 'o', color='darkblue', markersize=6, alpha=0.7, label='|Cov|')
        if hurst_fuk is not None and not np.isnan(hurst_fuk):
            power_law_fit_fuk = (cov_values[0]) * (cov_lags / cov_lags[0]) ** (hurst_fuk + 1.5)
            subplots[1, 1].loglog(cov_lags, power_law_fit_fuk, 'b-', linewidth=2.5, label=f'Fit: H = {hurst_fuk:.4f}')
        subplots[1, 1].legend(fontsize=10)
    else:
        subplots[1, 1].text(0.5, 0.5, 'Fukasawa method skipped', ha='center', va='center', transform=subplots[1, 1].transAxes)
    
    subplots[1, 1].set_title("Fukasawa's Method: log Cov = (H+3/2) log k + c", fontsize=12, fontweight='bold')
    subplots[1, 1].set_xlabel('Lag k')
    subplots[1, 1].set_ylabel('Covariance magnitude')
    subplots[1, 1].grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    return figure

def estimate_hurst_from_volatility(volatility_input, price_input=None, realized_variance_input=None, verbose=True):
    
    log_volatility = np.asarray(volatility_input).flatten()
    
    if verbose:
        print("\n" + "="*75)
        print("HURST PARAMETER ESTIMATION FROM UNIVARIATE DATA")
        print("="*75)
    
    h_monofractal, zeta_q, q_values = estimate_hurst_monofractal_scaling(log_volatility)
    
    if verbose:
        print(f"\n[1] Monofractal Scaling Method")
        print(f"    Estimated H = {h_monofractal:.4f}")
    
    h_acf, acf_vals, acf_lag_vals = estimate_hurst_acf_method(log_volatility)
    
    if verbose:
        print(f"\n[2] ACF Method (Bennedsen, Lunde, Pakkanen 2022)")
        print(f"    Estimated H = {h_acf:.4f}")
    
    h_fukasawa = None
    cov_vals = None
    cov_lag_vals = None
    
    if price_input is not None and realized_variance_input is not None:
        price = np.asarray(price_input).flatten()
        rv = np.asarray(realized_variance_input).flatten()
        
        if len(price) == len(rv) and len(price) > 50:
            h_fukasawa, cov_vals, cov_lag_vals = estimate_hurst_fukasawa_robust(price, rv)
            
            if verbose:
                print(f"\n[3] Fukasawa's Robust Method")
                print(f"    Estimated H = {h_fukasawa:.4f}")
        else:
            if verbose:
                print(f"\n[3] Fukasawa's Robust Method")
                print(f"    Skipped (price and RV must have equal length and ≥50 samples)")
    else:
        if verbose:
            print(f"\n[3] Fukasawa's Robust Method")
            print(f"    Skipped (requires price and realized_variance arguments)")
    
    if verbose:
        print("\n" + "="*75)
        print("SUMMARY OF ESTIMATES")
        print("="*75)
        print(f"Monofractal:       H = {h_monofractal:.4f}")
        print(f"ACF:               H = {h_acf:.4f}")
        if h_fukasawa is not None:
            print(f"Fukasawa:          H = {h_fukasawa:.4f}")
            h_average = np.nanmean([h_monofractal, h_acf, h_fukasawa])
            print(f"\nAverage H:         {h_average:.4f}")
        else:
            h_average = np.nanmean([h_monofractal, h_acf])
            print(f"\nAverage H:         {h_average:.4f}")
        print("="*75 + "\n")
    
    results = {
        'hurst_monofractal': h_monofractal,
        'hurst_acf': h_acf,
        'hurst_fukasawa': h_fukasawa,
        'zeta_scaling': zeta_q,
        'moment_orders': q_values,
        'acf_values': acf_vals,
        'acf_lags': acf_lag_vals,
        'cov_values': cov_vals,
        'cov_lags': cov_lag_vals
    }
    
    return results

def load_from_csv(file_path, column_name=None):
    import pandas as pd
    dataframe = pd.read_csv(file_path)
    
    if column_name is None:
        series = dataframe.iloc[:, 0].values
    else:
        series = dataframe[column_name].values
    
    return series.flatten()

if __name__ == '__main__':
    
    np.random.seed(42)
    
    true_hurst_parameter = 0.15
    number_of_samples = 5000
    
    fbm_time_series = generate_fbm_mandelbrot_van_ness(number_of_samples, true_hurst_parameter)
    log_volatility_series = create_log_volatility_from_fbm(fbm_time_series, volatility_scale=0.25)
    
    price_time_series = np.exp(np.cumsum(0.008 * np.random.randn(number_of_samples)))
    realized_variance_time_series = np.abs(np.diff(log_volatility_series)) + 0.005
    
    estimation_results = estimate_hurst_from_volatility(
        log_volatility_series,
        price_input=price_time_series,
        realized_variance_input=realized_variance_time_series,
        verbose=True
    )
    
    figure = plot_estimation_results(
        log_volatility_series,
        estimation_results['hurst_monofractal'],
        estimation_results['zeta_scaling'],
        estimation_results['moment_orders'],
        estimation_results['hurst_acf'],
        estimation_results['acf_values'],
        estimation_results['acf_lags'],
        estimation_results['hurst_fukasawa'],
        estimation_results['cov_values'],
        estimation_results['cov_lags']
    )
    
    plt.show()