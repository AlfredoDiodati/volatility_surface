import pandas as pd
import numpy as np
from pathlib import Path
from crossover import process_bucket

def extract_nu_from_file(filepath):
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('Estimated nu:'):
                    nu_value = float(line.split(':')[1].strip())
                    return nu_value
    except:
        return None
    return None

def write_summary_file(base_output, variance_windows):
    summary_file = base_output / "nu_summary.txt"
    
    nu_values = []
    
    for var_window in variance_windows:
        window_dir = base_output / f"window_{var_window}"
        estimate_file = window_dir / "underlying_nu_estimate.txt"
        nu = extract_nu_from_file(estimate_file)
        nu_values.append(nu if nu is not None else np.nan)
    
    with open(summary_file, 'w') as f:
        f.write("window nu_estimate\n")
        for window, nu in zip(variance_windows, nu_values):
            nu_str = f"{nu:.6f}" if not np.isnan(nu) else "N/A"
            f.write(f"{window} {nu_str}\n")

def estimate_underlying(subfolder, variance_windows=[10, 20, 30, 50], fit_start_lag=5, fit_end_lag=100):
    
    price_data = pd.read_parquet("data/" + subfolder + "/put/underlying.parquet")
    price = pd.to_numeric(price_data.iloc[:, 0], errors='coerce').dropna().values
    
    log_returns = np.diff(np.log(price))
    
    base_output = Path("plot") / subfolder / "underlying" / "scaling" / "crossover"
    base_output.mkdir(parents=True, exist_ok=True)
    
    for var_window in variance_windows:
        window_dir = base_output / f"window_{var_window}"
        window_dir.mkdir(parents=True, exist_ok=True)
        
        result = process_bucket(
            log_returns,
            bucket_name="underlying",
            output_dir=str(window_dir),
            variance_window=var_window,
            fit_start_lag=fit_start_lag,
            fit_end_lag=fit_end_lag
        )
    
    write_summary_file(base_output, variance_windows)

if __name__ == '__main__':
    subfolder = "SPX"
    estimate_underlying(subfolder, variance_windows=[10, 20, 30, 50])