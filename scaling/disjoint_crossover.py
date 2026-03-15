import pandas as pd
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

def write_summary_file(base_output, variance_windows, bucket_names):
    summary_file = base_output / "nu_summary.txt"
    
    data_dict = {'bucket': bucket_names}
    
    for var_window in variance_windows:
        window_dir = base_output / f"window_{var_window}"
        nu_values = []
        
        for bucket_name in bucket_names:
            estimate_file = window_dir / f"{bucket_name}_nu_estimate.txt"
            nu = extract_nu_from_file(estimate_file)
            nu_values.append(nu if nu is not None else 'N/A')
        
        data_dict[f'window_{var_window}'] = nu_values
    
    summary_df = pd.DataFrame(data_dict)
    
    with open(summary_file, 'w') as f:
        f.write(summary_df.to_string(index=False))

def estimate_all_buckets(subfolder, variance_windows=[10, 20, 30, 50], fit_start_lag=5, fit_end_lag=100):
    
    data = pd.read_parquet("data/" + subfolder + "/put/bucket_matrix.parquet").set_index("DATE")
    
    base_output = Path("plot") / subfolder / "put" / "scaling" / "crossover"
    base_output.mkdir(parents=True, exist_ok=True)
    
    bucket_names = data.columns.tolist()
    
    for var_window in variance_windows:
        window_dir = base_output / f"window_{var_window}"
        window_dir.mkdir(parents=True, exist_ok=True)
        
        for bucket_name in bucket_names:
            iv_values = data[bucket_name].values
            
            result = process_bucket(
                iv_values,
                bucket_name=bucket_name,
                output_dir=str(window_dir),
                variance_window=var_window,
                fit_start_lag=fit_start_lag,
                fit_end_lag=fit_end_lag
            )
    
    write_summary_file(base_output, variance_windows, bucket_names)

if __name__ == '__main__':
    subfolder = "SPX"
    estimate_all_buckets(subfolder, variance_windows=[10, 20, 30, 50])