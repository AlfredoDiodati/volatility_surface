from univariate_scaling import moment_scaling
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    """Computes realized power variation and scaling
    exponent from daily observations to 6 months"""
    subfolder = "SPY"
    data = pd.read_parquet("data/" + subfolder + "/put/bucket_matrix.parquet")
    print(f"nans in dataset {np.sum(data.isna().to_numpy())}")
    test = data["mat1_mon1"]
    moments = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    scaling_dict = moment_scaling(test, 1.0, 126.0, moments)

    for q in moments:
        ...

if __name__ == "__main__":
    main()