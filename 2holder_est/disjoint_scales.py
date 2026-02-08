from univariate_scaling import moment_scaling

import pandas as pd
import numpy as np

def main():
    "From daily observations to 6 months"
    subfolder = "SPY"
    data = pd.read_parquet("data/" + subfolder + "/put/bucket_matrix.parquet")
    print(np.sum(data.isna().to_numpy()))
    test = data["mat1_mon1"]
    moments = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    scaling_dict = moment_scaling(test, 1.0, 126.0, moments)

if __name__ == "__main__":
    main()