import pandas as pd
import numpy as np

old = pd.read_parquet("data/SPY/put/bucket_matrix_pd.parquet")
new = pd.read_parquet("data/SPY/put/bucket_matrix.parquet").set_index("DATE")

old.index.name = None
new.index.name = None

common_dates = old.index.intersection(new.index)
common_cols = sorted(old.columns.intersection(new.columns))

old_aligned = old.loc[common_dates, common_cols].sort_index()
new_aligned = new.loc[common_dates, common_cols].sort_index()

diff = np.abs(old_aligned.values - new_aligned.values)
rows, cols = np.where(diff > 0)
print(f"differing cells: {len(rows)}")
for r, c in zip(rows[:20], cols[:20]):
    print(f"date={old_aligned.index[r]} col={common_cols[c]} old={old_aligned.values[r,c]:.6f} new={new_aligned.values[r,c]:.6f}")