import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
import re

OUTPUT_DIR = Path("out/otm/pca2/")
DATA_PATH = Path("data/SPX/otm/bucket_matrix_fine.parquet")
N_COMPONENTS_TO_PLOT = 6

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 150,
})


def parse_column_metadata(column_names):
    pattern = re.compile(r"mat(\d+)_mon(\d+)")
    mat_indices, mon_indices = [], []
    for col in column_names:
        match = pattern.match(col)
        if not match:
            raise ValueError(f"Column {col} does not match expected format mat{{i}}_mon{{j}}")
        mat_indices.append(int(match.group(1)))
        mon_indices.append(int(match.group(2)))
    unique_maturities = sorted(set(mat_indices))
    unique_monomers = sorted(set(mon_indices))
    return mat_indices, mon_indices, unique_maturities, unique_monomers


def load_and_clean(data_path):
    raw = pl.read_parquet(data_path)
    if "DATE" in raw.columns:
        raw = raw.drop("DATE")
    return raw


def compute_first_differences(data_frame):
    column_names = data_frame.columns
    values = data_frame.to_numpy().astype(float)
    differenced = np.full_like(values, np.nan)
    differenced[1:] = values[1:] - values[:-1]
    return pl.DataFrame({col: differenced[:, i] for i, col in enumerate(column_names)})


def drop_nan_rows(matrix):
    nan_mask = np.any(np.isnan(matrix), axis=1)
    return matrix[~nan_mask]


def build_eigenvector_grid(evec, column_names, mat_indices, mon_indices,
                            unique_maturities, unique_monomers):
    n_mat = len(unique_maturities)
    n_mon = len(unique_monomers)
    mat_to_idx = {v: i for i, v in enumerate(unique_maturities)}
    mon_to_idx = {v: i for i, v in enumerate(unique_monomers)}
    grid = np.full((n_mat, n_mon), np.nan)
    for col_idx, col in enumerate(column_names):
        pattern = re.compile(r"mat(\d+)_mon(\d+)")
        match = pattern.match(col)
        mat, mon = int(match.group(1)), int(match.group(2))
        grid[mat_to_idx[mat], mon_to_idx[mon]] = evec[col_idx]
    return grid


def decompose_grid_by_dimension(grid):
    valid_rows = ~np.all(np.isnan(grid), axis=1)
    valid_cols = ~np.all(np.isnan(grid), axis=0)
    subgrid = grid[np.ix_(valid_rows, valid_cols)]
    subgrid_filled = np.where(np.isnan(subgrid), 0.0, subgrid)
    U, s, Vt = np.linalg.svd(subgrid_filled, full_matrices=False)
    maturity_profile = U[:, 0] * s[0]
    moneyness_profile = Vt[0, :]
    svr = s[0] / s[1] if len(s) > 1 and s[1] > 1e-12 else np.inf
    row_indices = np.where(valid_rows)[0]
    col_indices = np.where(valid_cols)[0]
    return maturity_profile, moneyness_profile, svr, row_indices, col_indices


def save_dimensional_profiles(pca, n_components, column_names, mat_indices, mon_indices,
                               unique_maturities, unique_monomers, label, output_dir):
    maturity_values = np.array(unique_maturities)
    moneyness_values = np.array(unique_monomers)

    fig_mat, axes_mat = plt.subplots(1, n_components, figsize=(4 * n_components, 4), sharey=False)
    fig_mon, axes_mon = plt.subplots(1, n_components, figsize=(4 * n_components, 4), sharey=False)

    for k in range(n_components):
        evec = pca.components_[k]
        grid = build_eigenvector_grid(evec, column_names, mat_indices, mon_indices,
                                       unique_maturities, unique_monomers)
        mat_profile, mon_profile, svr, row_idx, col_idx = decompose_grid_by_dimension(grid)

        axes_mat[k].plot(maturity_values[row_idx], mat_profile, marker="o", markersize=4)
        axes_mat[k].axhline(0, color="gray", linewidth=0.6, linestyle="--")
        axes_mat[k].set_xlabel("Maturity")
        axes_mat[k].set_title(f"PC {k+1} | SVR={svr:.2f}")
        if k == 0:
            axes_mat[k].set_ylabel("Maturity profile (left SV)")

        axes_mon[k].plot(moneyness_values[col_idx], mon_profile, marker="o", markersize=4)
        axes_mon[k].axhline(0, color="gray", linewidth=0.6, linestyle="--")
        axes_mon[k].set_xlabel("Moneyness")
        axes_mon[k].set_title(f"PC {k+1} | SVR={svr:.2f}")
        if k == 0:
            axes_mon[k].set_ylabel("Moneyness profile (right SV)")

    fig_mat.suptitle(f"Maturity profiles of PCs — {label}", fontsize=14)
    fig_mat.tight_layout()
    fig_mat.savefig(output_dir / f"maturity_profiles_{label.replace(' ', '_')}.pdf",
                    format="pdf", bbox_inches="tight")
    plt.close(fig_mat)

    fig_mon.suptitle(f"Moneyness profiles of PCs — {label}", fontsize=14)
    fig_mon.tight_layout()
    fig_mon.savefig(output_dir / f"moneyness_profiles_{label.replace(' ', '_')}.pdf",
                    format="pdf", bbox_inches="tight")
    plt.close(fig_mon)


def save_reconstruction_quality(pca, n_components, column_names, mat_indices, mon_indices,
                                 unique_maturities, unique_monomers, label, output_dir):
    fig, axes = plt.subplots(1, n_components, figsize=(4 * n_components, 4))
    for k in range(n_components):
        evec = pca.components_[k]
        grid = build_eigenvector_grid(evec, column_names, mat_indices, mon_indices,
                                       unique_maturities, unique_monomers)
        _, _, svr, row_idx, col_idx = decompose_grid_by_dimension(grid)

        valid_rows = ~np.all(np.isnan(grid), axis=1)
        valid_cols = ~np.all(np.isnan(grid), axis=0)
        subgrid = grid[np.ix_(valid_rows, valid_cols)]
        subgrid_filled = np.where(np.isnan(subgrid), 0.0, subgrid)
        U, s, Vt = np.linalg.svd(subgrid_filled, full_matrices=False)
        rank1_approx = np.outer(U[:, 0], Vt[0, :]) * s[0]
        residual = subgrid_filled - rank1_approx

        vmax = np.max(np.abs(subgrid_filled))
        im = axes[k].imshow(residual, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        axes[k].set_title(f"PC {k+1} residual | SVR={svr:.2f}")
        axes[k].set_xlabel("Moneyness index")
        if k == 0:
            axes[k].set_ylabel("Maturity index")
        fig.colorbar(im, ax=axes[k], shrink=0.8)

    fig.suptitle(f"Rank-one residuals of PC grids — {label}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / f"rank1_residuals_{label.replace(' ', '_')}.pdf",
                format="pdf", bbox_inches="tight")
    plt.close(fig)


data_frame = load_and_clean(DATA_PATH)
column_names = data_frame.columns
mat_indices, mon_indices, unique_maturities, unique_monomers = parse_column_metadata(column_names)

differences_matrix = compute_first_differences(data_frame).to_numpy().astype(float)
clean_differences = drop_nan_rows(differences_matrix)

pca = PCA()
pca.fit(clean_differences)

save_dimensional_profiles(pca, N_COMPONENTS_TO_PLOT, column_names, mat_indices, mon_indices,
                           unique_maturities, unique_monomers, "first differences", OUTPUT_DIR)

save_reconstruction_quality(pca, N_COMPONENTS_TO_PLOT, column_names, mat_indices, mon_indices,
                             unique_maturities, unique_monomers, "first differences", OUTPUT_DIR)

print("Done. Results saved to:", OUTPUT_DIR.resolve())