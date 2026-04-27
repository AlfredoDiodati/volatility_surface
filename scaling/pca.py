import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend
from sklearn.decomposition import PCA
from scipy import stats
from pathlib import Path
import re
import sys

OUTPUT_DIR = Path("out/otm/pca")
DATA_PATH = Path("data/SPX/otm/bucket_matrix.parquet")

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
    result = pl.DataFrame({col: differenced[:, i] for i, col in enumerate(column_names)})
    return result


def drop_nan_rows(matrix):
    nan_mask = np.any(np.isnan(matrix), axis=1)
    return matrix[~nan_mask]


def run_pca(matrix):
    pca = PCA()
    pca.fit(matrix)
    return pca


def save_elbow_plot(pca, label, output_dir):
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    n_components = len(explained)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(range(1, n_components + 1), explained, marker="o", markersize=3, linewidth=1)
    axes[0].set_xlabel("Component rank")
    axes[0].set_ylabel("Explained variance ratio")
    axes[0].set_title(f"Scree plot — {label}")
    axes[0].set_yscale("log")
    axes[0].set_xscale("log")

    axes[1].plot(range(1, n_components + 1), cumulative, marker="o", markersize=3, linewidth=1)
    axes[1].axhline(0.9, color="gray", linestyle="--", linewidth=0.8, label="90%")
    axes[1].axhline(0.95, color="black", linestyle="--", linewidth=0.8, label="95%")
    axes[1].set_xlabel("Component rank")
    axes[1].set_ylabel("Cumulative explained variance")
    axes[1].set_title(f"Cumulative variance — {label}")
    axes[1].legend()

    fig.tight_layout()
    output_path = output_dir / f"elbow_{label.replace(' ', '_')}.pdf"
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def eigenvector_rank_matrix(pca, n_components, unique_maturities, unique_monomers, column_names,
                             mat_indices, mon_indices):
    n_mat = len(unique_maturities)
    n_mon = len(unique_monomers)
    mat_to_idx = {v: i for i, v in enumerate(unique_maturities)}
    mon_to_idx = {v: i for i, v in enumerate(unique_monomers)}

    col_to_grid = {}
    for col, mat, mon in zip(column_names, mat_indices, mon_indices):
        col_to_grid[col] = (mat_to_idx[mat], mon_to_idx[mon])

    eigenvector_matrices = []
    for k in range(n_components):
        evec = pca.components_[k]
        grid = np.full((n_mat, n_mon), np.nan)
        for col_idx, col in enumerate(column_names):
            i, j = col_to_grid[col]
            grid[i, j] = evec[col_idx]
        eigenvector_matrices.append(grid)

    return eigenvector_matrices


def singular_value_ratios(eigenvector_matrices):
    ratios = []
    for grid in eigenvector_matrices:
        valid_rows = ~np.all(np.isnan(grid), axis=1)
        valid_cols = ~np.all(np.isnan(grid), axis=0)
        subgrid = grid[np.ix_(valid_rows, valid_cols)]
        subgrid = np.nan_to_num(subgrid, nan=0.0)
        sv = np.linalg.svd(subgrid, compute_uv=False)
        if len(sv) > 1 and sv[1] > 1e-12:
            ratios.append(sv[0] / sv[1])
        else:
            ratios.append(np.inf)
    return ratios


def save_eigenvector_plots(eigenvector_matrices, singular_value_ratios_list, n_to_plot,
                            unique_maturities, unique_monomers, label, output_dir):
    n_to_plot = min(n_to_plot, len(eigenvector_matrices))
    fig, axes = plt.subplots(2, n_to_plot // 2 + n_to_plot % 2, figsize=(4 * n_to_plot, 8))
    axes = np.array(axes).flatten()

    for k in range(n_to_plot):
        grid = eigenvector_matrices[k]
        ax = axes[k]
        im = ax.imshow(grid, aspect="auto", cmap="RdBu_r",
                       vmin=-np.nanmax(np.abs(grid)), vmax=np.nanmax(np.abs(grid)))
        ax.set_title(f"PC {k+1} | SVR={singular_value_ratios_list[k]:.2f}")
        ax.set_xlabel("Monomer index")
        ax.set_ylabel("Maturity index")
        fig.colorbar(im, ax=ax, shrink=0.8)

    for k in range(n_to_plot, len(axes)):
        axes[k].set_visible(False)

    fig.suptitle(f"Eigenvector grids — {label}", fontsize=14)
    fig.tight_layout()
    output_path = output_dir / f"eigenvectors_{label.replace(' ', '_')}.pdf"
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def hilbert_schmidt_separability_distance(covariance_matrix, n_mat, n_mon, column_names,
                                           mat_indices, mon_indices, unique_maturities, unique_monomers):
    mat_to_idx = {v: i for i, v in enumerate(unique_maturities)}
    mon_to_idx = {v: i for i, v in enumerate(unique_monomers)}

    n_cols = len(column_names)
    col_grid = np.array([(mat_to_idx[mat_indices[i]], mon_to_idx[mon_indices[i]])
                          for i in range(n_cols)])

    C_theta = np.zeros((n_mat, n_mat))
    C_theta_counts = np.zeros((n_mat, n_mat))
    for i in range(n_cols):
        for j in range(n_cols):
            ri, rj = col_grid[i, 0], col_grid[j, 0]
            C_theta[ri, rj] += covariance_matrix[i, j]
            C_theta_counts[ri, rj] += 1
    nonzero = C_theta_counts > 0
    C_theta[nonzero] /= C_theta_counts[nonzero]

    C_mon = np.zeros((n_mon, n_mon))
    C_mon_counts = np.zeros((n_mon, n_mon))
    for i in range(n_cols):
        for j in range(n_cols):
            ci, cj = col_grid[i, 1], col_grid[j, 1]
            C_mon[ci, cj] += covariance_matrix[i, j]
            C_mon_counts[ci, cj] += 1
    nonzero = C_mon_counts > 0
    C_mon[nonzero] /= C_mon_counts[nonzero]

    separable_approx = np.kron(C_theta / np.linalg.norm(C_theta),
                                C_mon / np.linalg.norm(C_mon))
    n_full = covariance_matrix.shape[0]
    separable_approx_aligned = separable_approx[:n_full, :n_full]

    hs_distance = np.linalg.norm(covariance_matrix - separable_approx_aligned, "fro")
    hs_relative = hs_distance / np.linalg.norm(covariance_matrix, "fro")

    return hs_distance, hs_relative, C_theta, C_mon


def save_separability_table(results_dict, output_dir):
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Separability diagnostics}",
        r"\begin{tabular}{lcc}",
        r"\hline",
        r"Dataset & HS distance & Relative HS distance \\",
        r"\hline",
    ]
    for label, (hs_dist, hs_rel) in results_dict.items():
        lines.append(f"{label} & {hs_dist:.4f} & {hs_rel:.4f} \\\\")
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    output_path = output_dir / "separability_table.tex"
    output_path.write_text("\n".join(lines))


def save_svr_plot(svr_levels, label, output_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    finite_svr = [v if np.isfinite(v) else np.nan for v in svr_levels]
    ax.plot(range(1, len(finite_svr) + 1), finite_svr, marker="o", markersize=3)
    ax.axhline(10, color="gray", linestyle="--", linewidth=0.8, label="SVR=10 threshold")
    ax.set_xlabel("Component rank")
    ax.set_ylabel("Singular value ratio (first / second)")
    ax.set_title(f"Eigenvector rank-one test — {label}")
    ax.legend()
    fig.tight_layout()
    output_path = output_dir / f"svr_{label.replace(' ', '_')}.pdf"
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def analyze(matrix, label, column_names, mat_indices, mon_indices,
             unique_maturities, unique_monomers, output_dir, n_evec_to_plot=8):
    clean_matrix = drop_nan_rows(matrix)

    pca = run_pca(clean_matrix)
    save_elbow_plot(pca, label, output_dir)

    n_components_to_inspect = min(20, clean_matrix.shape[1])
    evec_matrices = eigenvector_rank_matrix(
        pca, n_components_to_inspect, unique_maturities, unique_monomers,
        column_names, mat_indices, mon_indices
    )

    svr_list = singular_value_ratios(evec_matrices)
    save_svr_plot(svr_list, label, output_dir)
    save_eigenvector_plots(evec_matrices, svr_list, n_evec_to_plot,
                           unique_maturities, unique_monomers, label, output_dir)

    covariance_matrix = np.cov(clean_matrix.T)
    n_mat = len(unique_maturities)
    n_mon = len(unique_monomers)
    hs_dist, hs_rel, C_theta, C_mon = hilbert_schmidt_separability_distance(
        covariance_matrix, n_mat, n_mon, column_names, mat_indices, mon_indices,
        unique_maturities, unique_monomers
    )

    return hs_dist, hs_rel


data_frame = load_and_clean(DATA_PATH)
column_names = data_frame.columns
mat_indices, mon_indices, unique_maturities, unique_monomers = parse_column_metadata(column_names)

levels_matrix = data_frame.to_numpy().astype(float)
differenced_frame = compute_first_differences(data_frame)
differences_matrix = differenced_frame.to_numpy().astype(float)

separability_results = {}

hs_dist_lev, hs_rel_lev = analyze(
    levels_matrix, "levels", column_names, mat_indices, mon_indices,
    unique_maturities, unique_monomers, OUTPUT_DIR
)
separability_results["Levels"] = (hs_dist_lev, hs_rel_lev)

hs_dist_dif, hs_rel_dif = analyze(
    differences_matrix, "first differences", column_names, mat_indices, mon_indices,
    unique_maturities, unique_monomers, OUTPUT_DIR
)
separability_results["First differences"] = (hs_dist_dif, hs_rel_dif)

save_separability_table(separability_results, OUTPUT_DIR)

print("Done. Results saved to:", OUTPUT_DIR.resolve())