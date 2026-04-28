import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
import re

OUTPUT_DIR = Path("out/otm/pca3/")
DATA_PATH = Path("data/SPX/otm/bucket_matrix_fine.parquet")

MONEYNESS_BOUNDARY_INDEX = 9

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
            raise ValueError(f"Column {col} does not match expected format")
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


def build_index_map(column_names, mat_indices, mon_indices,
                     unique_maturities, unique_monomers):
    mat_to_idx = {v: i for i, v in enumerate(unique_maturities)}
    mon_to_idx = {v: i for i, v in enumerate(unique_monomers)}
    col_to_mat = {}
    col_to_mon = {}
    for col_idx, col in enumerate(column_names):
        col_to_mat[col_idx] = mat_to_idx[mat_indices[col_idx]]
        col_to_mon[col_idx] = mon_to_idx[mon_indices[col_idx]]
    return col_to_mat, col_to_mon


def marginalize_covariance_over_moneyness(full_cov, col_to_mat, col_to_mon,
                                           n_mat, n_mon):
    mat_cov = np.zeros((n_mat, n_mat))
    mat_cov_counts = np.zeros((n_mat, n_mat))
    n_cols = full_cov.shape[0]
    for i in range(n_cols):
        for j in range(n_cols):
            ri = col_to_mat[i]
            rj = col_to_mat[j]
            mat_cov[ri, rj] += full_cov[i, j]
            mat_cov_counts[ri, rj] += 1
    nonzero = mat_cov_counts > 0
    mat_cov[nonzero] /= mat_cov_counts[nonzero]
    return mat_cov


def marginalize_covariance_over_maturity(full_cov, col_to_mat, col_to_mon,
                                          n_mat, n_mon, mon_subset=None):
    mon_cov = np.zeros((n_mon, n_mon))
    mon_cov_counts = np.zeros((n_mon, n_mon))
    n_cols = full_cov.shape[0]
    for i in range(n_cols):
        for j in range(n_cols):
            ci = col_to_mon[i]
            cj = col_to_mon[j]
            if mon_subset is not None:
                if ci not in mon_subset or cj not in mon_subset:
                    continue
            mon_cov[ci, cj] += full_cov[i, j]
            mon_cov_counts[ci, cj] += 1
    nonzero = mon_cov_counts > 0
    mon_cov[nonzero] /= mon_cov_counts[nonzero]
    return mon_cov


def eigenvalue_decay_plot(eigenvalues, label, xlabel, output_dir):
    n = len(eigenvalues)
    ranks = np.arange(1, n + 1)
    eigenvalues_normalized = eigenvalues / eigenvalues[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(ranks, eigenvalues_normalized, marker="o", markersize=4, label="empirical")
    q2_ref = ranks[0] ** 2 / ranks ** 2
    q4_ref = ranks[0] ** 4 / ranks ** 4
    axes[0].plot(ranks, q2_ref, linestyle="--", linewidth=0.8, color="gray", label=r"$q^{-2}$")
    axes[0].plot(ranks, q4_ref, linestyle=":", linewidth=0.8, color="black", label=r"$q^{-4}$")
    axes[0].set_xlabel("Rank")
    axes[0].set_ylabel("Normalized eigenvalue")
    axes[0].set_title(f"Eigenvalue decay — {label} (linear)")
    axes[0].legend()

    axes[1].loglog(ranks, eigenvalues_normalized, marker="o", markersize=4, label="empirical")
    axes[1].loglog(ranks, q2_ref, linestyle="--", linewidth=0.8, color="gray", label=r"$q^{-2}$")
    axes[1].loglog(ranks, q4_ref, linestyle=":", linewidth=0.8, color="black", label=r"$q^{-4}$")
    axes[1].set_xlabel("Rank (log)")
    axes[1].set_ylabel("Normalized eigenvalue (log)")
    axes[1].set_title(f"Eigenvalue decay — {label} (log-log)")
    axes[1].legend()

    fig.tight_layout()
    output_path = output_dir / f"eigenvalue_decay_{label.replace(' ', '_')}.pdf"
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def eigenvector_plot(eigenvectors, eigenvalues, axis_values, axis_label, label, n_to_plot, output_dir):
    n_to_plot = min(n_to_plot, eigenvectors.shape[1])
    fig, axes = plt.subplots(1, n_to_plot, figsize=(4 * n_to_plot, 4), sharey=False)
    if n_to_plot == 1:
        axes = [axes]
    variance_fractions = eigenvalues / eigenvalues.sum()
    for k in range(n_to_plot):
        axes[k].plot(axis_values, eigenvectors[:, k], marker="o", markersize=4)
        axes[k].axhline(0, color="gray", linewidth=0.6, linestyle="--")
        axes[k].set_xlabel(axis_label)
        axes[k].set_title(f"Mode {k+1} | {variance_fractions[k]*100:.1f}%")
        if k == 0:
            axes[k].set_ylabel("Eigenvector loading")
    fig.suptitle(f"Eigenmodes — {label}", fontsize=14)
    fig.tight_layout()
    output_path = output_dir / f"eigenmodes_{label.replace(' ', '_')}.pdf"
    fig.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def power_law_fit(eigenvalues):
    ranks = np.arange(1, len(eigenvalues) + 1)
    log_ranks = np.log(ranks)
    log_eigs = np.log(eigenvalues / eigenvalues[0])
    slope, intercept = np.polyfit(log_ranks, log_eigs, 1)
    return slope


def save_operator_summary(results, output_dir):
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Estimated power-law exponent of eigenvalue decay by dimension}",
        r"\begin{tabular}{lcc}",
        r"\hline",
        r"Dimension & Estimated exponent & Closest reference \\",
        r"\hline",
    ]
    for label, slope in results.items():
        closest = r"$q^{-2}$" if abs(slope + 2) < abs(slope + 4) else r"$q^{-4}$"
        lines.append(f"{label} & {slope:.2f} & {closest} \\\\")
    lines += [
        r"\hline",
        r"\end{tabular}",
        r"\end{table}",
    ]
    output_path = output_dir / "operator_order_table.tex"
    output_path.write_text("\n".join(lines))


data_frame = load_and_clean(DATA_PATH)
column_names = data_frame.columns
mat_indices, mon_indices, unique_maturities, unique_monomers = parse_column_metadata(column_names)

differences_matrix = compute_first_differences(data_frame).to_numpy().astype(float)
clean_matrix = drop_nan_rows(differences_matrix)

n_mat = len(unique_maturities)
n_mon = len(unique_monomers)
maturity_values = np.array(unique_maturities)
moneyness_values = np.array(unique_monomers)

full_cov = np.cov(clean_matrix.T)

col_to_mat, col_to_mon = build_index_map(
    column_names, mat_indices, mon_indices, unique_maturities, unique_monomers
)

mat_cov = marginalize_covariance_over_moneyness(full_cov, col_to_mat, col_to_mon, n_mat, n_mon)
mat_eigenvalues, mat_eigenvectors = np.linalg.eigh(mat_cov)
mat_eigenvalues = mat_eigenvalues[::-1]
mat_eigenvectors = mat_eigenvectors[:, ::-1]

eigenvalue_decay_plot(mat_eigenvalues, "maturity direction", "Rank", OUTPUT_DIR)
eigenvector_plot(mat_eigenvectors, mat_eigenvalues, maturity_values,
                  "Maturity bucket", "maturity direction", 6, OUTPUT_DIR)

mon_cov_full = marginalize_covariance_over_maturity(
    full_cov, col_to_mat, col_to_mon, n_mat, n_mon, mon_subset=None
)

put_indices = set(range(MONEYNESS_BOUNDARY_INDEX))
call_indices = set(range(MONEYNESS_BOUNDARY_INDEX, n_mon))

mon_cov_puts = marginalize_covariance_over_maturity(
    full_cov, col_to_mat, col_to_mon, n_mat, n_mon, mon_subset=put_indices
)
mon_cov_calls = marginalize_covariance_over_maturity(
    full_cov, col_to_mat, col_to_mon, n_mat, n_mon, mon_subset=call_indices
)

for mon_cov, label, mon_values in [
    (mon_cov_full[np.ix_(range(n_mon), range(n_mon))], "moneyness full", moneyness_values),
    (mon_cov_puts[np.ix_(sorted(put_indices), sorted(put_indices))],
     "moneyness puts", moneyness_values[sorted(put_indices)]),
    (mon_cov_calls[np.ix_(sorted(call_indices), sorted(call_indices))],
     "moneyness calls", moneyness_values[sorted(call_indices)]),
]:
    eigs, evecs = np.linalg.eigh(mon_cov)
    eigs = eigs[::-1]
    evecs = evecs[:, ::-1]
    eigenvalue_decay_plot(eigs, label, "Rank", OUTPUT_DIR)
    eigenvector_plot(evecs, eigs, mon_values, "Moneyness bucket", label, 6, OUTPUT_DIR)

power_law_results = {}
for mat_label, eigs in [
    ("Maturity", mat_eigenvalues),
]:
    slope = power_law_fit(eigs)
    power_law_results[mat_label] = slope
    print(f"{mat_label}: estimated exponent = {slope:.2f}")

for mon_label, mon_cov, idx_set in [
    ("Moneyness full", mon_cov_full, range(n_mon)),
    ("Moneyness puts", mon_cov_puts, sorted(put_indices)),
    ("Moneyness calls", mon_cov_calls, sorted(call_indices)),
]:
    sub = mon_cov[np.ix_(list(idx_set), list(idx_set))]
    eigs, _ = np.linalg.eigh(sub)
    eigs = eigs[::-1]
    slope = power_law_fit(eigs)
    power_law_results[mon_label] = slope
    print(f"{mon_label}: estimated exponent = {slope:.2f}")

save_operator_summary(power_law_results, OUTPUT_DIR)
print("Done. Results saved to:", OUTPUT_DIR.resolve())