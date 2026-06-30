# Volatility Surface

Code for modeling and forecasting the implied volatility (IV) surface of SPX (S&P 500 index) out-of-the-money options. The project fits and compares a family of score-driven and state-space panel models using both a classical model confidence set (MCS) and a sequential MCS (sMCS). It also includes a Monte Carlo study and thesis-figure scripts.

---

## Directory structure

```
cleaning/       data ingestion and filtering pipeline
models/         model implementations and shared internals
fit/            estimation, forecasting, evaluation, and Monte Carlo scripts
plot/           standalone visualization and table scripts
test/           smoke tests and unit tests
data/           raw and processed option data (not tracked)
out/            model outputs and results (not tracked)
```

---

## Device selection

Before running any model script, run the one-time device benchmark:

```bash
python benchmark_device.py
```

This benchmarks a representative `lax.scan` kernel (p×p linear solve + score update, T=2000 steps) on both CPU and GPU, then writes `device.json` with the faster choice. All entry-point scripts read this file via `_device.py` at startup and call `jax.config.update("jax_default_device", ...)` accordingly — no device logic lives inside the model scripts themselves.

If no GPU is detected, or if free GPU VRAM is below 4000 MiB, the benchmark skips GPU and writes `{"device": "cpu"}` directly. The 4000 MiB threshold is chosen to exclude GPUs with 4 GiB of total VRAM (e.g. the NVIDIA GeForce GTX 1050 Ti, the GPU on the development machine), which reliably run out of memory during the filter scans in this project despite appearing to have enough free memory at process start.

`device.json` is machine-specific and gitignored. Re-run `benchmark_device.py` whenever the hardware changes or after a system reboot where baseline GPU memory usage differs.

---

## Pipeline

### Step 1 — Data cleaning

Run in order from the project root:

```bash
python cleaning/filter_all.py
python cleaning/structure_all.py
python cleaning/head_builder.py   # optional: generates CSV previews
```

### Step 2 — Monte Carlo study

```bash
python fit/MC_lmSD.py
```

Simulates data from the FI-SD (lmSD) model with oracle parameters, fits all models on each simulated sample, and produces LaTeX performance tables in `out/SPX/mc/simulate_lmSD_sjit/`.

### Step 3 — Estimation and forecasting

```bash
python fit/full_forecast_fixed.py
```

Fits every model on the training half of the sample and produces one-step-ahead and multi-step-ahead predictions for the test half. Results land in `out/SPX/otm/full_performance_fixed/`.

### Step 4a — Classical MCS

```bash
python fit/full_mcs_fixed.py
```

Runs the Hansen (2011) model confidence set on the predictions from step 3, using block-bootstrap with multiple block lengths and significance levels. Writes LaTeX tables to `out/SPX/otm/full_mcs_fixed/`.

### Step 4b — Sequential MCS

```bash
python fit/full_smcs_fixed.py
```

Runs the Arnold, Gavrilopoulos, Schulz & Ziegel (2026) sequential MCS on the same predictions. Writes LaTeX tables and multi-page PDFs (e-process trajectories, membership heatmaps) to `out/SPX/otm/full_smcs_fixed/` and `plot/SPX/otm/full_smcs_fixed/`.

---

## File reference

### Root utilities

| File | Description |
|------|-------------|
| `benchmark_device.py` | One-time device benchmark. Queries free GPU VRAM, runs a representative `lax.scan` kernel on CPU and GPU (with JIT warmup), and writes the faster device to `device.json`. See the Device selection section above. |
| `_device.py` | Reads `device.json` and returns the corresponding `jax.Device`. Falls back to CPU with a warning if `device.json` is missing or if the selected GPU is no longer available. Imported by all entry-point scripts. |

### `cleaning/`

| File | Description |
|------|-------------|
| `filter_all.py` | Reads raw end-of-day SPX option CSVs from `data/SPX/raw/`, separates puts and calls, computes moneyness (K/S) and calendar maturity (days), applies quality screens (IV in [0.05, 0.70], maturity in [7, 360] days, OTM only via delta), assigns 4 maturity buckets and 6 moneyness buckets by delta, and streams results to `data/SPX/otm/filtered.parquet` and `data/SPX/otm/checks.parquet`. |
| `structure_all.py` | Reads `filtered.parquet`, computes log-IV, normalises maturity to fractions of 255 days, and creates the flat `data/SPX/otm/full.parquet` (columns: DATE, logIV, level, moneyness, maturity, bucket\_idx) used as input by all models. Also saves per-bucket descriptive stats. |
| `head_builder.py` | Creates a 20-row CSV preview of each Parquet file in `data/SPX/otm/` into `data/SPX/otm/head/`, for quick inspection without loading full files. |
| `flatten.sh` | After unzipping OptionDX archives (which place each year's CSVs in a subfolder), this script moves all files from nested subdirectories up to the target root directory and removes the empty subdirectories. Usage: `bash cleaning/flatten.sh data/SPX/raw`. |

### `models/`

#### Public model modules

Each public module exposes `fit`, `forecast`, `forecast_rolling_h`, and `simulate_panel` (or `simulate`). All score-driven models use Student-t innovations with estimated degrees of freedom `nu`.

| File | Label | Description |
|------|-------|-------------|
| `ss.py` | SS | Linear Gaussian state-space model. Factor loadings evolve as an AR(1) latent state. Uses the Kalman filter with a collapsed-observation trick (`fit_collapsed`) for efficiency. |
| `adjSD.py` | SD | Score-driven model with AR(1) coefficient dynamics (transition matrix `B` + score innovation `A`). One-step Fisher-score update re-weighted by the Student-t likelihood. |
| `lmSD.py` | FI-SD | Fractionally integrated score-driven model. Coefficient dynamics are driven by a truncated power-law convolution of past scores stored in a circular buffer, giving long-memory persistence. |
| `ff_SD.py` | Mk-SD | Multi-frequency score-driven model. Coefficient dynamics are represented as a weighted sum of `K+1` exponential components whose decay rates `{λ_k}` and weights `{w_k}` approximate a power-law kernel. Score propagation uses the kappa fixed-point system. |
| `ff_SS.py` | Mk-SS | Multi-frequency state-space analogue of Mk-SD. The `K+1` exponential components form the latent state; the Kalman filter (Woodbury vectorised variant) is used for inference. |
| `MSMSD.py` | MSM | Markov-switching multifractal score-driven model. `K` binary Markov chains govern the volatility regime of the score innovations; the filter marginalises over all `2^K` hidden states in log-space. |

All models support a `bucket_idx` column in the covariate matrix `M` that selects a bucket-specific intercept shift `omega[bucket_idx]`, giving 24 separate level corrections (4 maturity × 6 moneyness buckets).

The multi-frequency models (Mk-SD, Mk-SS, MSM) are run for K ∈ {1, 2, 3, 5, 10}.

#### No-bucket variants

Separate implementations of each model used by the Monte Carlo study (`fit/MC_lmSD.py`). They differ from the bucket versions in two structural ways:

**1. No bucket-intercept mechanism.** The bucket versions carry a `bucket_idx` column as the last column of `M` and an `omega_load` (or `omega`) parameter vector. At every filter step they index `omega_load[bidx_t]` and concatenate the result onto the base covariates to form the design row. The no-bucket versions drop this entirely — there is no `omega` parameter and the design matrix `Z_t` equals the base covariates directly.

**2. Pre-assembled design matrix passed into the scan.** Because the MC simulation uses a fixed synthetic observation grid (same N observations every period), the full `T×N×p` design matrix can be constructed once outside the filter. The no-bucket filter receives the pre-built `base_covariates` tensor as a scan input, so each step just slices `Z_t = base_covariates[t]` without any dynamic concatenation. The bucket filter instead constructs `Z_t` inside the scan by looking up and appending the bucket column at each step. This pre-assembly eliminates per-step allocation and makes the computation cheaper when N and p are small and fixed.

As a consequence, the no-bucket filters also do not store the intermediate `b_hist` array (the per-step latent state history), since the MC study does not need it. The `fit` return value therefore omits `b_hist`, keeping memory usage low over many Monte Carlo replications.

| File | Used by |
|------|---------|
| `_adjSD_nobucket.py` | `fit/MC_lmSD.py` |
| `_lmSD_nobucket.py` | `fit/MC_lmSD.py` |
| `_ff_SD_nobucket.py` | `fit/MC_lmSD.py` |
| `_ff_SS_nobucket.py` | `fit/MC_lmSD.py` |
| `_MSMSD_nobucket.py` | `fit/MC_lmSD.py` |
| `_ss_nobucket.py` | `fit/MC_lmSD.py` |

#### Shared internals

| File | Description |
|------|-------------|
| `_solver.py` | Custom JAX L-BFGS implementation with circular history buffer, two-loop recursion, Wolfe backtracking line search, and best-iterate tracking. All optimisation runs through this. |
| `_kalman.py` | Multiple Kalman filter variants: `_filter` (full, stores all quantities), `_filter_light` (log-likelihood only), `_filter_light_univariate` (univariate treatment, replaces O(p³) Cholesky with p scalar divisions), `_filter_light_vec` / `_filter_vec` (Woodbury-identity vectorised form, efficient when state dim ≪ N), `_filter_light_chol` / `_filter_chol` (direct N×N Cholesky, stable when state dim ≥ N). Also includes `_fit` (generic MLE driver), `_fit_collapsed` (collapsed-observation MLE), and `_simulation`. Based on Durbin & Koopman (2012). |
| `__init__.py` | Empty. |

### `fit/`

| File | Description |
|------|-------------|
| `full_forecast_fixed.py` | Main estimation and forecasting script. Splits sample at midpoint. For each model (SS, SD, FI-SD, Mk-SD K∈{1,2,3,5,10}, Mk-SS K∈{1,2,3,5,10}, MSM K∈{1,2,3,5,10}) fits on the training half and rolls through the test half to produce: one-step-ahead predictions, multi-step-ahead predictions at h∈{5,22,66,260}, out-of-sample log-likelihoods, and latent state histories. All outputs are written as Parquet to `out/SPX/otm/full_performance_fixed/`. |
| `full_mcs_fixed.py` | Classical MCS evaluation. Loads predictions from `full_forecast_fixed.py`, computes MSE, MAE, neg-log-lik, AIC, and BIC losses, and runs the Hansen (2011) MCS via stationary block bootstrap (`mcs.py`) at multiple block lengths (1,5,10,15,20) and significance levels (0.05, 0.10, 0.25). Writes LaTeX tables to `out/SPX/otm/full_mcs_fixed/`. |
| `full_smcs_fixed.py` | Sequential MCS evaluation. Same inputs as above, but uses `smcs.py` to run the Arnold, Gavrilopoulos, Schulz & Ziegel (2026) sMCS. For each (horizon, metric) combination: computes adjusted e-values per model over time, saves a CSV grid, generates e-process trajectory PDFs and membership-heatmap PDFs, and writes LaTeX tables with sMCS significance stars. |
| `smcs.py` | Sequential MCS implementation (Arnold, Gavrilopoulos, Schulz & Ziegel, 2026). Input: T×M loss matrix (Polars DataFrame or JAX array). Constructs pairwise loss-differential e-processes (strong or uniformly-weak hypothesis), merges them per model, adjusts for simultaneous inference via isotonic regression in log-space, and returns surviving models, e-process history, and adjusted e-values. Optionally enforces running intersection. |
| `mcs.py` | Classical MCS implementation. Supports `TR` (pairwise t-statistic range) and `Tmax` (dot-average t-statistic) elimination rules. Uses stationary block bootstrap with HAC variance estimation (Bartlett kernel). Also exposes `loss_differentials` (pairwise loss difference DataFrame) and `diebold_mariano_test` (two-sided DM test). |
| `MC_lmSD.py` | Monte Carlo study. Oracle FI-SD parameters are hard-coded. For each sample length T∈{2000}: simulates a panel from the lmSD model, fits all models (including an oracle lmSD that starts at true parameters with 0 iterations), evaluates one-step and multi-step MSE/MAE/log-lik/AIC/BIC, caches fitted parameters (`.npz`), incrementally writes results to `.jsonl`, and produces LaTeX performance tables. GPU memory management and sleep inhibition included. |
| `_forecast_metrics.py` | Forecast metric helpers. `per_step_mse`, `per_step_mae` (per-date averages as lists for sMCS/MCS input), `per_step_aic`, `per_step_bic` (per-step penalised losses), `compute_mse`, `compute_mae`, `compute_aic`, `compute_bic` (scalar aggregates). |

### `plot/`

All plot scripts are standalone — run from the project root and write PDFs or show figures interactively.

| File | Description |
|------|-------------|
| `fig1.py` | Approximation quality of the multi-frequency sum-of-exponentials kernel to the power-law target τ^{-η}. Shows convergence as K increases, with and without truncation. Thesis figure 1. |
| `fig3.py` | Same approximation, restructured figure layout for the thesis (2×2 panel). |
| `table_buckets.py` | Computes and prints descriptive statistics (mean IV, std, moneyness, maturity) for each of the 24 moneyness × maturity bucket cells. Suitable for a LaTeX `tabular`. |
| `betas_over_time.py` | Time-series of estimated factor betas (intercept, moneyness, maturity, bucket offset) from multiple fitted models. Highlights crisis periods and the in-sample/out-of-sample split. Uses fitted parameter files from `out/SPX/otm/full_performance_fixed/`. |
| `bis_ffSD_K5_over_time.py` | Time-series of the K+1 latent score-component states b_i(s) for Mk-SD with K=5, plotted per coefficient dimension, coloured by component index. Reads `b_ffSD_K5.parquet`. |
| `bis_ffSS_K5_over_time.py` | Same as above for Mk-SS K=5. Reads `b_ffSS_K5.parquet`. |
| `bis_ffSD_ffSS_K5_over_time.py` | Combined side-by-side plot of b_i(s) states for both Mk-SD and Mk-SS at K=5, with crisis period shading and a train/test split line. |

### `test/`

| File | Description |
|------|-------------|
| `test_ffSD_jit.py` | Smoke test for `ff_SD`. Generates synthetic panel data, runs `fit`, `forecast`, and `forecast_rolling_h`, and checks that outputs have correct shapes and that a second JIT-compiled call is faster than the first. |
| `test_solve_kappas.py` | Unit test for the kappa fixed-point solver in `ff_SD`. Verifies that the fixed-point residual is below 1e-6 and that the solve completes in under 200ms across several (p, K, η, φ) combinations. |

---

## Data

Raw data files (`data/SPX/raw/*.txt`) are end-of-day SPX option chains covering 2010–2021, stored as 7-zip archives in `data/SPX/archives/zip/`. Extract before running the cleaning pipeline.

The processed files produced by the pipeline are:

| File | Description |
|------|-------------|
| `data/SPX/otm/filtered.parquet` | All OTM options after quality filtering, with bucket assignments and flag columns. |
| `data/SPX/otm/checks.parquet` | Cumulative filter diagnostic counts (rows removed by each criterion). |
| `data/SPX/otm/full.parquet` | Final model input: DATE, logIV, level, moneyness, maturity, bucket\_idx. |
| `data/SPX/otm/underlying.parquet` | Daily SPX underlying price series. |
| `data/SPX/otm/head/*.csv` | 20-row CSV previews of each Parquet file. |

---

## Requirements

- Python ≥ 3.11
- JAX
- Polars
- PyArrow
- NumPy
- Pandas
- Matplotlib
