# Non-technical summary

Code for modeling and forecasting the implied volatility (IV) surface of SPX (S&P 500 index) out-of-the-money options used for my Econometrics and Operations Research Masters Thesis at VU Amsterdam. 

The aim of the project is to approximate Fractionally Integrated (FI) factor models for the IV surface with a Structural, Markovian representation. This is possible because the auto-covariance of FI processes can be effectively approximated by a sum of exponential functions. This means that the fractionally integrated process can be approximated by a finite sum of affine processes, which are much cheaper to track than the true model, which instead has an expanding state dimension that increases linearly with the sample size, which makes computations expensive in multivariate models, or structural State-Space models that require a FI component.

The advantage of the specific approximation approach taken in the thesis is that the approximation requires the same number of parameters regardless of the lags of interest to approximate, whereas similar approaches (e.g. Markovian lifting) typically require an increasing number of parameters to capture higher lag horizons (effectively making their model semi-nonparametric). 

The code implements various parameter-driven and observation-driven models for the IV surface, both short-memory and fractionally integrated, and their Markovian approximation counterparts. Then a Monte-Carlo simulation study confirms that the approximation procedure is statistically identical to the true FI process, whereas short-memory processes diverge in terms of Mean Squared Error. Finally, a forecasting exercise on real SPX Options is made.

A copy of the thesis is also available in the projects directory.

## Abstract of the thesis

Fractional Integration and Random Multiplicative Cascades are the most widely used methods to model time series with Long Range Dependence, but are often challenging to use due to their computational requirements. Instead, this thesis presents a procedure that approximates Long Range Dependence parsimoniously. Unlike Fractional Integration, this procedure generates long-range State-Space models with a finite state dimension, and unlike multiplicative cascades it can be efficiently estimated by the Kalman Filter. The approximation is also adjusted for Score-Driven models, making it possible to use non-Gaussian features without simulation-based estimation of the parameters. The methodology is applied to Score-Driven and State-Space dynamic factor models for the implied volatility surface of the S&P 500. Monte Carlo studies demonstrate that the approximation procedure effectively replicates true Long Range Dependence characteristics, outperforming short-memory models in point forecasts at multiple horizons. When tested on real option data it is found that both the approximating and true models have similar forecasting performance to short-memory models at different horizons.

---

## Installation

### 1. Get the source code

**From GitHub:**
```bash
git clone git@github.com:AlfredoDiodati/volatility_surface.git
cd volatility_surface
```

**From a ZIP archive:** extract the archive and navigate into the resulting folder:
```bash
unzip volatility_surface.zip
cd volatility_surface
```

### 2. Create a Python environment

Python 3.11 or later is required. Using a dedicated virtual environment is strongly recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Or with conda:

```bash
conda create -n volsurface python=3.11
conda activate volsurface
```

### 3. Install JAX

JAX installation depends on your hardware. From inside the activated environment:

**CPU only:**
```bash
pip install -U jax
```

**GPU (CUDA 12):**
```bash
pip install -U "jax[cuda12]"
```

See the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for other CUDA versions or platform-specific instructions.

### 4. Install the remaining dependencies

```bash
pip install polars pyarrow numpy pandas matplotlib
```

### 5. Install the package in editable mode

```bash
pip install -e .
```

This registers the project root as a package so that imports such as `from models.ff_SD import fit` and `from fit.smcs import smcs` resolve correctly from any entry-point script.

### 6. Create the untracked directories

`data/` and `out/` are gitignored and must be created manually before running any script:

```bash
mkdir -p data/SPX/raw data/SPX/archives/zip out
```

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

Download the SPX end-of-day option chain data from [OptionsDX](https://www.optionsdx.com). Place the downloaded ZIP archives in `data/SPX/archives/zip/`, then extract them. OptionsDX archives place each year's files in a subdirectory; flatten everything into a single folder:

```bash
bash cleaning/flatten.sh data/SPX/raw
```

After flattening, `data/SPX/raw/` should contain one `.txt` file per trading day. Then run the cleaning scripts in order from the project root:

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

## Script reference

### Device setup

#### `benchmark_device.py`

Run once before using any model script. Checks for an NVIDIA GPU via `nvidia-smi`; if found, queries free VRAM. If free VRAM is below 4000 MiB the GPU is skipped immediately and `{"device": "cpu"}` is written to `device.json`. Otherwise it benchmarks a representative `lax.scan` kernel (T=2000 timesteps, N=30 observations, p=4 coefficients, 3 timed repetitions with JIT warmup) on both CPU and GPU using `jax.block_until_ready` for accurate timing, then writes the faster device. The XLA memory environment variables (`XLA_PYTHON_CLIENT_PREALLOCATE=false`, `TF_GPU_ALLOCATOR=cuda_malloc_async`) are set before JAX is imported whenever `nvidia-smi` is present, to prevent JAX from pre-allocating the entire GPU memory and skewing the VRAM check.

`device.json` is machine-specific and gitignored. Re-run whenever the hardware configuration changes.

#### `_device.py`

Reads `device.json` and returns a `jax.Device`. Imported at module load time by every entry-point script, which then calls `jax.config.update("jax_default_device", ...)`. Falls back to CPU with a printed warning if `device.json` is missing (prompting the user to run `benchmark_device.py` first) or if the JSON names a GPU that is not available at runtime (e.g. the script is run on a different machine).

---

### Step 1 — Cleaning scripts

#### `cleaning/flatten.sh`

Bash script that moves all files from nested subdirectories of a target directory up to the target root, then removes the resulting empty subdirectories. Takes the target path as its only argument. Handles filename collisions by appending a numeric counter suffix. This is needed because OptionsDX delivers ZIP archives with one subdirectory per year; after extraction the raw CSVs sit one level too deep for `filter_all.py` to find them.

```bash
bash cleaning/flatten.sh data/SPX/raw
```

After running, `data/SPX/raw/` should contain one `.txt` file per trading day with no subdirectories.

#### `cleaning/filter_all.py`

Reads every `.txt` CSV from `data/SPX/raw/` (sorted alphabetically, one file per trading day) and processes them one at a time to keep peak memory usage bounded. For each file the script:

1. Parses the CSV, strips whitespace and bracket characters from column names, and casts numeric fields from string.
2. Separates puts and calls, computing moneyness `K/S`, calendar maturity in days `(expire − quote)`, and a `DATE` string in YYYYMMDD format.
3. Assigns **4 maturity buckets**: [7, 45], (45, 90], (90, 180], (180, 360].
4. Assigns **6 moneyness buckets** by delta: deep OTM put (−0.125, 0), OTM put (−0.375, −0.125], ATM put (−0.5, −0.375]; ATM call [0.375, 0.5), OTM call [0.125, 0.375), deep OTM call (0, 0.125).
5. Flags rows for removal: maturity outside [7, 360] days, IV outside [0.05, 0.70], any missing value in key fields, and non-OTM options (puts with delta ≤ −0.5 or ≥ 0; calls with delta ≥ 0.5 or ≤ 0).
6. Streams two outputs via PyArrow `ParquetWriter` so the full multi-year dataset is never loaded into memory at once:
   - **`data/SPX/otm/filtered.parquet`** — rows passing all filters, with all flag and bucket columns retained.
   - **`data/SPX/otm/checks/all.parquet`** — one cumulative row with counts for every filter flag and moneyness/maturity bin, accumulated across all trading days. Useful for diagnosing how many rows each criterion removes.

#### `cleaning/structure_all.py`

Reads `data/SPX/otm/filtered.parquet` and produces three output files.

**`data/SPX/otm/full.parquet`** is the primary model input, built lazily via Polars `scan_parquet` / `sink_parquet`:
- `logIV = log(IV)` (natural log of implied volatility).
- `level = 1` (intercept covariate).
- `moneyness = K/S`.
- `maturity = MATURITY / 255` (maturity in fractions of a trading year).
- `bucket_idx = (MATURITY_BUCKET − 1) × 6 + (MONEYNESS_BUCKET − 1)`, ranging from 0 to 23. Ordered first by maturity bucket, then by moneyness bucket.

**`data/SPX/otm/underlying.parquet`** — one row per trading day with the SPX underlying price.

**`data/SPX/otm/bucket.parquet`** — one representative option per (DATE, MATURITY_BUCKET, MONEYNESS_BUCKET) cell. The representative is the option closest to the bucket centroid by the weighted distance `10 × (Δ_midpoint − Δ)² + (τ_midpoint − τ)²`, where the factor 10 down-weights maturity deviations relative to delta deviations. Additional columns include `moneyness²`, `maturity × moneyness` interaction, and a full set of one-hot bucket dummies. **`data/SPX/otm/bucket_matrix.parquet`** pivots this to a DATE × joint-bucket matrix of logIV values.

#### `cleaning/head_builder.py`

For every `.parquet` file found in `data/SPX/otm/`, reads the first 20 rows with PyArrow and writes a CSV to `data/SPX/otm/head/` with the same stem. Skips files whose head CSV already exists. Useful for quickly inspecting column names and dtypes without loading full datasets. No modelling dependency; safe to run at any point after `filter_all.py`.

---

### Step 2 — Monte Carlo

#### `fit/MC_lmSD.py`

Monte Carlo simulation study designed to assess finite-sample performance of all models when the data-generating process is the FI-SD (lmSD) model. Oracle lmSD parameters are hard-coded from a prior empirical fit on the full SPX sample.

The synthetic panel uses a fixed observation grid of N=24 options per period: 6 moneyness levels × 4 maturities, matching the empirical bucket structure but without bucket indices (the no-bucket model variants are used throughout).

For each sample length T ∈ {2000}:

1. **Simulate** a panel from the lmSD model with oracle parameters and the fixed grid.
2. **Fit** every model: SS, SD, FI-SD, Mk-SD K∈{1,2,3,5,10}, Mk-SS K∈{1,2,3,5,10}, MSM K∈{1,2,3,5,10}, plus an oracle lmSD initialised at the true parameters with 0 optimisation iterations (zero-iteration oracle serves as an upper bound on in-class performance).
3. **Evaluate** one-step and multi-step (h∈{5,22,66,260}) forecast accuracy: MSE, MAE, neg-log-lik, AIC, BIC.
4. **Cache** fitted parameter dicts as `.npz` files and append results incrementally to a `.jsonl` file so that a partially completed run can be resumed. On restart, if a model's `.npz` cache exists for a given replication, that model is skipped.
5. **Tabulate** mean and standard deviation of each metric across all replications into LaTeX `tabular` environments in `out/SPX/mc/simulate_lmSD_sjit/`.

Sets XLA memory env vars before importing JAX if nvidia-smi is present. Uses `_device.py` for device selection.

---

### Step 3 — Estimation and forecasting

#### `fit/full_forecast_fixed.py`

Main empirical script. Reads `data/SPX/otm/full.parquet`, reshapes the flat long-format observations into a T×N panel (NaN-padded to the maximum daily observation count), and splits at the temporal midpoint into training and test halves.

**Initialisation.** OLS beta and residual variance are computed on the training half. Bucket-specific intercept offsets (`omega`) are initialised as per-bucket mean OLS residuals relative to bucket 0. Each model family has a dedicated initial-guess constructor using these estimates.

**Models.** Each runner function is JIT-compiled once before training. The models fitted are:

| Name | Model |
|------|-------|
| `ss` | Linear Gaussian state-space (collapsed-observation Kalman MLE) |
| `adjSD` | Score-driven with AR(1) beta dynamics (SD) |
| `lmSD` | Fractionally integrated score-driven (FI-SD) |
| `ffSD_K{k}` | Multi-frequency score-driven for K∈{1,2,3,5,10} (Mk-SD) |
| `ffSS_K{k}` | Multi-frequency state-space for K∈{1,2,3,5,10} (Mk-SS) |
| `msmSD_K{k}` | Markov-switching multifractal score-driven for K∈{1,2,3,5,10} (MSM) |

All models use L-BFGS with up to 5000 iterations, tolerance 1e-4, and learning rate 1.0. Convergence and iteration count are recorded.

**Skip logic.** If a model's output files (`predictions_`, `step_`, `params_`) already exist in the output directory, that model is skipped. This allows resuming a partially completed run. Use `--models model1 model2` to run a specific subset, and `--suffix _tag` to write to non-default filenames.

**Outputs** (all written to `out/SPX/otm/full_performance_fixed/`):

| File | Content |
|------|---------|
| `predictions_{model}.parquet` | One-step-ahead point predictions, one row per test date |
| `predictions_h_{model}.parquet` | Multi-step predictions for h∈{5,22,66,260}, columns: date, horizon, predictions |
| `step_{model}.parquet` | Per-date: OOS log-lik, predictive variance, VaR, training log-lik, n_params, niter, converged |
| `params_{model}.json` | Full fitted parameter dict serialised to JSON |
| `b_{model}.parquet` | (Mk-SD and Mk-SS only) Full latent state history in long format: date, is_insample, k (component), j (coefficient), b (value) |
| `summary.parquet` | One row per model: aggregate MSE, MAE, total OOS log-lik, AIC, BIC |

---

### Step 4 — Evaluation

#### `fit/full_mcs_fixed.py`

Classical MCS evaluation. Discovers available models automatically by scanning `out/SPX/otm/full_performance_fixed/` for `predictions_*.parquet` and `predictions_h_*.parquet` files — only models present in both sets are included, so no model list needs to be specified.

**Loss computation.** Per-step MSE and MAE for h∈{1, 5, 22, 66, 260}. Per-step neg-OOS-log-lik, AIC, and BIC for h=1 only, loaded from `step_{model}.parquet`; if those files are missing the likelihood-based metrics are skipped with a printed warning.

**MCS.** Runs the Hansen (2011) TR-statistic MCS via `fit/mcs.py`: stationary block bootstrap with 10,000 replications, seed 42, Bartlett-kernel HAC variance, across all combinations of block lengths {1, 5, 10, 15, 20} and significance levels {0.05, 0.10, 0.25}. The canonical results use block length l=10 and α=0.10.

**LaTeX output** (to `out/SPX/otm/full_mcs_fixed/`): one table per metric with columns for each horizon and one row per model, with star annotations for MCS membership (`*` at α=0.05, `**` at α=0.10, `***` at α=0.25). A separate table covers the five one-step-ahead metrics. Tables are written for each block length.

#### `fit/full_smcs_fixed.py`

Sequential MCS (sMCS) evaluation. Same model discovery logic as `full_mcs_fixed.py`. For each (horizon, metric) combination it assembles a T×M loss matrix (rows = test dates, columns = models) and passes it to `fit/smcs.py`.

Runs the Arnold, Gavrilopoulos, Schulz & Ziegel (2026) sMCS under the weak hypothesis at α∈{0.05, 0.10, 0.25}. Results include terminal-date model sets, full binary membership histories over time, and adjusted e-value time series.

**Outputs** (to `out/SPX/otm/full_smcs_fixed/` and `plot/SPX/otm/full_smcs_fixed/`):

| File | Content |
|------|---------|
| `smcs_grid_{metric}_h{h}.csv` | Matrix of terminal sMCS membership for each (model, α) |
| LaTeX tables | sMCS membership stars per model, analogous to MCS tables |
| E-process PDFs | Adjusted e-value trajectories over the test period for each model, one PDF per metric, with α-level threshold lines in three colours |
| Heatmap PDFs | For each model: time series of the highest α at which it remains in the sMCS, colour-coded (green = all three αs, yellow = α≥0.10, orange = α=0.05 only, red = excluded) |

---

## Requirements

- Python ≥ 3.11
- JAX
- Polars
- PyArrow
- NumPy
- Pandas
- Matplotlib
