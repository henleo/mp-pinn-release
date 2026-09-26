# Bayesian physics-informed neural networks via message passing

Code to reproduce every figure in `figures_onepass/advection` (1D advection) and
`figures_onepass/diffusion` (1D Fisher-KPP reaction-diffusion), and the main results table
(`figures_onepass/main_results_table.py`), from scratch: the message passing method ("Ours"), the
HMC, VI and gradient-descent baselines, the ablations, and the figure and table scripts.

The models and inference code are written in Julia (`lib/`, `pde/`). The data download, the
experiment driver and the figure scripts are written in Python.

## Requirements

- Julia 1.12.6 (the version `Manifest.toml` was resolved with), with `julia` on the `PATH`, e.g. installed via [juliaup](https://github.com/JuliaLang/juliaup)
- Python 3.10 (the pinned packages in `requirements.txt` have no wheels for Python ≥ 3.13; if
  `python3.10` is not installed, e.g. `uv python install 3.10`, then use `uv venv --python 3.10 .venv`
  in place of the `python3.10 -m venv .venv` line below)
- About 110 GB of disk space for the PDEBench data and about 8 GB for the experiment outputs
- 16 GB RAM is enough for 4 parallel runs (a single run at wave speed b = 7 needs up to about 3.5 GB)

All experiments run on the CPU; no GPU is needed.

## Setup

Run all commands from the repository root.

```bash
# Python environment
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Julia 1.12.6 (with juliaup) and the Julia environment (versions pinned in Manifest.toml)
juliaup add 1.12.6
juliaup override set 1.12.6
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
```

In every new shell, run `source .venv/bin/activate` in the repository root before the commands below.
`run_experiments.py` also selects Julia 1.12.6 itself (through juliaup's `JULIAUP_CHANNEL`) and
warns if a different Julia version runs.

## Data

The experiments use 21 files of PDEBench (Takamoto et al., NeurIPS 2022 Datasets and Benchmarks): 1D
advection at wave speeds b ∈ {0.1, 0.4, 1, 2, 7} and 1D reaction-diffusion at
ν ∈ {0.5, 1, 2, 5} × ρ ∈ {1, 2, 5, 10}. The script downloads them from PDEBench's official data
repository into `data/1D/Advection/Train/` and `data/1D/ReactionDiffusion/Train/` and checks their md5 sums:

```bash
python pde/download_pdebench_subset.py              # all 21 files, about 107 GB
python pde/download_pdebench_subset.py advection    # or only the 5 advection files (about 41 GB)
python pde/download_pdebench_subset.py reacdiff     # or only the 16 reaction-diffusion files (about 66 GB)
```

An interrupted download can be restarted with the same command: complete files are skipped, and
the interrupted file is downloaded again from the start (its partial `.part` file is overwritten).

## Running the experiments

`run_experiments.py` runs every experiment and then every figure and table script:

```bash
python run_experiments.py all --jobs 4 --big-jobs 2
```

`--jobs` is the number of Julia processes that run at the same time, and `--big-jobs` is how many of
them may be memory-heavy b = 7 runs. Every run writes one HDF5 file to `artifacts/`, with its console
output in a `.runlog` file next to it. Finished runs are skipped, so an interrupted command can simply
be started again (on Ctrl-C or `kill`, the driver stops its running Julia processes and removes
their partial outputs). The whole pipeline takes roughly 100 core-hours, about one to two days with 4
parallel jobs.

The steps can also be run one at a time, in this order:

```bash
python run_experiments.py references                  # float64 reference solutions for reaction-diffusion (minutes)
python run_experiments.py adv-ours adv-gd adv-hmc adv-vi adv-wide adv-ablation adv-epsilon
python run_experiments.py diff-ours diff-gd diff-hmc diff-vi diff-wide diff-gauss
python run_experiments.py timing                      # on an otherwise idle machine, after all runs above
python run_experiments.py figures
```

| Step | What it runs | Runs |
|---|---|---|
| `references` | float64 reference solutions of the reaction-diffusion problems | 96 |
| `adv-ours` | our one-pass message passing, advection | 55 |
| `adv-gd` | gradient-descent PINN baseline (Adam; "SGD" in the paper), advection | 50 |
| `adv-hmc` | multichain HMC reference, advection | 50 |
| `adv-vi` | mean-field VI baseline, advection | 50 |
| `adv-wide` | our method with H = 2, 3, 4 hidden units, advection | 150 |
| `adv-ablation` | subgraph ablation (prior only, initial condition only, PDE residual only), advection | 50 |
| `adv-epsilon` | PDE noise scale ε sweep, advection | 450 |
| `diff-ours` | our one-pass message passing, reaction-diffusion | 80 |
| `diff-gd` | gradient-descent PINN baseline (Adam; "SGD" in the paper), reaction-diffusion | 80 |
| `diff-hmc` | multichain HMC reference, reaction-diffusion (needs `diff-ours`) | 80 |
| `diff-vi` | mean-field VI baseline, reaction-diffusion | 80 |
| `diff-wide` | our method with H = 2, 3, 4 hidden units, reaction-diffusion | 132 |
| `diff-gauss` | our method on a Gaussian initial condition, reaction-diffusion (no data needed) | 16 |
| `timing` | steady-state inference times, written to `figures_onepass/*/training_inference_time_steady_state.csv` | 3 |
| `figures` | every figure script and the main results table (checks first that all steps above, including `timing`, are complete) | 22 |

`python run_experiments.py <step> --dry-run` prints the command of every run of a step that is
still missing, without running anything. The driver runs Julia single-threaded
(`OPENBLAS_NUM_THREADS=1`). This matters because the figures compare posteriors of different
methods coordinate by coordinate in a shared eigenbasis, whose sign convention depends on the BLAS
threading. All runs use fixed random seeds. Wall-clock times (the timing panels of `training_inference_time.pdf` and `wider_model.pdf`)
depend on the machine and, for training times, on `--jobs`, so these panels reproduce the protocol
but not the exact numbers.

## Figures

Each figure script reads the finished runs from `artifacts/` (and stops with an error naming the
`run_experiments.py` step if a run is missing) and writes its PDF(s) and a LaTeX table with the
plotted numbers (`*_table.tex`) next to itself; `run_experiments.py figures` also saves each
script's console output as `*.stdout.log`. A single figure can be regenerated with, e.g.

```bash
MPLBACKEND=Agg python figures_onepass/advection/combined_solution_grid.py
```

| Script (in `figures_onepass/advection/` and `figures_onepass/diffusion/`) | Output |
|---|---|
| `combined_solution_grid.py` | exact solution, our predictive mean and variance |
| `combined_solution_grid_gaussian.py` (reaction-diffusion) | the same for a Gaussian initial condition |
| `dataset_grid.py` (reaction-diffusion) | reference solution vs. our predictive mean on all 16 (ν, ρ) settings |
| `mp_vs_gd_scatter_grid.py` | MSE and PDE residual, ours vs. the gradient-descent baseline |
| `mp_vs_gd_scatter_grid_mse.py` (advection) | the MSE over all points of panel (a) of that figure (printed only) |
| `calibration_comparison.py` | calibration and uncertainty structure, ours vs. HMC and VI (`calibration_variance.pdf`) |
| `posterior_recovery_comparison.py` | posterior means and standard deviations, pairwise between ours, HMC and VI |
| `posterior_fit_metrics.py` | posterior fit to the HMC reference |
| `uncertainty_growth_curve.py` | predictive uncertainty over time |
| `training_inference_time.py` | training and inference time, ours vs. HMC and VI |
| `wider_model.py` | width ablation (H = 1 to 4 hidden units) |
| `subgraph_ablation_plot.py` (advection) | prior-only / initial-condition-only / residual-only subgraphs vs. the full model (`space_analysis.pdf`) |
| `mp_pinn_epsilon_dependence.py` (advection) | dependence on the PDE noise scale ε (`mse_residual_vs_epsilon.pdf`) |
| `figures_onepass/main_results_table.py` | main results table: RMSE, 90 % coverage and CRPS of all methods on both PDEs (`main_results_table.tex`) |

`python run_experiments.py figures` also sets `CALIBRATION_ALL_PANELS=1` and `SUBGRAPH_ALL_PANELS=1`,
which make the calibration and ablation scripts write their additional panel sets
(`calibration_comparison.pdf`, `variance_consistency.pdf`, `space_analysis_all_panels.pdf`).

## Repository layout

```
lib/                  factor graph, Gaussian messages and factors (Julia)
pde/*.jl              the PINN model with message passing, and the gradient-descent, HMC and VI baselines
pde/advection/        advection runners (*.jl) and their Python loaders
pde/diffusion/        reaction-diffusion runners (*.jl), loaders and the float64 reference solver
pde/*/diagnostics/steady_state_timing/   steady-state inference timing
figures_onepass/      figure scripts
run_experiments.py    runs all experiments and figures
```
