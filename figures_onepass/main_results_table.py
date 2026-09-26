"""
Main results table: accuracy and calibration of every method against the TRUE solution, pooled
over all runs of the advection and Fisher-KPP experiments, for both PDEs side by side.

Rows (metrics), columns (method x PDE):
  RMSE      root-mean-square error of the predictive mean (the point prediction for SGD)
            against the truth over the full (x,t) grid of one run, averaged over the runs.
  Cov@90    empirical coverage of the nominal 90% Gaussian predictive interval: the share of all
            grid points of all runs whose truth lies inside mean +- 1.645 sd (every run has the
            same grid, so this is also the mean of the per-run coverages). Target: 90%.
  CRPS      closed-form Gaussian CRPS (continuous ranked probability score; Gneiting & Raftery
            2007), mean over all grid points of all runs. For the non-Bayesian SGD baseline the predictive
            is a point mass, whose CRPS is exactly the absolute error, so its entry is the MAE and
            directly comparable; its coverage is undefined ('--').
  Methods: Ours = one-pass EP (expectation propagation), VI = mean-field variational inference,
  HMC = the multichain Hamiltonian Monte Carlo reference (exact inference for the same model),
  SGD = the non-Bayesian PINN baseline trained by stochastic gradient descent (Adam) on the standard
  PINN loss (the paper's "PINN baseline"; the controlled baseline of mp_vs_gd_scatter_grid).

Run sets and truth: exactly those of the two calibration_comparison.py scripts, whose loaders are
used here (so the Ours/HMC/VI numbers equal their tables' pooled rows): advection = wave speeds
b in {0.1, 0.4, 1, 2, 7} x the ten canonical trajectories idxs(10) (50 runs; truth = the stored
PDEBench trajectory); Fisher-KPP = the 16 (nu, rho) cells x idxs(5) (80 runs; truth = u_ref, the
float64 re-solve of the nominal PDE, never the stored trajectory with its frozen tail). A run whose
HMC reference fails the convergence gate is dropped for every method, SGD included; the number of
runs is printed in the log. The SGD artifacts of the same runs (gd_pinn_beta*, gd_pinn_reacdiff_*)
are checked to hold the same trajectory. Reads cached artifacts only and never trains; a missing
artifact is an error naming the runner that makes it.

Formatting: each metric row carries one power-of-ten scale for all eight entries (chosen from the
data so the smallest entry is >= 1, stated in the row label), three significant figures (coverage:
one decimal, in %). HMC, the exact-inference reference, is set in grey and does not compete; the
best of Ours, VI and SGD per PDE and row is bold (lowest RMSE / CRPS, coverage closest to 90%;
SGD has no coverage). Also printed (not in the table): the RMSE
pooled over all grid points of all runs, as a cross-check against the per-run mean.

Output: figures_onepass/main_results_table.tex (a complete table environment for \\input, needs
booktabs); stdout is the log (`python run_experiments.py figures` saves it as
figures_onepass/main_results_table.stdout.log).

Run: MPLBACKEND=Agg python figures_onepass/main_results_table.py
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys
from pathlib import Path

import numpy as np

FIGURES_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURES_DIR.parent
OUTPUT_TEX = FIGURES_DIR / "main_results_table.tex"


def _load_module(name: str, path: Path):
    """Import a figure script under a unique name (both calibration scripts share a file name)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


adv = _load_module("adv_calibration", FIGURES_DIR / "advection" / "calibration_comparison.py")
fkpp = _load_module("fkpp_calibration", FIGURES_DIR / "diffusion" / "calibration_comparison.py")
import gd_pinn_baseline_sweep as adv_gd       # noqa: E402  (pde/advection, on sys.path via adv)
import reacdiff_baselines_sweep as fkpp_sweep  # noqa: E402  (pde/diffusion, on sys.path via fkpp)

METHODS = ("Ours", "VI", "HMC", "SGD")   # column order within each PDE
BAYESIAN = ("Ours", "VI", "HMC")
REFERENCE = "HMC"   # exact inference: shown in grey, never bold
GREY = r"\textcolor[gray]{0.5}"   # the gray model works with both the color and the xcolor package
PDES = (("adv", "Advection"), ("fkpp", "Fisher--KPP"))
LEVEL = 0.9
MIN_VAR = adv.MIN_VAR
assert fkpp.MIN_VAR == MIN_VAR


# ----------------------------------------------------------------------------- per-run statistics


def run_stats(true_u: np.ndarray, pred_u: np.ndarray, pred_var: np.ndarray | None) -> dict:
    """Per-run sums, so pooled means over runs follow exactly. pred_var=None: point prediction."""
    err = true_u - pred_u
    out = dict(n=err.size, sse=float(np.sum(err ** 2)), rmse=float(np.sqrt(np.mean(err ** 2))))
    if pred_var is None:
        out.update(inside=None, crps_sum=float(np.sum(np.abs(err))))   # CRPS of a point mass = |error|
    else:
        var = np.maximum(pred_var, MIN_VAR)
        z = err / np.sqrt(var)
        out.update(inside=int(np.sum(np.abs(z) <= adv.z_quantile(LEVEL))),
                   crps_sum=adv.crps_gaussian(true_u, pred_u, pred_var) * err.size)
    return out


def advection_stats() -> tuple[dict[str, list[dict]], list[dict]]:
    with contextlib.redirect_stdout(io.StringIO()):   # the loader's per-run progress lines
        records, excluded = adv.collect_records()
    stats = {m: [] for m in METHODS}
    for rec in records:
        truth = rec["Ours"]["true_u"]
        for m in BAYESIAN:
            stats[m].append(run_stats(rec[m]["true_u"], rec[m]["pred_u"], rec[m]["pred_var"]))
        path = adv_gd._cache_path(rec["beta"], rec["idx"])
        if not path.exists():
            raise FileNotFoundError(f"missing SGD artifact {path}; generate it with {adv_gd.JULIA_RUNNER.name}")
        gd = adv_gd.load_cached(rec["beta"], rec["idx"])
        if gd["idx"] != rec["idx"] or not np.array_equal(gd["true_u"], truth):
            raise ValueError(f"b={rec['beta']:g} idx={rec['idx']}: the SGD artifact holds a different trajectory")
        stats["SGD"].append(run_stats(truth, gd["pred_u"], None))
    return stats, excluded


def fisher_stats() -> tuple[dict[str, list[dict]], list[dict]]:
    with contextlib.redirect_stdout(io.StringIO()):
        records, excluded = fkpp.collect_records(list(fkpp.ALL_CELLS))
    stats = {m: [] for m in METHODS}
    for rec in records:
        truth = rec["Ours"]["true_u"]   # u_ref, the float64 re-solve
        for m in BAYESIAN:
            stats[m].append(run_stats(rec[m]["true_u"], rec[m]["pred_u"], rec[m]["pred_var"]))
        path = fkpp_sweep.cache_path("GD", rec["nu"], rec["rho"], rec["idx"])
        if not path.exists():
            raise FileNotFoundError(f"missing SGD artifact {path}; generate it with {fkpp_sweep.RUNNERS['GD'].name}")
        gd = fkpp_sweep.load_cached("GD", rec["nu"], rec["rho"], rec["idx"])
        if gd["idx"] != rec["idx"] or not np.array_equal(gd["u_ref"], truth):
            raise ValueError(f"nu={rec['nu']:g} rho={rec['rho']:g} idx={rec['idx']}: the SGD artifact's truth differs")
        stats["SGD"].append(run_stats(truth, gd["pred_u"], None))
    return stats, excluded


# ----------------------------------------------------------------------------- aggregation


def aggregate(runs: list[dict]) -> dict:
    n = sum(r["n"] for r in runs)
    return dict(
        runs=len(runs),
        rmse=float(np.mean([r["rmse"] for r in runs])),
        rmse_pooled=float(np.sqrt(sum(r["sse"] for r in runs) / n)),
        cov=None if runs[0]["inside"] is None else sum(r["inside"] for r in runs) / n,
        crps=sum(r["crps_sum"] for r in runs) / n,
    )


def _scale_exponent(values: list[float]) -> int:
    """Power of ten that makes the smallest value >= 1 (e.g. 0.037 -> 2, printed as 3.70)."""
    return -int(np.floor(np.log10(min(values))))


def _sig3(value: float) -> str:
    """Three significant figures, trailing zeros kept (38.95 -> '39.0', 160.13 -> '160', 0.012 -> '0.0120')."""
    if value == 0:
        return "0.0"
    decimals = max(0, 2 - int(np.floor(np.log10(abs(value)))))
    s = f"{value:.{decimals}f}"
    if abs(float(s)) >= 10 ** (3 - decimals):   # rounding carried into a new digit (99.95 -> 100.0)
        s = f"{value:.{max(0, decimals - 1)}f}"
    return s


def _cell(text: str | None, bold: bool) -> str:
    if text is None:
        return "--"
    return rf"\textbf{{{text}}}" if bold else text


def write_table(results: dict[str, dict[str, dict]], n_runs: dict[str, int]) -> None:
    rows = []
    for key, label in (("rmse", "RMSE"), ("cov", "Cov@90"), ("crps", "CRPS")):
        fmt = (lambda v: f"{v:.1f}") if key == "cov" else _sig3
        cells = []
        if key == "cov":
            head = r"Cov.\ 90\,\% (\%)"
            scale = 100.0
        else:
            e = _scale_exponent([results[p][m][key] for p, _ in PDES for m in METHODS])
            head = rf"{label} ($\times 10^{{-{e}}}$)"
            scale = 10.0 ** e
        for p, _ in PDES:
            vals = {m: results[p][m][key] for m in METHODS}
            valid = {m: v for m, v in vals.items() if v is not None and m != REFERENCE}
            best = (min(valid, key=lambda m: abs(valid[m] - LEVEL)) if key == "cov"
                    else min(valid, key=valid.get))
            # a tie at the printed precision bolds every tied entry
            text = {m: None if v is None else fmt(v * scale) for m, v in vals.items()}
            for m in METHODS:
                c = _cell(text[m], m != REFERENCE and text[m] is not None and text[m] == text[best])
                cells.append(rf"{GREY}{{{c}}}" if m == REFERENCE else c)
        rows.append(f"{head} & " + " & ".join(cells) + r" \\")
    header_pdes = " & ".join(rf"\multicolumn{{4}}{{c}}{{{name}}}" for p, name in PDES)
    method_heads = [rf"{GREY}{{{m}}}" if m == REFERENCE else m for m in METHODS]
    lines = [
        r"% Auto-generated by figures_onepass/main_results_table.py -- do not edit by hand.",
        r"% Every entry is against the TRUE solution on the full (x,t) grid of each run. Run sets = those of",
        r"%   figures_onepass/{advection,diffusion}/calibration_comparison.py: advection b in {0.1,0.4,1,2,7} x idxs(10),",
        r"%   Fisher-KPP 16 (nu, rho) cells x idxs(5) (truth u_ref). RMSE = mean over runs of the per-run RMSE; coverage =",
        r"%   pooled share of grid points inside the nominal 90% Gaussian interval; CRPS = mean closed-form Gaussian CRPS;",
        r"%   SGD (Adam, point estimate): CRPS of a point mass = mean absolute error, coverage undefined.",
        r"% RMSE pooled over all grid points of all runs (for reference, not shown): "
        + "; ".join(f"{name.replace('--', '-')} " + ", ".join(f"{m} {results[p][m]['rmse_pooled']:.4g}" for m in METHODS)
                    for p, name in PDES) + ".",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l cccc cccc}",
        r"\toprule",
        rf" & {header_pdes} \\",
        r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}",
        " & " + " & ".join(method_heads * 2) + r" \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        (r"\caption{Accuracy and calibration against the true solution, pooled over all runs: "
         r"advection (wave speeds $b\in\{0.1,0.4,1,2,7\}$, ten trajectories each) and Fisher--KPP "
         r"($\nu\in\{0.5,1,2,5\}\times\rho\in\{1,2,5,10\}$, five trajectories each). "
         r"RMSE: mean over runs of the per-run root-mean-square error of the predictive mean; "
         r"Cov.\ 90\,\%: share of grid points inside the nominal 90\,\% predictive interval (target 90\,\%); "
         r"CRPS: continuous ranked probability score. "
         r"SGD is the non-Bayesian PINN baseline trained by stochastic gradient descent (Adam); its CRPS "
         r"reduces to the mean absolute error. HMC (grey) is exact inference for the same model and serves as "
         r"the reference; the best of Ours, VI and SGD per PDE is in bold.}"),
        r"\label{tab:main_results}",
        r"\end{table}",
        "",
    ]
    OUTPUT_TEX.write_text("\n".join(lines))
    print(f"Saved table to {OUTPUT_TEX}")


def main() -> None:
    results, n_runs = {}, {}
    for p, name, fn in (("adv", "Advection", advection_stats), ("fkpp", "Fisher-KPP", fisher_stats)):
        stats, excluded = fn()
        results[p] = {m: aggregate(stats[m]) for m in METHODS}
        n_runs[p] = results[p]["Ours"]["runs"]
        print(f"{name}: {n_runs[p]} runs admitted by the HMC gate; dropped: "
              + (", ".join(str({k: r[k] for k in r if k in ('beta', 'nu', 'rho', 'idx')}) for r in excluded) or "none"))
        for m in METHODS:
            a = results[p][m]
            assert a["runs"] == n_runs[p]
            cov = "--" if a["cov"] is None else f"{100 * a['cov']:.2f}%"
            print(f"  {m:5s} RMSE (mean of per-run) {a['rmse']:.5f} | RMSE pooled {a['rmse_pooled']:.5f} | "
                  f"Cov@90 {cov} | CRPS {a['crps']:.5f}" + ("  (point estimate: CRPS = MAE)" if m == "SGD" else ""))
    write_table(results, n_runs)


if __name__ == "__main__":
    main()
