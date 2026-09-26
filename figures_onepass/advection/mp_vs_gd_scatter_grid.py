"""
Statistical comparison of MP-PINN against the controlled gradient-descent baseline (same basis,
architecture, and graph construction -- see gd_pinn_model.jl) across many trajectories per wave
speed: MSE and PDE-residual RMS, scattered vs. beta on a log-log scale; two panels,
"Ours" vs. the controlled baseline.

Trajectory indices come from the canonical seeded draw (`trajectory_selection.idxs`), shared
with every other multi-trajectory figure, so all figures use nested subsets of the same
trajectories and the development trajectory (idx=-1) is excluded. Raising the per-beta count
extends the existing sample (prefix property) instead of resampling. The figure uses
N_TRAJ_PER_BETA = 10 trajectories at each of the five wave speeds b = 0.1, 0.4, 1, 2, 7.

Run:  python figures_onepass/advection/mp_vs_gd_scatter_grid.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURES_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "pde" / "advection"))

import mp_pinn_beta_sweep as mp_sweep
import gd_pinn_baseline_sweep as gd_sweep
from trajectory_selection import idxs

OUTPUT_PDF = FIGURES_DIR / "mp_vs_gd_scatter_grid.pdf"

BETAS = [0.1, 0.4, 1.0, 2.0, 7.0]
N_TRAJ_PER_BETA = 10


def _idxs_for(beta: float) -> list[int]:
    return idxs(N_TRAJ_PER_BETA)


def load_mp(beta: float, idx: int) -> dict:
    path = mp_sweep._cache_path(beta, idx)
    if not path.exists():
        raise FileNotFoundError(f"missing artifact {path}; run: python run_experiments.py adv-ours")
    return mp_sweep.load_cached(beta, idx)


def load_gd(beta: float, idx: int) -> dict:
    path = gd_sweep._cache_path(beta, idx)
    if not path.exists():
        raise FileNotFoundError(f"missing artifact {path}; run: python run_experiments.py adv-gd")
    return gd_sweep.load_cached(beta, idx)


def collect_results() -> tuple[list[dict], list[dict]]:
    mp_results, gd_results = [], []
    for beta in BETAS:
        for idx in _idxs_for(beta):
            print(f"[beta={beta} idx={idx}] loading MP + controlled baseline...")
            mp_results.append(load_mp(beta, idx))
            gd_results.append(load_gd(beta, idx))
    return mp_results, gd_results


def make_scatter_grid(mp_results: list[dict], gd_results: list[dict], output_path: Path) -> None:
    plt.rcParams["mathtext.fontset"] = "cm"
    fig, (ax_mse, ax_res) = plt.subplots(1, 2, figsize=(15, 6))

    def scatter_one(ax, results: list[dict], marker: str, color: str, label: str, metric: str) -> None:
        betas_plot, values = [], []
        for r in results:
            if metric == "mse":
                v = np.mean((r["true_u"] - r["pred_u"]) ** 2)
            else:
                v = np.sqrt(np.mean(r["residual"] ** 2))
            betas_plot.append(r["beta"])
            values.append(v)
        ax.scatter(betas_plot, values, marker=marker, color=color, alpha=0.75, s=90,
                   edgecolors="black", linewidths=0.5, label=label)

    for metric, ax, ylabel, title in (
        ("mse", ax_mse, r"$\mathrm{MSE}$", r"$\mathrm{(a)\ MSE\ vs.}\ b$"),
        ("residual", ax_res, r"$\mathrm{RMS\ residual}$", r"$\mathrm{(b)\ PDE\ residual\ vs.}\ b$"),
    ):
        scatter_one(ax, mp_results, "o", "blue", "Ours", metric)
        scatter_one(ax, gd_results, "^", "orangered", "Baseline", metric)
        ax.set_xscale("log")
        ax.set_xlabel(r"$b$", fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.set_title(title, fontsize=22)
        ax.tick_params(axis="both", labelsize=14)
        ax.grid(True, which="both", alpha=0.3)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.yaxis.get_offset_text().set_fontsize(14)

    handles, labels = ax_mse.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=18, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved grid to {output_path}")


WAVE_SPEED_GROUPS = (r"$b \le 1$", lambda b: b <= 1), (r"$b > 1$", lambda b: b > 1)


def _metric_values(results: list[dict], metric: str, in_group) -> np.ndarray:
    values = []
    for r in results:
        if not in_group(r["beta"]):
            continue
        if metric == "mse":
            values.append(np.mean((r["true_u"] - r["pred_u"]) ** 2))
        else:
            values.append(np.sqrt(np.mean(r["residual"] ** 2)))
    return np.array(values)


def _format_mean_std(values: np.ndarray, scale: float) -> str:
    mean, std = np.mean(values) / scale, np.std(values) / scale
    return rf"${mean:.2f} \pm {std:.2f}$"


def make_latex_table(mp_results: list[dict], gd_results: list[dict], output_path: Path) -> None:
    """Mean +/- std of MSE and PDE-residual RMS, aggregated over every cached trajectory,
    grouped by wave speed b<=1 vs b>1 -- the quantitative counterpart to the scatter grid.
    Every value shares one common order-of-magnitude scale (set by the smallest value in the
    table), divided out once and reported in the caption, rather than repeating "x10^-n" in
    every cell."""
    all_groups = [
        (metric, in_group, _metric_values(results, metric, in_group))
        for _, results in (("Ours", mp_results), ("Baseline", gd_results))
        for metric in ("mse", "residual")
        for _, in_group in WAVE_SPEED_GROUPS
    ]
    smallest = min(
        abs(v) for _, _, values in all_groups for v in (np.mean(values), np.std(values)) if v != 0
    )
    exp = int(np.floor(np.log10(smallest)))
    scale = 10.0**exp

    rows = []
    for label, results in (("Ours", mp_results), ("Baseline", gd_results)):
        cells = [
            _format_mean_std(_metric_values(results, metric, in_group), scale)
            for metric in ("mse", "residual")
            for _, in_group in WAVE_SPEED_GROUPS
        ]
        rows.append((label, cells))

    group_labels = [label for label, _ in WAVE_SPEED_GROUPS]
    lines = [
        r"% Auto-generated by figures_onepass/advection/mp_vs_gd_scatter_grid.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l cc cc}",
        r"\toprule",
        r" & \multicolumn{2}{c}{MSE} & \multicolumn{2}{c}{PDE residual} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
        rf"Method & {group_labels[0]} & {group_labels[1]} & {group_labels[0]} & {group_labels[1]} \\",
        r"\midrule",
    ]
    for label, cells in rows:
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    n_pairs = min(len(mp_results), len(gd_results))
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{Mean $\pm$ std of MSE and PDE residual RMS (all values $\times 10^{{{exp}}}$), "
        rf"MP-PINN (Ours) vs.\ the controlled baseline, aggregated over {n_pairs} (beta, "
        rf"trajectory) pairs -- however many are cached when this table was generated, not "
        rf"necessarily the full N\_TRAJ\_PER\_BETA={N_TRAJ_PER_BETA} target -- grouped by "
        r"$b\le 1$ and $b>1$.}",
        r"\label{tab:mp_vs_gd_scatter}",
        r"\end{table}",
    ]
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Saved LaTeX table to {output_path}")


def main() -> None:
    mp_results, gd_results = collect_results()
    make_scatter_grid(mp_results, gd_results, OUTPUT_PDF)
    make_latex_table(mp_results, gd_results, FIGURES_DIR / "mp_vs_gd_scatter_grid_table.tex")


if __name__ == "__main__":
    main()
