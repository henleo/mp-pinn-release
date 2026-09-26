"""
Statistical comparison of MP-PINN ("Ours": the message-passing PINN [physics-informed neural
network: a network fitted to the initial condition and to the PDE residual at collocation points],
here the one-pass EP [expectation propagation] posterior mean) against the controlled
gradient-descent baseline ("Baseline": a conventional PINN trained by Adam [a stochastic gradient
optimizer] on the same basis, architecture, graph construction and rho-continuation ladder [the
model is fitted at a fixed sequence of increasing reaction strengths, each fit warm-starting the
next], see pde/diffusion/gd_run_reacdiff.jl) across the five canonical trajectories of every cell of the
PDEBench Fisher-KPP grid (nu in NUS x rho in RHOS, 16 cells, 80 (cell, trajectory) pairs; the
Fisher-KPP equation is u_t = nu u_xx + rho u (1 - u), nu the diffusion coefficient, rho the
reaction strength): MSE [mean squared error] against the truth and PDE-residual RMS [root mean
square], scattered vs. rho on a log-log scale. This is the reaction-diffusion port of figures_onepass/advection/mp_vs_gd_scatter_grid.py;
the methods, the metrics, the colours (blue = Ours, orange-red = Baseline) and the table format
are inherited, only the axis changes: the wave speed b of the advection figure becomes
the reaction strength rho (log axis, 4 values), and the diffusion coefficient nu is pooled into
every rho point (a rho point holds the 4 x 5 = 20 pairs of its four nu values) and shown by the
marker shape (nu = 0.5 circle, 1 square, 2 diamond, 5 triangle). Around each rho the points are
spread horizontally by nu and by method so that overlapping markers stay legible; that horizontal
spread carries no information beyond rho.

Metrics (identical to the advection script):
  * MSE  = mean over the (x, t) grid of (u_ref - pred_u)^2, where u_ref is the float64 re-solve of
    the nominal PDE from the dataset's own initial condition (pde/diffusion/reacdiff_reference.py),
    never the stored PDEBench trajectory, whose late-time freeze is a data artifact;
  * PDE residual RMS = root mean square over the grid of the model's own residual (u_t - nu u_xx -
    rho u (1 - u) evaluated at the point estimate; the `residual` array every runner stores).
The model uses the normalized coefficients rho_norm = rho T_basis = 2 rho and nu_norm =
nu T_basis / L^2 = 2 nu; the axis and the table show the physical rho and nu.

Both methods use the same trajectories (IDXS = trajectory_selection.idxs(5), the seeded draw
shared by every multi-trajectory figure; the development trajectory idx=-1 is excluded) and, per
trajectory, the identical problem, graph and eigenbasis [the eigenvectors of the Gram matrix of the
fitting problem, the coordinates in which the weights are inferred] (checked with `assert_same_basis` on
every (Ours, Baseline) pair, and the two artifacts' stored trajectories are compared as well).
Like the advection figure this figure has NO convergence gate: it compares point predictions
and no HMC run is involved, so every cached pair enters; the log states this explicitly, and
the number of pairs per cell is in the table. Reads cached artifacts only; never trains: a
missing artifact raises FileNotFoundError naming the file.

The companion table (mp_vs_gd_scatter_grid_table.tex) lists all 16 (nu, rho) cells explicitly
(mean +/- std over the n = 5 trajectories of the cell, both methods, both metrics), then one
pooled row per rho (over the four nu values, n = 20) and the grand pooled row (all cells, n = 80).
Every value of one metric shares one common order-of-magnitude scale (set by the smallest value
of that metric in the table), divided out once and stated in the column header.

Cell override: the environment variable REACDIFF_CELLS, a comma-separated list of "nu:rho" pairs
(for example REACDIFF_CELLS="0.5:1.0,0.5:2.0"), restricts the run to those cells (the outputs are
then marked PARTIAL); the default (unset or empty) is all 16 cells and never skips a cell. In
override mode the log, the table comment and a title line on the figure say which cells were used,
the pooled rows say which cells they pool, and the other cells appear in the table as "not run". Within a cell every one of the five canonical
trajectories must exist for both methods in both modes.

Run: python figures_onepass/diffusion/mp_vs_gd_scatter_grid.py [--png out.png]
The console output is also written to mp_vs_gd_scatter_grid.log next to the outputs.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter

FIGURES_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURES_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "pde" / "diffusion"))
sys.path.insert(0, str(REPO_ROOT / "pde" / "advection"))

import reacdiff_baselines_sweep as sweep  # noqa: E402  (load_cached / cache_path / assert_same_basis)
from trajectory_selection import idxs  # noqa: E402

OUTPUT_PDF = FIGURES_DIR / "mp_vs_gd_scatter_grid.pdf"
OUTPUT_TEX = FIGURES_DIR / "mp_vs_gd_scatter_grid_table.tex"
OUTPUT_LOG = FIGURES_DIR / "mp_vs_gd_scatter_grid.log"

NUS = list(sweep.NUS)  # diffusion coefficients of the dataset grid (0.5, 1, 2, 5): the marker shape
RHOS = list(sweep.RHOS)  # reaction strengths of the dataset grid (1, 2, 5, 10): the x-axis
ALL_CELLS = [(nu, rho) for nu in NUS for rho in RHOS]
IDXS = idxs(5)  # the five canonical trajectories of every cell (the same seeded draw as the advection figures)
CELLS_ENV = "REACDIFF_CELLS"
METHODS = ("Ours", "GD")  # artifact names in reacdiff_baselines_sweep; GD is shown as "Baseline"
LABELS = {"Ours": "Ours", "GD": "Baseline"}
COLORS = {"Ours": "blue", "GD": "orangered"}  # the advection figure's colours
NU_MARKERS = {0.5: "o", 1.0: "s", 2.0: "D", 5.0: "^"}  # nu shown by the marker shape
METRICS = ("mse", "residual")
METRIC_TITLES = {"mse": "MSE", "residual": "PDE residual"}
# Horizontal spread around each rho (in log10 units) so the four nu markers and the two methods do
# not sit on top of each other: x = rho * 10^(NU_OFFSET[nu] + METHOD_OFFSET[method]).
NU_OFFSET = {nu: off for nu, off in zip(NUS, (-0.105, -0.035, 0.035, 0.105))}
METHOD_OFFSET = {"Ours": -0.014, "GD": 0.014}


# ----------------------------------------------------------------------------- log file


class _Tee:
    """Write everything printed to stdout to the log file as well (the log ships with the figure)."""

    def __init__(self, path: Path):
        self._stdout = sys.stdout
        self._file = open(path, "w")

    def write(self, s: str) -> None:
        self._stdout.write(s)
        self._file.write(s)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def close(self) -> None:
        sys.stdout = self._stdout
        self._file.close()


# ----------------------------------------------------------------------------- cells


def selected_cells() -> tuple[list[tuple[float, float]], bool]:
    """(cells to run, override?) from the REACDIFF_CELLS environment variable; default = all 16."""
    raw = os.environ.get(CELLS_ENV, "").strip()
    if not raw:
        return list(ALL_CELLS), False
    cells = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            nu_s, rho_s = item.split(":")
            nu, rho = float(nu_s), float(rho_s)
        except ValueError as e:
            raise ValueError(f"{CELLS_ENV} entry {item!r} is not of the form nu:rho") from e
        if (nu, rho) not in ALL_CELLS:
            raise ValueError(f"{CELLS_ENV} entry {item!r} is not a grid cell; nu in {NUS}, rho in {RHOS}")
        if (nu, rho) not in cells:
            cells.append((nu, rho))
    return cells, True


def cell_label(nu: float, rho: float) -> str:
    return f"nu={nu:g} rho={rho:g}"


# ----------------------------------------------------------------------------- loading


def load_pair(nu: float, rho: float, idx: int) -> tuple[dict, dict]:
    """The cached (Ours, Baseline) records of one trajectory; never trains. Checks that both were
    built on the same trajectory, the same eigenbasis and the same truth grid."""
    for method in METHODS:
        path = sweep.cache_path(method, nu, rho, idx)
        if not path.exists():
            raise FileNotFoundError(
                f"missing {LABELS[method]} artifact {path} ({cell_label(nu, rho)}, idx={idx}); "
                f"generate it with pde/diffusion/{sweep.RUNNERS[method].name} (this script never trains)")
    ours, gd = (sweep.load_cached(method, nu, rho, idx) for method in METHODS)
    if ours["idx"] != idx or gd["idx"] != idx:
        raise ValueError(f"{cell_label(nu, rho)}: artifact idx attrs ({ours['idx']}, {gd['idx']}) differ from {idx}")
    sweep.assert_same_basis(ours, gd)
    # u_ref is recomputed from (nu, rho, idx) by the loader, so it is identical by construction; the
    # stored PDEBench trajectory true_u is what each runner actually read, so compare that.
    if ours["true_u"].shape != gd["true_u"].shape or not np.allclose(ours["true_u"], gd["true_u"]):
        raise ValueError(f"{cell_label(nu, rho)} idx={idx}: the two artifacts' stored trajectories differ")
    if ours["pred_u"].shape != ours["u_ref"].shape or gd["pred_u"].shape != gd["u_ref"].shape:
        raise ValueError(f"{cell_label(nu, rho)} idx={idx}: prediction and truth grids have different shapes")
    return ours, gd


def collect_results(cells: list[tuple[float, float]]) -> tuple[list[dict], list[dict]]:
    mp_results, gd_results = [], []
    for nu, rho in cells:
        for idx in IDXS:
            print(f"[{cell_label(nu, rho)} idx={idx}] loading Ours + controlled baseline...")
            ours, gd = load_pair(nu, rho, idx)
            mp_results.append(ours)
            gd_results.append(gd)
    return mp_results, gd_results


# ----------------------------------------------------------------------------- metrics


def metric_value(r: dict, metric: str) -> float:
    if metric == "mse":
        return float(np.mean((r["u_ref"] - r["pred_u"]) ** 2))
    if metric == "residual":
        return float(np.sqrt(np.mean(r["residual"] ** 2)))
    raise ValueError(metric)


def _metric_values(results: list[dict], metric: str, in_group) -> np.ndarray:
    """All values of `metric` of the records whose (nu, rho) satisfies `in_group`."""
    return np.array([metric_value(r, metric) for r in results if in_group(r["nu"], r["rho"])])


# ----------------------------------------------------------------------------- figure


def make_scatter_grid(mp_results: list[dict], gd_results: list[dict], output_path: Path,
                      png_path: Path | None = None, note: str | None = None) -> None:
    plt.rcParams["mathtext.fontset"] = "cm"
    fig, (ax_mse, ax_res) = plt.subplots(1, 2, figsize=(15, 6))

    def scatter_one(ax, results: list[dict], method: str, metric: str) -> None:
        for nu in NUS:
            xs = [r["rho"] * 10 ** (NU_OFFSET[nu] + METHOD_OFFSET[method]) for r in results if r["nu"] == nu]
            ys = [metric_value(r, metric) for r in results if r["nu"] == nu]
            if not xs:
                continue
            ax.scatter(xs, ys, marker=NU_MARKERS[nu], color=COLORS[method], alpha=0.75, s=60,
                       edgecolors="black", linewidths=0.5)

    for metric, ax, ylabel, title in (
        ("mse", ax_mse, r"$\mathrm{MSE}$", r"$\mathrm{(a)\ MSE\ vs.}\ \rho$"),
        ("residual", ax_res, r"$\mathrm{RMS\ residual}$", r"$\mathrm{(b)\ PDE\ residual\ vs.}\ \rho$"),
    ):
        for method, results in (("Ours", mp_results), ("GD", gd_results)):
            scatter_one(ax, results, method, metric)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(RHOS)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(RHOS[0] / 1.6, RHOS[-1] * 1.6)
        ax.set_xlabel(r"$\rho$", fontsize=20)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.set_title(title, fontsize=22)
        ax.tick_params(axis="both", labelsize=14)
        ax.grid(True, which="both", alpha=0.3)

    method_handles = [Line2D([], [], marker="o", linestyle="none", color=COLORS[m], markeredgecolor="black",
                             markersize=11, label=LABELS[m]) for m in METHODS]
    nu_handles = [Line2D([], [], marker=NU_MARKERS[nu], linestyle="none", color="0.45", markeredgecolor="black",
                         markersize=10, label=rf"$\nu={nu:g}$") for nu in NUS]
    # Both legends hang directly below the figure's bottom edge (anchor in figure fractions, the
    # legend's top edge at the anchor); bbox_inches="tight" extends the saved page to include them.
    fig.legend(handles=method_handles, loc="upper center", ncol=2, fontsize=18, frameon=False,
               bbox_to_anchor=(0.27, 0.0), borderaxespad=0, title=r"$\mathrm{colour:\ method}$", title_fontsize=16)
    fig.legend(handles=nu_handles, loc="upper center", ncol=4, fontsize=18, frameon=False,
               bbox_to_anchor=(0.72, 0.0), borderaxespad=0,
               title=r"$\mathrm{marker:\ diffusion\ coefficient}\ \nu$", title_fontsize=16)
    if note:
        fig.suptitle(note, fontsize=16, color="#b00020", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    if png_path is not None:
        fig.savefig(png_path, bbox_inches="tight", dpi=110)
        print(f"Saved PNG to {png_path}")
    plt.close(fig)
    print(f"Saved grid to {output_path}")


# ----------------------------------------------------------------------------- table


def _format_mean_std(values: np.ndarray, scale: float) -> str:
    mean, std = np.mean(values) / scale, np.std(values) / scale
    return rf"${mean:.2f} \pm {std:.2f}$"


def _common_scale(values_by_group: list[np.ndarray]) -> tuple[float, int]:
    """One order-of-magnitude scale for a metric: 10^floor(log10(smallest non-zero mean or std))."""
    smallest = min(abs(v) for values in values_by_group if len(values)
                   for v in (np.mean(values), np.std(values)) if v != 0)
    exp = int(np.floor(np.log10(smallest)))
    return 10.0 ** exp, exp


def make_latex_table(mp_results: list[dict], gd_results: list[dict], cells: list[tuple[float, float]],
                     override: bool, output_path: Path) -> None:
    """Mean +/- std of MSE and PDE-residual RMS per (nu, rho) cell (over its n trajectories, both
    methods), then pooled per rho over nu and pooled over the whole grid -- the quantitative
    counterpart to the scatter grid. Per metric, every value shares one common order-of-magnitude
    scale (set by the smallest value of that metric in the table), divided out once and stated in
    the column header, rather than repeating "x10^-n" in every cell."""
    by_method = (("Ours", mp_results), ("GD", gd_results))
    # rows: (kind, label cells, in_group); "cell" rows for all 16 cells, a pooled row after each rho block
    row_specs = []
    for rho in RHOS:
        for nu in NUS:
            row_specs.append(("cell", (nu, rho), lambda n, r, nu=nu, rho=rho: n == nu and r == rho))
        row_specs.append(("rho", rho, lambda n, r, rho=rho: r == rho))
    row_specs.append(("all", None, lambda n, r: True))

    scales = {}
    for metric in METRICS:
        groups = [_metric_values(results, metric, in_group) for _, results in by_method for _, _, in_group in row_specs]
        scales[metric] = _common_scale(groups)

    def group_cells(in_group) -> list[tuple[float, float]]:
        return [c for c in cells if in_group(*c)]

    lines = [
        r"% Auto-generated by figures_onepass/diffusion/mp_vs_gd_scatter_grid.py",
        r"% Per cell: mean +/- std over the n canonical trajectories (the same trajectories for both methods).",
        r"% Pooled rows: 'all nu | rho' pools the four cells nu in {0.5, 1, 2, 5} of that rho;",
        r"% 'all cells' pools the whole grid. MSE is against the float64 nominal-PDE reference u_ref.",
        r"% No convergence gate (no HMC involved): every cached (cell, trajectory) pair enters; no run dropped.",
    ]
    if override:
        lines.append(rf"% OVERRIDE MODE ({CELLS_ENV}): only these cells were used: "
                     + ", ".join(cell_label(*c) for c in cells) + "; pooled rows pool the used cells only.")
    lines += [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l l r cc cc}",
        r"\toprule",
        rf" & & & \multicolumn{{2}}{{c}}{{MSE ($\times 10^{{{scales['mse'][1]}}}$)}} & "
        rf"\multicolumn{{2}}{{c}}{{PDE residual ($\times 10^{{{scales['residual'][1]}}}$)}} \\",
        r"\cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r"$\nu$ & $\rho$ & $n$ & Ours & Baseline & Ours & Baseline \\",
        r"\midrule",
    ]
    for kind, key, in_group in row_specs:
        used = group_cells(in_group)
        if kind == "cell":
            nu, rho = key
            head = rf"{nu:g} & {rho:g}"
        elif kind == "rho":
            head = rf"all $\nu$ & {key:g}"
            if override and used:
                head = rf"$\nu \in \{{{', '.join(f'{c[0]:g}' for c in used)}\}}$ & {key:g}"
            lines.append(r"\addlinespace[2pt]")
        else:
            head = r"\multicolumn{2}{l}{all cells}"
            if override:
                head = rf"\multicolumn{{2}}{{l}}{{all {len(used)} used cells}}"
            lines.append(r"\midrule")
        if not used:
            lines.append(rf"{head} & 0 & \multicolumn{{4}}{{c}}{{not run}} \\")
            continue
        n_per_method = [len(_metric_values(results, "mse", in_group)) for _, results in by_method]
        if len(set(n_per_method)) != 1:
            raise RuntimeError(f"unequal run counts between methods in row {kind} {key}: {n_per_method}")
        vals = [_format_mean_std(_metric_values(results, metric, in_group), scales[metric][0])
                for metric in METRICS for _, results in by_method]
        lines.append(rf"{head} & {n_per_method[0]} & " + " & ".join(vals) + r" \\")
        if kind == "rho":
            lines.append(r"\addlinespace[2pt]")
    n_pairs = min(len(mp_results), len(gd_results))
    caption = (
        rf"Mean $\pm$ std of the mean squared error (MSE, against the float64 reference solution of the "
        rf"nominal PDE) and of the PDE residual RMS (root mean square), MP-PINN (Ours; the message-passing "
        rf"physics-informed neural network) vs.\ the "
        rf"controlled gradient-descent baseline on the Fisher-KPP "
        rf"grid, per $(\nu,\rho)$ cell over its $n$ canonical trajectories (the same trajectories for both "
        rf"methods), pooled per $\rho$ over $\nu$, and pooled over the grid; {n_pairs} (cell, trajectory) "
        rf"pairs in total. Every value of one metric is divided by the common scale in its column header. "
        rf"The model uses the normalized coefficients $\rho_{{\mathrm{{norm}}}}=2\rho$, "
        rf"$\nu_{{\mathrm{{norm}}}}=2\nu$; the table shows the physical values."
    )
    if override:
        caption += (rf" PARTIAL: generated in override mode ({CELLS_ENV}) on {len(cells)} of {len(ALL_CELLS)} "
                    "cells; the pooled rows pool only the cells listed in their first column.")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{caption}}}",
        r"\label{tab:mp_vs_gd_scatter_reacdiff}",
        r"\end{table}",
    ]
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Saved LaTeX table to {output_path}")


# ----------------------------------------------------------------------------- log summary


def print_summary(mp_results: list[dict], gd_results: list[dict], cells: list[tuple[float, float]]) -> None:
    print("\nRuns dropped: none. This figure has no convergence gate (it compares point predictions of Ours and "
          "the GD baseline; no HMC run is involved), so every cached (cell, trajectory) pair enters.")
    print(f"Pairs used: {len(mp_results)} = {len(cells)} cells x {len(IDXS)} trajectories {IDXS}")
    print("\nPer cell (mean +- std over the trajectories; rel. L2 = relative L2 error of the prediction, "
          "vs u_ref and, for transparency, vs the stored PDEBench trajectory):")
    print(f"{'cell':>18s} {'method':>8s} {'n':>2s} {'MSE':>20s} {'residual RMS':>20s} {'relL2 ref':>10s} {'relL2 stored':>12s}")
    for nu, rho in cells:
        for method, results in (("Ours", mp_results), ("GD", gd_results)):
            rs = [r for r in results if r["nu"] == nu and r["rho"] == rho]
            mse = np.array([metric_value(r, "mse") for r in rs])
            res = np.array([metric_value(r, "residual") for r in rs])
            l2r = np.mean([r["rel_l2_ref"] for r in rs])
            l2s = np.mean([r["rel_l2_stored"] for r in rs])
            print(f"{cell_label(nu, rho):>18s} {LABELS[method]:>8s} {len(rs):2d} "
                  f"{mse.mean():9.3e} +- {mse.std():8.2e} {res.mean():9.3e} +- {res.std():8.2e} {l2r:10.4f} {l2s:12.4f}")
    print("\nPooled per rho (over the used nu values):")
    for rho in RHOS:
        used = [c for c in cells if c[1] == rho]
        if not used:
            print(f"  rho={rho:g}: not run")
            continue
        for method, results in (("Ours", mp_results), ("GD", gd_results)):
            mse = _metric_values(results, "mse", lambda n, r, rho=rho: r == rho)
            res = _metric_values(results, "residual", lambda n, r, rho=rho: r == rho)
            print(f"  rho={rho:g} nu in {{{', '.join(f'{c[0]:g}' for c in used)}}} {LABELS[method]:>8s} n={len(mse):2d} "
                  f"MSE {mse.mean():.3e} +- {mse.std():.2e}  residual {res.mean():.3e} +- {res.std():.2e}")
    # Cells where the baseline's error is far from Ours': worth a look (e.g. a diverged Adam run).
    flagged = []
    for nu, rho in cells:
        m = np.mean([metric_value(r, "mse") for r in mp_results if r["nu"] == nu and r["rho"] == rho])
        g = np.mean([metric_value(r, "mse") for r in gd_results if r["nu"] == nu and r["rho"] == rho])
        if max(m, g) / max(min(m, g), 1e-300) > 10:
            flagged.append(f"{cell_label(nu, rho)} (MSE Ours {m:.2e}, Baseline {g:.2e})")
    if flagged:
        print("\nCells where the two methods' mean MSE differ by more than 10x (check the runs): "
              + "; ".join(flagged))


# ----------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--png", type=Path, default=None, help="also render the figure to this PNG (for inspection)")
    args = parser.parse_args()
    tee = _Tee(OUTPUT_LOG)
    sys.stdout = tee
    try:
        cells, override = selected_cells()
        if override:
            print(f"OVERRIDE MODE: {CELLS_ENV}={os.environ[CELLS_ENV]!r} -> running only the cells "
                  f"{', '.join(cell_label(*c) for c in cells)} ({len(cells)} of {len(ALL_CELLS)}). "
                  "The outputs are PARTIAL and marked as such; unset the variable for the full figure.")
        else:
            print(f"Running all {len(ALL_CELLS)} cells (nu in {NUS} x rho in {RHOS}), trajectories {IDXS}")
        mp_results, gd_results = collect_results(cells)
        note = (f"PARTIAL ({CELLS_ENV}): cells {', '.join(cell_label(*c) for c in cells)} only" if override else None)
        make_scatter_grid(mp_results, gd_results, OUTPUT_PDF, args.png, note)
        make_latex_table(mp_results, gd_results, cells, override, OUTPUT_TEX)
        print_summary(mp_results, gd_results, cells)
        print(f"Log written to {OUTPUT_LOG}")
    except BaseException as e:  # record the reason in the log (the traceback goes to stderr only)
        print(f"FAILED: {type(e).__name__}: {e}")
        raise
    finally:
        tee.close()


if __name__ == "__main__":
    main()
