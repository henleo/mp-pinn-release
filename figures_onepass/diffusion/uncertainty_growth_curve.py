"""
Posterior uncertainty vs. time, segregated by reaction strength rho, on the PDEBench 1D
reaction-diffusion (Fisher-KPP) dataset, u_t = nu u_xx + rho u (1 - u), over the 4 x 4 grid of
settings nu in {0.5, 1, 2, 5} (diffusion coefficient) x rho in {1, 2, 5, 10} (reaction strength)
and the five canonical trajectories `idxs(5)` per setting (the same seeded draw the advection
figures use, see `trajectory_selection.py`). This is the reaction-diffusion port of
`figures_onepass/advection/uncertainty_growth_curve.py`: what the wave speed b is
there, rho is here, and nu is the second dimension. The model (`mp_pinn_reacdiff_dataset_run.jl`,
"Ours" = one-pass EP, expectation propagation: a Gaussian posterior built by one sweep of local
moment matching) trains on the initial condition (t=0) as its only labeled observation, plus
unsupervised PDE-residual constraints elsewhere, so the question is whether the predictive
standard deviation (std) grows with the distance from the observations. The model sees the
normalized values rho_norm = 2 rho and nu_norm = 2 nu (the legend shows the physical rho).

The panel: the predictive std |sigma| of Ours vs. t, one bold curve per rho. As in the advection
figure, every (trajectory, grid point) of a rho is pooled -- here over the 4 nu x 5 trajectories =
20 runs of that rho -- into N_TIME_BINS equal-width time bins, the predictive variance is averaged
within each bin and its square root is plotted at the bin center. The shaded band of each rho is
the spread over nu: the same binned curve computed per cell (nu, rho) from its 5 trajectories,
band = min to max over the 4 nu (chosen over per-nu thin lines: 16 extra lines would hide the
four pooled curves; the per-cell curves are in the table instead). Only Ours is shown (as in the
advection figure); there is no HMC reference in this figure, so no run gate applies and nothing is dropped.
Self-contained (reads the Ours cache through `reacdiff_baselines_sweep.load_cached`); reads cached
artifacts only, never trains; a missing artifact raises FileNotFoundError naming the file.

Cell selection: by default all 16 cells (every artifact must exist).
REACDIFF_CELLS="0.5:1.0,0.5:2.0" (comma-separated nu:rho pairs) restricts the script to
those cells; the log then states which cells were used and the table marks the others as not
selected. The default run is the one to report.

The companion table (`uncertainty_growth_curve_table.tex`) holds the numbers behind the panel:
block (a) the plotted pooled curve per rho at every time bin together with the band edges (min and
max over nu); block (b) every one of the 16 cells (nu, rho) explicitly -- its binned std at five
bin centers spanning [0, 1], its end-to-start ratio and n (the number of pooled trajectories) --
plus one pooled row per rho (pools the 4 nu of that rho) and one over all cells. The script writes
`uncertainty_growth_curve.log` (a copy of everything it prints) next to the outputs.

Run: python figures_onepass/diffusion/uncertainty_growth_curve.py [--png out.png]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURES_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "pde" / "diffusion"))
sys.path.insert(0, str(REPO_ROOT / "pde" / "advection"))

import reacdiff_baselines_sweep as sweep  # noqa: E402
from trajectory_selection import idxs  # noqa: E402

OUTPUT_PDF = FIGURES_DIR / "uncertainty_growth_curve.pdf"
OUTPUT_TABLE = FIGURES_DIR / "uncertainty_growth_curve_table.tex"
OUTPUT_LOG = FIGURES_DIR / "uncertainty_growth_curve.log"

NUS, RHOS = sweep.NUS, sweep.RHOS
ALL_CELLS = [(nu, rho) for rho in RHOS for nu in NUS]  # rho outer, nu inner: the table order
IDXS = idxs(5)  # the five canonical trajectories of every cell

MIN_VAR = 1e-12
N_TIME_BINS = 25
BAND_ALPHA = 0.18
N_TABLE_BINS = 5  # block (b) of the table quotes this many bin centers, evenly spread over [0, 1]

# Manually chosen, maximally contrastive colors (not a continuous colormap) -- one per reaction
# strength, in increasing rho order (the advection figure's colours, first four used). The colour
# is fixed per rho (RHOS.index), so an override run keeps the default run's colours.
CONTRASTIVE_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#9a6324",
]


# ----------------------------------------------------------------------------- cell selection
def selected_cells() -> tuple[list[tuple[float, float]], bool]:
    """(cells, override): all 16 cells, or the subset named by REACDIFF_CELLS ("nu:rho,nu:rho,...")."""
    raw = os.environ.get("REACDIFF_CELLS", "").strip()
    if not raw:
        return list(ALL_CELLS), False
    cells = []
    for item in raw.split(","):
        try:
            nu_s, rho_s = item.strip().split(":")
            cell = (float(nu_s), float(rho_s))
        except ValueError as e:
            raise ValueError(f"REACDIFF_CELLS entry {item!r} is not 'nu:rho'") from e
        if cell not in ALL_CELLS:
            raise ValueError(f"REACDIFF_CELLS entry {item!r} is not a grid cell (nu in {NUS}, rho in {RHOS})")
        if cell not in cells:
            cells.append(cell)
    return [c for c in ALL_CELLS if c in cells], True


def cell_text(nu: float, rho: float) -> str:
    return f"nu={nu:g} rho={rho:g}"


def rhos_of(cells) -> list[float]:
    return [rho for rho in RHOS if any(c[1] == rho for c in cells)]


# ----------------------------------------------------------------------------- data
def load_all_cached(cells) -> list[dict]:
    """One Ours record per (nu, rho, idx) of the selected cells, from the cache only. A missing
    artifact raises FileNotFoundError naming it (this script never trains)."""
    results = []
    for nu, rho in cells:
        for idx in IDXS:
            path = sweep.cache_path("Ours", nu, rho, idx)
            if not path.exists():
                raise FileNotFoundError(f"[{cell_text(nu, rho)} idx={idx}] missing cached artifact {path} "
                                        "(this script never trains; generate it with "
                                        "pde/diffusion/mp_pinn_reacdiff_dataset_run.jl, or restrict the "
                                        "cells with REACDIFF_CELLS)")
            results.append(sweep.load_cached("Ours", nu, rho, idx))
    return results


def pooled_t_and_var(results: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Per-point (t, pred_var) pooled across every (trajectory, grid point) in `results`."""
    t, pred_var = [], []
    for r in results:
        n_x, n_t = r["u_ref"].shape
        t.append(np.broadcast_to(r["t"][None, :], (n_x, n_t)).ravel())
        pred_var.append(np.maximum(r["pred_var"].ravel(), MIN_VAR))
    return np.concatenate(t), np.concatenate(pred_var)


def binned_mean_std(t: np.ndarray, pred_var: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(t.min(), t.max(), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(np.digitize(t, edges) - 1, 0, n_bins - 1)
    mean_var = np.full(n_bins, np.nan)
    for b in range(n_bins):
        mask = bin_idx == b
        if mask.any():
            mean_var[b] = pred_var[mask].mean()
    return centers, np.sqrt(mean_var)


def compute_curves(results: list[dict], cells) -> tuple[list[dict], list[dict], dict]:
    """(rho_curves, cell_curves, all_curve) -- the single source both the figure and its companion
    table draw from. rho_curves: one dict per rho (increasing rho) with the pooled binned sigma(t) over every
    run of that rho (`std`), the number of pooled runs (`n`), and the band edges `lo` / `hi` = min /
    max over nu of the per-cell curves. cell_curves: one dict per selected cell (nu, rho) with its
    own pooled binned sigma(t) over its trajectories and `n`. all_curve: the same over every loaded run
    (the last row of the table), pooled point by point like the other curves."""
    cell_curves = []
    for nu, rho in cells:
        group = [r for r in results if r["nu"] == nu and r["rho"] == rho]
        t, pred_var = pooled_t_and_var(group)
        centers, std_curve = binned_mean_std(t, pred_var, N_TIME_BINS)
        cell_curves.append(dict(nu=nu, rho=rho, n=len(group), centers=centers, std=std_curve))
    rho_curves = []
    for rho in rhos_of(cells):  # rho is PDEBench's own reaction-strength parameter
        group = [r for r in results if r["rho"] == rho]
        t, pred_var = pooled_t_and_var(group)
        centers, std_curve = binned_mean_std(t, pred_var, N_TIME_BINS)
        per_nu = np.stack([c["std"] for c in cell_curves if c["rho"] == rho])  # (n_nu, n_bins)
        rho_curves.append(dict(rho=rho, n=len(group), nus=[c["nu"] for c in cell_curves if c["rho"] == rho],
                               centers=centers, std=std_curve,
                               lo=np.nanmin(per_nu, axis=0), hi=np.nanmax(per_nu, axis=0)))
    t, pred_var = pooled_t_and_var(results)
    centers, std_curve = binned_mean_std(t, pred_var, N_TIME_BINS)
    all_curve = dict(n=len(results), centers=centers, std=std_curve)
    return rho_curves, cell_curves, all_curve


# ----------------------------------------------------------------------------- figure
def make_figure(rho_curves: list[dict], output_paths: list[Path]) -> None:
    plt.rcParams["mathtext.fontset"] = "cm"

    fig, ax = plt.subplots(figsize=(7, 5))
    for c in rho_curves:
        color = CONTRASTIVE_COLORS[RHOS.index(c["rho"])]
        ax.fill_between(c["centers"], c["lo"], c["hi"], color=color, alpha=BAND_ALPHA, lw=0)
        ax.plot(c["centers"], c["std"], color=color, lw=1.5, marker="^", ms=6, label=rf"$\rho={c['rho']:g}$")
    ax.fill_between([], [], [], color="0.5", alpha=BAND_ALPHA * 1.5, lw=0,
                    label=r"min–max over $\nu$")

    ax.set_xlabel(r"$t$", fontsize=14)
    ax.set_ylabel(r"$\mathrm{predictive\ std}$", fontsize=14)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    for path in output_paths:
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Saved uncertainty growth curve to {path}")
    plt.close(fig)


# ----------------------------------------------------------------------------- table
def _fmt(v: float, scale: float) -> str:
    return "--" if np.isnan(v) else f"{v / scale:.2f}"


def write_table(rho_curves: list[dict], cell_curves: list[dict], all_curve: dict, cells, override: bool,
                output_path: Path) -> None:
    """The plotted sigma(t) curves as numbers. Block (a): one row per time-bin center, per rho the
    pooled curve and its band edges (min / max over nu). Block (b): every one of the 16 cells, its
    binned sigma at N_TABLE_BINS bin centers, the end/start ratio and n, plus one pooled row per rho
    and one over all selected cells. Every value shares one common order-of-magnitude scale (set by
    the smallest value), divided out once and reported in the caption, as in the advection table."""
    centers = rho_curves[0]["centers"]
    # One row per bin center only makes sense if every group was binned on the same t grid.
    assert all(np.allclose(c["centers"], centers) for c in rho_curves + cell_curves + [all_curve]), "bin centers differ"

    all_sigma = np.concatenate([np.concatenate([c["std"], c["lo"], c["hi"]]) for c in rho_curves]
                               + [c["std"] for c in cell_curves])
    exp = int(np.floor(np.log10(np.nanmin(all_sigma[all_sigma > 0]))))
    scale = 10.0**exp
    half_width = 0.5 * (centers[1] - centers[0])
    t_lo, t_hi = (round(float(v), 6) for v in (centers[0] - half_width, centers[-1] + half_width))
    table_bins = np.linspace(0, len(centers) - 1, N_TABLE_BINS).round().astype(int)
    rhos = rhos_of(cells)

    lines = [
        f"% Auto-generated by figures_onepass/diffusion/{Path(__file__).name}",
        "% " + (f"OVERRIDE MODE (REACDIFF_CELLS): only the cells {', '.join(cell_text(*c) for c in cells)} "
                "were run; the other cells are marked 'not selected'. NOT the default run over all 16 cells."
                if override else f"default run: all {len(ALL_CELLS)} cells, {len(IDXS)} trajectories each"),
        "% No run gate applies (Ours only, no HMC reference in this figure); no run was dropped.",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\textbf{(a) The plotted curves.}\\[2pt]",
        rf"\begin{{tabular}}{{l {' '.join('ccc' for _ in rhos)}}}",
        r"\toprule",
        r"$t$ & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{$\rho={rho:g}$}}" for rho in rhos) + r" \\",
        " & ".join(["", *["pooled & min & max"] * len(rhos)]) + r" \\",
        r"\midrule",
    ]
    for i, center in enumerate(centers):
        cells_i = []
        for c in rho_curves:
            cells_i += [_fmt(c["std"][i], scale), _fmt(c["lo"][i], scale), _fmt(c["hi"][i], scale)]
        lines.append(f"{center:.2f} & " + " & ".join(cells_i) + r" \\")
    lines += [
        r"\midrule",
        r"$n_{\mathrm{runs}}$ & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{c['n']}}}" for c in rho_curves) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\\[6pt]",
        r"\textbf{(b) Per cell.}\\[2pt]",
        rf"\begin{{tabular}}{{l l {'c' * N_TABLE_BINS} c c}}",
        r"\toprule",
        r"$\nu$ & $\rho$ & " + " & ".join(rf"$t={centers[b]:.2f}$" for b in table_bins)
        + r" & $\sigma_{\mathrm{end}}/\sigma_{\mathrm{start}}$ & $n$ \\",
        r"\midrule",
    ]

    def block_row(label_nu: str, label_rho: str, std: np.ndarray, n: int) -> str:
        ratio = std[-1] / std[0] if std[0] > 0 else np.nan
        return (f"{label_nu} & {label_rho} & " + " & ".join(_fmt(std[b], scale) for b in table_bins)
                + f" & {ratio:.2f} & {n}" + r" \\")

    by_cell = {(c["nu"], c["rho"]): c for c in cell_curves}
    for rho in RHOS:
        for nu in NUS:
            if (nu, rho) in by_cell:
                c = by_cell[(nu, rho)]
                lines.append(block_row(f"{nu:g}", f"{rho:g}", c["std"], c["n"]))
            else:
                lines.append(f"{nu:g} & {rho:g} & " + " & ".join(["--"] * N_TABLE_BINS)
                             + r" & -- & 0 \quad\emph{(not selected)} \\")
        if rho in rhos:
            c = next(c for c in rho_curves if c["rho"] == rho)
            lines.append(r"\addlinespace[1pt]")
            lines.append(block_row(r"all (pooled over $\nu$)", f"{rho:g}", c["std"], c["n"]))
        lines.append(r"\midrule")
    lines.append(block_row(r"all (pooled over $\nu$)", r"all (pooled over $\rho$)", all_curve["std"], all_curve["n"]))
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{Binned posterior predictive standard deviation $|\sigma|$ of Ours (one-pass EP, "
        r"expectation propagation: a Gaussian posterior from one sweep of local moment matching; "
        rf"all values $\times 10^{{{exp}}}$) vs.\ time on the Fisher-KPP dataset. Within each of the "
        rf"{len(centers)} equal-width bins of $t\in[{t_lo:g}, {t_hi:g}]$ (rows labeled by bin center), "
        r"the predictive variance is averaged over every (trajectory, grid point) pooled for the group "
        r"and its square root is reported. Block (a), the plotted curves: per reaction strength $\rho$ "
        r"the pooled curve (bold line; pools the 4 diffusion coefficients $\nu$ and the "
        rf"{len(IDXS)} trajectories per cell, $n_{{\mathrm{{runs}}}}$ runs in total) and the plotted "
        r"band edges (min and max over $\nu$ of the per-cell curves). Block (b), every cell "
        r"$(\nu,\rho)$ of the $4\times4$ grid: its own curve (pooling its $n$ trajectories) at "
        rf"{N_TABLE_BINS} of the bin centers, the ratio of its last to its first bin, and $n$; the "
        r"row after each $\rho$ pools that $\rho$'s four cells (the block-(a) curve), the last row "
        r"pools every listed cell. The model sees $\rho_{\mathrm{norm}}=2\rho$ and "
        r"$\nu_{\mathrm{norm}}=2\nu$. No run gate applies (Ours only); no run was dropped."
        + (" OVERRIDE MODE: only the selected cells were run." if override else "") + "}",
        r"\label{tab:uncertainty_growth_curve_reacdiff}",
        r"\end{table}",
    ]
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Saved LaTeX table to {output_path}")


def print_summary(rho_curves: list[dict], cell_curves: list[dict]) -> None:
    print("Binned predictive std of Ours, start bin -> end bin (ratio); band = min..max over nu at the end bin:")
    for c in rho_curves:
        print(f"  rho={c['rho']:g}: n_runs={c['n']} (nu in {[f'{v:g}' for v in c['nus']]}), "
              f"sigma {c['std'][0]:.3e} -> {c['std'][-1]:.3e} (x{c['std'][-1] / c['std'][0]:.2f}), "
              f"end band [{c['lo'][-1]:.3e}, {c['hi'][-1]:.3e}]")
    print("Per cell (n trajectories): sigma start -> end (ratio)")
    for c in cell_curves:
        print(f"  {cell_text(c['nu'], c['rho'])} (n={c['n']}): {c['std'][0]:.3e} -> {c['std'][-1]:.3e} "
              f"(x{c['std'][-1] / c['std'][0]:.2f})")


class _Tee:
    """Copies everything printed to `path` (the script's .log), while still printing to the terminal."""

    def __init__(self, path: Path, stream):
        self.file = open(path, "w")
        self.stream = stream

    def write(self, s):
        self.stream.write(s)
        self.file.write(s)

    def flush(self):
        self.stream.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--png", type=Path, default=None, help="also render the figure to this PNG path")
    args = parser.parse_args()
    cells, override = selected_cells()
    tee = _Tee(OUTPUT_LOG, sys.stdout)
    sys.stdout = tee
    try:
        if override:
            print(f"REACDIFF_CELLS override: using {len(cells)} of {len(ALL_CELLS)} cells: "
                  + ", ".join(cell_text(*c) for c in cells) + " (NOT the default run over all 16 cells)")
        else:
            print(f"default run: all {len(ALL_CELLS)} cells, {len(IDXS)} trajectories each")
        results = load_all_cached(cells)
        print(f"Loaded {len(results)} cached Ours (one-pass EP) runs; no run gate applies (no HMC reference "
              "in this figure), no run dropped.")
        rho_curves, cell_curves, all_curve = compute_curves(results, cells)
        make_figure(rho_curves, [OUTPUT_PDF] + ([args.png] if args.png else []))
        print_summary(rho_curves, cell_curves)
        write_table(rho_curves, cell_curves, all_curve, cells, override, OUTPUT_TABLE)
    finally:
        sys.stdout = tee.stream
        tee.close()
        print(f"Log written to {OUTPUT_LOG}")


if __name__ == "__main__":
    main()
