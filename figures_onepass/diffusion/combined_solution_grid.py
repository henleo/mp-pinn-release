"""
Solution grid for the Fisher-KPP (reaction-diffusion) dataset problems: what our method's
posterior looks like on one representative trajectory, across the reaction strength rho.

Port of `figures_onepass/advection/combined_solution_grid.py` (there: one column per wave speed
b). Here the reaction strength rho plays the role of b: one column per rho in
{1, 2, 5, 10} at diffusion coefficient nu = 1 (the model sees the normalized values
rho_norm = 2 rho, nu_norm = 2 nu), trajectory idx 9394 (the first of the five canonical seeded
trajectories, `trajectory_selection.idxs(5)`, the same one `dataset_grid.py` shows). Rows:
  (a) exact solution: the float64 re-solve of the nominal PDE from the dataset's initial
      condition (`u_ref`, reacdiff_reference.py) -- NOT the stored PDEBench trajectory, whose
      late-time freeze is a data artifact;
  (b) Ours: the posterior predictive mean (the mean of the solution under the fitted
      approximate posterior) of the one-pass EP run (expectation propagation, the message-passing
      posterior approximation that is our method);
  (c) our predicted variance: the posterior predictive variance (the spread of the solution
      under that posterior) of the same run, on a log colour scale.
Rows (a)-(b) share one colour scale per column; row (c) has its own. Space x in [0, 1) with 1024
points, time t in [0, 1] with 101 steps (the dataset's own grid).

Only nu = 1 is drawn, so the companion table (`combined_solution_grid_table.tex`) has two parts:
the colour-scale extents and the rel-L2 error (||Ours - exact||_2 / ||exact||_2 over the plotted
(x, t) grid) of every drawn column, exactly as in the advection table; and, so nothing is lost
by showing one nu, the rel-L2 vs u_ref and the mean predictive variance (mean over the (x, t)
grid) for ALL 16 (nu, rho) cells, each averaged over the 5 canonical trajectories (n = 5 per
cell, mean +/- std), with pooled rows per rho (pooling the 4 nu, n = 20) and overall (n = 80).
The rel-L2 against the stored PDEBench trajectory is listed too, for transparency only.

No HMC-convergence gate applies to this figure: like the advection figure it shows only our
method (no HMC comparison), so no run is dropped; the log says so.

Reads cached artifacts only (`artifacts/mp_pinn_reacdiff_ds2_nu*_rho*_idx*.h5`); a missing
artifact raises FileNotFoundError naming the file. Never trains.

Run:  python figures_onepass/diffusion/combined_solution_grid.py [--png preview.png]
      leaves combined_solution_grid.{pdf,log} and combined_solution_grid_table.tex next to it.
Env:  REACDIFF_CELLS="1:1,1:2" (comma-separated nu:rho pairs) restricts the run to those cells
      (a partial build; the columns are the selected cells at nu = 1, the table
      lists the selected cells). Default: all 16 cells, every artifact required.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from matplotlib.ticker import FixedFormatter, FixedLocator, LogLocator, MaxNLocator

FIGURES_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURES_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "pde" / "diffusion"))
sys.path.insert(0, str(REPO_ROOT / "pde" / "advection"))

import reacdiff_baselines_sweep as rd  # noqa: E402
from trajectory_selection import idxs  # noqa: E402

OUTPUT_PDF = FIGURES_DIR / "combined_solution_grid.pdf"
OUTPUT_TABLE = FIGURES_DIR / "combined_solution_grid_table.tex"
OUTPUT_LOG = FIGURES_DIR / "combined_solution_grid.log"
PNG_PREVIEW: Path | None = None  # set by `--png <path>`: also write a PNG preview there (not shipped)

NUS = list(rd.NUS)      # [0.5, 1, 2, 5]
RHOS = list(rd.RHOS)    # [1, 2, 5, 10]  -- the figure's columns (at SHOW_NU)
IDXS = idxs(5)          # [9394, 898, 2398, 5906, 2343]
SHOW_NU = 1.0           # the nu drawn in the figure; all nu are in the table
SHOW_IDX = IDXS[0]      # 9394, the representative trajectory drawn
ALL_CELLS = [(nu, rho) for nu in NUS for rho in RHOS]

ROW_TITLES = [
    r"$\mathrm{(a)\ Exact}$" + "\n" + r"$\mathrm{solution}$",
    r"$\mathrm{(b)\ Ours}$",
    r"$\mathrm{(c)\ Our}$" + "\n" + r"$\mathrm{predicted}$" + "\n" + r"$\mathrm{variance}$",
]
BLOCK_BOTTOM_ROWS = {1, 2}  # rows 0-1 share a color scale (bottom=row 1); row 2 is its own block

# Manual vertical layout (figure-fraction units) -- matplotlib's uniform `hspace` can't give
# the row-1/row-2 gap (which must also fit the variance color scale) more room than the row-0/
# row-1 gap without inflating both, so both rows are positioned explicitly instead.
ROW0_TOP = 0.81
LEFT, RIGHT = 0.27, 0.97
SMALL_GAP_EXTRA = 0.01        # visual buffer beyond the xlabel's own footprint, row 0-1
GAP_ROW1_TO_SCALE = 0.05      # whitespace between "Ours" and the variance color scale
TICK_ROOM = 0.018             # room for a color scale's own tick labels, below the bar
GAP_SCALE_TO_ROW2 = 0.02      # whitespace between that scale and the variance row
BOTTOM_BUFFER = 0.015
CBAR_HEIGHT = 0.014
CBAR_GAP = 0.016              # whitespace between the rho=... title and the shared color scale
U_TICKS = [0.2, 0.5, 0.8]     # shared u scale: Fisher-KPP solutions live in [0, 1]; three short labels
                              # that fit a column-wide bar at this fontsize (0.25/0.50/0.75 collide)
MAX_VAR_TICKS = 3             # variance scale: at most this many labels per column-wide bar


class _Tee:
    """Duplicate stdout into the script's .log so the printed numbers ship with the figure."""

    def __init__(self, path: Path):
        self._file = open(path, "w")
        self._stdout = sys.stdout

    def write(self, s):
        self._stdout.write(s)
        self._file.write(s)

    def flush(self):
        self._stdout.flush()
        self._file.flush()


def selected_cells() -> tuple[list[tuple[float, float]], bool]:
    """(cells, override): all 16 cells by default, or the `REACDIFF_CELLS` subset."""
    raw = os.environ.get("REACDIFF_CELLS", "").strip()
    if not raw:
        return list(ALL_CELLS), False
    cells = []
    for item in raw.split(","):
        nu_s, rho_s = item.strip().split(":")
        cell = (float(nu_s), float(rho_s))
        if cell not in ALL_CELLS:
            raise ValueError(f"REACDIFF_CELLS entry {item!r} is not a grid cell; nu in {NUS}, rho in {RHOS}")
        cells.append(cell)
    return cells, True


def load_ours(nu: float, rho: float, idx: int) -> dict:
    """The cached one-pass EP artifact; raises (never trains) when it is missing."""
    path = rd.cache_path("Ours", nu, rho, idx)
    if not path.exists():
        raise FileNotFoundError(f"missing artifact {path} (nu={nu}, rho={rho}, idx={idx}); "
                                "generate it with pde/diffusion/mp_pinn_reacdiff_dataset_run.jl")
    return rd.load_cached("Ours", nu, rho, idx)


def variance_ticks(var_min: float, var_max: float) -> tuple[list[float], list[str], int | None]:
    """Tick positions, labels and the common exponent e for one column's log variance scale.
    The advection figure labels powers of ten only; here a column's variance often spans less
    than a decade with no power of ten inside, which would leave the bar unlabeled. So: if at
    least two powers of ten fall inside the range, those (labels 10^e, exponent None); else the
    1/2/5 multiples, else every integer multiple, else the range's own ends rounded inwards --
    then the labels are short mantissas v / 10^e and the bar gets one 'x10^e' text above its
    right end. At most MAX_VAR_TICKS ticks, thinned evenly."""

    def thin(ticks: list[float]) -> list[float]:
        if len(ticks) <= MAX_VAR_TICKS:
            return ticks
        pick = np.round(np.linspace(0, len(ticks) - 1, MAX_VAR_TICKS)).astype(int)
        return [ticks[i] for i in pick]

    def inside(subs) -> list[float]:
        cand = LogLocator(subs=subs, numticks=100).tick_values(var_min, var_max)
        out = []
        for v in cand:
            m = v / 10 ** np.floor(np.log10(v))
            if var_min <= v <= var_max and abs(m - round(m)) < 1e-6:  # integer multiples only
                out.append(float(v))
        return sorted(set(out))

    decades = inside((1.0,))
    if len(decades) >= 2:
        ticks = thin(decades)
        return ticks, [rf"$10^{{{int(round(np.log10(v)))}}}$" for v in ticks], None

    ticks = []
    for subs in ((1.0, 2.0, 5.0), tuple(float(k) for k in range(1, 10))):
        cand = inside(subs)
        if len(cand) >= 2:
            ticks = thin(cand)
            break
    if len(ticks) < 2:  # range too narrow for integer multiples: its ends, rounded inwards
        step = 10.0 ** np.floor(np.log10(var_max - var_min))
        lo, hi = float(np.ceil(var_min / step) * step), float(np.floor(var_max / step) * step)
        ticks = [lo, hi] if hi > lo else [lo]
    e = int(np.floor(np.log10(ticks[-1])))
    return ticks, [f"{round(v / 10 ** e, 6):g}" for v in ticks], e


def scale_limits(mp_r: dict) -> tuple[float, float, float, float]:
    """(vmin, vmax) of the color scale shared by rows 0-1 (exact/ours) and (var_min, var_max)
    of row 2's log scale -- the exact numbers the figure's colorbars span, so the companion
    table reports them from the same source. Exact = u_ref (float64 nominal-PDE reference)."""
    true_u, mp_pred, mp_var = mp_r["u_ref"], mp_r["pred_u"], mp_r["pred_var"]
    vmin = min(true_u.min(), mp_pred.min())
    vmax = max(true_u.max(), mp_pred.max())
    return vmin, vmax, max(mp_var.min(), 1e-300), mp_var.max()


def make_combined_grid(mp_results: list[dict], output_path: Path, png_path: Path | None = None) -> None:
    plt.rcParams["mathtext.fontset"] = "cm"

    n_cols = len(mp_results)
    n_rows = len(ROW_TITLES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.6 * n_cols, 6.2 * n_rows), squeeze=False)
    fig.subplots_adjust(left=LEFT, right=RIGHT, top=ROW0_TOP, bottom=0.05, hspace=0.8, wspace=0.3)

    shared_meshes = [None] * n_cols
    var_meshes = [None] * n_cols

    for col, mp_r in enumerate(mp_results):
        rho = mp_r["rho"]
        x, t = mp_r["x"], mp_r["t"]
        true_u, mp_pred, mp_var = mp_r["u_ref"], mp_r["pred_u"], mp_r["pred_var"]

        # Rows 0-1 (exact/ours) share one color scale per column (rho).
        vmin, vmax, var_min, var_max = scale_limits(mp_r)
        var_norm = LogNorm(vmin=var_min, vmax=var_max)

        panels = [
            (true_u, dict(vmin=vmin, vmax=vmax)),
            (mp_pred, dict(vmin=vmin, vmax=vmax)),
            (mp_var, dict(norm=var_norm)),
        ]

        for row, (grid, norm_kwargs) in enumerate(panels):
            ax = axes[row][col]
            mesh = ax.pcolormesh(t, x, grid, cmap="turbo", shading="auto", rasterized=True,
                                  **norm_kwargs)
            if row < 2:
                shared_meshes[col] = mesh  # rows 0-1 share one scale -> one horizontal colorbar
            else:
                var_meshes[col] = mesh  # row 2 (variance) has its own scale -> its own colorbar
            if row == 0:
                ax.set_title(rf"$\rho={rho:g}$", fontsize=56, pad=180)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            ax.tick_params(axis="both", labelsize=28)

            # x-axis (spatial) label/ticks: only the leftmost column of each row.
            if col == 0:
                ax.set_ylabel(r"$s$", fontsize=44)
            else:
                ax.tick_params(axis="y", labelleft=False)

            # t-axis (temporal) label/ticks: only the bottom row of each color-scale block
            # (rows 0-1 share one scale -> row 1 is its bottom; row 2 is its own block).
            if row in BLOCK_BOTTOM_ROWS:
                ax.set_xlabel(r"$t$", fontsize=44)
            else:
                ax.tick_params(axis="x", labelbottom=False)

    # Measure how much room each row's own xlabel/title needs below/above its axes (constant
    # for a given fontsize + figure height, independent of where the axes currently sits).
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()

    def bottom_of(artist):
        return artist.get_window_extent(renderer).transformed(inv).y0

    ref_pos = axes[1][0].get_position()
    xlabel_drop = ref_pos.y0 - bottom_of(axes[1][0].xaxis.label)

    # The rho=... title/shared-scale sit ABOVE ROW0_TOP, so they don't consume any of the budget
    # below it -- only each row's own trailing xlabel (and, for row 1, the variance scale) does.
    small_gap = xlabel_drop + SMALL_GAP_EXTRA
    big_gap = xlabel_drop + GAP_ROW1_TO_SCALE + CBAR_HEIGHT + TICK_ROOM + GAP_SCALE_TO_ROW2

    # Lay out all 3 rows explicitly: same height throughout, tight between rows 0-1, wide
    # between row 1 and row 2 (to fit the variance color scale plus the requested whitespace).
    row_height = (ROW0_TOP - small_gap - big_gap - xlabel_drop - BOTTOM_BUFFER) / n_rows

    row_tops = [ROW0_TOP]
    for gap in (small_gap, big_gap):
        row_tops.append(row_tops[-1] - row_height - gap)

    for row, top in enumerate(row_tops):
        for col in range(n_cols):
            old = axes[row][col].get_position()
            axes[row][col].set_position([old.x0, top - row_height, old.width, row_height])

    # Re-measure after moving the axes (label positions shift with them).
    fig.canvas.draw()
    label_bottom_row1 = bottom_of(axes[1][0].xaxis.label)
    var_cbar_top = label_bottom_row1 - GAP_ROW1_TO_SCALE
    title_bottom = bottom_of(axes[0][0].title)
    shared_cbar_top = title_bottom - CBAR_GAP

    for col in range(n_cols):
        width_pos = axes[0][col].get_position()

        # Shared (rows 0-1) color scale, above row 0 -- seen BEFORE its pictures, not after.
        y0 = shared_cbar_top - CBAR_HEIGHT
        cax = fig.add_axes([width_pos.x0, y0, width_pos.width, CBAR_HEIGHT])
        cbar = fig.colorbar(shared_meshes[col], cax=cax, orientation="horizontal")
        vmin, vmax, var_min, var_max = scale_limits(mp_results[col])
        cbar.set_ticks([v for v in U_TICKS if vmin <= v <= vmax])
        cbar.ax.tick_params(labelsize=28)

        # Variance color scale, above row 2 (between it and row 1), likewise before its picture.
        y0v = var_cbar_top - CBAR_HEIGHT
        cax_v = fig.add_axes([width_pos.x0, y0v, width_pos.width, CBAR_HEIGHT])
        cbar_v = fig.colorbar(var_meshes[col], cax=cax_v, orientation="horizontal")
        # Explicit ticks (see `variance_ticks`): a column's variance spans less than a decade
        # here, so the decade-only locator of the advection figure would label nothing, and the
        # default locator's major+minor ticks would overlap at this fontsize.
        ticks, labels, exponent = variance_ticks(var_min, var_max)
        cbar_v.locator = FixedLocator(ticks)
        cbar_v.formatter = FixedFormatter(labels)
        cbar_v.update_ticks()
        cbar_v.ax.minorticks_off()
        if exponent is not None:  # the ticks' common factor, above the bar's right end
            fig.text(width_pos.x1, y0v + CBAR_HEIGHT + 0.004, rf"$\times 10^{{{exponent}}}$",
                     ha="right", va="bottom", fontsize=28)
        cbar_v.ax.tick_params(labelsize=28)

    # Row labels as separate figure text (not crammed into the ylabel) so they don't overlap
    # neighboring rows at this fontsize. Horizontal, parallel to the figure's bottom edge.
    for row in range(n_rows):
        pos = axes[row][0].get_position()
        # Right-aligned just before the axes' own tight bbox (which includes its y-tick labels
        # and ylabel, sticking out left of `LEFT`) -- a flat offset from `LEFT` itself would
        # either overlap those or leave a gap, depending on row; this adapts per row instead.
        left_edge = axes[row][0].get_tightbbox(renderer).transformed(inv).x0
        fig.text(left_edge - 0.01, (pos.y0 + pos.y1) / 2, ROW_TITLES[row], rotation=0,
                  va="center", ha="right", fontsize=56)

    with PdfPages(output_path) as pdf:
        pdf.savefig(fig)
    print(f"Saved grid to {output_path}")
    if png_path is not None:
        fig.savefig(png_path, dpi=40)
        print(f"Saved PNG preview to {png_path}")
    plt.close(fig)


def write_table(mp_results: list[dict], cell_results: dict[tuple[float, float], list[dict]],
                path: Path, override: bool) -> None:
    """LaTeX companion table, two parts. Part 1 (as in the advection table): per drawn column
    (rho at nu = SHOW_NU, idx = SHOW_IDX), the color-scale extents its three panels are drawn
    with -- the shared exact/ours range and the log-scale variance range -- plus the rel-L2 of
    panel (b) against panel (a), all from the plotted arrays. Part 2: every (nu, rho) cell,
    rel-L2 vs u_ref and vs the stored data and the mean predictive variance, mean +/- std over
    the cell's trajectories, with pooled rows per rho and overall."""
    n_traj = len(IDXS)
    lines = [
        r"% Auto-generated by figures_onepass/diffusion/combined_solution_grid.py",
        rf"% Part 1: one column per reaction strength rho at nu={SHOW_NU:g} (trajectory idx={SHOW_IDX}).",
        r"% [u_min, u_max] is the color scale shared by the exact/Ours panels; [var_min, var_max] the",
        r"% variance panel's log scale; rel-L2 = ||Ours - exact||_2 / ||exact||_2 over the plotted",
        r"% (x, t) grid, exact = float64 re-solve of the nominal PDE (u_ref), not the stored data.",
        r"% Model-side normalized parameters: rho_norm = 2 rho, nu_norm = 2 nu.",
    ]
    if override:
        lines.append(f"% RESTRICTED RUN (REACDIFF_CELLS): only the cells listed below were used.")
    lines += [
        r"\begin{tabular}{r r r rr rr r}",
        r"\toprule",
        r" & & & \multicolumn{2}{c}{$u$ scale} & \multicolumn{2}{c}{$\sigma^2$ scale} & \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"$\nu$ & $\rho$ & $n_x \times n_t$ & min & max & min & max & rel-$L_2$ \\",
        r"\midrule",
    ]
    for mp_r in mp_results:
        vmin, vmax, var_min, var_max = scale_limits(mp_r)
        true_u, mp_pred = mp_r["u_ref"], mp_r["pred_u"]
        rel_l2 = np.linalg.norm(mp_pred - true_u) / np.linalg.norm(true_u)
        n_x, n_t = true_u.shape
        lines.append(
            f"{mp_r['nu']:g} & {mp_r['rho']:g} & ${n_x} \\times {n_t}$ & {vmin:.3f} & {vmax:.3f} & "
            f"{var_min:.2e} & {var_max:.2e} & {rel_l2:.4f} \\\\")
        print(f"nu={mp_r['nu']:g} rho={mp_r['rho']:g} idx={mp_r['idx']}: u in [{vmin:.4f}, {vmax:.4f}], "
              f"var in [{var_min:.3e}, {var_max:.3e}], rel_l2={rel_l2:.5f}")
    lines += [r"\bottomrule", r"\end{tabular}", ""]

    # Part 2: all cells, so the nu that are not drawn are not lost.
    lines += [
        rf"% Part 2: every (nu, rho) cell, n = number of trajectories per cell (the {n_traj} canonical",
        rf"% indices {IDXS}); mean +/- std over them of rel-L2 vs u_ref, rel-L2 vs the stored PDEBench",
        r"% trajectory (transparency only: its late-time freeze is a data artifact) and the mean",
        r"% predictive variance (mean of the posterior predictive variance over the (x, t) grid).",
        (r"% Pooled rows: 'all nu' pools the SELECTED cells at that rho; 'all' pools every selected cell"
         " (RESTRICTED RUN)." if override else
         r"% Pooled rows: 'all nu' pools the 4 nu at that rho (n = 4 x 5 = 20); 'all' pools every cell."),
        r"% No run is dropped: this figure shows only our method, so no HMC-convergence gate applies.",
        r"\begin{tabular}{r r r r r r}",
        r"\toprule",
        r"$\nu$ & $\rho$ & $n$ & rel-$L_2$ vs.\ $u_\mathrm{ref}$ & rel-$L_2$ vs.\ stored & mean $\sigma^2$ \\",
        r"\midrule",
    ]

    def stats(recs: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rr = np.array([r["rel_l2_ref"] for r in recs])
        rs = np.array([r["rel_l2_stored"] for r in recs])
        pv = np.array([float(r["pred_var"].mean()) for r in recs])
        return rr, rs, pv

    def row(label_nu: str, label_rho: str, recs: list[dict]) -> str:
        rr, rs, pv = stats(recs)
        return (rf"{label_nu} & {label_rho} & {len(recs)} & ${rr.mean():.4f} \pm {rr.std():.4f}$ & "
                rf"${rs.mean():.4f} \pm {rs.std():.4f}$ & ${pv.mean():.2e} \pm {pv.std():.2e}$ \\")

    for nu, rho in ALL_CELLS:
        recs = cell_results.get((nu, rho))
        if recs is None:
            if not override:
                raise RuntimeError(f"cell nu={nu} rho={rho} has no results in default mode")
            continue
        lines.append(row(f"{nu:g}", f"{rho:g}", recs))
        rr, rs, pv = stats(recs)
        print(f"cell nu={nu:g} rho={rho:g} n={len(recs)}: rel_l2_ref {rr.mean():.4f}+-{rr.std():.4f}, "
              f"rel_l2_stored {rs.mean():.4f}+-{rs.std():.4f}, mean pred_var {pv.mean():.3e}+-{pv.std():.3e}")
    lines.append(r"\midrule")
    for rho in RHOS:
        pooled = [r for (nu, rh), recs in cell_results.items() if rh == rho for r in recs]
        if pooled:
            lines.append(row("all $\\nu$", f"{rho:g}", pooled))
            rr, rs, pv = stats(pooled)
            print(f"pooled rho={rho:g} (all nu) n={len(pooled)}: rel_l2_ref {rr.mean():.4f}+-{rr.std():.4f}, "
                  f"mean pred_var {pv.mean():.3e}+-{pv.std():.3e}")
    everything = [r for recs in cell_results.values() for r in recs]
    lines.append(row("all", "all", everything))
    rr, rs, pv = stats(everything)
    print(f"pooled all cells n={len(everything)}: rel_l2_ref {rr.mean():.4f}+-{rr.std():.4f}, "
          f"mean pred_var {pv.mean():.3e}+-{pv.std():.3e}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"Saved LaTeX table to {path}")


def main() -> None:
    cells, override = selected_cells()
    if override:
        print(f"REACDIFF_CELLS override: using {len(cells)} cell(s): "
              + ", ".join(f"nu={nu:g}:rho={rho:g}" for nu, rho in cells))
    else:
        print(f"default mode: all {len(cells)} cells, {len(IDXS)} trajectories each (idx {IDXS})")
    print("no HMC-convergence gate applies (our method only); no run is dropped")

    # Every artifact of every selected cell: the table needs all trajectories, the figure one.
    cell_results: dict[tuple[float, float], list[dict]] = {}
    for nu, rho in cells:
        cell_results[(nu, rho)] = [load_ours(nu, rho, idx) for idx in IDXS]

    # Figure columns: the selected cells at SHOW_NU, in rho order, trajectory SHOW_IDX.
    columns = [rho for rho in RHOS if (SHOW_NU, rho) in cell_results]
    if not columns:
        raise RuntimeError(f"no selected cell at nu={SHOW_NU:g}; the figure needs at least one (REACDIFF_CELLS)")
    mp_results = [next(r for r in cell_results[(SHOW_NU, rho)] if r["idx"] == SHOW_IDX) for rho in columns]
    print(f"figure columns: nu={SHOW_NU:g}, rho in {columns}, idx={SHOW_IDX}")

    make_combined_grid(mp_results, OUTPUT_PDF, PNG_PREVIEW)
    write_table(mp_results, cell_results, OUTPUT_TABLE, override)


if __name__ == "__main__":
    if "--png" in sys.argv:
        PNG_PREVIEW = Path(sys.argv[sys.argv.index("--png") + 1])
    sys.stdout = _Tee(OUTPUT_LOG)
    main()
    sys.stdout.flush()
