from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator, MaxNLocator

FIGURES_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURES_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "pde" / "advection"))

import mp_pinn_beta_sweep as mp_sweep

OUTPUT_PDF = FIGURES_DIR / "combined_solution_grid.pdf"
OUTPUT_TABLE = FIGURES_DIR / "combined_solution_grid_table.tex"

BETAS = [0.1, 0.4, 1.0, 2.0, 7.0]  # = mp_sweep.BETAS; pass a different list to main() to override
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
CBAR_GAP = 0.016              # whitespace between the b=... title and the shared color scale


def scale_limits(mp_r: dict) -> tuple[float, float, float, float]:
    """(vmin, vmax) of the color scale shared by rows 0-1 (exact/ours) and (var_min, var_max)
    of row 2's log scale -- the exact numbers the figure's colorbars span, so the companion
    table reports them from the same source."""
    true_u, mp_pred, mp_var = mp_r["true_u"], mp_r["pred_u"], mp_r["pred_var"]
    vmin = min(true_u.min(), mp_pred.min())
    vmax = max(true_u.max(), mp_pred.max())
    return vmin, vmax, max(mp_var.min(), 1e-300), mp_var.max()


def make_combined_grid(mp_results: list[dict], output_path: Path) -> None:
    plt.rcParams["mathtext.fontset"] = "cm"

    n_cols = len(mp_results)
    n_rows = len(ROW_TITLES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.6 * n_cols, 6.2 * n_rows), squeeze=False)
    fig.subplots_adjust(left=LEFT, right=RIGHT, top=ROW0_TOP, bottom=0.05, hspace=0.8, wspace=0.3)

    shared_meshes = [None] * n_cols
    var_meshes = [None] * n_cols

    for col, mp_r in enumerate(mp_results):
        beta = mp_r["beta"]
        x, t = mp_r["x"], mp_r["t"]
        true_u, mp_pred, mp_var = mp_r["true_u"], mp_r["pred_u"], mp_r["pred_var"]

        # Rows 0-1 (exact/ours) share one color scale per column (beta).
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
                ax.set_title(rf"$b={beta}$", fontsize=56, pad=180)
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

    # The b=... title/shared-scale sit ABOVE ROW0_TOP, so they don't consume any of the budget
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
        cbar.ax.tick_params(labelsize=28)

        # Variance color scale, above row 2 (between it and row 1), likewise before its picture.
        y0v = var_cbar_top - CBAR_HEIGHT
        cax_v = fig.add_axes([width_pos.x0, y0v, width_pos.width, CBAR_HEIGHT])
        cbar_v = fig.colorbar(var_meshes[col], cax=cax_v, orientation="horizontal")
        # Cap tick count -- a narrow-range column (e.g. b=7.0's variance spans <1 decade) would
        # otherwise pack in enough LogLocator major+minor ticks to overlap at this fontsize.
        cbar_v.locator = LogLocator(subs=(1.0,), numticks=10)
        cbar_v.update_ticks()
        cbar_v.ax.minorticks_off()
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
    plt.close(fig)
    print(f"Saved grid to {output_path}")


def write_table(mp_results: list[dict], path: Path) -> None:
    """LaTeX companion table: per column (beta) of the grid, the color-scale extents its
    three panels are drawn with -- the shared exact/ours range and the log-scale variance
    range -- plus the rel-L2 of panel (b) against panel (a), all from the plotted arrays."""
    lines = [
        r"% Auto-generated by figures_onepass/advection/combined_solution_grid.py",
        r"% One column per wave speed b (trajectory idx=6369). [u_min, u_max] is the color",
        r"% scale shared by the exact/Ours panels; [var_min, var_max] the variance panel's log",
        r"% scale; rel-L2 = ||Ours - exact||_2 / ||exact||_2 over the plotted (x, t) grid.",
        r"\begin{tabular}{r r rr rr r}",
        r"\toprule",
        r" & & \multicolumn{2}{c}{$u$ scale} & \multicolumn{2}{c}{$\sigma^2$ scale} & \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
        r"$b$ & $n_x \times n_t$ & min & max & min & max & rel-$L_2$ \\",
        r"\midrule",
    ]
    for mp_r in mp_results:
        vmin, vmax, var_min, var_max = scale_limits(mp_r)
        true_u, mp_pred = mp_r["true_u"], mp_r["pred_u"]
        rel_l2 = np.linalg.norm(mp_pred - true_u) / np.linalg.norm(true_u)
        n_x, n_t = true_u.shape
        lines.append(
            f"{mp_r['beta']:g} & ${n_x} \\times {n_t}$ & {vmin:.3f} & {vmax:.3f} & "
            f"{var_min:.2e} & {var_max:.2e} & {rel_l2:.4f} \\\\")
        print(f"b={mp_r['beta']:g}: u in [{vmin:.4f}, {vmax:.4f}], "
              f"var in [{var_min:.3e}, {var_max:.3e}], rel_l2={rel_l2:.5f}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"Saved LaTeX table to {path}")


def run_mp(beta: float, idx: int) -> dict:
    if not mp_sweep._cache_path(beta, idx).exists():
        mp_sweep.train_one_beta(beta, idx)
    return mp_sweep.load_cached(beta, idx)


def main(betas: list[float] = BETAS) -> None:
    # idx=6369 (not the sweep's own default of -1): one fixed illustrative trajectory; run_mp
    # trains it only when its cached artifact is missing.
    idx = 6369
    mp_results = [run_mp(beta, idx) for beta in betas]
    make_combined_grid(mp_results, OUTPUT_PDF)
    write_table(mp_results, OUTPUT_TABLE)


if __name__ == "__main__":
    main()
