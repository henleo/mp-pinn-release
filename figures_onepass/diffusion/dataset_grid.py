"""
Fisher-KPP on the REAL PDEBench 1D reaction-diffusion dataset: the full 4x4 (nu, rho) grid of
dataset settings, five random-IC trajectories each (canonical seeded indices shared with the
advection sweeps). For every setting, one representative trajectory is shown as a
True (float64 nominal-PDE reference) / Ours (MP-PINN posterior mean) heatmap pair, annotated
with the mean +/- std of rel-L2 over all five trajectories; a companion LaTeX table
(`dataset_grid_table.tex`) carries the full numbers, including the error against the STORED
PDEBench trajectory for transparency.

Ground truth is the float64 re-solve of the nominal PDE from the same initial condition
(`reacdiff_reference.py`), NOT the stored trajectory: the stored PDEBench trajectories freeze
at non-physical values once diffusion homogenizes the field (float32 stall of the generator's
exact-logistic substep -- e.g. u=0.625 is an exact fixed point of its float32 map at nu=5,
rho=1). The reference agrees with the stored data to ~1e-3 before the freeze.

Run: python figures_onepass/diffusion/dataset_grid.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURES_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "pde" / "diffusion"))
sys.path.insert(0, str(REPO_ROOT / "pde" / "advection"))

import mp_pinn_reacdiff_dataset_sweep as ds_sweep
from trajectory_selection import idxs

OUTPUT_PDF = FIGURES_DIR / "dataset_grid.pdf"
OUTPUT_TABLE = FIGURES_DIR / "dataset_grid_table.tex"

NUS = ds_sweep.NUS
RHOS = ds_sweep.RHOS
IDXS = idxs(5)
SHOW_IDX = IDXS[0]      # representative trajectory per cell
FAIL_THRESHOLD = 0.3    # mean rel_l2 above this is flagged as a breakdown (red border)


def collect() -> dict[tuple[float, float], list[dict]]:
    results = {}
    for nu in NUS:
        for rho in RHOS:
            cell = []
            for idx in IDXS:
                path = ds_sweep._cache_path(nu, rho, idx)
                if not path.exists():
                    raise FileNotFoundError(f"missing artifact {path}; run: python run_experiments.py diff-ours")
                cell.append(ds_sweep.load_cached(nu, rho, idx))
            results[(nu, rho)] = cell
    return results


def make_grid(results, output_path: Path) -> None:
    plt.rcParams["mathtext.fontset"] = "cm"
    n_rows, n_cols = len(RHOS), 2 * len(NUS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 3.4 * n_rows), squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.965, top=0.9, bottom=0.05, hspace=0.5, wspace=0.45)

    for i, rho in enumerate(RHOS):
        for k, nu in enumerate(NUS):
            cell = results.get((nu, rho))
            true_ax, ours_ax = axes[i][2 * k], axes[i][2 * k + 1]
            if not cell:
                for ax in (true_ax, ours_ax):
                    ax.text(0.5, 0.5, "not run", ha="center", va="center", fontsize=13,
                            color="gray", transform=ax.transAxes)
                    ax.set_xticks([]); ax.set_yticks([])
                continue
            r = next((c for c in cell if c["idx"] == SHOW_IDX), cell[0])
            rels = np.array([c["rel_l2_ref"] for c in cell])
            failed = rels.mean() > FAIL_THRESHOLD
            vmin, vmax = r["u_ref"].min(), r["u_ref"].max()
            ours_mesh = None
            for sub, (kind, ax, grid) in enumerate((("True", true_ax, r["u_ref"]),
                                                     ("Ours", ours_ax, r["pred_u"]))):
                mesh = ax.pcolormesh(r["t"], r["x"], grid, cmap="jet", shading="auto",
                                      vmin=vmin, vmax=vmax, rasterized=True)
                if kind == "Ours":
                    ours_mesh = mesh
                color = "crimson" if (failed and kind == "Ours") else "black"
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(2.0 if (failed and kind == "Ours") else 0.8)
                ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=3))
                ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=3))
                ax.tick_params(axis="both", labelsize=11)
                ax.set_xlabel(r"$t$", fontsize=16)
                if i == 0:
                    ax.set_title(rf"$\nu={nu:g}$" + "\n" + kind, fontsize=18)
                if k == 0 and sub == 0:
                    ax.set_ylabel(rf"$\rho={rho:g}$" + "\n" + r"$s$", fontsize=16)
                elif sub == 0:
                    ax.set_ylabel(r"$s$", fontsize=16)
            ours_ax.text(0.5, -0.42, rf"rel-$L_2$: ${rels.mean():.3f} \pm {rels.std():.3f}$",
                         transform=ours_ax.transAxes, ha="center", fontsize=12)
            if ours_mesh is not None:
                cbar = fig.colorbar(ours_mesh, ax=[true_ax, ours_ax], fraction=0.046, pad=0.03)
                cbar.ax.tick_params(labelsize=10)

    fig.suptitle(r"$\mathrm{Fisher}$-$\mathrm{KPP\ (PDEBench\ dataset\ ICs)}$", fontsize=24)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved grid to {output_path}")


def make_table(results, output_path: Path) -> None:
    lines = [
        r"% Auto-generated by figures_onepass/diffusion/dataset_grid.py",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{ll ccc}",
        r"\toprule",
        r"$\nu$ & $\rho$ & rel-$L_2$ vs.\ nominal PDE & rel-$L_2$ vs.\ stored data & train time (s) \\",
        r"\midrule",
    ]
    for nu in NUS:
        for rho in RHOS:
            cell = results.get((nu, rho))
            if not cell:
                continue
            rr = np.array([c["rel_l2_ref"] for c in cell])
            rs = np.array([c["rel_l2_stored"] for c in cell])
            tt = np.array([c["train_time_seconds"] for c in cell])
            lines.append(
                rf"{nu:g} & {rho:g} & ${rr.mean():.3f} \pm {rr.std():.3f}$ & "
                rf"${rs.mean():.3f} \pm {rs.std():.3f}$ & ${tt.mean():.0f}$ \\"
            )
    n = sum(len(v) for v in results.values())
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{Fisher--KPP on PDEBench dataset initial conditions ({n} runs, "
        r"5 trajectories per setting). Errors are reported against a float64 re-solve of the "
        r"nominal PDE from the same initial condition; the stored-data column is inflated at "
        r"late times by the dataset's frozen-reaction artifact and shown for transparency.}",
        r"\label{tab:reacdiff_dataset}",
        r"\end{table}",
    ]
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Saved LaTeX table to {output_path}")


def main() -> None:
    results = collect()
    if not results:
        raise SystemExit("no cached dataset runs found -- run the sweep first")
    make_grid(results, OUTPUT_PDF)
    make_table(results, OUTPUT_TABLE)
    pooled = np.array([c["rel_l2_ref"] for cell in results.values() for c in cell])
    print(f"pooled rel_l2 vs nominal PDE: median {np.median(pooled):.3f}, "
          f"90th pct {np.quantile(pooled, 0.9):.3f}, n={len(pooled)}")


if __name__ == "__main__":
    main()
