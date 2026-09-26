"""
Posterior uncertainty vs. time, segregated by wave speed c_norm = 4*beta. mp_pinn_model.jl
trains on the initial condition (t=0) as its only labeled observation, plus unsupervised
PDE-residual constraints elsewhere. Self-contained (reads mp_pinn_beta_sweep's cache directly).

Trajectory set: the same ten canonical trajectories idxs(10) per wave speed b in
BETAS as every other figure of this directory (pde/advection/trajectory_selection.py, one seeded
draw). Both the trajectory ids and n per b are stated in
the table comment. Every artifact must exist (no silent gaps).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURES_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "pde" / "advection"))

import mp_pinn_beta_sweep as mp_sweep  # noqa: E402
from trajectory_selection import idxs  # noqa: E402

OUTPUT_PDF = FIGURES_DIR / "uncertainty_growth_curve.pdf"
OUTPUT_TABLE = FIGURES_DIR / "uncertainty_growth_curve_table.tex"

BETAS = [0.1, 0.4, 1.0, 2.0, 7.0]
IDXS = idxs(10)  # the canonical set shared with every other figure here
MIN_VAR = 1e-12
N_TIME_BINS = 25

# Manually chosen, maximally contrastive colors (not a continuous colormap) -- one per wave
# speed, in increasing beta order. Legend labels and table columns use PDEBench's beta (b), the
# same label as the other advection figures; the physical wave speed is c_norm = 4*b.
CONTRASTIVE_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#9a6324",
]


def load_all_cached() -> list[dict]:
    """The canonical trajectories only (IDXS at every b in BETAS); a missing artifact is an error."""
    results = []
    for beta in BETAS:
        for idx in IDXS:
            path = mp_sweep._cache_path(beta, idx)
            if not path.exists():
                raise FileNotFoundError(f"missing one-pass artifact {path} (run pde/advection/mp_pinn_run_beta.jl {beta} {idx})")
            results.append(mp_sweep.load_cached(beta, idx))
    return results


def pooled_t_and_var(results: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Per-point (t, pred_var) pooled across every (trajectory, grid point) in `results`."""
    t, pred_var = [], []
    for r in results:
        n_x, n_t = r["true_u"].shape
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


def compute_curves(results: list[dict]) -> list[tuple[float, int, np.ndarray, np.ndarray]]:
    """One (beta, n_trajectories, bin centers, binned sigma(t)) per wave speed, in increasing
    beta order -- the single source both the figure and its companion table draw from."""
    curves = []
    for beta in BETAS:  # beta is PDEBench's own wave-speed parameter
        group = [r for r in results if r["beta"] == beta]
        assert len(group) == len(IDXS), (beta, len(group))
        t, pred_var = pooled_t_and_var(group)
        centers, std_curve = binned_mean_std(t, pred_var, N_TIME_BINS)
        curves.append((beta, len(group), centers, std_curve))
    return curves


def make_figure(curves: list[tuple[float, int, np.ndarray, np.ndarray]], output_path: Path) -> None:
    plt.rcParams["mathtext.fontset"] = "cm"

    fig, ax = plt.subplots(figsize=(7, 5))
    for color, (beta, _, centers, std_curve) in zip(CONTRASTIVE_COLORS, curves):
        ax.plot(centers, std_curve, color=color, lw=1.5, marker="^", ms=6, label=rf"$b={beta:g}$")

    ax.set_xlabel(r"$t$", fontsize=14)
    ax.set_ylabel(r"$\mathrm{predictive\ std}$", fontsize=14)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved uncertainty growth curve to {output_path}")


def write_table(curves: list[tuple[float, int, np.ndarray, np.ndarray]], output_path: Path) -> None:
    """The plotted sigma(t) curves as numbers: one column per wave speed, one row per time-bin
    center, plus the number of pooled trajectories per b. Every value shares one common
    order-of-magnitude scale (set by the smallest value), divided out once and reported in the
    caption, as in mp_vs_gd_scatter_grid_table.tex."""
    betas = [beta for beta, _, _, _ in curves]
    n_traj = [n for _, n, _, _ in curves]
    centers = curves[0][2]
    # One row per bin center only makes sense if every wave speed was binned on the same t grid.
    assert all(np.allclose(c, centers) for _, _, c, _ in curves), "bin centers differ across b"

    sigma = np.column_stack([s for _, _, _, s in curves])  # (n_bins, n_betas)
    exp = int(np.floor(np.log10(np.nanmin(sigma[sigma > 0]))))
    scale = 10.0**exp
    half_width = 0.5 * (centers[1] - centers[0])
    t_lo, t_hi = (round(float(v), 6) for v in (centers[0] - half_width, centers[-1] + half_width))

    lines = [
        f"% Auto-generated by figures_onepass/advection/{Path(__file__).name}",
        f"% Trajectories: the canonical set idxs({len(IDXS)}) = {IDXS} (pde/advection/trajectory_selection.py) at every",
        f"% wave speed, the same set as every other figure of this directory; n per b: "
        + ", ".join(f"{n} at b={beta:g}" for beta, n in zip(betas, n_traj)) + ".",
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{l {'c' * len(betas)}}}",
        r"\toprule",
        r"$t$ & " + " & ".join(rf"$b={beta:g}$" for beta in betas) + r" \\",
        r"\midrule",
    ]
    for center, row in zip(centers, sigma):
        cells = ["--" if np.isnan(v) else f"{v / scale:.2f}" for v in row]
        lines.append(f"{center:.2f} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\midrule",
        r"$n_{\mathrm{traj}}$ & " + " & ".join(str(n) for n in n_traj) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{Binned posterior predictive standard deviation $|\sigma|$ (all values "
        rf"$\times 10^{{{exp}}}$) vs.\ time, one column per wave speed $b$: within each of the "
        rf"{len(centers)} equal-width bins of $t\in[{t_lo:g}, {t_hi:g}]$ (rows labeled by bin "
        r"center), the "
        r"predictive variance is averaged over every (trajectory, grid point) pooled for that "
        r"$b$ and its square root is reported. The last row is the number of trajectories pooled "
        r"per $b$: the canonical seeded set shared with every other figure of this set.}",
        r"\label{tab:uncertainty_growth_curve}",
        r"\end{table}",
    ]
    output_path.write_text("\n".join(lines) + "\n")
    print(f"Saved LaTeX table to {output_path}")


def main() -> None:
    results = load_all_cached()
    print(f"Loaded {len(results)} cached MP-PINN trajectories.")
    curves = compute_curves(results)
    make_figure(curves, OUTPUT_PDF)
    write_table(curves, OUTPUT_TABLE)


if __name__ == "__main__":
    main()
