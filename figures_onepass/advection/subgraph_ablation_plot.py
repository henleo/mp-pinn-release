"""
Solution spaces of the one-pass MP-PINN (advection) for the prior and for the posteriors of the
two subgraphs and of the full factor graph: predictive mean surface (colour) and +/- one
predictive std (grey shade): the paper's space_analysis.pdf (Figure "Solution spaces").

Regimes, all on the SAME factor graph (basis, collocation points, prior, schedule, seeds of every
one-pass run; `pde/advection/mp_pinn_subgraph_ablation.jl`):
  prior  untrained
  ic     trained on the data subgraph only (labeled initial-condition factors)
  res    trained on the PDE subgraph only (residual factors)
  full   trained on both = the one-pass posterior of every other figure (read from the canonical
         artifact mp_pinn_beta{b}_idx{i}.h5, which the ablation runner's own full regime
         reproduces bit for bit; checked on b = 0.1 trajectory 9394 and b = 2 trajectory 898)

Shipped figure (space_analysis.pdf, 2 x 2, the paper's panels): (a) prior, (b) data subgraph,
(c) full graph, (d) exact solution, for b = 0.1 on the first canonical trajectory. With
SUBGRAPH_ALL_PANELS=1 the script also writes space_analysis_all_panels.pdf (2 x 3): the same
panels plus the PDE-subgraph regime, which the paper text describes but the four-panel figure does
not show, and a panel with the spatial-mean predictive std of all four regimes over
time (the +/- std shades of the narrow regimes are thinner than a line in 3D).

Details:
  * Shown run: b = 0.1 on the first canonical trajectory idxs(10)[0] = 9394. The companion table
    covers all ten canonical trajectories at each of the five wave speeds of this directory
    (50 runs).
  * The +/- std shades are not clipped to the range of the exact solution; each
    panel's vertical axis spans its whole shade and the solution's range.
  * Predictive mean and std from the one-pass closed-form moments, i.e. the same
    numbers a per-point computation gives.
  * Companion table and log; a missing artifact is an error.

Artifacts: `pde/advection/mp_pinn_subgraph_ablation_sweep.py` (loader; `python
pde/advection/mp_pinn_subgraph_ablation_sweep.py` runs the missing ablations). No training here.

Run: python figures_onepass/advection/subgraph_ablation_plot.py [--png DIR]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

FIGURES_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURES_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "pde" / "advection"))

import mp_pinn_subgraph_ablation_sweep as abl  # noqa: E402

OUTPUT_PDF = FIGURES_DIR / "space_analysis.pdf"
OUTPUT_PDF_ALL_PANELS = FIGURES_DIR / "space_analysis_all_panels.pdf"
OUTPUT_TABLE = FIGURES_DIR / "space_analysis_table.tex"
OUTPUT_LOG = FIGURES_DIR / "subgraph_ablation_plot.log"
ALL_PANELS = os.environ.get("SUBGRAPH_ALL_PANELS", "").strip() == "1"

BETAS = abl.BETAS
IDXS = abl.IDXS
REGIMES = abl.REGIMES                    # prior, ic, res, full
SHOW_BETA = 0.1
SHOW_IDX = IDXS[0]
MAX_EPOCHS = 300
Z90 = float(norm.ppf(0.95))

REGIME_NAMES = {"prior": "prior", "ic": "data subgraph", "res": "PDE subgraph", "full": "full graph"}
REGIME_COLORS = {"prior": "0.45", "ic": "#1f77b4", "res": "#d62728", "full": "#2ca02c"}
# full graph dashed on top: its std curve lies on the PDE subgraph's
REGIME_LINES = {"prior": dict(linestyle="-", linewidth=5), "ic": dict(linestyle="-", linewidth=5),
                "res": dict(linestyle="-", linewidth=7), "full": dict(linestyle=(0, (4, 3)), linewidth=5)}
PANEL_TITLES = {
    "prior": "Prior space",
    "ic": "Posterior space after\ntraining on the data subgraph",
    "res": "Posterior space after\ntraining on the PDE subgraph",
    "full": "Posterior space after\ntraining on the full graph",
    "true": "Exact solution",
    "std": "Mean predictive std\nover time",
}
PANELS_SHIPPED = ("prior", "ic", "full", "true")
PANELS_ALL = ("prior", "ic", "res", "full", "true", "std")
FADE_SETTINGS = {"prior": dict(n_layers=40, alpha=0.05), "ic": dict(n_layers=40, alpha=0.015),
                 "res": dict(n_layers=40, alpha=0.015), "full": dict(n_layers=40, alpha=0.015)}
X_STRIDE, T_STRIDE = 16, 3   # downsampling for a renderable 3D surface (full grid 1024 x 201)


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


# ----------------------------------------------------------------------------- statistics


def regime_stats(r: dict, key: str) -> dict:
    err = r["mean"][key] - r["true_u"]
    sig = r["std"][key]
    return dict(
        mse_t0=float(np.mean(err[:, 0] ** 2)),
        mse=float(np.mean(err ** 2)),
        std_t0=float(np.mean(sig[:, 0])),
        std_end=float(np.mean(sig[:, -1])),
        cov90=float(np.mean(np.abs(err) <= Z90 * sig)),
        amplitude=float(np.std(r["mean"][key][:, -1]) / np.std(r["true_u"][:, -1])),
        epochs=r["epochs"][key],
    )


def collect() -> list[dict]:
    missing = [abl.cache_path(b, i) for b in BETAS for i in IDXS if not abl.cache_path(b, i).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} ablation artifacts missing, first: {missing[0]} "
                                "(python pde/advection/mp_pinn_subgraph_ablation_sweep.py runs them)")
    records = []
    for b in BETAS:
        for i in IDXS:
            r = abl.load(b, i)
            for key in REGIMES:
                if not (np.isfinite(r["mean"][key]).all() and np.isfinite(r["std"][key]).all()):
                    raise ValueError(f"b={b:g} idx={i}: non-finite predictive in regime {key}")
            r["stats"] = {key: regime_stats(r, key) for key in REGIMES}
            records.append(r)
    return records


def median_stats(records: list[dict], b: float, key: str) -> dict:
    rs = [r for r in records if r["beta"] == b]
    out = {s: float(np.median([r["stats"][key][s] for r in rs])) for s in rs[0]["stats"][key]}
    out["n"] = len(rs)
    out["n_capped"] = sum(r["stats"][key]["epochs"] >= MAX_EPOCHS for r in rs)
    return out


# ----------------------------------------------------------------------------- figure


def _label_3d_axes(ax) -> None:
    """Extra label/tick padding -- mplot3d's defaults crowd axis labels against tick numbers."""
    ax.set_xlabel("t", labelpad=30, fontsize=60)
    ax.set_ylabel("s", labelpad=30, fontsize=60)   # space is s in the paper (x = (s, t))
    ax.set_zlabel("u", labelpad=26, fontsize=60)
    ax.xaxis.set_tick_params(pad=8)
    ax.yaxis.set_tick_params(pad=8)
    ax.zaxis.set_tick_params(pad=10)
    ax.tick_params(axis="both", labelsize=20)


def _grids(r: dict):
    xi = np.arange(0, len(r["x"]), X_STRIDE)
    ti = np.arange(0, len(r["t"]), T_STRIDE)
    X, T = np.meshgrid(r["x"][xi], r["t"][ti], indexing="ij")
    return xi, ti, X, T


def draw_regime(ax, r: dict, key: str, title: str) -> tuple[float, float]:
    """Mean surface (jet on the exact solution's range) inside a grey +/- one-std shade; the
    vertical axis spans the whole shade and the solution's range. Returns the z limits."""
    xi, ti, X, T = _grids(r)
    vmin, vmax = r["true_u"].min(), r["true_u"].max()
    mean = r["mean"][key][np.ix_(xi, ti)]
    std = r["std"][key][np.ix_(xi, ti)]
    lower, upper = mean - std, mean + std
    settings = FADE_SETTINGS[key]
    for frac in np.linspace(0.0, 1.0, settings["n_layers"]):
        surf = ax.plot_surface(T, X, lower + frac * (upper - lower), color="#3d7fc7", alpha=settings["alpha"],
                               linewidth=0, antialiased=True)
        surf.set_rasterized(True)
    ax.plot_surface(T, X, mean, cmap="jet", vmin=vmin, vmax=vmax, linewidth=0, antialiased=True)
    zlo, zhi = min(vmin, lower.min()), max(vmax, upper.max())
    ax.set_zlim(zlo, zhi)
    _label_3d_axes(ax)
    ax.set_title(title, fontsize=64, pad=28)
    return zlo, zhi


def draw_exact(ax, r: dict, title: str) -> None:
    xi, ti, X, T = _grids(r)
    vmin, vmax = r["true_u"].min(), r["true_u"].max()
    ax.plot_surface(T, X, r["true_u"][np.ix_(xi, ti)], cmap="jet", vmin=vmin, vmax=vmax, linewidth=0,
                    antialiased=True)
    ax.set_zlim(vmin, vmax)
    _label_3d_axes(ax)
    ax.set_title(title, fontsize=64, pad=28)


def draw_std_over_time(ax, r: dict, title: str) -> None:
    for key in REGIMES:
        ax.plot(r["t"], r["std"][key].mean(axis=0), color=REGIME_COLORS[key], label=REGIME_NAMES[key],
                **REGIME_LINES[key])
    ax.set_yscale("log")
    ax.set_xlabel("t", fontsize=60, labelpad=12)
    ax.set_ylabel("predictive std", fontsize=44, labelpad=12)
    ax.tick_params(axis="both", labelsize=32)
    ax.grid(True, which="major", alpha=0.3)
    # in the empty band between the prior and the trained regimes
    ax.legend(fontsize=34, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.92), handlelength=2.5)
    ax.set_title(title, fontsize=64, pad=28)


def make_figure(r: dict, panels: tuple[str, ...], path: Path, png_path: Path | None = None) -> dict:
    """2 x 2 (four panels) or 2 x 3 (six panels) grid, letters in panel order; returns the z
    limits of every regime panel (for the log)."""
    n_cols = 2 if len(panels) == 4 else 3
    fig = plt.figure(figsize=(17 * n_cols, 30))
    gs = fig.add_gridspec(2, n_cols)
    zlims = {}
    for k, key in enumerate(panels):
        title = f"({chr(ord('a') + k)}) {PANEL_TITLES[key]}"
        pos = gs[k // n_cols, k % n_cols]
        if key == "std":
            # a 2D panel, inset so its tick labels line up with the 3D boxes' visual extent
            ax = fig.add_subplot(pos)
            draw_std_over_time(ax, r, title)
            continue
        ax = fig.add_subplot(pos, projection="3d")
        if key == "true":
            draw_exact(ax, r, title)
        else:
            zlims[key] = draw_regime(ax, r, key, title)
    fig.subplots_adjust(left=0.03, right=0.97, top=0.90, bottom=0.03, hspace=0.28, wspace=0.08)
    if "std" in panels:
        # the 2D panel needs room for its own tick labels, which the 3D boxes do not
        std_ax = fig.axes[panels.index("std")]
        box = std_ax.get_position()
        std_ax.set_position([box.x0 + 0.06 * box.width, box.y0 + 0.12 * box.height,
                             0.84 * box.width, 0.76 * box.height])
    fig.savefig(path, dpi=150)
    if png_path is not None:
        fig.savefig(png_path, dpi=30)
    plt.close(fig)
    print(f"Saved figure to {path}")
    return zlims


# ----------------------------------------------------------------------------- table


def _sci(v: float) -> str:
    """Compact scientific notation, e.g. 1.38e-3."""
    m, ex = f"{v:.2e}".split("e")
    return f"{m}e{int(ex)}"


def same_initial_conditions(records: list[dict]) -> bool:
    """True when every trajectory index has the identical initial condition at every b."""
    first = {}
    for r in records:
        ref = first.setdefault(r["idx"], r["true_u"][:, 0])
        if not np.array_equal(ref, r["true_u"][:, 0]):
            return False
    return True


def write_table(records: list[dict], shown: dict, path: Path) -> None:
    ids = ", ".join(str(i) for i in IDXS)
    header = [
        r"% Auto-generated by figures_onepass/advection/subgraph_ablation_plot.py",
        r"% Companion table of space_analysis.pdf: the one-pass MP-PINN (Ours) on one factor graph, untrained (prior),",
        r"%   trained on the data subgraph only (labeled initial-condition factors), on the PDE subgraph only (residual",
        r"%   factors), or on both (full graph = the one-pass posterior of every other figure).",
        rf"% Block 1: the run shown in the figure (b={SHOW_BETA:g}, trajectory {SHOW_IDX}); panels (a), (b), (c) = rows prior,",
        r"%   data subgraph, full graph (the PDE subgraph is shown only in space_analysis_all_panels.pdf).",
        rf"% Block 2: median over the canonical trajectories idxs({len(IDXS)}) = [{ids}]",
        r"%   (pde/advection/trajectory_selection.py) at each b; n = 10 per b, no run excluded.",
        r"% All against the true trajectory on the full 1024 x 201 (x,t) grid. MSE(0) / MSE: mean squared error of the",
        r"%   predictive mean at t = 0 (the labeled time) / over the whole grid. std(0) / std(end): spatial mean of the",
        r"%   predictive std at the first / last time. cov90: share of grid points whose truth lies in the central 90%",
        r"%   Gaussian predictive interval. amp.: spatial std of the predictive mean at the last time / that of the",
        r"%   truth (~0 = a constant function, 1 = the truth's amplitude). epochs: sweeps until the KL tolerance",
        rf"%   (median; capped = runs stopped at the {MAX_EPOCHS}-epoch cap).",
    ]
    if same_initial_conditions(records):
        header += [
            r"% PDEBench uses the same initial condition for a trajectory index at every b (checked on all runs here),",
            r"%   so MSE(0) of the regimes that see the data barely depends on b.",
        ]
    header += [
        r"\begin{tabular}{l l l l l r r r}",
        r"\toprule",
        r"regime & MSE(0) & MSE & std(0) & std(end) & cov90 & amp. & epochs \\",
    ]
    lines = list(header)
    for b in BETAS:
        for key in ("ic", "res"):
            s = median_stats(records, b, key)
            if s["epochs"] <= 1:
                prior = median_stats(records, b, "prior")
                rs = [r for r in records if r["beta"] == b]
                mu_max = max(r["w_absmean_max"][key] for r in rs)
                sd_min = min(r["w_std_min"][key] for r in rs)
                lines += [
                    rf"% NOTE b={b:g}, {REGIME_NAMES[key]}: the schedule stops after {s['epochs']:.0f} epoch (median). From this b's prior",
                    rf"%   (predictive std {prior['std_t0']:.1e}) every single factor update changes a weight belief by a KL below the",
                    r"%   tolerance 1e-5 (KL_divergence reports such changes as 0), so the stopping rule of train! ends the run at",
                    rf"%   once: the weights barely move (largest |posterior mean| {mu_max:.1e}, smallest posterior std {sd_min:.3f};",
                    r"%   prior 0 and 1), i.e. this row is an essentially untrained model. Its predictive mean still differs from",
                    r"%   the prior row's, because under this wide prior the output reacts strongly even to these small weight",
                    r"%   changes. The residual factors alone do shrink the weight spread (PDE-subgraph row), and the full graph,",
                    r"%   which interleaves both factor sets in every sweep, fits the data.",
                ]
    lines += [r"\midrule", rf"\multicolumn{{8}}{{l}}{{\emph{{shown run: $b={SHOW_BETA:g}$, trajectory {SHOW_IDX}}}}}\\"]
    for key in REGIMES:
        s = shown["stats"][key]
        lines.append(f"{REGIME_NAMES[key]} & {_sci(s['mse_t0'])} & {_sci(s['mse'])} & {_sci(s['std_t0'])} & "
                     f"{_sci(s['std_end'])} & {s['cov90']:.2f} & {s['amplitude']:.2f} & {s['epochs']} \\\\")
    for b in BETAS:
        lines += [r"\midrule", rf"\multicolumn{{8}}{{l}}{{\emph{{median over 10 trajectories, $b={b:g}$}}}}\\"]
        for key in REGIMES:
            s = median_stats(records, b, key)
            capped = f" ({s['n_capped']} capped)" if s["n_capped"] else ""
            lines.append(f"{REGIME_NAMES[key]} & {_sci(s['mse_t0'])} & {_sci(s['mse'])} & {_sci(s['std_t0'])} & "
                         f"{_sci(s['std_end'])} & {s['cov90']:.2f} & {s['amplitude']:.2f} & "
                         f"{s['epochs']:.0f}{capped} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines))
    print(f"Saved table to {path}")


def print_summary(records: list[dict]) -> None:
    print("per run and regime: MSE(0), MSE, std(0), std(end), cov90, amplitude, epochs")
    for r in records:
        print(f"b={r['beta']:g} idx={r['idx']} (K={r['n_kept']})")
        for key in REGIMES:
            s = r["stats"][key]
            print(f"  {REGIME_NAMES[key]:14s} MSE(0) {s['mse_t0']:.3e}  MSE {s['mse']:.3e}  std(0) {s['std_t0']:.3e}  "
                  f"std(end) {s['std_end']:.3e}  cov90 {s['cov90']:.3f}  amp {s['amplitude']:.3f}  epochs {s['epochs']}")
    print("\nmedian over the trajectories of each b:")
    for b in BETAS:
        print(f"b={b:g}")
        for key in REGIMES:
            s = median_stats(records, b, key)
            print(f"  {REGIME_NAMES[key]:14s} MSE(0) {s['mse_t0']:.3e}  MSE {s['mse']:.3e}  std(0) {s['std_t0']:.3e}  "
                  f"std(end) {s['std_end']:.3e}  cov90 {s['cov90']:.3f}  amp {s['amplitude']:.3f}  "
                  f"epochs {s['epochs']:.0f} (capped {s['n_capped']}/{s['n']})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", type=Path, default=None, help="also render the figures as PNG into this directory")
    args = ap.parse_args()
    tee = _Tee(OUTPUT_LOG)
    sys.stdout = tee
    try:
        records = collect()
        shown = next(r for r in records if r["beta"] == SHOW_BETA and r["idx"] == SHOW_IDX)
        if args.png is not None:
            args.png.mkdir(parents=True, exist_ok=True)
        png = (args.png / (OUTPUT_PDF.stem + ".png")) if args.png else None
        zlims = make_figure(shown, PANELS_SHIPPED, OUTPUT_PDF, png)
        if ALL_PANELS:
            png = (args.png / (OUTPUT_PDF_ALL_PANELS.stem + ".png")) if args.png else None
            zlims.update(make_figure(shown, PANELS_ALL, OUTPUT_PDF_ALL_PANELS, png))
        print(f"shown run b={SHOW_BETA:g} idx={SHOW_IDX}: exact solution range "
              f"[{shown['true_u'].min():.4f}, {shown['true_u'].max():.4f}]; vertical axis per panel (mean -/+ std "
              f"and the solution's range): " + ", ".join(f"{REGIME_NAMES[k]} [{lo:.4f}, {hi:.4f}]"
                                                        for k, (lo, hi) in zlims.items()))
        write_table(records, shown, OUTPUT_TABLE)
        print_summary(records)
    finally:
        tee.close()


if __name__ == "__main__":
    main()
