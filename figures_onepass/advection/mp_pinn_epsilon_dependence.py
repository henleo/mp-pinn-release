"""
Effect of the PDE tolerance epsilon on the one-pass MP-PINN (advection): prediction error and PDE
residual against epsilon, one line per wave speed (the paper's
mse_residual_vs_epsilon.pdf).

epsilon is the residual noise scale of the PDE factors, R ~ N(0, epsilon^2) (the runner's
beta_pde, the paper's epsilon_C); everything else is the one-pass pipeline of every other figure
here (same runner `mp_pinn_run_beta.jl`, prior, IC noise, basis rule, schedule and seeds). Note
that epsilon also enters the choice of the rotated basis (the weighted Gram matrix G of Appendix
"Differentiable and decorrelated basis"), so the number of kept eigen-directions can change with
it; the table lists it.

Protocol:
  * Runs: the ten canonical trajectories idxs(10) at each of the five wave speeds b = 0.1, 0.4,
    1, 2, 7 of this directory (50 runs per epsilon). Lines are the median over the ten
    trajectories, the shaded band their interquartile range.
  * epsilon grid: one 1-2-5 series for every b, 0.005 ... 5. The default epsilon = 0.05 of every
    other figure is marked; its runs are the canonical one-pass artifacts themselves (not re-run).
  * The predictive mean is E[g(z)] of the one-pass posterior, as in every other
    figure; the PDE residual is that of the network at the posterior-mean weights,
    R = g'(z) (z_t + c z_x) in the model's normalized time (the quantity `load_cached` of
    mp_pinn_beta_sweep.py rebuilds).
  * Companion table and log; a missing artifact is an error (nothing is skipped silently).

Per-run statistics, all over the full 1024 x 201 (x, t) grid against the true trajectory:
  MSE        mean of (predictive mean - truth)^2                          (panel a)
  mean R^2   mean of the squared residual                                 (panel b)
  cov90      share of grid points whose truth lies in the 90% predictive interval (table)
  amplitude  spatial std of the predictive mean at the last time over that of the truth (table;
             near 0 = collapsed to a constant function, the failure mode the paper describes)

Artifacts: `pde/advection/mp_pinn_epsilon_sweep.py` (loader; `python
pde/advection/mp_pinn_epsilon_sweep.py` trains missing runs). No training here.

Run: python figures_onepass/advection/mp_pinn_epsilon_dependence.py [--png DIR]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

FIGURES_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURES_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "pde" / "advection"))

import mp_pinn_epsilon_sweep as eps_sweep  # noqa: E402

OUTPUT_PDF = FIGURES_DIR / "mse_residual_vs_epsilon.pdf"
OUTPUT_TABLE = FIGURES_DIR / "mse_residual_vs_epsilon_table.tex"
OUTPUT_LOG = FIGURES_DIR / "mp_pinn_epsilon_dependence.log"

BETAS = eps_sweep.BETAS
EPSILONS = eps_sweep.EPSILONS
IDXS = eps_sweep.IDXS
DEFAULT_EPS = eps_sweep.DEFAULT_EPS
MAX_EPOCHS = 300               # the runner's epoch cap (train! default)
MIN_VAR = 1e-10                # variance floor for the interval (as in calibration_comparison.py)
Z90 = float(norm.ppf(0.95))    # half-width of the central 90% Gaussian interval in stds


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


def run_stats(r: dict) -> dict:
    err = r["pred_u"] - r["true_u"]
    sig = np.sqrt(np.maximum(r["pred_var"], MIN_VAR))
    finite = bool(np.isfinite(r["pred_u"]).all() and np.isfinite(r["pred_var"]).all()
                  and np.isfinite(r["residual"]).all())
    return dict(
        beta=r["beta"], eps=r["eps"], idx=r["idx"], finite=finite,
        mse=float(np.mean(err ** 2)),
        res2=float(np.mean(r["residual"] ** 2)),
        cov90=float(np.mean(np.abs(err) <= Z90 * sig)),
        amplitude=float(np.std(r["pred_u"][:, -1]) / np.std(r["true_u"][:, -1])),
        epochs=r["epochs"], n_kept=r["n_kept"], n_raw=r["n_raw"],
    )


def collect() -> list[dict]:
    """Per-run statistics of every (b, epsilon, trajectory); a missing artifact is an error."""
    missing = [eps_sweep.cache_path(b, e, i) for b in BETAS for e in EPSILONS for i in IDXS
               if not eps_sweep.cache_path(b, e, i).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} epsilon-sweep artifacts missing, first: {missing[0]} "
                                "(python pde/advection/mp_pinn_epsilon_sweep.py trains them)")
    stats = []
    for b in BETAS:
        for e in EPSILONS:
            for i in IDXS:
                stats.append(run_stats(eps_sweep.load_cached(b, e, i)))
    return stats


def group(stats: list[dict], b: float, e: float) -> list[dict]:
    return [s for s in stats if s["beta"] == b and s["eps"] == e]


def quartiles(values) -> tuple[float, float, float]:
    q1, med, q3 = np.percentile(np.asarray(values, dtype=float), [25, 50, 75])
    return float(med), float(q1), float(q3)


def summarize(stats: list[dict]) -> dict:
    """Per (b, epsilon): median and quartiles over the trajectories of every statistic, and counts."""
    out = {}
    for b in BETAS:
        for e in EPSILONS:
            g = group(stats, b, e)
            n_kept = sorted({s["n_kept"] for s in g})
            out[(b, e)] = dict(
                n=len(g), n_nonfinite=sum(not s["finite"] for s in g),
                n_capped=sum(s["epochs"] >= MAX_EPOCHS for s in g),
                n_kept=n_kept, n_raw=g[0]["n_raw"],
                **{k: quartiles([s[k] for s in g]) for k in ("mse", "res2", "cov90", "amplitude", "epochs")},
            )
    return out


def _b_label(b: float) -> str:
    return f"{b:g}"


def make_figure(summary: dict, path: Path, png_path: Path | None = None) -> None:
    plt.rcParams["mathtext.fontset"] = "cm"
    cmap = plt.get_cmap("viridis")
    colors = {b: cmap(i / max(1, len(BETAS) - 1)) for i, b in enumerate(BETAS)}
    fig, axes = plt.subplots(1, 2, figsize=(17, 7.6))
    eps = np.asarray(EPSILONS)
    for ax, key in ((axes[0], "mse"), (axes[1], "res2")):
        for b in BETAS:
            med = np.array([summary[(b, e)][key][0] for e in EPSILONS])
            q1 = np.array([summary[(b, e)][key][1] for e in EPSILONS])
            q3 = np.array([summary[(b, e)][key][2] for e in EPSILONS])
            ax.fill_between(eps, q1, q3, color=colors[b], alpha=0.07, linewidth=0)
            ax.plot(eps, med, color=colors[b], marker="o", markersize=7, linewidth=2.5, label=rf"$b={_b_label(b)}$")
        ax.axvline(DEFAULT_EPS, color="0.35", linestyle=":", linewidth=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\epsilon$", fontsize=24)
        ax.tick_params(axis="both", labelsize=18)
        ax.grid(True, which="major", alpha=0.3)
    for ax in axes:
        lo, hi = ax.get_ylim()
        ax.text(DEFAULT_EPS * 1.12, hi / (hi / lo) ** 0.04, r"$\mathrm{default}$", fontsize=16, color="0.35",
                va="top", ha="left")
    axes[0].set_ylabel(r"$\mathrm{MSE}$", fontsize=24)
    axes[1].set_ylabel(r"$\mathrm{mean\ squared\ residual}$", fontsize=24)
    axes[0].set_title(r"$\mathrm{(a)\ Squared\ error\ vs.}\ \epsilon$", fontsize=26)
    axes[1].set_title(r"$\mathrm{(b)\ Squared\ PDE\ residual\ vs.}\ \epsilon$", fontsize=26)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title=r"$\mathrm{wave\ speed}$", fontsize=18, title_fontsize=18,
               loc="lower center", ncol=len(BETAS), bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.13, 1, 1])
    fig.savefig(path)
    if png_path is not None:
        fig.savefig(png_path, dpi=80)
    plt.close(fig)
    print(f"Saved figure to {path}")


def _sci(v: float) -> str:
    """Compact scientific notation, e.g. 1.38e-3."""
    m, ex = f"{v:.2e}".split("e")
    return f"{m}e{int(ex)}"


def _range(q: tuple[float, float, float], fmt) -> str:
    return f"{fmt(q[0])} [{fmt(q[1])}, {fmt(q[2])}]"


def write_table(summary: dict, path: Path) -> None:
    ids = ", ".join(str(i) for i in IDXS)
    lines = [
        r"% Auto-generated by figures_onepass/advection/mp_pinn_epsilon_dependence.py",
        r"% Companion table of mse_residual_vs_epsilon.pdf: the one-pass MP-PINN (Ours) trained with PDE tolerance",
        r"%   epsilon (residual noise scale, R ~ N(0, epsilon^2)); everything else as in every other figure of this",
        rf"%   directory. epsilon = {DEFAULT_EPS:g} is the default (its rows are the canonical one-pass runs).",
        rf"% Runs: the canonical trajectories idxs({len(IDXS)}) = [{ids}]",
        r"%   (pde/advection/trajectory_selection.py) at every b and epsilon; n = 10 per row, no run excluded.",
        r"% All statistics over the full 1024 x 201 (x,t) grid of each run, against the true trajectory;",
        r"%   median [first quartile, third quartile] over the 10 trajectories.",
        r"% MSE: mean of (predictive mean - truth)^2 (panel a). R^2: mean squared PDE residual of the network at the",
        r"%   posterior-mean weights, in the model's normalized time (panel b).",
        r"% cov90: share of grid points whose truth lies in the central 90% Gaussian predictive interval (median).",
        r"% amp.: spatial std of the predictive mean at the last time / that of the truth (median; ~0 = collapsed",
        r"%   to a constant function).",
        rf"% capped: runs that stopped at the {MAX_EPOCHS}-epoch cap instead of the KL tolerance.",
        r"% K: eigen-directions kept by the rotated basis (the same for every trajectory of a b), of "
        + " / ".join(str(summary[(b, EPSILONS[0])]["n_raw"]) for b in BETAS),
        r"%   raw basis functions at b = " + " / ".join(_b_label(b) for b in BETAS) + ".",
        r"\begin{tabular}{r l l r r r r}",
        r"\toprule",
        r"$\epsilon$ & MSE & mean $R^2$ & cov90 & amp. & capped & $K$ \\",
    ]
    for b in BETAS:
        lines += [r"\midrule", rf"\multicolumn{{7}}{{l}}{{\emph{{$b={_b_label(b)}$}}}}\\"]
        for e in EPSILONS:
            s = summary[(b, e)]
            k = "/".join(str(v) for v in s["n_kept"])
            eps_str = rf"\textbf{{{e:g}}}" if e == DEFAULT_EPS else f"{e:g}"
            lines.append(
                f"{eps_str} & {_range(s['mse'], _sci)} & {_range(s['res2'], _sci)} & "
                f"{s['cov90'][0]:.2f} & {s['amplitude'][0]:.2f} & {s['n_capped']}/{s['n']} & {k} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines))
    print(f"Saved table to {path}")


def print_summary(stats: list[dict], summary: dict) -> None:
    print("per run: b, epsilon, idx, MSE, mean R^2, cov90, amplitude, epochs, K")
    for s in stats:
        print(f"  b={s['beta']:g} eps={s['eps']:g} idx={s['idx']}: MSE {s['mse']:.3e}  R^2 {s['res2']:.3e}  "
              f"cov90 {s['cov90']:.3f}  amp {s['amplitude']:.3f}  epochs {s['epochs']}  K {s['n_kept']}"
              + ("" if s["finite"] else "  NON-FINITE"))
    print("\nmedian over the 10 trajectories (IQR in brackets):")
    for b in BETAS:
        print(f"b={b:g}")
        for e in EPSILONS:
            s = summary[(b, e)]
            print(f"  eps={e:<6g} MSE {s['mse'][0]:.3e} [{s['mse'][1]:.2e}, {s['mse'][2]:.2e}]  "
                  f"R^2 {s['res2'][0]:.3e} [{s['res2'][1]:.2e}, {s['res2'][2]:.2e}]  cov90 {s['cov90'][0]:.3f}  "
                  f"amp {s['amplitude'][0]:.3f}  capped {s['n_capped']}/{s['n']}  K {s['n_kept']}"
                  + (f"  NON-FINITE {s['n_nonfinite']}" if s["n_nonfinite"] else ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", type=Path, default=None, help="also render the figure as PNG into this directory")
    args = ap.parse_args()
    tee = _Tee(OUTPUT_LOG)
    sys.stdout = tee
    try:
        stats = collect()
        if not all(s["finite"] for s in stats):
            bad = [f"b={s['beta']:g} eps={s['eps']:g} idx={s['idx']}" for s in stats if not s["finite"]]
            raise ValueError(f"non-finite predictive in {len(bad)} runs: {', '.join(bad)}")
        summary = summarize(stats)
        png = None
        if args.png is not None:
            args.png.mkdir(parents=True, exist_ok=True)
            png = args.png / (OUTPUT_PDF.stem + ".png")
        make_figure(summary, OUTPUT_PDF, png)
        write_table(summary, OUTPUT_TABLE)
        print_summary(stats, summary)
    finally:
        tee.close()


if __name__ == "__main__":
    main()
