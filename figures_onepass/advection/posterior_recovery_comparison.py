"""
Weight-space posterior recovery: pairwise scatter comparisons of the three inference methods,
aggregated over the canonical seeded trajectory set per wave speed (b in {0.1, 0.4, 1.0, 2.0,
7.0}; the set is shared with every other multi-trajectory figure and the development
trajectory is excluded). Every b uses the same ten trajectories, b = 7 included (one b = 7
run costs about an hour of HMC and two hours of VI). The number of trajectories per wave
speed is stated in the log, the summary JSON and every companion table.

  * Ours: the single-pass (one-pass) EP (expectation propagation) posterior of MP-PINN. For
    every eigenbasis coordinate (one of the rotated weight coordinates all three methods
    share) the runner stores the mean and the std of that coordinate's marginal belief.
  * HMC (Hamiltonian Monte Carlo): the asymptotically exact reference. 4 chains with a dense
    Gauss-Newton mass matrix (a curvature-based preconditioner) and overdispersed
    initializations (`bpinn_run_beta_multichain.jl`). A coordinate is admitted into every
    comparison only when split-R-hat < 1.05 and bulk ESS (effective sample size) > 100; these
    two chain convergence diagnostics certify that the chains actually converged there (the
    admitted fraction is printed and saved alongside the figures). A run (one (b, idx)) enters
    the statistics only with at least MIN_ADMITTED admitted coordinates; which runs are
    dropped by this gate, and which runs enter with only a partially admitted coordinate set,
    is computed from the caches at run time and written to the log, the summary JSON and the
    table comments (never hard-coded here).
  * VI (variational inference): mean-field (one independent Gaussian per coordinate) Gaussian
    variational inference, Algorithm 2 of Yang et al. 2021. At b = 7 VI's predictive mean
    collapses (rel-L2 about 1): its posterior means are near zero for every
    coordinate, so the VI panels at b = 7 show the collapse rather than a fit. The VI points
    are shown nevertheless (the axes are wide enough), and the tables print the resulting
    extreme numbers instead of hiding them.

Three separate PDFs, one per pair of methods, each a 1x2 figure (left: posterior means,
right: posterior stds; y=x dashed in both):

  posterior_recovery_ours_vs_hmc.pdf   x = HMC, y = Ours
  posterior_recovery_ours_vs_vi.pdf    x = VI,  y = Ours
  posterior_recovery_vi_vs_hmc.pdf     x = HMC, y = VI

Each point is one eigenbasis coordinate of one trajectory's posterior, pooling all betas and
all trajectories (viridis colour per b). A random coordinate subsample per run (drawn ONCE
per (b, idx) among the admitted HMC coordinates and reused in all three figures, so the same
coordinates appear everywhere) keeps ~1000 points per panel legible. Mean panels use symlog
axes (linear near zero, logarithmic beyond; the linear region is |mu| < 1e-2) with symmetric,
data-driven limits: 1.15 times the largest |mean| over the plotted points of all three figures,
and the same limits in every figure so the panels stay comparable and no point is clipped (the
value is printed in the log). Std panels are log-log (both axes logarithmic), so a constant
multiplicative std bias appears as a parallel offset from y=x.

A machine-readable summary (per-beta Pearson correlations (the linear correlation coefficient)
of the posterior means, median std ratios, median mean mismatch in units of the x-method's
posterior std, admitted fractions, n per wave speed, dropped runs, for all three pairs) and one
LaTeX companion table per figure derived from that same summary are written next to the PDFs.
All statistics use only admitted HMC coordinates, in every pair, so the three tables describe
the same coordinate set. No training is triggered: the script only reads caches. A run whose
cache is missing (for example an HMC run that has not finished yet) is dropped with a
printed reason and listed in the tables, never fabricated.

Run: python figures_onepass/advection/posterior_recovery_comparison.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURES_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "pde" / "advection"))

import mp_pinn_beta_sweep as mp_sweep
import bpinn_multichain_sweep as hmc_sweep
import vi_baseline_sweep as vi_sweep
from trajectory_selection import idxs

OUTPUT_SUMMARY = FIGURES_DIR / "posterior_recovery_comparison_summary.json"

# (x-method, y-method) -> output stem; the method keys index the per-run records below.
PAIRS = {
    ("hmc", "mp"): "posterior_recovery_ours_vs_hmc",
    ("vi", "mp"): "posterior_recovery_ours_vs_vi",
    ("hmc", "vi"): "posterior_recovery_vi_vs_hmc",
}
NAMES = {"mp": "Ours", "hmc": "HMC", "vi": "VI"}
MODULES = {"mp": mp_sweep, "hmc": hmc_sweep, "vi": vi_sweep}
# One-line explanation of each method for the table comments (first use of every term).
GLOSSARY = {
    "mp": "Ours = the single-pass EP (expectation propagation) posterior of MP-PINN",
    "hmc": "HMC = Hamiltonian Monte Carlo, the asymptotically exact reference",
    "vi": "VI = mean-field (one independent Gaussian per coordinate) variational inference",
}

BETAS = [0.1, 0.4, 1.0, 2.0, 7.0]
# The same ten canonical trajectories (one seeded draw) at every wave speed.
IDXS_BY_BETA = {b: idxs(10) for b in (0.1, 0.4, 1.0, 2.0, 7.0)}  # the same ten trajectories at every b
N_POINTS_PER_RUN = 25   # random eigencoordinate subsample per (beta, idx), for legibility
RHAT_MAX = 1.05
ESS_MIN = 100.0
RNG_SEED = 0
MIN_ADMITTED = 3        # a trajectory enters the statistics only with >= this many coordinates
MEAN_LIM_PAD = 1.15     # symlog mean axes: symmetric limits = this factor x the largest plotted |mean|
MEAN_LINTHRESH = 1e-2   # symlog mean axes: |mu| below this is drawn on a linear scale


def _load(method: str, beta: float, idx: int) -> dict | None:
    """Load a cached posterior, or None when the cache is missing; never train from a figure
    script."""
    module = MODULES[method]
    path = module._cache_path(beta, idx)
    if not path.exists():
        print(f"  MISSING {NAMES[method]} cache for beta={beta} idx={idx}: {path}")
        return None
    return module.load_cached(beta, idx)


def _run_label(beta: float, idx: int) -> str:
    return f"b={beta:g} idx {idx}"


def _accept_str(hmc: dict) -> str:
    return "/".join(f"{float(a):.2f}" for a in hmc["accept_rates"])


def collect_results() -> tuple[list[dict], list[dict]]:
    """Returns (records, dropped). One record per loaded (beta, idx): the three posteriors,
    the HMC convergence mask, the coordinate subsample used by all three figures (drawn once
    here, seeded) and whether the run passes the run gate (>= MIN_ADMITTED admitted
    coordinates). `dropped` lists every planned run that does not enter the statistics, with
    its reason: a missing cache, or the HMC gate (with admitted fraction and chain acceptance
    rates). Runs that enter with a partially admitted coordinate set are flagged in the
    record (`partial`) so the log and tables can name them."""
    rng = np.random.RandomState(RNG_SEED)
    records, dropped = [], []
    for beta in BETAS:
        for idx in IDXS_BY_BETA[beta]:
            print(f"[beta={beta} idx={idx}] loading MP + multichain HMC + VI...")
            loaded = {k: _load(k, beta, idx) for k in NAMES}
            missing = [NAMES[k] for k, v in loaded.items() if v is None]
            if missing:
                reason = f"cache missing ({', '.join(missing)}; run not finished or being re-run)"
                dropped.append(dict(beta=beta, idx=idx, reason=reason, admitted_fraction=None))
                print(f"  DROPPED {_run_label(beta, idx)}: {reason}")
                continue
            hmc = loaded["hmc"]
            ok = (hmc["rhat"] < RHAT_MAX) & (hmc["ess"] > ESS_MIN)
            n_ok = int(ok.sum())
            admitted = float(ok.mean())
            take = (rng.choice(np.flatnonzero(ok), size=min(N_POINTS_PER_RUN, n_ok), replace=False)
                    if n_ok > 0 else np.zeros(0, dtype=int))
            included = n_ok >= MIN_ADMITTED
            if not included:
                reason = (f"HMC gate: {n_ok} < {MIN_ADMITTED} admitted coordinates "
                          f"({admitted:.1%} admitted, chain acceptance {_accept_str(hmc)})")
                dropped.append(dict(beta=beta, idx=idx, reason=reason, admitted_fraction=admitted))
                print(f"  DROPPED {_run_label(beta, idx)}: {reason}")
            records.append(dict(beta=beta, idx=idx, mp=loaded["mp"], hmc=hmc, vi=loaded["vi"],
                                ok=ok, take=take, admitted_fraction=admitted,
                                included=included, partial=included and admitted < 1.0))
    return records, dropped


def dropped_lines(dropped: list[dict]) -> list[str]:
    """Human-readable list of dropped runs (one string per run), 'none' when empty."""
    return [f"{_run_label(d['beta'], d['idx'])} ({d['reason']})" for d in dropped] or ["none"]


def partial_lines(records: list[dict]) -> list[str]:
    """Runs that enter the statistics with only part of their HMC coordinates admitted."""
    return [
        f"{_run_label(r['beta'], r['idx'])} ({r['admitted_fraction']:.1%} admitted, "
        f"chain acceptance {_accept_str(r['hmc'])})"
        for r in records if r["partial"]
    ] or ["none"]


def mean_axis_limit(records) -> float:
    """Half-width of the symmetric symlog mean axes shared by all three figures: MEAN_LIM_PAD
    times the largest |posterior mean| over the plotted (subsampled) coordinates of all three
    methods, so no point is clipped in any figure and the mean panels stay comparable."""
    largest = max(
        float(np.abs(r[k]["w_eigen_mean"][r["take"]]).max())
        for r in records if r["included"] and r["take"].size > 0 for k in NAMES
    )
    return MEAN_LIM_PAD * largest


def pooled_scatter(ax, records, x_method: str, y_method: str, field: str, title: str,
                   beta_colors, mean_lim: float) -> None:
    """Scatter of `y_method` against `x_method` for `field` in {"mean", "std"}, one point per
    subsampled admitted coordinate of every run that passes the run gate (the same runs as in
    the tables), coloured by wave speed. `mean_lim` is the
    shared half-width of the symlog mean axes (see `mean_axis_limit`)."""
    for r in records:
        take = r["take"]
        if not r["included"] or take.size == 0:   # same run gate as the tables
            continue
        ax.scatter(r[x_method][f"w_eigen_{field}"][take], r[y_method][f"w_eigen_{field}"][take],
                   s=22, alpha=0.45, color=beta_colors[r["beta"]], edgecolors="none")

    if field == "mean":
        ax.set_xscale("symlog", linthresh=MEAN_LINTHRESH)
        ax.set_yscale("symlog", linthresh=MEAN_LINTHRESH)
        lims = (-mean_lim, mean_lim)
    else:
        ax.set_xscale("log")
        ax.set_yscale("log")
        all_stds = np.concatenate([
            np.concatenate([r[k][f"w_eigen_{field}"][r["ok"]] for k in (x_method, y_method)])
            for r in records if r["included"]
        ])
        lims = (0.7 * all_stds.min(), 1.5 * all_stds.max())
    ax.plot(lims, lims, "k--", linewidth=1.8, alpha=0.6, zorder=0, label=r"$y=x$")
    ax.set_xlim(*lims)
    ax.set_ylim(*lims)
    label = r"\ posterior\ mean" if field == "mean" else r"\ posterior\ std"
    ax.set_xlabel(rf"$\mathrm{{{NAMES[x_method]}{label}}}$", fontsize=20)
    ax.set_ylabel(rf"$\mathrm{{{NAMES[y_method]}{label}}}$", fontsize=20)
    ax.set_title(title, fontsize=22)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=14)


def _pair_stats(rs: list[dict], x_method: str, y_method: str) -> dict:
    """Per-trajectory statistics of y against x over admitted coordinates: Pearson
    correlation of the posterior means, median |mu_y - mu_x| / sigma_x, median sigma_y / sigma_x."""
    corr, z, ratio = [], [], []
    for r in rs:
        if not r["included"]:
            continue
        ok = r["ok"]
        m_x, m_y = r[x_method]["w_eigen_mean"][ok], r[y_method]["w_eigen_mean"][ok]
        s_x, s_y = r[x_method]["w_eigen_std"][ok], r[y_method]["w_eigen_std"][ok]
        corr.append(float(np.corrcoef(m_y, m_x)[0, 1]))
        z.append(float(np.median(np.abs(m_y - m_x) / s_x)))
        ratio.append(float(np.median(s_y / s_x)))
    return dict(corr=corr, z=z, ratio=ratio)


def summarize(records, dropped) -> dict:
    """Per-beta agreement statistics over CONVERGED (admitted) HMC coordinates only, for the
    three method pairs: the Ours-vs-HMC statistics, the Ours-vs-VI std ratio, the admitted
    fraction and the HMC acceptance rates, followed by the planned/loaded trajectory counts, the
    Ours-vs-VI mean statistics, the VI-vs-HMC statistics and the dropped/partial run lists."""
    summary = {"rhat_max": RHAT_MAX, "ess_min": ESS_MIN, "min_admitted_coordinates": MIN_ADMITTED,
               "idxs_by_beta": {str(b): ids for b, ids in IDXS_BY_BETA.items()},
               "n_points_per_run": N_POINTS_PER_RUN, "rng_seed": RNG_SEED,
               "dropped_runs": [dict(beta=d["beta"], idx=d["idx"], reason=d["reason"]) for d in dropped],
               "partial_runs": [dict(beta=r["beta"], idx=r["idx"], admitted_fraction=r["admitted_fraction"],
                                     accept_rates=[float(a) for a in r["hmc"]["accept_rates"]])
                                for r in records if r["partial"]],
               "per_beta": {}}
    for beta in BETAS:
        rs = [r for r in records if r["beta"] == beta]
        admitted = np.concatenate([r["ok"] for r in rs]) if rs else np.zeros(0, dtype=bool)
        oh = _pair_stats(rs, "hmc", "mp")   # Ours vs HMC
        ov = _pair_stats(rs, "vi", "mp")    # Ours vs VI
        vh = _pair_stats(rs, "hmc", "vi")   # VI vs HMC
        rng_or_nan = lambda v: [min(v), max(v)] if v else [float("nan"), float("nan")]
        summary["per_beta"][str(beta)] = dict(
            # --- Ours vs HMC, Ours vs VI std ratio, admitted fraction, HMC acceptance ---
            admitted_fraction=float(admitted.mean()) if admitted.size else float("nan"),
            mean_corr_range=rng_or_nan(oh["corr"]),
            median_std_ratio_ours_over_hmc=oh["ratio"],
            median_std_ratio_ours_over_vi=ov["ratio"],
            median_abs_mean_diff_in_hmc_sigmas=oh["z"],
            accept_rates=[[float(a) for a in r["hmc"]["accept_rates"]] for r in rs],
            # --- trajectory counts, dropped runs, per-trajectory lists of the other two pairs ---
            n_trajectories=len(oh["corr"]),
            n_trajectories_planned=len(IDXS_BY_BETA[beta]),
            n_trajectories_loaded=len(rs),
            dropped_runs=[_run_label(d["beta"], d["idx"]) for d in dropped if d["beta"] == beta],
            mean_corr_ours_vs_hmc=oh["corr"],
            mean_corr_ours_vs_vi=ov["corr"],
            mean_corr_ours_vs_vi_range=rng_or_nan(ov["corr"]),
            median_abs_mean_diff_ours_vi_in_vi_sigmas=ov["z"],
            mean_corr_vi_vs_hmc=vh["corr"],
            mean_corr_vi_vs_hmc_range=rng_or_nan(vh["corr"]),
            median_abs_mean_diff_vi_hmc_in_hmc_sigmas=vh["z"],
            median_std_ratio_vi_over_hmc=vh["ratio"],
        )
    return summary


def _fmt(v: float) -> str:
    """Table number: two decimals when that is readable, otherwise two significant digits in
    scientific notation (VI at b = 7 produces ratios far outside [0.1, 1000))."""
    if not np.isfinite(v):
        return "--"
    if 0.1 <= abs(v) < 1000:
        return f"{v:.2f}"
    mant, exp = f"{v:.1e}".split("e")
    return rf"${mant}\times10^{{{int(exp)}}}$"


def _n_per_beta_str(summary: dict) -> str:
    return ", ".join(f"{float(b):g}: {s['n_trajectories']}/{s['n_trajectories_planned']}"
                     for b, s in summary["per_beta"].items())


def _table_header(stem: str, x_method: str, y_method: str, with_admitted: bool,
                  summary: dict) -> list[str]:
    """Comment lines explaining every number of one companion table; each technical term is
    explained in brackets on its first use. The trajectory counts and the dropped/partial run
    lists come from the summary (computed from the caches), not from hard-coded prose."""
    X, Y = NAMES[x_method], NAMES[y_method]
    # Methods in the order they are named below; HMC always appears (it defines admission).
    methods = list(dict.fromkeys([y_method, x_method, "hmc"]))
    if x_method == "hmc":
        unit = r"%   (mean mismatch measured in stds of the HMC reference posterior)."
    else:
        unit = (rf"%   (mean mismatch measured in stds of the {X} posterior; {X} is a baseline, "
                r"not a reference posterior).")
    planned = ", ".join(f"{b:g}: {len(ids)}" for b, ids in IDXS_BY_BETA.items())
    dropped = [f"{_run_label(d['beta'], d['idx'])} ({d['reason']})" for d in summary["dropped_runs"]] or ["none"]
    partial = [f"{_run_label(p['beta'], p['idx'])} ({p['admitted_fraction']:.1%} admitted, chain acceptance "
               + "/".join(f"{a:.2f}" for a in p["accept_rates"]) + ")"
               for p in summary["partial_runs"]] or ["none"]
    lines = [
        rf"% Auto-generated by figures_onepass/advection/posterior_recovery_comparison.py ({stem}).",
        rf"% Weight-space posterior of {Y} (y) against {X} (x), per eigenbasis coordinate (one of the",
        r"%   rotated weight coordinates all three methods share).",
        rf"% Trajectories per wave speed b (canonical seeded set): {planned}.",
        r"%   (the same seeded draw at every b; one b = 7 run costs about an hour of HMC and two hours of VI).",
        r"% Methods:",
        *[rf"%   {GLOSSARY[k]};" for k in methods],
        rf"% Only HMC-converged coordinates enter (split-R-hat < {RHAT_MAX} and bulk ESS > {ESS_MIN:g}; these",
        r"%   two chain convergence diagnostics certify that the HMC chains converged at that coordinate),",
        r"%   for every pair, so all three posterior_recovery tables describe the same coordinate set.",
        rf"% Run gate: a run (one (b, idx)) enters only with >= {MIN_ADMITTED} admitted coordinates.",
        r"% Dropped by the HMC gate or a missing cache (computed at run time):",
        *[rf"%   {d}" for d in dropped],
        r"% Runs entering with only part of their HMC coordinates admitted:",
        *[rf"%   {p}" for p in partial],
        rf"% corr: min to max over trajectories of the Pearson correlation (the linear correlation",
        rf"%   coefficient) between the posterior means of {Y} and {X}.",
        rf"% |dmu|/sigma_{X}: median over trajectories of the per-trajectory median of |mu_{Y} - mu_{X}| / sigma_{X}",
        unit,
        rf"% sigma_{Y}/sigma_{X}: median over trajectories of the per-trajectory median std ratio (1 = same width).",
    ]
    if with_admitted:
        lines.append(r"% admitted: fraction of all coordinates (all loaded trajectories of that b) passing the HMC convergence test.")
    lines.append(rf"% n: trajectories entering the statistics / trajectories planned for that b ({_n_per_beta_str(summary)}).")
    if x_method == "vi" or y_method == "vi":
        lines.append(r"% VI at b = 7: the VI predictive mean collapses (rel-L2 about 1), so its posterior means")
        lines.append(r"%   are near zero for every coordinate; the b = 7 VI numbers describe that collapse, not a fit.")
    return lines


def write_table(summary: dict, x_method: str, y_method: str, path: Path) -> None:
    """LaTeX companion table for one pair, from the same per-beta summary the JSON holds."""
    X, Y = NAMES[x_method], NAMES[y_method]
    key = (x_method, y_method)
    corr_key = {("hmc", "mp"): "mean_corr_range", ("vi", "mp"): "mean_corr_ours_vs_vi_range",
                ("hmc", "vi"): "mean_corr_vi_vs_hmc_range"}[key]
    z_key = {("hmc", "mp"): "median_abs_mean_diff_in_hmc_sigmas",
             ("vi", "mp"): "median_abs_mean_diff_ours_vi_in_vi_sigmas",
             ("hmc", "vi"): "median_abs_mean_diff_vi_hmc_in_hmc_sigmas"}[key]
    ratio_key = {("hmc", "mp"): "median_std_ratio_ours_over_hmc",
                 ("vi", "mp"): "median_std_ratio_ours_over_vi",
                 ("hmc", "vi"): "median_std_ratio_vi_over_hmc"}[key]
    with_admitted = x_method == "hmc"

    cols = "r c r r" + (" r" if with_admitted else "") + " r"
    head = (rf"$b$ & $\mathrm{{corr}}(\mu_{{\mathrm{{{Y}}}}}, \mu_{{\mathrm{{{X}}}}})$ & "
            rf"$|\Delta\mu|/\sigma_{{\mathrm{{{X}}}}}$ & $\sigma_{{\mathrm{{{Y}}}}}/\sigma_{{\mathrm{{{X}}}}}$ & "
            + (r"admitted & " if with_admitted else "") + r"$n$ \\")
    lines = _table_header(path.stem, x_method, y_method, with_admitted, summary)
    lines += [rf"\begin{{tabular}}{{{cols}}}", r"\toprule", head, r"\midrule"]
    for beta, s in summary["per_beta"].items():
        n_str = f"{s['n_trajectories']}/{s['n_trajectories_planned']}"
        if s["n_trajectories"] == 0:
            lines.append(f"{float(beta):g} & -- & -- & -- & " + ("-- & " if with_admitted else "") + f"{n_str} \\\\")
            continue
        lo, hi = s[corr_key]
        # math-mode numbers give a proper minus sign (VI at b = 7 has negative correlations);
        # the spaced en dash keeps "-0.041 -- 0.092" from reading as a subtraction.
        row = (f"{float(beta):g} & ${lo:.3f}$ -- ${hi:.3f}$ & "
               f"{_fmt(float(np.median(s[z_key])))} & {_fmt(float(np.median(s[ratio_key])))} & ")
        if with_admitted:
            row += f"{100 * s['admitted_fraction']:.1f}\\% & "
        row += f"{n_str} \\\\"
        lines.append(row)
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"Saved table to {path}")


def make_figure(records, x_method: str, y_method: str, output_path: Path,
                mean_lim: float) -> None:
    plt.rcParams["mathtext.fontset"] = "cm"
    cmap = plt.get_cmap("viridis")
    beta_colors = {b: cmap(i / max(1, len(BETAS) - 1)) for i, b in enumerate(BETAS)}
    X, Y = NAMES[x_method], NAMES[y_method]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.0))
    pooled_scatter(axes[0], records, x_method, y_method, "mean",
                   rf"$\mathrm{{(a)\ Posterior\ means:\ {Y}\ vs.\ {X}}}$", beta_colors, mean_lim)
    pooled_scatter(axes[1], records, x_method, y_method, "std",
                   rf"$\mathrm{{(b)\ Posterior\ stds:\ {Y}\ vs.\ {X}}}$", beta_colors, mean_lim)

    handles = [plt.Line2D([], [], marker="o", linestyle="none", color=beta_colors[b],
                          markersize=10, label=rf"$b={b:g}$") for b in BETAS]
    handles.append(plt.Line2D([], [], linestyle="--", color="k", label=r"$y=x$"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=17,
               bbox_to_anchor=(0.5, -0.06), title=r"$\mathrm{wave\ speed}$", title_fontsize=17)
    fig.tight_layout(rect=(0, 0.11, 1, 1))   # the legend band below must clear the x-axis labels
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path}")


def main() -> None:
    records, dropped = collect_results()
    mean_lim = mean_axis_limit(records)
    print(f"symlog mean axes in all three figures: [-{mean_lim:.3f}, {mean_lim:.3f}] "
          f"({MEAN_LIM_PAD} x largest plotted |mean| = {mean_lim / MEAN_LIM_PAD:.3f})")
    for (x_method, y_method), stem in PAIRS.items():
        make_figure(records, x_method, y_method, FIGURES_DIR / f"{stem}.pdf", mean_lim)
    summary = summarize(records, dropped)
    OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2))
    print(f"Saved summary to {OUTPUT_SUMMARY}")
    for (x_method, y_method), stem in PAIRS.items():
        write_table(summary, x_method, y_method, FIGURES_DIR / f"{stem}_table.tex")

    print("trajectories per wave speed (entering / planned): " + _n_per_beta_str(summary))
    print("dropped runs (HMC gate or missing cache):")
    for line in dropped_lines(dropped):
        print(f"  {line}")
    print("runs entering with only part of their HMC coordinates admitted:")
    for line in partial_lines(records):
        print(f"  {line}")
    for beta, s in summary["per_beta"].items():
        print(f"beta={beta}: admitted {s['admitted_fraction']:.1%} of coordinates, "
              f"n={s['n_trajectories']}/{s['n_trajectories_planned']}")
        if s["n_trajectories"] == 0:
            continue
        print(f"  Ours vs HMC: mean-corr {s['mean_corr_range'][0]:.3f}..{s['mean_corr_range'][1]:.3f}, "
              f"|dmu|/sigma_HMC {np.median(s['median_abs_mean_diff_in_hmc_sigmas']):.3g}, "
              f"sigma_Ours/sigma_HMC {np.median(s['median_std_ratio_ours_over_hmc']):.3g}")
        print(f"  Ours vs VI:  mean-corr {s['mean_corr_ours_vs_vi_range'][0]:.3f}.."
              f"{s['mean_corr_ours_vs_vi_range'][1]:.3f}, "
              f"|dmu|/sigma_VI {np.median(s['median_abs_mean_diff_ours_vi_in_vi_sigmas']):.3g}, "
              f"sigma_Ours/sigma_VI {np.median(s['median_std_ratio_ours_over_vi']):.3g}")
        print(f"  VI vs HMC:   mean-corr {s['mean_corr_vi_vs_hmc_range'][0]:.3f}.."
              f"{s['mean_corr_vi_vs_hmc_range'][1]:.3f}, "
              f"|dmu|/sigma_HMC {np.median(s['median_abs_mean_diff_vi_hmc_in_hmc_sigmas']):.3g}, "
              f"sigma_VI/sigma_HMC {np.median(s['median_std_ratio_vi_over_hmc']):.3g}")


if __name__ == "__main__":
    main()
