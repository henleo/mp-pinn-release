"""
Weight-space posterior recovery on the Fisher-KPP (reaction-diffusion) dataset: pairwise
scatter comparisons of the three inference methods, aggregated over the canonical seeded
trajectory set on the 16-cell grid nu in {0.5, 1, 2, 5} x rho in {1, 2, 5, 10} (nu = diffusion
coefficient, rho = reaction strength; the model sees the normalized values rho_norm = 2 rho and
nu_norm = 2 nu). Every cell uses the same FIVE trajectories, `trajectory_selection.idxs(5)`
(the seeded draw shared with every other multi-trajectory figure; the development trajectory
is excluded), so a cell holds 5 runs and the whole grid 80. The number of runs per cell is
stated in the log, the summary JSON and every companion table.

  * Ours: the single-pass (one-pass) EP (expectation propagation) posterior of MP-PINN with the
    rho-continuation ladder (the reaction strength is raised in steps during the pass). For
    every eigenbasis coordinate (one of the rotated weight coordinates all three methods
    share) the runner stores the mean and the std of that coordinate's marginal belief.
  * HMC (Hamiltonian Monte Carlo): the asymptotically exact reference. Multiple chains with a
    dense Gauss-Newton mass matrix (a curvature-based preconditioner), MAP (maximum a posteriori,
    the posterior mode) warm start through the same ladder and warm-up step adaptation (`bpinn_run_reacdiff_multichain.jl`). A
    coordinate is admitted into every comparison only when split-R-hat < 1.05 and bulk ESS
    (effective sample size) > 100; these two chain convergence diagnostics certify that the
    chains actually converged there (the admitted fraction is printed and saved alongside the
    figures). A run (one (nu, rho, idx)) enters the statistics only with at least MIN_ADMITTED
    admitted coordinates; which runs are dropped by this gate, and which runs enter with only
    a partially admitted coordinate set, is computed from the artifacts at run time and written
    to the log, the summary JSON and the table comments (never hard-coded here).
  * VI (variational inference): mean-field (one independent Gaussian per coordinate) Gaussian
    variational inference with the same ladder, Algorithm 2 of Yang et al. 2021. Where VI's
    posterior means collapse the VI panels show the collapse rather than a fit; the VI points
    are shown nevertheless (the axes are wide enough), and the tables print the resulting
    extreme numbers instead of hiding them.

All three methods are built on the identical problem, graph and eigenbasis V;
`assert_same_basis` is called on every (Ours, HMC, VI) triple before it is compared (the
artifacts' `v_fingerprint`), so a coordinate index means the same weight direction in all
three posteriors.

Three separate PDFs, one per pair of methods, each a 1x2 figure (left: posterior means,
right: posterior stds; y=x dashed in both):

  posterior_recovery_ours_vs_hmc.pdf   x = HMC, y = Ours
  posterior_recovery_ours_vs_vi.pdf    x = VI,  y = Ours
  posterior_recovery_vi_vs_hmc.pdf     x = HMC, y = VI

Each point is one eigenbasis coordinate of one trajectory's posterior, pooling all 16 cells
and all trajectories. As in every figure of this set the reaction strength rho plays the role
of the advection wave speed: the point COLOUR is rho (4 viridis colours) and the diffusion
coefficient nu, pooled into each rho, is shown by the MARKER shape (nu = 0.5 circle, 1 square,
2 diamond, 5 triangle); both are in the legend. A random coordinate subsample per run (drawn
ONCE per (nu, rho, idx) among the admitted HMC coordinates with a run-specific seed, reused in
all three figures and stored in the summary JSON, so the same coordinates appear everywhere and
can be traced) keeps ~2000 points per panel legible. Mean panels use symlog axes (linear near
zero, logarithmic beyond; the linear region is |mu| < 1e-2) with symmetric, data-driven limits:
1.15 times the largest |mean| over the plotted points of all three figures, and the same limits
in every figure so the panels stay comparable and no point is clipped (the value is printed in
the log). Std panels are log-log (both axes logarithmic), so a constant multiplicative std
bias appears as a parallel offset from y=x.

A machine-readable summary (per-cell and per-rho (pooled over nu) Pearson correlations (the
linear correlation coefficient) of the posterior means, median std ratios, median mean
mismatch in units of the x-method's posterior std, admitted fractions, n per cell, dropped
runs, the coordinate subsample of every run, for all three pairs) and one LaTeX companion
table per figure derived from that same summary are written next to the PDFs. Every table
lists all 16 cells explicitly, one pooled row per rho (pooling the four nu) and one row
pooling the whole grid, with n per row. All statistics use only admitted HMC coordinates, in
every pair, so the three tables describe the same coordinate set.

No training is triggered: the script only reads artifacts. In the default run (all 16 cells) a
missing artifact raises an error naming the file (nothing is skipped silently). The environment
variable REACDIFF_CELLS (comma-separated "nu:rho" pairs, e.g. REACDIFF_CELLS=0.5:1.0,0.5:2.0)
restricts the script to those cells; in
that override mode a run whose artifact is missing is dropped with a printed reason and listed
in the tables (never fabricated), and the cells used are printed and written to the log, the
JSON and the table comments. The log is written by the script itself
(posterior_recovery_comparison.log next to the outputs).

Run: python figures_onepass/diffusion/posterior_recovery_comparison.py [--png DIR]
  --png DIR additionally renders every figure as PNG into DIR (for inspection).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURES_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "pde" / "diffusion"))
sys.path.insert(0, str(REPO_ROOT / "pde" / "advection"))

from reacdiff_baselines_sweep import NUS, RHOS, assert_same_basis, cache_path, load_cached  # noqa: E402
from trajectory_selection import idxs  # noqa: E402

OUTPUT_SUMMARY = FIGURES_DIR / "posterior_recovery_comparison_summary.json"
OUTPUT_LOG = FIGURES_DIR / "posterior_recovery_comparison.log"

# (x-method, y-method) -> output stem; the method keys index the per-run records below.
PAIRS = {
    ("hmc", "mp"): "posterior_recovery_ours_vs_hmc",
    ("vi", "mp"): "posterior_recovery_ours_vs_vi",
    ("hmc", "vi"): "posterior_recovery_vi_vs_hmc",
}
NAMES = {"mp": "Ours", "hmc": "HMC", "vi": "VI"}
LOADER_METHOD = {"mp": "Ours", "hmc": "HMC", "vi": "VI"}   # keys of reacdiff_baselines_sweep
# One-line explanation of each method for the table comments (first use of every term).
GLOSSARY = {
    "mp": ("Ours = the single-pass EP (expectation propagation) posterior of MP-PINN with the rho-continuation "
           "ladder (the reaction strength is raised in steps during the pass)"),
    "hmc": ("HMC = Hamiltonian Monte Carlo, the asymptotically exact reference (same model; warm-started at the "
            "MAP (maximum a posteriori, the posterior mode) point found through the same ladder)"),
    "vi": "VI = mean-field (one independent Gaussian per coordinate) variational inference, same ladder",
}

# The grid: rho plays the role of the advection wave speed (colour), nu is pooled into
# each rho and shown by the marker shape.
IDXS = idxs(5)                       # canonical trajectories, the same five in every cell
NU_MARKERS = {0.5: "o", 1.0: "s", 2.0: "D", 5.0: "^"}
N_POINTS_PER_RUN = 25   # random eigencoordinate subsample per (nu, rho, idx), for legibility
RHAT_MAX = 1.05
ESS_MIN = 100.0
RNG_SEED = 0
MIN_ADMITTED = 3        # a run enters the statistics only with >= this many coordinates
MEAN_LIM_PAD = 1.15     # symlog mean axes: symmetric limits = this factor x the largest plotted |mean|
MEAN_LINTHRESH = 1e-2   # symlog mean axes: |mu| below this is drawn on a linear scale


class _Tee:
    """Duplicates stdout into the log file so the script leaves its own .log next to the outputs."""

    def __init__(self, path: Path):
        self._file = path.open("w")
        self._stdout = sys.stdout

    def write(self, s: str) -> None:
        self._stdout.write(s)
        self._file.write(s)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def all_cells() -> list[tuple[float, float]]:
    """The 16 (nu, rho) cells, rho outer (the colour), nu inner (the marker)."""
    return [(float(nu), float(rho)) for rho in RHOS for nu in NUS]


def cells_from_env() -> tuple[list[tuple[float, float]], bool]:
    """(cells, override): all 16 cells by default; REACDIFF_CELLS="nu:rho,nu:rho" selects a subset
    (override mode, see the docstring). Unknown cells raise."""
    raw = os.environ.get("REACDIFF_CELLS", "").strip()
    if not raw:
        return all_cells(), False
    cells = []
    for item in raw.split(","):
        parts = item.strip().split(":")
        if len(parts) != 2:
            raise ValueError(f"REACDIFF_CELLS entry {item!r} is not of the form nu:rho (e.g. 0.5:1.0)")
        try:
            cell = (float(parts[0]), float(parts[1]))
        except ValueError:
            raise ValueError(f"REACDIFF_CELLS entry {item!r} is not of the form nu:rho (e.g. 0.5:1.0)") from None
        if cell not in all_cells():
            raise ValueError(f"REACDIFF_CELLS entry {item!r} is not a grid cell; nu in {NUS}, rho in {RHOS}")
        if cell not in cells:
            cells.append(cell)
    cells.sort(key=lambda c: (RHOS.index(c[1]), NUS.index(c[0])))
    return cells, True


def _run_label(nu: float, rho: float, idx: int) -> str:
    return f"nu={nu:g} rho={rho:g} idx {idx}"


def _cell_key(nu: float, rho: float) -> str:
    return f"nu={nu:g},rho={rho:g}"


def _accept_str(hmc: dict) -> str:
    return "/".join(f"{float(a):.2f}" for a in hmc["accept_rates"])


def _load(method: str, nu: float, rho: float, idx: int, override: bool) -> dict | None:
    """Load a cached posterior. A missing artifact raises in the default run (naming the file) and
    returns None in override mode (the run is then dropped with a printed reason); a figure
    script never trains."""
    path = cache_path(LOADER_METHOD[method], nu, rho, idx)
    if not path.exists():
        msg = f"missing {NAMES[method]} artifact for {_run_label(nu, rho, idx)}: {path}"
        if not override:
            raise FileNotFoundError(msg + " (the figure scripts never train; generate it with the "
                                    f"runner listed in pde/diffusion/reacdiff_baselines_sweep.py)")
        print(f"  MISSING {msg}")
        return None
    return load_cached(LOADER_METHOD[method], nu, rho, idx)


def collect_results(cells: list[tuple[float, float]], override: bool) -> tuple[list[dict], list[dict]]:
    """Returns (records, dropped). One record per loaded (nu, rho, idx): the three posteriors
    (basis-checked against each other), the HMC convergence mask, the coordinate subsample used
    by all three figures (drawn once here with a run-specific seed, so it does not depend on
    which other cells are present) and whether the run passes the run gate (>= MIN_ADMITTED
    admitted coordinates). `dropped` lists every planned run that does not enter the statistics,
    with its reason: a missing artifact (override mode only; the default run raises), or the
    HMC gate (with admitted fraction and chain acceptance rates). Runs that enter with a
    partially admitted coordinate set are flagged in the record (`partial`) so the log and
    tables can name them."""
    records, dropped = [], []
    for nu, rho in cells:
        for idx in IDXS:
            print(f"[{_run_label(nu, rho, idx)}] loading Ours + multichain HMC + VI...")
            loaded = {k: _load(k, nu, rho, idx, override) for k in NAMES}
            missing = [NAMES[k] for k, v in loaded.items() if v is None]
            if missing:
                reason = f"artifact missing ({', '.join(missing)}; override mode)"
                dropped.append(dict(nu=nu, rho=rho, idx=idx, reason=reason, admitted_fraction=None))
                print(f"  DROPPED {_run_label(nu, rho, idx)}: {reason}")
                continue
            assert_same_basis(loaded["mp"], loaded["hmc"], loaded["vi"])   # same V in all three
            hmc = loaded["hmc"]
            ok = (hmc["rhat"] < RHAT_MAX) & (hmc["ess"] > ESS_MIN)
            n_ok = int(ok.sum())
            admitted = float(ok.mean())
            rng = np.random.RandomState([RNG_SEED, int(round(10 * nu)), int(round(10 * rho)), idx])
            take = (rng.choice(np.flatnonzero(ok), size=min(N_POINTS_PER_RUN, n_ok), replace=False)
                    if n_ok > 0 else np.zeros(0, dtype=int))
            take = np.sort(take)
            included = n_ok >= MIN_ADMITTED
            if not included:
                reason = (f"HMC gate: {n_ok} < {MIN_ADMITTED} admitted coordinates "
                          f"({admitted:.1%} admitted, chain acceptance {_accept_str(hmc)})")
                dropped.append(dict(nu=nu, rho=rho, idx=idx, reason=reason, admitted_fraction=admitted))
                print(f"  DROPPED {_run_label(nu, rho, idx)}: {reason}")
            records.append(dict(nu=nu, rho=rho, idx=idx, mp=loaded["mp"], hmc=hmc, vi=loaded["vi"],
                                ok=ok, take=take, admitted_fraction=admitted,
                                included=included, partial=included and admitted < 1.0))
    return records, dropped


def dropped_lines(dropped: list[dict]) -> list[str]:
    """Human-readable list of dropped runs (one string per run), 'none' when empty."""
    return [f"{_run_label(d['nu'], d['rho'], d['idx'])} ({d['reason']})" for d in dropped] or ["none"]


def partial_lines(records: list[dict]) -> list[str]:
    """Runs that enter the statistics with only part of their HMC coordinates admitted."""
    return [
        f"{_run_label(r['nu'], r['rho'], r['idx'])} ({r['admitted_fraction']:.1%} admitted, "
        f"chain acceptance {_accept_str(r['hmc'])})"
        for r in records if r["partial"]
    ] or ["none"]


def mean_axis_limit(records) -> float:
    """Half-width of the symmetric symlog mean axes shared by all three figures: MEAN_LIM_PAD
    times the largest |posterior mean| over the plotted (subsampled) coordinates of all three
    methods, so no point is clipped in any figure and the mean panels stay comparable."""
    plotted = [r for r in records if r["included"] and r["take"].size > 0]
    if not plotted:
        raise RuntimeError("no run passes the HMC gate; nothing to plot")
    largest = max(float(np.abs(r[k]["w_eigen_mean"][r["take"]]).max()) for r in plotted for k in NAMES)
    return MEAN_LIM_PAD * largest


def pooled_scatter(ax, records, x_method: str, y_method: str, field: str, title: str,
                   rho_colors, mean_lim: float) -> None:
    """Scatter of `y_method` against `x_method` for `field` in {"mean", "std"}, one point per
    subsampled admitted coordinate of every run that passes the run gate (the same runs as in
    the tables), coloured by reaction strength rho, marker shape by diffusion coefficient nu.
    `mean_lim` is the shared half-width of the symlog mean axes (see `mean_axis_limit`)."""
    for r in records:
        take = r["take"]
        if not r["included"] or take.size == 0:   # same run gate as the tables
            continue
        ax.scatter(r[x_method][f"w_eigen_{field}"][take], r[y_method][f"w_eigen_{field}"][take],
                   s=24, alpha=0.45, color=rho_colors[r["rho"]], marker=NU_MARKERS[r["nu"]],
                   edgecolors="none")

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
    """Per-run statistics of y against x over admitted coordinates: Pearson correlation of the
    posterior means, median |mu_y - mu_x| / sigma_x, median sigma_y / sigma_x."""
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


def _group_summary(rs: list[dict], dropped: list[dict], n_planned: int, pools: str) -> dict:
    """Agreement statistics of one group of runs (a cell, one rho pooled over nu, or the whole
    grid) over CONVERGED (admitted) HMC coordinates only, for the three method pairs. The keys
    follow the advection summary (per_beta) so downstream readers stay the same."""
    admitted = np.concatenate([r["ok"] for r in rs]) if rs else np.zeros(0, dtype=bool)
    oh = _pair_stats(rs, "hmc", "mp")   # Ours vs HMC
    ov = _pair_stats(rs, "vi", "mp")    # Ours vs VI
    vh = _pair_stats(rs, "hmc", "vi")   # VI vs HMC
    rng_or_nan = lambda v: [min(v), max(v)] if v else [float("nan"), float("nan")]
    return dict(
        pools=pools,
        admitted_fraction=float(admitted.mean()) if admitted.size else float("nan"),
        mean_corr_range=rng_or_nan(oh["corr"]),
        median_std_ratio_ours_over_hmc=oh["ratio"],
        median_std_ratio_ours_over_vi=ov["ratio"],
        median_abs_mean_diff_in_hmc_sigmas=oh["z"],
        accept_rates=[[float(a) for a in r["hmc"]["accept_rates"]] for r in rs],
        n_trajectories=len(oh["corr"]),
        n_trajectories_planned=n_planned,
        n_trajectories_loaded=len(rs),
        dropped_runs=[_run_label(d["nu"], d["rho"], d["idx"]) for d in dropped],
        mean_corr_ours_vs_hmc=oh["corr"],
        mean_corr_ours_vs_vi=ov["corr"],
        mean_corr_ours_vs_vi_range=rng_or_nan(ov["corr"]),
        median_abs_mean_diff_ours_vi_in_vi_sigmas=ov["z"],
        mean_corr_vi_vs_hmc=vh["corr"],
        mean_corr_vi_vs_hmc_range=rng_or_nan(vh["corr"]),
        median_abs_mean_diff_vi_hmc_in_hmc_sigmas=vh["z"],
        median_std_ratio_vi_over_hmc=vh["ratio"],
    )


def summarize(records, dropped, cells: list[tuple[float, float]], override: bool) -> dict:
    """Per-cell (all 16 (nu, rho)), per-rho (pooled over nu) and whole-grid agreement statistics
    over admitted HMC coordinates, plus the run gate bookkeeping and the coordinate subsample of
    every run (so the plotted coordinates can be traced across the three figures)."""
    summary = {"rhat_max": RHAT_MAX, "ess_min": ESS_MIN, "min_admitted_coordinates": MIN_ADMITTED,
               "nus": [float(v) for v in NUS], "rhos": [float(v) for v in RHOS], "idxs": IDXS,
               "cells_used": [_cell_key(nu, rho) for nu, rho in cells], "override_mode": override,
               "n_points_per_run": N_POINTS_PER_RUN, "rng_seed": RNG_SEED,
               "dropped_runs": [dict(nu=d["nu"], rho=d["rho"], idx=d["idx"], reason=d["reason"]) for d in dropped],
               "partial_runs": [dict(nu=r["nu"], rho=r["rho"], idx=r["idx"], admitted_fraction=r["admitted_fraction"],
                                     accept_rates=[float(a) for a in r["hmc"]["accept_rates"]])
                                for r in records if r["partial"]],
               "subsample_by_run": {_run_label(r["nu"], r["rho"], r["idx"]): [int(i) for i in r["take"]]
                                    for r in records if r["included"]},
               "per_cell": {}, "per_rho": {}}
    for nu, rho in cells:
        rs = [r for r in records if r["nu"] == nu and r["rho"] == rho]
        dr = [d for d in dropped if d["nu"] == nu and d["rho"] == rho]
        summary["per_cell"][_cell_key(nu, rho)] = _group_summary(rs, dr, len(IDXS), "one cell") | dict(nu=nu, rho=rho)
    for rho in RHOS:
        rho = float(rho)
        cells_rho = [c for c in cells if c[1] == rho]
        rs = [r for r in records if r["rho"] == rho]
        dr = [d for d in dropped if d["rho"] == rho]
        summary["per_rho"][f"{rho:g}"] = _group_summary(
            rs, dr, len(IDXS) * len(cells_rho), f"the {len(cells_rho)} nu cells at rho={rho:g}") | dict(rho=rho)
    summary["all_cells"] = _group_summary(records, dropped, len(IDXS) * len(cells), f"all {len(cells)} cells")
    return summary


def _fmt(v: float) -> str:
    """Table number: two decimals when that is readable, otherwise two significant digits in
    scientific notation (a collapsed VI posterior produces ratios far outside [0.1, 1000))."""
    if not np.isfinite(v):
        return "--"
    if 0.1 <= abs(v) < 1000:
        return f"{v:.2f}"
    mant, exp = f"{v:.1e}".split("e")
    return rf"${mant}\times10^{{{int(exp)}}}$"


def _n_per_cell_str(summary: dict) -> str:
    return ", ".join(f"({k}): {s['n_trajectories']}/{s['n_trajectories_planned']}"
                     for k, s in summary["per_cell"].items())


def _table_header(stem: str, x_method: str, y_method: str, with_admitted: bool,
                  summary: dict) -> list[str]:
    """Comment lines explaining every number of one companion table; each technical term is
    explained in brackets on its first use. The run counts and the dropped/partial run lists
    come from the summary (computed from the artifacts), not from hard-coded prose."""
    X, Y = NAMES[x_method], NAMES[y_method]
    # Methods in the order they are named below; HMC always appears (it defines admission).
    methods = list(dict.fromkeys([y_method, x_method, "hmc"]))
    if x_method == "hmc":
        unit = r"%   (mean mismatch measured in stds of the HMC reference posterior)."
    else:
        unit = (rf"%   (mean mismatch measured in stds of the {X} posterior; {X} is a baseline, "
                r"not a reference posterior).")
    dropped = [f"{_run_label(d['nu'], d['rho'], d['idx'])} ({d['reason']})" for d in summary["dropped_runs"]] or ["none"]
    partial = [f"{_run_label(p['nu'], p['rho'], p['idx'])} ({p['admitted_fraction']:.1%} admitted, chain acceptance "
               + "/".join(f"{a:.2f}" for a in p["accept_rates"]) + ")"
               for p in summary["partial_runs"]] or ["none"]
    nus = ", ".join(f"{v:g}" for v in NUS)
    rhos = ", ".join(f"{v:g}" for v in RHOS)
    lines = [
        rf"% Auto-generated by figures_onepass/diffusion/posterior_recovery_comparison.py ({stem}).",
        rf"% Weight-space posterior of {Y} (y) against {X} (x), per eigenbasis coordinate (one of the",
        r"%   rotated weight coordinates all three methods share; the artifacts' basis fingerprints were",
        r"%   checked to agree for every (Ours, HMC, VI) triple).",
        rf"% Fisher-KPP (reaction-diffusion) dataset grid: nu (diffusion coefficient) in {{{nus}}} x rho (reaction",
        rf"%   strength) in {{{rhos}}}; the model uses the normalized values rho_norm = 2 rho and nu_norm = 2 nu.",
        rf"% Trajectories per cell (canonical seeded set, the same in every cell): {len(IDXS)} = {IDXS}.",
    ]
    if summary["override_mode"]:
        lines.append(rf"% OVERRIDE MODE (REACDIFF_CELLS): only these cells were used: {', '.join(summary['cells_used'])}.")
    lines += [
        r"% Methods:",
        *[rf"%   {GLOSSARY[k]};" for k in methods],
        rf"% Only HMC-converged coordinates enter (split-R-hat < {RHAT_MAX} and bulk ESS > {ESS_MIN:g}; these",
        r"%   two chain convergence diagnostics certify that the HMC chains converged at that coordinate),",
        r"%   for every pair, so all three posterior_recovery tables describe the same coordinate set.",
        rf"% Run gate: a run (one (nu, rho, idx)) enters only with >= {MIN_ADMITTED} admitted coordinates.",
        r"% Dropped by the HMC gate or a missing artifact (computed at run time):",
        *[rf"%   {d}" for d in dropped],
        r"% Runs entering with only part of their HMC coordinates admitted:",
        *[rf"%   {p}" for p in partial],
        rf"% corr: min to max over runs of the Pearson correlation (the linear correlation",
        rf"%   coefficient) between the posterior means of {Y} and {X}.",
        rf"% |dmu|/sigma_{X}: median over runs of the per-run median of |mu_{Y} - mu_{X}| / sigma_{X}",
        unit,
        rf"% sigma_{Y}/sigma_{X}: median over runs of the per-run median std ratio (1 = same width).",
    ]
    if with_admitted:
        lines.append(r"% admitted: fraction of all coordinates (all loaded runs of that row) passing the HMC convergence test.")
    lines.append(r"% n: runs entering the statistics / runs planned for that row"
                 + (" ('--' = cell not requested in override mode)." if summary["override_mode"] else "."))
    lines.append(r"% Rows: one per cell (nu, rho), all 16 listed; 'all nu' rows pool the four nu cells of that rho")
    lines.append(r"%   (the four markers of one colour in the figure); the last row pools the whole grid.")
    lines.append(rf"% n per cell: {_n_per_cell_str(summary)}.")
    if x_method == "vi" or y_method == "vi":
        lines.append(r"% Where VI's posterior means collapse (near zero for every coordinate) the VI numbers describe")
        lines.append(r"%   that collapse, not a fit; they are printed rather than hidden.")
    return lines


def _table_row(first: str, s: dict, corr_key: str, z_key: str, ratio_key: str, with_admitted: bool) -> str:
    n_str = f"{s['n_trajectories']}/{s['n_trajectories_planned']}"
    if s["n_trajectories"] == 0:
        return f"{first} & -- & -- & -- & " + ("-- & " if with_admitted else "") + f"{n_str} \\\\"
    lo, hi = s[corr_key]
    # math-mode numbers give a proper minus sign (a collapsed VI has negative correlations);
    # the spaced en dash keeps "-0.041 -- 0.092" from reading as a subtraction.
    row = (f"{first} & ${lo:.3f}$ -- ${hi:.3f}$ & "
           f"{_fmt(float(np.median(s[z_key])))} & {_fmt(float(np.median(s[ratio_key])))} & ")
    if with_admitted:
        row += f"{100 * s['admitted_fraction']:.1f}\\% & "
    return row + f"{n_str} \\\\"


def write_table(summary: dict, x_method: str, y_method: str, path: Path) -> None:
    """LaTeX companion table for one pair, from the same per-cell / per-rho summary the JSON
    holds: 16 cell rows grouped by rho, one 'all nu' pooled row per rho, one whole-grid row."""
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

    cols = "r r c r r" + (" r" if with_admitted else "") + " r"
    head = (rf"$\rho$ & $\nu$ & $\mathrm{{corr}}(\mu_{{\mathrm{{{Y}}}}}, \mu_{{\mathrm{{{X}}}}})$ & "
            rf"$|\Delta\mu|/\sigma_{{\mathrm{{{X}}}}}$ & $\sigma_{{\mathrm{{{Y}}}}}/\sigma_{{\mathrm{{{X}}}}}$ & "
            + (r"admitted & " if with_admitted else "") + r"$n$ \\")
    lines = _table_header(path.stem, x_method, y_method, with_admitted, summary)
    lines += [rf"\begin{{tabular}}{{{cols}}}", r"\toprule", head, r"\midrule"]
    used = {(s["nu"], s["rho"]) for s in summary["per_cell"].values()}
    for rho in RHOS:
        rho = float(rho)
        for nu in NUS:
            nu = float(nu)
            if (nu, rho) not in used:   # override mode: the cell was not requested
                lines.append(f"{rho:g} & {nu:g} & \\multicolumn{{{3 + int(with_admitted)}}}{{c}}{{not run (REACDIFF\\_CELLS override)}} & -- \\\\")
                continue
            s = summary["per_cell"][_cell_key(nu, rho)]
            lines.append(_table_row(f"{rho:g} & {nu:g}", s, corr_key, z_key, ratio_key, with_admitted))
        s = summary["per_rho"][f"{rho:g}"]
        lines.append(_table_row(rf"{rho:g} & all $\nu$", s, corr_key, z_key, ratio_key, with_admitted))
        lines.append(r"\midrule")
    lines.append(_table_row(r"all & all", summary["all_cells"], corr_key, z_key, ratio_key, with_admitted))
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines) + "\n")
    print(f"Saved table to {path}")


def make_figure(records, x_method: str, y_method: str, output_path: Path,
                mean_lim: float, png_dir: Path | None = None) -> None:
    plt.rcParams["mathtext.fontset"] = "cm"
    cmap = plt.get_cmap("viridis")
    rho_colors = {float(rho): cmap(i / max(1, len(RHOS) - 1)) for i, rho in enumerate(RHOS)}
    X, Y = NAMES[x_method], NAMES[y_method]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.4))
    pooled_scatter(axes[0], records, x_method, y_method, "mean",
                   rf"$\mathrm{{(a)\ Posterior\ means:\ {Y}\ vs.\ {X}}}$", rho_colors, mean_lim)
    pooled_scatter(axes[1], records, x_method, y_method, "std",
                   rf"$\mathrm{{(b)\ Posterior\ stds:\ {Y}\ vs.\ {X}}}$", rho_colors, mean_lim)

    # Two legends: colour = reaction strength rho (the axis of this figure set), marker = the
    # pooled diffusion coefficient nu; the y=x line joins the marker legend.
    rho_handles = [plt.Line2D([], [], marker="o", linestyle="none", color=rho_colors[float(rho)],
                              markersize=10, label=rf"$\rho={rho:g}$") for rho in RHOS]
    nu_handles = [plt.Line2D([], [], marker=NU_MARKERS[float(nu)], linestyle="none", color="0.45",
                             markersize=9, label=rf"$\nu={nu:g}$") for nu in NUS]
    nu_handles.append(plt.Line2D([], [], linestyle="--", color="k", label=r"$y=x$"))
    fig.legend(handles=rho_handles, loc="lower center", ncol=len(rho_handles), fontsize=16,
               bbox_to_anchor=(0.235, -0.06), title=r"$\mathrm{colour:\ reaction\ strength}\ \rho$",
               title_fontsize=16)
    fig.legend(handles=nu_handles, loc="lower center", ncol=len(nu_handles), fontsize=16,
               bbox_to_anchor=(0.755, -0.06), title=r"$\mathrm{marker:\ diffusion\ coefficient}\ \nu$",
               title_fontsize=16)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved figure to {output_path}")
    if png_dir is not None:
        png_path = png_dir / (output_path.stem + ".png")
        fig.savefig(png_path, bbox_inches="tight", dpi=110)
        print(f"Saved PNG to {png_path}")
    plt.close(fig)


def main(png_dir: Path | None = None) -> None:
    cells, override = cells_from_env()
    if override:
        print(f"OVERRIDE MODE (REACDIFF_CELLS): using {len(cells)}/16 cells: "
              + ", ".join(_cell_key(nu, rho) for nu, rho in cells))
    else:
        print("default mode: all 16 cells (a missing artifact raises)")
    print(f"trajectories per cell: {len(IDXS)} = {IDXS}")
    records, dropped = collect_results(cells, override)
    mean_lim = mean_axis_limit(records)
    print(f"symlog mean axes in all three figures: [-{mean_lim:.3f}, {mean_lim:.3f}] "
          f"({MEAN_LIM_PAD} x largest plotted |mean| = {mean_lim / MEAN_LIM_PAD:.3f})")
    if png_dir is not None:
        png_dir.mkdir(parents=True, exist_ok=True)
    for (x_method, y_method), stem in PAIRS.items():
        make_figure(records, x_method, y_method, FIGURES_DIR / f"{stem}.pdf", mean_lim, png_dir)
    summary = summarize(records, dropped, cells, override)
    OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2))
    print(f"Saved summary to {OUTPUT_SUMMARY}")
    for (x_method, y_method), stem in PAIRS.items():
        write_table(summary, x_method, y_method, FIGURES_DIR / f"{stem}_table.tex")

    if override:
        print("cells used (override mode): " + ", ".join(summary["cells_used"]))
    print("runs per cell (entering / planned): " + _n_per_cell_str(summary))
    print("dropped runs (HMC gate or missing artifact):")
    for line in dropped_lines(dropped):
        print(f"  {line}")
    print("runs entering with only part of their HMC coordinates admitted:")
    for line in partial_lines(records):
        print(f"  {line}")

    def _print_group(label: str, s: dict) -> None:
        print(f"{label}: admitted {s['admitted_fraction']:.1%} of coordinates, "
              f"n={s['n_trajectories']}/{s['n_trajectories_planned']} ({s['pools']})")
        if s["n_trajectories"] == 0:
            return
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

    for key, s in summary["per_cell"].items():
        _print_group(f"cell {key}", s)
    for rho, s in summary["per_rho"].items():
        _print_group(f"rho={rho} (pooled over nu)", s)
    _print_group("all cells", summary["all_cells"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="posterior recovery comparison on the Fisher-KPP grid")
    parser.add_argument("--png", type=Path, default=None, metavar="DIR",
                        help="also render every figure as PNG into this directory (for inspection)")
    args = parser.parse_args()
    tee = _Tee(OUTPUT_LOG)
    sys.stdout = tee
    try:
        main(args.png)
    except BaseException as exc:   # the log must show why the run stopped (default mode raises on a missing artifact)
        print(f"FAILED: {type(exc).__name__}: {exc}")
        raise
    finally:
        sys.stdout = tee._stdout
        tee.close()
