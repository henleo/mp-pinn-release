"""
Does the one-pass EP (expectation propagation) method still work on a slightly wider network? The
single-hidden-unit model u = g(w.phi) is compared with u = sum_h a_h g(w_h.phi), H = 2, 3 and 4
hidden units with FIXED readout weights a (H=2: (1,-1); H=3: (1,-1,0.5); H=4: (1,-1,0.5,-0.5)),
trained by the SAME one-pass EP schedule, priors, noise levels and tolerances (runner:
pde/advection/mp_pinn_run_beta_wide.jl; the H=1 points are the existing single-neuron artifacts).
Wave speeds b in {0.1, 0.4, 1.0, 2.0, 7.0}. Trajectory sets (pde/advection/trajectory_selection.py,
one fixed random draw): every b uses the same ten canonical trajectories idxs(10) (one b = 7 run
of the multichain HMC (Hamiltonian Monte Carlo) reference costs about an hour, a wide H = 3 or 4 EP
run 40 to 65 minutes). Everything is evaluated against
the TRUE trajectory on the full (x,t) grid.

The shipped figure (wider_model.pdf) has four panels, 2 x 2:

  (a) rel-L2 error of the predictive mean vs truth: one point per trajectory, line through medians.
  (b) empirical coverage of the nominal 90% (solid) and 95% (dashed) predictive intervals vs the
      true trajectory, pooled over every grid point of the admitted trajectories per b (the
      formula of calibration_comparison.py).
  (c) predictive NLL (negative log-likelihood of the truth under the Gaussian predictive, mean of
      the per-trajectory means, as in calibration_comparison.py).
  (d) training wall time per run: points per trajectory, line through medians.
  The grey series with x markers in (b) and (c) is the multichain HMC posterior of the H=1 model.

Two further panels are not in wider_model.pdf: their numbers are in the companion table, and
WIDER_MODEL_ALL_PANELS=1 builds the extended six-panel figure, with its letters (a)-(f), into
wider_model_all_panels.pdf (wider_model.pdf is then left untouched); their plotting functions are
plot_mean_std and plot_neuron_usage:

  - mean predictive std at the last time step, panel (b) of the six-panel figure (the spatial mean of
    sqrt(pred_var) at t=end, then the across-trajectory mean, i.e. the t=end value of
    panel (c) of calibration_variance.pdf, calibration_comparison.py); points = per trajectory.
  - neuron usage, panel (f) of the six-panel figure: each neuron's share of the predictive mean,
    rms(a_h E[g(z_h)]) / rms(u_pred) over the grid, median across trajectories. It shows whether
    every neuron contributes or one neuron carries everything; the dotted lines mark the equal
    share 1/H and the small digits at the right end of the lines are the neuron indices (1..H, in
    the order of a).

How to read the coverage and NLL panels (and the mean std): the wide predictive variance is the
mean-field sum over independent neurons, Var[u] = sum_h a_h^2 Var[g(z_h)], because the EP beliefs
factorize over all weights of all neurons and carry no covariance between neurons. The predictive
std and the coverage therefore rise, and the NLL falls, with H mainly because more independent
variance terms are added. This is an independence effect and must not be read as better-calibrated
inference: the true posterior of the wide model has strongly anti-correlated neuron pairs,
whose negative covariance the mean-field sum leaves out. The HMC series is grey
because it is the posterior of a different model, the single-neuron model (its exact posterior),
and is shown for scale only, not as the reference for the wide models.

The companion table (wider_model_table.tex) holds every number: per (b, H) the median / min / max
rel-L2, the mean std at t=end, coverage at 90% / 95%, NLL, epochs and training time, plus the
per-neuron usage block (share of the mean, pre-activation range, median posterior std). The
'train [s]' entry of the HMC rows is the MAP (maximum a posteriori) warm start plus the summed
wall time of the four chains, which run one after the other; it is not comparable to EP epochs.

Run set: the same rule as calibration_comparison.py. A trajectory enters only when its multichain
HMC reference converged (at least HMC_MIN_ADMITTED of the eigenbasis (the rotated weight
coordinates all methods share) coordinates pass split-R-hat (the chain-agreement diagnostic)
< 1.05 and bulk ESS (effective sample size) > 100). A trajectory that fails this gate is dropped
for EVERY model, H=1..4 and the HMC line alike, so every statistic here uses one run set and the
H=1 rows agree with the "Ours" rows of calibration_comparison_table.tex. Which runs are dropped is
NOT hard-coded: it is computed from the cached HMC diagnostics at run time, printed in the log
(together with the dropped run's own per-model numbers, for the record) and written into the
table comment with the resulting n per wave speed, so the list follows the caches (a run enters
as soon as its cached HMC reference passes the gate, without any edit here). A run counts as
diverged when it hit the 300-epoch cap and its per-epoch max KL (Kullback-Leibler divergence, in
nats) still exceeded 1 within its last 50 epochs. Median epochs are rounded half up to an integer
everywhere (table, log and printed summary). The NLL panel uses a log y axis while every mean NLL
is positive and falls back to symlog (linear near zero, log beyond +-NLL_LINTHRESH) otherwise, so
a negative NLL (small predictive std, smaller error) is never dropped silently. Reads cached
artifacts only; never trains, and raises if an artifact is missing.

Run: python figures_onepass/advection/wider_model.py
     (six-panel figure: WIDER_MODEL_ALL_PANELS=1 in front of the same command)

Environment variable WIDER_MODEL_BETAS (comma-separated wave speeds, default = all five, e.g.
WIDER_MODEL_BETAS=0.1,0.4,1.0,2.0) restricts the wave speeds, so the figure can be built from the
artifacts of a subset of the wave speeds. The restriction is announced in the
log and in the table comment; the shipped figure always uses the default (all five).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

FIGURES_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURES_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "pde" / "advection"))

import mp_pinn_beta_sweep as mp_sweep
import mp_pinn_wide_sweep as wide_sweep
import bpinn_multichain_sweep as hmc_sweep
from trajectory_selection import idxs

OUTPUT_PDF = FIGURES_DIR / "wider_model.pdf"
OUTPUT_PDF_ALL_PANELS = FIGURES_DIR / "wider_model_all_panels.pdf"   # WIDER_MODEL_ALL_PANELS=1 only
OUTPUT_TEX = FIGURES_DIR / "wider_model_table.tex"

# Panels in figure order; the letters (a), (b), ... follow the order. "mean_std" and "neuron_usage"
# appear only in the six-panel figure (WIDER_MODEL_ALL_PANELS=1, see the docstring).
PANELS_SHIPPED = ("rel_l2", "coverage", "nll", "train_time")
PANELS_ALL = ("rel_l2", "mean_std", "coverage", "nll", "train_time", "neuron_usage")
ALL_PANELS = os.environ.get("WIDER_MODEL_ALL_PANELS", "").strip() == "1"

BETAS_ALL = [0.1, 0.4, 1.0, 2.0, 7.0]
# The same ten canonical trajectories (one seeded draw) at every wave speed.
IDXS_BY_BETA = {b: idxs(10) for b in (0.1, 0.4, 1.0, 2.0, 7.0)}  # the same ten trajectories at every b


def _betas_from_env() -> list[float]:
    """WIDER_MODEL_BETAS restricts the wave speeds (a build from the artifacts of a subset of them)."""
    raw = os.environ.get("WIDER_MODEL_BETAS", "").strip()
    if not raw:
        return list(BETAS_ALL)
    betas = [float(s) for s in raw.split(",") if s.strip()]
    unknown = [b for b in betas if b not in IDXS_BY_BETA]
    if unknown or not betas:
        raise ValueError(f"WIDER_MODEL_BETAS={raw!r}: allowed wave speeds are {BETAS_ALL}")
    return sorted(set(betas))


BETAS = _betas_from_env()
BETAS_RESTRICTED = BETAS != BETAS_ALL
WIDTHS = [1, 2, 3, 4]
MIN_VAR = 1e-10           # same floor on the predictive variance as calibration_comparison.py
COVERAGE_LEVELS = [0.9, 0.95]
RHAT_MAX, ESS_MIN = 1.05, 100.0  # per-coordinate HMC convergence gate (as in calibration_comparison.py)
HMC_MIN_ADMITTED = 0.9           # a trajectory is used only if at least this fraction of its HMC coordinates pass
COLORS = {1: "#4a3aa7", 2: "#2b8cbe", 3: "#1b9e77", 4: "#c51b7d"}   # H=1 keeps the "Ours" purple
MARKERS = {1: "o", 2: "s", 3: "D", 4: "^"}
HMC_STYLE = dict(color="#7f7f7f", linestyle="--", marker="x", markersize=8, lw=1.8)
# Multiplicative offsets of the per-trajectory points on the log-b axis: a fixed factor is a fixed
# visual distance on a log axis, so the four H groups stay equally separated at every b, b=7 included.
X_JITTER = {1: 0.90, 2: 0.965, 3: 1.035, 4: 1.11}
X_JITTER_F = {2: 0.85, 3: 1.0, 4: 1.18}  # wider offsets in the neuron-usage panel only, so the neuron-index labels fit between the H groups
X_PAD = 1.35          # log-b axis limits: BETAS[0] / X_PAD .. BETAS[-1] * X_PAD, so the jittered points at b=0.1 and b=7 are not clipped
LABEL_GAP_F = 0.022   # minimum vertical distance (data units of the neuron-usage panel) between two neuron-index labels of one H
F_YMAX = 0.72         # y-limit of the neuron-usage panel: the shares stay below 0.55, so the data fill the panel
LEGEND_BAND = 0.30    # mean-std and NLL panels: fraction of the axes height kept free of data at the top, where the legend sits
NLL_LINTHRESH = 10.0  # the NLL panel falls back from log to symlog (linear inside +-NLL_LINTHRESH, log outside) when a mean NLL is <= 0


def load_one(beta: float, idx: int, H: int) -> dict:
    """Cached artifact only (no training is ever triggered here); H=1 = the single-neuron artifact."""
    if H == 1:
        path = mp_sweep._cache_path(beta, idx)
        if not path.exists():
            raise FileNotFoundError(f"missing single-neuron artifact {path} (b={beta:g}, idx={idx})")
        r = mp_sweep.load_cached(beta, idx)
        r.update(H=1, a=np.array([1.0]), rel_l2=float(np.linalg.norm(r["pred_u"] - r["true_u"]) / np.linalg.norm(r["true_u"])),
                 neuron_out_rms=np.array([float(np.sqrt(np.mean(r["pred_u"] ** 2)))]))
        return r
    path = wide_sweep._cache_path(beta, idx, H)
    if not path.exists():
        raise FileNotFoundError(f"missing wide artifact {path} (b={beta:g}, idx={idx}, H={H}); "
                                "generate it with pde/advection/mp_pinn_run_beta_wide.jl")
    return wide_sweep.load_cached(beta, idx, H)


def collect_hmc() -> tuple[dict[float, list[dict]], dict[float, list[int]], list[dict]]:
    """Multichain HMC posterior of the H=1 model per b, the trajectories it admits, and the runs the
    gate dropped. A trajectory whose HMC reference did not converge (fewer than HMC_MIN_ADMITTED of
    its eigenbasis coordinates pass the R-hat / ESS gate) is dropped for EVERY model of this
    figure, exactly as calibration_comparison.py does, so the reference line and all H rows share
    one run set. Each dropped run is returned as a dict (beta, idx, admitted, accept_rates) for the
    table comment."""
    if BETAS_RESTRICTED:
        print(f"NOTE: WIDER_MODEL_BETAS restricts this build to b in {BETAS} (default: {BETAS_ALL}).")
    hmc, used, dropped = {}, {}, []
    for beta in BETAS:
        hmc[beta], used[beta] = [], []
        for idx in IDXS_BY_BETA[beta]:
            path = hmc_sweep._cache_path(beta, idx)
            if not path.exists():
                raise FileNotFoundError(f"missing multichain HMC artifact {path} (b={beta:g}, idx={idx})")
            r = hmc_sweep.load_cached(beta, idx)
            admitted = float(np.mean((r["rhat"] < RHAT_MAX) & (r["ess"] > ESS_MIN)))
            if admitted < HMC_MIN_ADMITTED:
                rates = np.round(r["accept_rates"], 3).tolist()
                print(f"[b={beta} idx={idx}] EXCLUDED for every model: only {admitted:.0%} of the HMC coordinates "
                      f"converged (max split-R-hat {np.nanmax(r['rhat']):.3g}, min ESS {np.nanmin(r['ess']):.3g}, "
                      f"chain acceptance rates {rates}), so there is no usable reference for this trajectory.")
                _report_excluded_run(beta, idx, r)
                dropped.append(dict(beta=beta, idx=idx, admitted=admitted, accept_rates=rates))
                continue
            hmc[beta].append(r)
            used[beta].append(idx)
        print(f"[b={beta} HMC H=1 reference] loaded {len(hmc[beta])} of {len(IDXS_BY_BETA[beta])} canonical runs "
              f"(n = {len(used[beta])}; admitted trajectories: {used[beta]})")
        if not used[beta]:
            raise RuntimeError(f"b={beta:g}: the HMC gate admitted no trajectory, nothing to plot")
    print("Dropped by the HMC gate: " + (_dropped_text(dropped) if dropped else "none"))
    print("n per wave speed: " + ", ".join(f"{len(used[b])} at b={b:g} (of {len(IDXS_BY_BETA[b])} canonical)" for b in BETAS))
    return hmc, used, dropped


def _dropped_text(dropped: list[dict]) -> str:
    return "; ".join(f"b={d['beta']:g} idx {d['idx']} ({d['admitted']:.0%} admitted, chain acceptance {d['accept_rates']})"
                     for d in dropped)


def _report_excluded_run(beta: float, idx: int, hmc_run: dict) -> None:
    """For the record only: the single-run numbers (vs truth) of every model on a dropped trajectory."""
    for H in WIDTHS:
        try:
            r = load_one(beta, idx, H)
        except FileNotFoundError as e:   # the record is informational; a dropped run never enters the figure
            print(f"    for the record, H={H} on this run alone: artifact missing ({e})")
            continue
        print(f"    for the record, H={H} on this run alone: rel-L2 {r['rel_l2']:.4f}, mean std t=end "
              f"{mean_std_end(r):.3e}, cov@90 {100 * coverage([r], 0.9):.1f}%, cov@95 {100 * coverage([r], 0.95):.1f}%, "
              f"NLL {nll(r):.1f}, epochs {r['epochs']}" + (" (diverged)" if diverged(r) else ""))
    print(f"    for the record, HMC (H=1 model) on this run alone: mean std t=end {mean_std_end(hmc_run):.3e}, "
          f"cov@90 {100 * coverage([hmc_run], 0.9):.1f}%, NLL {nll(hmc_run):.1f}")


MISSING: list[tuple[float, int, int]] = []   # (b, H, idx) of wide artifacts that do not exist (filled by collect)


def collect(used: dict[float, list[int]]) -> dict[tuple[float, int], list[dict]]:
    """One list of runs per (b, H), restricted to the trajectories `used[b]` admitted by collect_hmc.
    The single-neuron (H=1) artifacts must all exist. A missing WIDE artifact (b, H, idx) is recorded
    in MISSING, printed, and left out: the (b, H) group then has fewer than n runs (the table's n
    column and its comment say so), and a group with no run at all is shown as "not run", so a
    missing wide artifact is reported, never silently skipped or fabricated."""
    runs = {}
    for beta in BETAS:
        for H in WIDTHS:
            rs = []
            for idx in used[beta]:
                try:
                    rs.append(load_one(beta, idx, H))
                except FileNotFoundError as e:
                    if H == 1:
                        raise
                    MISSING.append((beta, H, idx))
                    print(f"[b={beta} H={H}] idx {idx}: wide artifact missing, left out ({e})")
            runs[(beta, H)] = rs
            print(f"[b={beta} H={H}] loaded {len(rs)} of {len(used[beta])} runs")
    if MISSING:
        print("Wide artifacts missing (b, H, idx): " + ", ".join(f"({b:g}, {H}, {i})" for b, H, i in MISSING))
    return runs


def _nanmed(x) -> float:
    return float(np.median(x)) if len(x) else float("nan")


# --- per-run / per-group statistics (formulas shared with calibration_comparison.py) -------------

def mean_std_end(r: dict) -> float:
    """Spatial mean of the predictive std at the last time step."""
    return float(np.sqrt(np.maximum(r["pred_var"][:, -1], 0.0)).mean())


def standardized_residuals(r: dict) -> np.ndarray:
    std = np.sqrt(np.maximum(r["pred_var"], MIN_VAR))
    return ((r["true_u"] - r["pred_u"]) / std).ravel()


def coverage(rs: list[dict], level: float) -> float:
    if not rs:
        return float("nan")
    z = np.concatenate([standardized_residuals(r) for r in rs])
    return float(np.mean(np.abs(z) <= stats.norm.ppf(0.5 + level / 2)))


def nll(r: dict) -> float:
    var = np.maximum(r["pred_var"], MIN_VAR)
    return float(np.mean(0.5 * np.log(2 * np.pi * var) + (r["true_u"] - r["pred_u"]) ** 2 / (2 * var)))


def mean_nll(rs: list[dict]) -> float:
    return float(np.mean([nll(r) for r in rs])) if rs else float("nan")


def diverged(r: dict) -> bool:
    """A run that hit the 300-epoch cap AND whose per-epoch max KL still exceeded 1 within its last
    50 epochs: the EP iteration blew up instead of settling. (Runs that converged are never counted,
    even though their first few epochs routinely have KL spikes above 1.)"""
    return r["epochs"] >= 300 and bool(np.max(r["loss_y"][-50:]) > 1.0)


def neuron_shares(r: dict) -> np.ndarray:
    """rms(a_h E[g(z_h)]) / rms(u_pred) per neuron."""
    return r["neuron_out_rms"] / float(np.sqrt(np.mean(r["pred_u"] ** 2)))


def _round_half_up(x: float) -> int:
    """The one rounding rule for the median epochs in the table, the log and the printed summary:
    halves round up (150.5 -> 151), unlike int() (truncates) and format .0f (rounds half to even)."""
    return int(np.floor(x + 0.5))


# --- panels ---------------------------------------------------------------------------------------

def _reserve_top(ax, frac: float) -> None:
    """Raise the upper y-limit so that the top `frac` of the axes height holds no data; the legend
    goes there and covers no marker or line. Uses the scale (log or linear) the axis already has."""
    lo, _ = ax.get_ylim()
    ymax = ax.dataLim.y1
    if ax.get_yscale() == "log":
        ax.set_ylim(top=10 ** ((np.log10(ymax) - frac * np.log10(lo)) / (1 - frac)))
    elif ax.get_yscale() == "linear":
        ax.set_ylim(top=(ymax - frac * lo) / (1 - frac))
    else:   # symlog etc.: the same construction in the axis' own transformed coordinates
        tr = ax.yaxis.get_transform()
        lo_t, ymax_t = tr.transform([lo, ymax])
        ax.set_ylim(top=float(tr.inverted().transform([(ymax_t - frac * lo_t) / (1 - frac)])[0]))


def _spread(vals, gap: float) -> np.ndarray:
    """Label positions for the values `vals`: neighbours at least `gap` apart (pushed apart in
    sorted order), the mean of the positions equal to the mean of the values."""
    vals = np.asarray(vals, dtype=float)
    order = np.argsort(vals)
    y = vals[order].copy()
    for i in range(1, len(y)):
        y[i] = max(y[i], y[i - 1] + gap)
    y += vals.mean() - y.mean()
    out = np.empty_like(y)
    out[order] = y
    return out


def _points_and_median_line(ax, runs, key_fn, agg=np.median):
    for H in WIDTHS:
        vals = {b: [key_fn(r) for r in runs[(b, H)]] for b in BETAS}
        for b in BETAS:
            ax.scatter(np.full(len(vals[b]), b * X_JITTER[H]), vals[b], s=26, alpha=0.45,
                       color=COLORS[H], marker=MARKERS[H], edgecolors="none")
        ax.plot(BETAS, [agg(vals[b]) if len(vals[b]) else float("nan") for b in BETAS], color=COLORS[H], marker=MARKERS[H],
                markersize=10, lw=2.2, label=rf"$H={H}$")


def _finish(ax, title, ylabel, ylog=True, legend=True, legend_kw=None):
    ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.set_xticks(BETAS)
    ax.set_xticklabels([f"{b:g}" for b in BETAS])
    ax.minorticks_off()
    ax.set_xlim(BETAS[0] / X_PAD, BETAS[-1] * X_PAD)   # fixed margins on the log-b axis (the jittered points fit)
    ax.set_xlabel(r"$b$", fontsize=19)
    ax.set_ylabel(ylabel, fontsize=19)
    ax.set_title(title, fontsize=20)
    ax.grid(True, which="major", alpha=0.3)
    ax.tick_params(labelsize=13)
    if legend:
        ax.legend(**{"fontsize": 13, **(legend_kw or {})})


def plot_rel_l2(ax, runs, tag):
    _points_and_median_line(ax, runs, lambda r: r["rel_l2"])
    _finish(ax, r"$\mathrm{(%s)\ Relative\ } L_2 \mathrm{\ error\ vs.\ truth}$" % tag, r"$\mathrm{rel.}\ L_2\ \mathrm{error}$")


def plot_mean_std(ax, runs, hmc, tag):
    _points_and_median_line(ax, runs, mean_std_end, agg=np.mean)
    ax.plot(BETAS, [np.mean([mean_std_end(r) for r in hmc[b]]) for b in BETAS],
            label=r"$\mathrm{HMC}\ (H{=}1\ \mathrm{model})$", **HMC_STYLE)
    _finish(ax, r"$\mathrm{(%s)\ Mean\ predictive\ std\ at\ } t{=}t_{\mathrm{end}}$" % tag, r"$\mathrm{predictive\ std}$",
            legend=False)
    _reserve_top(ax, LEGEND_BAND)   # the legend sits in the empty band above the data
    ax.legend(fontsize=13, ncol=3, loc="upper left")


def plot_coverage(ax, runs, hmc, tag):
    for H in WIDTHS:
        for level, ls in zip(COVERAGE_LEVELS, ("-", "--")):
            ax.plot(BETAS, [coverage(runs[(b, H)], level) for b in BETAS], color=COLORS[H],
                    marker=MARKERS[H], markersize=10, lw=2.2, linestyle=ls,
                    label=rf"$H={H}$" if level == COVERAGE_LEVELS[0] else None)
    for level, ls in zip(COVERAGE_LEVELS, ("-", "--")):
        ax.plot(BETAS, [coverage(hmc[b], level) for b in BETAS], **{**HMC_STYLE, "linestyle": ls},
                label=r"$\mathrm{HMC}\ (H{=}1\ \mathrm{model})$" if level == COVERAGE_LEVELS[0] else None)
        ax.axhline(level, color="k", linestyle=ls, lw=1.0, alpha=0.45)
    ax.plot([], [], color="k", linestyle="-", lw=2, label=r"$90\%\ \mathrm{interval}$")
    ax.plot([], [], color="k", linestyle="--", lw=2, label=r"$95\%\ \mathrm{interval}$")
    ax.set_ylim(0, 1.02)
    _finish(ax, r"$\mathrm{(%s)\ Empirical\ coverage\ vs.\ truth}$" % tag, r"$\mathrm{empirical\ coverage\ of\ the\ truth}$", ylog=False)
    ax.legend(fontsize=11, ncol=2, loc="upper left", bbox_to_anchor=(0.01, 0.86))


def plot_nll(ax, runs, hmc, tag):
    vals = {H: [mean_nll(runs[(b, H)]) for b in BETAS] for H in WIDTHS}
    vals["hmc"] = [mean_nll(hmc[b]) for b in BETAS]
    for H in WIDTHS:
        ax.plot(BETAS, vals[H], color=COLORS[H], marker=MARKERS[H], markersize=10, lw=2.2, label=rf"$H={H}$")
    ax.plot(BETAS, vals["hmc"], label=r"$\mathrm{HMC}\ (H{=}1\ \mathrm{model})$", **HMC_STYLE)
    # A Gaussian NLL is negative when the predictive std is small and the error smaller still; a log
    # axis would drop such a point silently, so the axis is log only while every value is positive.
    all_positive = min(min(v) for v in vals.values()) > 0
    _finish(ax, r"$\mathrm{(%s)\ Predictive\ NLL\ vs.\ truth}$" % tag, r"$\mathrm{NLL}$",
            ylog=all_positive, legend=False)
    if not all_positive:
        print(f"[NLL panel] a mean NLL is <= 0 (min {min(min(v) for v in vals.values()):.3g}): symlog y axis "
              f"(linear inside +-{NLL_LINTHRESH:g}, log outside) instead of log")
        ax.set_yscale("symlog", linthresh=NLL_LINTHRESH)
    _reserve_top(ax, LEGEND_BAND)   # the legend sits in the empty band above the data
    ax.legend(fontsize=13, ncol=3, loc="upper left")


def plot_train_time(ax, runs, tag):
    _points_and_median_line(ax, runs, lambda r: r["train_time_seconds"])
    _finish(ax, r"$\mathrm{(%s)\ Training\ time}$" % tag, r"$\mathrm{time\ (s)}$")


def plot_neuron_usage(ax, runs, tag):
    """One line per (H, neuron); the dotted line per H is the equal share 1/H (in the legend next
    to its H); the neuron index is written at the right end of each line, pushed apart vertically
    where the lines of one H end close together."""
    handles = []   # (artist, label) pairs in legend order: each H followed by its 1/H reference line
    for H in WIDTHS[1:]:
        xs = np.array(BETAS) * X_JITTER_F[H]
        end_shares = []
        for h in range(H):
            shares = [_nanmed([neuron_shares(r)[h] for r in runs[(b, H)]]) for b in BETAS]
            line, = ax.plot(xs, shares, color=COLORS[H], marker=MARKERS[H], markersize=9, lw=1.2,
                            alpha=0.5 + 0.5 * (h == 0))
            if h == 0:
                handles.append((line, rf"$H={H}$"))
            end_shares.append(shares[-1])
        ref = ax.axhline(1.0 / H, color=COLORS[H], linestyle=":", lw=1.2, alpha=0.7)
        handles.append((ref, rf"$1/H = 1/{H}$"))
        for h, y in enumerate(_spread(end_shares, LABEL_GAP_F)):
            ax.text(xs[-1] * 1.06, y, f"{h + 1}", fontsize=9, color=COLORS[H], ha="left", va="center")
    ax.set_ylim(0, F_YMAX)
    _finish(ax, r"$\mathrm{(%s)\ Neuron\ share\ of\ the\ predictive\ mean}$" % tag,
            r"$\mathrm{neuron\ share}$", ylog=False, legend=False)
    ax.set_xlim(BETAS[0] * X_JITTER_F[WIDTHS[1]] / X_PAD, BETAS[-1] * X_JITTER_F[WIDTHS[-1]] * 1.3)   # room for the neuron-index labels
    ax.legend([a for a, _ in handles], [l for _, l in handles], fontsize=13, ncol=3, loc="upper center")


# --- table ----------------------------------------------------------------------------------------

def _sci(x: float) -> str:
    m, e = f"{x:.2e}".split("e")
    return rf"${m}{{\times}}10^{{{int(e)}}}$"


def write_table(runs, hmc, used, dropped, path: Path) -> None:
    n_by_beta = {b: len(used[b]) for b in BETAS}
    lines = [
        r"% Auto-generated by figures_onepass/advection/wider_model.py",
        r"% One-pass EP (expectation propagation) on u = sum_h a_h g(w_h.phi): H=1 is the single-neuron model (existing",
        r"%   artifacts), H=2 uses a=(1,-1), H=3 a=(1,-1,0.5), H=4 a=(1,-1,0.5,-0.5); same schedule, priors and noise levels for all H.",
        r"% 'capped' = number of the n runs that hit the 300-epoch limit without reaching the KL (Kullback-Leibler divergence,",
        r"%   in nats) tolerance 1e-5 (the same limit applies to H=1); 'div' = number of those capped runs whose per-epoch max KL",
        r"%   still exceeded 1 within the last 50 epochs, i.e. the EP iteration diverged instead of converging slowly.",
        r"% Trajectory set (pde/advection/trajectory_selection.py, one fixed random draw): the same ten canonical",
        r"%   trajectories idxs(10) at every b.",
        rf"% n per b = the canonical trajectories whose multichain HMC reference converged (at least",
        rf"%   {HMC_MIN_ADMITTED:.0%} of the eigenbasis (the rotated weight coordinates all methods share) coordinates with",
        rf"%   split-R-hat (the chain-agreement diagnostic) < {RHAT_MAX} and bulk ESS (effective sample size) > {ESS_MIN:g}, the rule",
        r"%   of calibration_comparison.py); computed from the cached HMC diagnostics when this table was written:",
        rf"%   n = {', '.join(f'{n_by_beta[b]} at b={b:g} (of {len(IDXS_BY_BETA[b])} canonical)' for b in BETAS)}.",
        (r"% Wide artifacts missing (a (b, H) row then has fewer than n runs, its own n column says how many): "
         + ", ".join(f"(b={b:g}, H={H}, idx {i})" for b, H, i in MISSING)) if MISSING else r"% Every (b, H, trajectory) wide artifact exists.",
        rf"%   Dropped by the HMC gate, for EVERY model (all H and the HMC rows): {_dropped_text(dropped) if dropped else 'none'}.",
        r"%   The dropped runs' own per-model numbers are in the printed log. Because the run set is shared, the H=1 rows",
        r"%   equal the 'Ours' rows of calibration_comparison_table.tex.",
    ]
    if BETAS_RESTRICTED:
        lines.append(rf"% NOTE: built with WIDER_MODEL_BETAS={os.environ['WIDER_MODEL_BETAS']!r}, i.e. b in {BETAS} only "
                     rf"(default: {BETAS_ALL}); rebuild without the variable for the shipped table.")
    lines += [
        r"% Everything is measured against the TRUE trajectory on the full 1024 x 201 (x,t) grid.",
        r"% Block 1, per (b, H): rel-L2 of the predictive mean (median [min, max] over trajectories);",
        r"%   mean std at t=end = across-trajectory mean of the spatial mean of sqrt(pred_var) at the last",
        r"%   time step (the t=end value of panel (c) of calibration_variance.pdf), with its min-max over trajectories;",
        rf"%   cov90 / cov95 = empirical coverage of the nominal 90% / 95% Gaussian predictive interval, pooled",
        r"%   over every grid point of the n trajectories; NLL = mean over trajectories of the per-trajectory mean",
        rf"%   Gaussian NLL (variance floored at {MIN_VAR:g}); epochs and training time = median [min, max]; the median",
        r"%   epochs are rounded half up to an integer (the same rule as in the printed log and summary).",
        r"%   The 'HMC (H=1)' rows are the multichain HMC posterior of the single-neuron model (its exact posterior). Their",
        r"%   'train [s]' = MAP (maximum a posteriori) warm start plus the summed wall time of the four chains, which run one",
        r"%   after the other (the step-size search is not counted); this is not comparable to EP epochs, so their epochs",
        r"%   cell is empty ('--').",
        r"% Reading the uncertainty columns (mean std, cov90, cov95, NLL) across H: the wide predictive variance is the",
        r"%   mean-field sum over independent neurons, Var[u] = sum_h a_h^2 Var[g(z_h)] (the EP beliefs factorize over all",
        r"%   weights of all neurons and carry no covariance between neurons). The rise of the mean std and of the coverage",
        r"%   and the fall of the NLL with H are therefore an independence effect (more independent variance terms are added)",
        r"%   and must NOT be read as better-calibrated inference: the true posterior of the wide model has strongly",
        r"%   anti-correlated neuron pairs, whose negative covariance the mean-field sum leaves out. The 'HMC (H=1)'",
        r"%   rows are a scale reference from a different model, not the reference posterior of the wide models.",
        r"% Block 2 (after the 'Neuron usage' heading), per (b, H, neuron): share = rms(a_h E[g(z_h)]) / rms(u_pred) over the",
        r"%   grid (median over trajectories, then its [min, max] over trajectories); z range = [min over trajectories of",
        r"%   min z_h, max over trajectories of max z_h] of the pre-activation z_h = w_h.phi at the posterior mean;",
        r"%   sigma_w = median over coordinates and trajectories of the posterior std of neuron h's eigenbasis weights.",
        r"\begin{tabular}{l l r r r r r r r r}",
        r"\toprule",
        r"$b$ & model & rel-$L_2$ median [min, max] & mean std $t{=}t_\mathrm{end}$ [min, max] & cov90 & cov95 & NLL & epochs (capped/div) & train [s] & $n$ \\",
        r"\midrule",
    ]
    for b in BETAS:
        for H in WIDTHS:
            rs = runs[(b, H)]
            if not rs:
                lines.append(f"{b:g} & $H={H}$ & \\multicolumn{{7}}{{l}}{{not run (0 of {len(used[b])} trajectories)}} & 0 \\\\")
                continue
            rl = np.array([r["rel_l2"] for r in rs])
            ms = np.array([mean_std_end(r) for r in rs])
            ep = np.array([r["epochs"] for r in rs])
            tt = np.array([r["train_time_seconds"] for r in rs])
            lines.append(
                f"{b:g} & $H={H}$ & {np.median(rl):.4f} [{rl.min():.4f}, {rl.max():.4f}] & "
                f"{_sci(ms.mean())} [{_sci(ms.min())}, {_sci(ms.max())}] & "
                f"{100 * coverage(rs, 0.9):.1f}\\% & {100 * coverage(rs, 0.95):.1f}\\% & {mean_nll(rs):.0f} & "
                f"{_round_half_up(np.median(ep))} [{ep.min()}, {ep.max()}] ({int((ep >= 300).sum())}/{sum(diverged(r) for r in rs)}) & {np.median(tt):.1f} [{tt.min():.1f}, {tt.max():.1f}] & {len(rs)} \\\\")
        hs = hmc[b]
        rl = np.array([np.linalg.norm(r["pred_u"] - r["true_u"]) / np.linalg.norm(r["true_u"]) for r in hs])
        ms = np.array([mean_std_end(r) for r in hs])
        tt = np.array([r["train_time_seconds"] for r in hs])
        lines.append(
            f"{b:g} & HMC ($H{{=}}1$) & {np.median(rl):.4f} [{rl.min():.4f}, {rl.max():.4f}] & "
            f"{_sci(ms.mean())} [{_sci(ms.min())}, {_sci(ms.max())}] & "
            f"{100 * coverage(hs, 0.9):.1f}\\% & {100 * coverage(hs, 0.95):.1f}\\% & {mean_nll(hs):.0f} & "
            f"-- & {np.median(tt):.0f} [{tt.min():.0f}, {tt.max():.0f}] & {len(hs)} \\\\")
        if b != BETAS[-1]:
            lines.append(r"\midrule")
    lines += [
        r"\midrule",
        r"\multicolumn{10}{l}{\emph{Neuron usage}, $H = 2, 3, 4$}\\",
        r"$b$ & model & neuron & $a_h$ & share (median) & share [min, max] & $z_h$ range & $\sigma_w$ median & & \\",
        r"\midrule",
    ]
    for b in BETAS:
        for H in WIDTHS[1:]:
            rs = runs[(b, H)]
            if not rs:
                continue
            for h in range(H):
                sh = np.array([neuron_shares(r)[h] for r in rs])
                zmin = min(float(r["neuron_z_min"][h]) for r in rs)
                zmax = max(float(r["neuron_z_max"][h]) for r in rs)
                sig = np.median(np.concatenate([r["w_eigen_std"][:, h] for r in rs]))
                lines.append(f"{b:g} & $H={H}$ & {h + 1} & {rs[0]['a'][h]:+.1f} & {np.median(sh):.3f} & "
                             f"[{sh.min():.3f}, {sh.max():.3f}] & [{zmin:.2f}, {zmax:.2f}] & {sig:.4f} & & \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines))
    print(f"Saved table to {path}")


def print_summary(runs, hmc, used, dropped) -> None:
    print(f"\n{sum(len(v) for v in used.values())} runs used "
          f"({', '.join(f'{len(used[b])} at b={b:g} (of {len(IDXS_BY_BETA[b])} canonical)' for b in BETAS)}); "
          "the same trajectories for every H and for the HMC reference. "
          "Dropped by the HMC gate: " + (_dropped_text(dropped) if dropped else "none") + ".")
    if BETAS_RESTRICTED:
        print(f"NOTE: WIDER_MODEL_BETAS restricted this build to b in {BETAS} (default: {BETAS_ALL}).")
    print("\nSummary per (b, H): rel-L2 median | mean std t=end | cov90 | cov95 | NLL | epochs median (capped at 300 / diverged) | train s median")
    for b in BETAS:
        for H in WIDTHS:
            rs = runs[(b, H)]
            if not rs:
                print(f"  b={b:<4g} H={H}: not run (0 of {len(used[b])} trajectories)")
                continue
            print(f"  b={b:<4g} H={H}: {np.median([r['rel_l2'] for r in rs]):.4f} | "
                  f"{np.mean([mean_std_end(r) for r in rs]):.3e} | {coverage(rs, 0.9):.3f} | {coverage(rs, 0.95):.3f} | "
                  f"{mean_nll(rs):8.1f} | {_round_half_up(np.median([r['epochs'] for r in rs]))} ({sum(r['epochs'] >= 300 for r in rs)}/{sum(diverged(r) for r in rs)}) | "
                  f"{np.median([r['train_time_seconds'] for r in rs]):.1f}")
        hs = hmc[b]
        print(f"  b={b:<4g} HMC(H=1): {np.median([np.linalg.norm(r['pred_u'] - r['true_u']) / np.linalg.norm(r['true_u']) for r in hs]):.4f} | "
              f"{np.mean([mean_std_end(r) for r in hs]):.3e} | {coverage(hs, 0.9):.3f} | {coverage(hs, 0.95):.3f} | {mean_nll(hs):8.1f}")
    print("\nNeuron usage (median share of the predictive mean per neuron; z range over trajectories):")
    for b in BETAS:
        for H in WIDTHS[1:]:
            rs = runs[(b, H)]
            if not rs:
                continue
            parts = []
            for h in range(H):
                sh = np.median([neuron_shares(r)[h] for r in rs])
                zmin = min(float(r["neuron_z_min"][h]) for r in rs)
                zmax = max(float(r["neuron_z_max"][h]) for r in rs)
                parts.append(f"n{h + 1}(a={rs[0]['a'][h]:+.1f}): share {sh:.3f}, z[{zmin:.2f},{zmax:.2f}]")
            print(f"  b={b:<4g} H={H}: " + "; ".join(parts))


PANEL_PLOTTERS = {
    "rel_l2": lambda ax, runs, hmc, tag: plot_rel_l2(ax, runs, tag),
    "mean_std": lambda ax, runs, hmc, tag: plot_mean_std(ax, runs, hmc, tag),
    "coverage": lambda ax, runs, hmc, tag: plot_coverage(ax, runs, hmc, tag),
    "nll": lambda ax, runs, hmc, tag: plot_nll(ax, runs, hmc, tag),
    "train_time": lambda ax, runs, hmc, tag: plot_train_time(ax, runs, tag),
    "neuron_usage": lambda ax, runs, hmc, tag: plot_neuron_usage(ax, runs, tag),
}


def make_figure(runs, hmc, panels=PANELS_SHIPPED):
    """Panels in row-major order, lettered (a), (b), ... in that order: 2 x 2 for the four shipped
    panels, 2 x 3 for all six (WIDER_MODEL_ALL_PANELS=1). Every panel keeps its 7 x 6 inch size."""
    plt.rcParams["mathtext.fontset"] = "cm"
    ncols = 2 if len(panels) == 4 else 3
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    for ax, name, tag in zip(axes.flat, panels, "abcdef"):
        PANEL_PLOTTERS[name](ax, runs, hmc, tag)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def main() -> None:
    hmc, used, dropped = collect_hmc()
    runs = collect(used)
    panels, output_pdf = (PANELS_ALL, OUTPUT_PDF_ALL_PANELS) if ALL_PANELS else (PANELS_SHIPPED, OUTPUT_PDF)
    if ALL_PANELS:
        print(f"NOTE: WIDER_MODEL_ALL_PANELS is set: the six-panel figure goes to {output_pdf}; "
              f"{OUTPUT_PDF} is not written.")
    fig = make_figure(runs, hmc, panels)
    fig.savefig(output_pdf, bbox_inches="tight")
    png = os.environ.get("WIDER_MODEL_PNG", "").strip()   # optional PNG copy for a visual check
    if png:
        fig.savefig(png, dpi=110, bbox_inches="tight")
        print(f"Saved PNG copy to {png}")
    plt.close(fig)
    print(f"Saved figure to {output_pdf}")
    write_table(runs, hmc, used, dropped, OUTPUT_TEX)
    print_summary(runs, hmc, used, dropped)


if __name__ == "__main__":
    main()
