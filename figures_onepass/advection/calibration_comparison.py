"""
Calibration and variance consistency of the one-pass EP (expectation propagation) posterior
predictive ("Ours") against the multi-chain HMC (Hamiltonian Monte Carlo) reference and the
mean-field VI (variational inference) baseline, on the ten canonical trajectories idxs(10) at each
wave speed b in {0.1, 0.4, 1, 2, 7} (50 runs). The script has two parts, calibration (part 1)
and variance consistency (part 2): the cached artifacts are loaded once, and
both parts use one run set (the HMC convergence gate below).

SHIPPED FIGURE, calibration_variance.pdf (2 x 2), with its companion table calibration_variance_table.tex:
  (a) CRPS (continuous ranked probability score) against the truth per b, Ours / HMC / VI
      = panel (d) of the full figure calibration_comparison.pdf (part 1).
  (b) empirical coverage of the truth by the nominal 90% (solid) / 95% (dashed) intervals per b
      = panel (e) of the full figure calibration_comparison.pdf (part 1).
  (c) Ours' mean predictive std vs time per b (band = min to max over the trajectories)
      = panel (a) of the full figure variance_consistency.pdf (part 2).
  (d) per-trajectory median of sigma_Ours / sigma_HMC, one point per (b, trajectory)
      = panel (d) of the full figure variance_consistency.pdf (part 2).
Panels (c) and (d) are drawn on the panel size (8.5 x 7.33 inch) and with the font sizes of
panels (a) and (b) (the full variance_consistency.pdf uses 7.25 x 5.75 inch panels, smaller fonts,
thinner lines and smaller markers; MERGED_VC_STYLE vs VC_STYLE), the log axis of (d) is labelled
with plain numbers like that of (a), and the wave speeds are written as in (a)/(b) (b=1, not
b=1.0). The plotted data are those of the corresponding panels of the full figures.

THE FULL FIGURES. CALIBRATION_ALL_PANELS=1 additionally writes
calibration_comparison.pdf (six panels a-f) and variance_consistency.pdf (four panels a-d), the
full figures of part 1 and part 2;
the shipped figure is written in both modes. The two full companion tables,
calibration_comparison_table.tex and variance_consistency_table.tex, are written on every run,
since they hold the numbers of every panel. The panel letters in PART 1 and PART 2 below refer to
those two full figures.

PART 1: calibration

Calibration (does the stated uncertainty match the actual error?) of the one-pass EP (expectation
propagation) posterior predictive ("Ours") against the multi-chain HMC (Hamiltonian Monte Carlo)
reference and the mean-field VI (variational inference; mean-field = every weight gets its own
independent Gaussian) baseline, over the full (x,t) grid of the canonical trajectories per wave
speed b in BETAS: the ten canonical trajectories idxs(10) at every b, including b = 7 (the HMC
reference costs about an hour and the VI baseline about two hours per b=7 run; see IDXS_BY_BETA).

Two levels of calibration are shown, because they answer two different questions:

  * MODEL-LEVEL, against the TRUE solution (panels a-e, table block 1): are the nominal posterior
    intervals honest about the distance to the true trajectory? Every method, including exact
    HMC, is far under-covered here (HMC's 90% interval covers roughly a quarter to a third of the
    grid points at every b), because the dominant error is model-form error shared by all three methods: the
    truncated Fourier basis cannot represent the true solution exactly, and that representation
    error is not part of the noise model (beta_ic = 0.02, beta_pde = 0.05). Coverage therefore
    mostly measures the model, not the inference method. The NLL (negative log-likelihood) gap
    between the methods is nevertheless an inference effect: the per-run residuals of the
    posterior mean against the truth are the same for Ours and HMC to 2-3 decimals (the table
    comment gives the numbers), so the gap (an order of magnitude in the pooled NLL) comes from
    the predictive variances alone. Block 1 is reported in full for honesty, with 90%/95% coverage
    and NLL kept prominent and CRPS (continuous ranked probability score, a proper scoring rule
    in the units of u) alongside.
  * INFERENCE-LEVEL, against the EXACT posterior (panel f, table block 2): how well does each
    approximate posterior reproduce the exact posterior predictive of the same model, which HMC
    samples? This isolates the approximation error of the inference method itself. It is computed
    in closed form from the stored predictive moments, treating the HMC predictive at each grid
    point as Gaussian with HMC's own mean m_H and variance v_H (an assumption, stated in the
    table comment as well; see VERIFICATION_NOTE below). With
    the method's mean m_q and std s_q, and z the two-sided standard-normal quantile of the level:
      coverage of the HMC predictive by the interval m_q +/- z s_q
          = Phi((m_q + z s_q - m_H)/s_H) - Phi((m_q - z s_q - m_H)/s_H),
      expected NLL of the method's Gaussian predictive under the HMC predictive
          = 0.5 ln(2 pi v_q) + (v_H + (m_H - m_q)^2) / (2 v_q),
    both averaged over the grid. HMC against itself gives nominal coverage and its own entropy
    (0.5 ln(2 pi v_H) + 0.5), which is the floor of the expected NLL.

Panels of calibration_comparison.pdf: (a) pooled coverage curve at nominal 50/80/90/95%; (b) standardized residuals (N(0,1) if
calibrated); (c) NLL per b (log axis); (d) CRPS per b (log axis); (e) 90%/95% coverage of the true
solution per b; (f) inference-level 90%/95% coverage of the HMC predictive per b. Panels (a)/(b)
pool every admitted run of every b (b=7 included); panels (c)-(f) show every b in BETAS on a log-b
axis. The companion table (calibration_comparison_table.tex) holds every number per b and pooled,
with n per b; the y-axes of (c)/(d) are chosen from the plotted values, so the large VI numbers
at b=7 (its predictive mean collapses there) stay inside the axes.

A run enters only when its HMC reference converged: at least HMC_MIN_ADMITTED of the eigenbasis
coordinates (the rotated weight coordinates all methods share) pass split-R-hat < 1.05 and bulk
ESS > 100 (split-R-hat compares the spread between the chain halves with the spread within them,
1 means they agree; bulk ESS is the effective sample size, the number of independent draws the
correlated chains are worth). A run that fails the gate is dropped for EVERY method, so all
comparisons share one run set; which runs are dropped is computed from the cached HMC diagnostics
at run time and is printed in the log and in the table comment (with the multichain HMC
references used here, every one of the 50 runs passes and nothing is dropped); if a run is
ever dropped, the table comment also lists Ours' and VI's model-level numbers with it put back.
Reads cached artifacts only; never trains.


PART 2: variance consistency

Is the predictive variance MEANINGFUL? Four questions, one figure (variance_consistency.pdf), over the same canonical
trajectories per wave speed as `posterior_recovery_comparison.py` (b in {0.1, 0.4, 1.0, 2.0, 7.0}):
the ten canonical trajectories `idxs(10)` (one seeded draw, see `trajectory_selection.py`) at
every b, b = 7 included (one b = 7 run costs about an hour of HMC and two hours of VI). "Ours" is the one-pass EP (expectation propagation) posterior
predictive; the reference is the multi-chain HMC (Hamiltonian Monte Carlo) posterior of the same
model; VI (variational inference) is the mean-field Gaussian baseline (a Gaussian with independent
weights, fitted by optimisation). All three share the model
and its noise model (β_ic, β_pde).

  (a) Does uncertainty grow with distance from the observations? Predictive std of Ours vs. time
      (observations exist only at t=0), per wave speed: solid = across-trajectory mean of the
      spatial-mean predictive std, band = min to max across the trajectories of that wave speed.
  (b) Does the predicted std track the actual error? Binned reliability: grid points pooled
      across all runs are binned by predicted std (per-method quantile bins, so every bin holds
      the same share of that method's points); each bin shows the empirical RMSE of those points
      vs. the bin's RMS predicted std. A calibrated model follows y=x; a constant multiplicative
      miscalibration is a parallel offset. Shown for Ours, HMC and VI: HMC's own offset separates
      "EP approximation error" from "shared modeling assumptions", and VI shows where a cheaper
      Gaussian approximation lands. The x limits are the padded std range of the plotted bins
      (LOG_PAD of its log-span on each side); the y axis runs from the bottom of that x range up
      to the padded RMSE range, so the y=x reference line stays visible across the whole x range
      while every bin sits well above it (RMSE/std, table block (b)). Caveat on the VI series:
      the pooled quantile bins sort strongly by wave speed for VI, whose predicted std grows with
      b while its mean gets worse with b (at b = 7 the VI mean collapses, rel-L2 (relative L2 error
      of the mean vs. the truth) about 1,
      so its RMSE vs. the truth is far above Ours and HMC there). The share of the top and
      bottom N_EDGE_BINS VI bins that comes from b in VI_HIGH_BETAS = {1, 2, 7}, and every
      method's RMSE vs. the truth per wave speed, are computed from the data at run time and
      written to the log and the table comment (with all 50 runs, the top four VI bins draw
      99% of their points from b in {1, 2, 7} and the bottom four 0%). The steep
      VI rise is therefore a between-wave-speed effect, not the predicted std tracking the error
      within a run. VI is never dropped from the panel: at b = 7 its points are extreme but still
      inside the log axes, whose limits are taken from the plotted bins of all three methods.
  (c) Is the miscalibration a consistent factor? The ratio σ_Ours/σ_HMC on the predictive
      grid vs. time (median over space and trajectories, per wave speed): a flat curve means
      Ours differs from the exact posterior by a roughly constant factor everywhere.
  (d) The same ratio's per-trajectory median, one point per (b, trajectory): consistency
      across trajectories rather than across the domain.

Run gate (the same rule and the same loader as part 1, so both parts share one run set): a run enters only when its HMC reference
converged, i.e. at least HMC_MIN_ADMITTED (90%) of the eigenbasis coordinates (the rotated weight
coordinates all methods share) have split-R-hat < 1.05 and bulk ESS > 100 (the two standard MCMC
convergence diagnostics: R-hat compares the chains' halves with each other, ESS is the number of
effectively independent samples). A run that fails the
gate is dropped for EVERY method, so every panel and table block uses the same run set. Which runs
are dropped is decided from the cached HMC diagnostics at run time: the log and the table comment
list them with their admitted share and chain acceptance rates, and state the number of runs used
per wave speed (with the HMC references used here every one of the 50 runs passes the gate, so
none is dropped; the list is recomputed from the cached diagnostics on every run). Reads cached
artifacts only; never trains (a missing artifact is an error).

The companion table (`variance_consistency_table.tex`) holds the numbers behind each panel:
(a) the mean-std curve at t=0 / t=1 / t=2 and its min-max band at t=2, (b) every reliability
bin for Ours, HMC and VI, (c) the ratio-vs-time curve at the same three times plus its range
over t, (d) the per-trajectory medians' spread and the per-beta and pooled grid-point ratio
quantiles. Panels (a), (c), (d) are about Ours vs. HMC only; VI appears in panel (b) alone.

Run: python figures_onepass/advection/calibration_comparison.py
     (+ the full figures: CALIBRATION_ALL_PANELS=1 in front of the same command)
"""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter
from scipy import stats

FIGURES_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURES_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "pde" / "advection"))

import mp_pinn_beta_sweep as mp_sweep
import bpinn_multichain_sweep as hmc_sweep
import vi_baseline_sweep as vi_sweep
from trajectory_selection import idxs

OUTPUT_PDF = FIGURES_DIR / "calibration_variance.pdf"          # the shipped (merged) figure
OUTPUT_TEX = FIGURES_DIR / "calibration_variance_table.tex"    # its companion table
CAL_PDF = FIGURES_DIR / "calibration_comparison.pdf"           # full figure of part 1 (CALIBRATION_ALL_PANELS=1 only)
CAL_TEX = FIGURES_DIR / "calibration_comparison_table.tex"     # full table of part 1 (every run)
VC_PDF = FIGURES_DIR / "variance_consistency.pdf"              # full figure of part 2 (CALIBRATION_ALL_PANELS=1 only)
VC_TEX = FIGURES_DIR / "variance_consistency_table.tex"        # full table of part 2 (every run)
ALL_PANELS = os.environ.get("CALIBRATION_ALL_PANELS", "").strip() == "1"

BETAS = [0.1, 0.4, 1.0, 2.0, 7.0]  # the wave speeds with a multichain HMC reference
# Canonical trajectories per wave speed: the same ten (one seeded draw) at every b.
IDXS_BY_BETA = {b: idxs(10) for b in (0.1, 0.4, 1.0, 2.0, 7.0)}  # the same ten trajectories at every b
assert set(IDXS_BY_BETA) == set(BETAS), "every wave speed in BETAS needs its trajectory list (and vice versa)"
METHODS = ("Ours", "HMC", "VI")
NOMINAL_LEVELS = [0.5, 0.8, 0.9, 0.95]
HEADLINE_LEVELS = [0.9, 0.95]  # the headline interval levels; panels (e)/(f) of the full calibration_comparison.pdf (panel (b) of the shipped figure) and both table blocks
MIN_VAR = 1e-10  # floor on predictive variance, avoids divide-by-zero (never binds on these runs)
Z_HIST_MAX = 10  # the residual histogram (panel (b) of the full calibration figure) drops |z| >= this
RHAT_MAX, ESS_MIN = 1.05, 100.0  # per-coordinate HMC convergence gate (as in posterior_recovery_comparison.py)
HMC_MIN_ADMITTED = 0.9  # a run is used only if at least this fraction of its HMC coordinates pass the gate
COLORS = {"Ours": "#4a3aa7", "HMC": "#eb6834", "VI": "#008300"}
MARKERS = {"Ours": "o", "HMC": "s", "VI": "^"}
LEVEL_STYLES = {0.9: "-", 0.95: "--"}  # solid = 90% interval, dashed = 95% interval, everywhere
# Candidate labelled ticks of the log y-axes of the NLL and CRPS panels (panels (c), (d) of the full
# calibration figure; the CRPS panel is (a) of the shipped figure), written as plain numbers; the
# ticks and limits actually used are chosen at run time from the plotted values (see _log_y_axis),
# because the range depends on which runs the HMC gate admits and on the VI numbers at b=7.
NLL_YTICK_CANDIDATES = [t * 10 ** k for k in range(-1, 6) for t in (1, 2, 5)]
CRPS_YTICK_CANDIDATES = [t * 10 ** k for k in range(-5, 2) for t in (1, 2, 3, 5)]
LOG_Y_MARGIN = 1.35  # the axis limits sit this factor beyond the smallest / largest plotted value
# The assumption behind the closed forms of block 2, written into the table comment.
VERIFICATION_NOTE = (
    "Assumption: the closed forms of block 2 assume a Gaussian HMC predictive at every grid point, "
    "with HMC's stored mean m_H and variance v_H; they use these two moments, not the HMC draws."
)


# ============================================================================= PART 1: calibration
# (the panel letters in the comments refer to calibration_comparison.pdf)


# ----------------------------------------------------------------------------- data


def _slim(r: dict) -> dict:
    """Keep only the (n_x, n_t) predictive arrays a calibration statistic needs."""
    return dict(true_u=r["true_u"], pred_u=r["pred_u"], pred_var=r["pred_var"])


def collect_records() -> tuple[list[dict], list[dict]]:
    """One record per (beta, idx) holding the three predictives, split into (admitted, excluded).
    Runs whose HMC reference did not converge are excluded for every method, so all statistics
    share one run set; the excluded records are kept only for the table comment that shows how
    little the exclusion changes the model-level numbers of Ours and VI. Also the loader of part 2
    (variance_records derives its arrays from these records). A missing artifact is an error."""
    records, excluded = [], []
    for beta in BETAS:
        for idx in IDXS_BY_BETA[beta]:
            print(f"[beta={beta} idx={idx}] loading Ours (one-pass EP) + multichain HMC + VI from cache...")
            hmc = hmc_sweep.load_cached(beta, idx)
            ours = mp_sweep.load_cached(beta, idx)
            vi = vi_sweep.load_cached(beta, idx)
            admitted = float(np.mean((hmc["rhat"] < RHAT_MAX) & (hmc["ess"] > ESS_MIN)))
            rec = dict(beta=beta, idx=idx, hmc_admitted=admitted, hmc_accept=np.round(hmc["accept_rates"], 3).tolist(),
                       t=ours["t"], Ours=_slim(ours), HMC=_slim(hmc), VI=_slim(vi))
            for m in METHODS[1:]:
                assert np.array_equal(rec[m]["true_u"], rec["Ours"]["true_u"]), "methods must share the trajectory"
            if admitted < HMC_MIN_ADMITTED:
                print(f"[beta={beta} idx={idx}] EXCLUDED from every method: only {admitted:.0%} of the HMC "
                      f"coordinates converged (chain acceptance rates "
                      f"{np.round(hmc['accept_rates'], 3).tolist()}), so there is no usable reference. "
                      "For the record, on this run alone (vs truth): "
                      + ", ".join(f"{name} NLL {nll(r['true_u'], *_moments(r)):.1f}, "
                                  f"cov@90 {100 * coverage(standardized_residuals(r), 0.9):.1f}%"
                                  for name, r in (("Ours", ours), ("HMC", hmc), ("VI", vi))))
                excluded.append(rec)
                continue
            records.append(rec)
    return records, excluded


def _select(records: list[dict], beta: float | None = None) -> list[dict]:
    return [r for r in records if beta is None or r["beta"] == beta]


def dropped_runs_text(excluded: list[dict]) -> str:
    """One line naming the runs the HMC gate dropped (with what the gate saw), or 'none'."""
    if not excluded:
        return "none"
    return "; ".join(f"b={r['beta']:g} idx {r['idx']} ({r['hmc_admitted']:.0%} of the HMC coordinates admitted, "
                     f"chain acceptance {r['hmc_accept']})" for r in excluded)


def n_per_beta_text(records: list[dict]) -> str:
    """'n = 10 of 10 at b=0.1, ..., 3 of 5 at b=7': admitted runs out of the canonical trajectories."""
    return "n = " + ", ".join(f"{len(_select(records, b))} of {len(IDXS_BY_BETA[b])} at b={b:g}" for b in BETAS)


def _moments(m: dict) -> tuple[np.ndarray, np.ndarray]:
    """(predictive mean, floored predictive variance) of one method on one run."""
    return m["pred_u"], np.maximum(m["pred_var"], MIN_VAR)


# ----------------------------------------------------------------------------- model level (vs truth)


def standardized_residuals(m: dict) -> np.ndarray:
    mean, var = _moments(m)
    return ((m["true_u"] - mean) / np.sqrt(var)).ravel()


def pooled_z(records: list[dict], method: str, beta: float | None = None) -> np.ndarray:
    """Standardized residuals of `method`, pooled over the runs of one beta or of all betas."""
    return np.concatenate([standardized_residuals(r[method]) for r in _select(records, beta)])


def z_quantile(level: float) -> float:
    """Half-width of the central `level` interval of N(0,1), in standard deviations."""
    return float(stats.norm.ppf(0.5 + level / 2))


def coverage(z: np.ndarray, level: float) -> float:
    return float(np.mean(np.abs(z) <= z_quantile(level)))


def nll(true_u: np.ndarray, pred_u: np.ndarray, pred_var: np.ndarray) -> float:
    var = np.maximum(pred_var, MIN_VAR)
    return float(np.mean(0.5 * np.log(2 * np.pi * var) + (true_u - pred_u) ** 2 / (2 * var)))


def crps_gaussian(true_u: np.ndarray, pred_u: np.ndarray, pred_var: np.ndarray) -> float:
    """Closed-form CRPS of a Gaussian predictive (Gneiting & Raftery 2007, eq. 5)."""
    std = np.sqrt(np.maximum(pred_var, MIN_VAR))
    z = (true_u - pred_u) / std
    crps = std * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))
    return float(np.mean(crps))


def mean_score(records: list[dict], method: str, score_fn, beta: float | None = None) -> float:
    """Mean of the per-run scores of `method` (every run has the same grid, so this equals the
    pooled grid-point mean), over one beta or all."""
    return float(np.mean([score_fn(r[method]["true_u"], r[method]["pred_u"], r[method]["pred_var"])
                          for r in _select(records, beta)]))


def rmse_of_mean(m: dict) -> float:
    """RMSE (root-mean-square error) of one method's posterior mean against the truth on one run;
    only used in the table comment, to show that Ours and HMC share the same residuals."""
    return float(np.sqrt(np.mean((m["true_u"] - m["pred_u"]) ** 2)))


# ----------------------------------------------------------------------------- inference level (vs HMC)


def hmc_coverage(rec: dict, method: str, level: float) -> float:
    """Probability mass of the HMC predictive (Gaussian with HMC's stored moments at each grid
    point) inside the method's central `level` interval, averaged over the grid. For
    method == "HMC" this is `level` by construction."""
    m_q, v_q = _moments(rec[method])
    m_h, v_h = _moments(rec["HMC"])
    s_q, s_h = np.sqrt(v_q), np.sqrt(v_h)
    z = z_quantile(level)
    return float(np.mean(stats.norm.cdf((m_q + z * s_q - m_h) / s_h) - stats.norm.cdf((m_q - z * s_q - m_h) / s_h)))


def hmc_expected_nll(rec: dict, method: str) -> float:
    """Expected NLL of the method's Gaussian predictive under the HMC predictive, averaged over
    the grid. For method == "HMC" this is HMC's own entropy, the attainable floor."""
    m_q, v_q = _moments(rec[method])
    m_h, v_h = _moments(rec["HMC"])
    return float(np.mean(0.5 * np.log(2 * np.pi * v_q) + (v_h + (m_h - m_q) ** 2) / (2 * v_q)))


def mean_inference_stat(records: list[dict], method: str, fn, beta: float | None = None, **kw) -> float:
    return float(np.mean([fn(r, method, **kw) for r in _select(records, beta)]))


# ----------------------------------------------------------------------------- panels


def _style(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, fontsize=24)
    ax.set_ylabel(ylabel, fontsize=24)
    ax.set_title(title, fontsize=26)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=18)


def _beta_axis(ax) -> None:
    ax.set_xscale("log")
    ax.set_xticks(BETAS)
    ax.set_xticklabels([f"${b:g}$" for b in BETAS])
    ax.xaxis.set_minor_formatter(NullFormatter())


def _log_y_axis(ax, tick_candidates: list[float], values: list[float],
                top_margin: float = LOG_Y_MARGIN) -> tuple[float, float]:
    """Log y-axis whose limits bracket every plotted value by the factor LOG_Y_MARGIN below and
    `top_margin` above (larger when a legend has to sit above the data), with the
    candidate ticks inside those limits labelled as plain numbers (no 10^k) and no minor tick
    labels. Returns the limits. A mean NLL can be negative in principle (a run whose predictive
    is both sharp and right, e.g. b=7 idx 2343 for Ours and HMC), so the log axis needs every
    plotted mean to be positive; if one is not, the axis falls back to a symmetric-log scale
    (linear near zero, logarithmic beyond 1) and says so on the axis."""
    lo, hi = min(values), max(values)
    if lo <= 0:
        print(f"WARNING: a plotted mean score is {lo:.3g} <= 0; using a symlog y-axis instead of a log axis")
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_title(ax.get_title() + r"$\ \mathrm{(symlog\ axis)}$", fontsize=26)
        return ax.get_ylim()
    ylim = (lo / LOG_Y_MARGIN, hi * top_margin)
    yticks = [t for t in tick_candidates if ylim[0] <= t <= ylim[1]]
    assert len(yticks) >= 2, f"too few candidate ticks inside {ylim}: extend the candidate list"
    ax.set_yscale("log")
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    return ylim


def _method_handle(method: str, label: str | None = None) -> Line2D:
    return Line2D([], [], color=COLORS[method], marker=MARKERS[method], markersize=12, lw=2.5, label=label or method)


def _level_handles() -> list[Line2D]:
    return [Line2D([], [], color="k", ls=LEVEL_STYLES[lev], lw=2.5, label=rf"$\mathrm{{{lev:.0%}\ interval}}$".replace("%", r"\%"))
            for lev in HEADLINE_LEVELS]


def plot_coverage(ax, records: list[dict], tag: str) -> None:
    ax.plot([0, 1], [0, 1], "k--", linewidth=2, alpha=0.5, label=r"$\mathrm{ideal}$")
    for i, level in enumerate(HEADLINE_LEVELS):  # vertical guides at the headline nominal levels
        ax.axvline(level, color="gray", ls=":", lw=1.5, alpha=0.8,
                   label=r"$\mathrm{nominal\ 90\%/95\%}$" if i == 0 else None)
    for method in METHODS:
        z_all = pooled_z(records, method)
        ax.plot(NOMINAL_LEVELS, [coverage(z_all, lev) for lev in NOMINAL_LEVELS], marker=MARKERS[method],
                color=COLORS[method], markersize=14, linewidth=2.5, label=method)
    ax.set_ylim(0, 1)
    _style(ax, r"$\mathrm{nominal\ coverage}$", r"$\mathrm{empirical\ coverage\ of\ the\ truth}$",
           r"$\mathrm{(%s)\ Coverage\ of\ the\ truth\ (pooled)}$" % tag)
    ax.legend(fontsize=18, loc="upper left")


def plot_residuals(ax, records: list[dict], tag: str) -> None:
    xs = np.linspace(-4, 4, 400)
    ax.plot(xs, stats.norm.pdf(xs), "k--", linewidth=2, alpha=0.6, label=r"$\mathcal{N}(0,1)$")
    for method in METHODS:
        z_all = pooled_z(records, method)
        z_all = z_all[np.abs(z_all) < Z_HIST_MAX]  # drop rare extreme outliers, keeps the histogram readable
        ax.hist(z_all, bins=80, range=(-4, 4), density=True, histtype="step", linewidth=2.5,
                color=COLORS[method], label=method)
    _style(ax, r"$\mathrm{standardized\ residual}$", r"$\mathrm{density}$",
           r"$\mathrm{(%s)\ Standardized\ residuals\ (vs.\ truth)}$" % tag)
    ax.legend(fontsize=18)


def plot_scoring_rule(ax, records: list[dict], score_fn, title: str, ylabel: str,
                      tick_candidates: list[float], legend_kw: dict, top_margin: float = LOG_Y_MARGIN) -> None:
    values = {method: [mean_score(records, method, score_fn, beta) for beta in BETAS] for method in METHODS}
    for method in METHODS:
        ax.plot(BETAS, values[method], marker=MARKERS[method], color=COLORS[method],
                markersize=14, linewidth=2.5, label=method)
    _beta_axis(ax)
    _style(ax, r"$b$", ylabel, title)
    ylim = _log_y_axis(ax, tick_candidates, [v for vals in values.values() for v in vals], top_margin)
    print(f"{title}: y-axis {ylim[0]:.4g} to {ylim[1]:.4g}; per-b means "
          + "; ".join(f"{m} {np.round(values[m], 4).tolist()}" for m in METHODS))
    ax.legend(fontsize=18, **legend_kw)


def plot_truth_coverage_per_b(ax, records: list[dict], tag: str) -> None:
    for method in METHODS:
        for level in HEADLINE_LEVELS:
            vals = [coverage(pooled_z(records, method, beta), level) for beta in BETAS]
            ax.plot(BETAS, vals, ls=LEVEL_STYLES[level], marker=MARKERS[method], color=COLORS[method],
                    markersize=12, linewidth=2.5)
    for level in HEADLINE_LEVELS:
        ax.axhline(level, color="k", ls=LEVEL_STYLES[level], lw=1.5, alpha=0.5)
    ax.text(BETAS[0], 0.925, r"$\mathrm{nominal}$", fontsize=16, color="k", alpha=0.7, va="center")
    _beta_axis(ax)
    ax.set_ylim(0, 1.0)
    _style(ax, r"$b$", r"$\mathrm{empirical\ coverage\ of\ the\ truth}$",
           r"$\mathrm{(%s)\ 90\%%/95\%%\ coverage\ of\ the\ truth}$" % tag)
    # the band between the data (below 0.4; HMC's 95% line reaches 0.35 at b=0.1) and the nominal lines
    # (0.9 / 0.95) is empty: put the legend there
    ax.legend(handles=[_method_handle(m) for m in METHODS] + _level_handles(), fontsize=16, ncol=2,
              loc="upper center", bbox_to_anchor=(0.5, 0.86))


def plot_hmc_coverage_per_b(ax, records: list[dict], tag: str) -> None:
    for method in ("Ours", "VI"):
        for level in HEADLINE_LEVELS:
            vals = [mean_inference_stat(records, method, hmc_coverage, beta, level=level) for beta in BETAS]
            ax.plot(BETAS, vals, ls=LEVEL_STYLES[level], marker=MARKERS[method], color=COLORS[method],
                    markersize=12, linewidth=2.5)
    for level in HEADLINE_LEVELS:  # HMC against itself is nominal by construction
        ax.axhline(level, color=COLORS["HMC"], ls=LEVEL_STYLES[level], lw=2.5, alpha=0.9)
    _beta_axis(ax)
    ax.set_ylim(0, 1.0)
    _style(ax, r"$b$", r"$\mathrm{coverage\ of\ the\ HMC\ predictive}$",
           r"$\mathrm{(%s)\ 90\%%/95\%%\ coverage\ of\ the\ HMC\ predictive}$" % tag)
    handles = [_method_handle("Ours"), _method_handle("VI"),
               Line2D([], [], color=COLORS["HMC"], lw=2.5, label=r"$\mathrm{HMC\ (nominal)}$")] + _level_handles()
    # same empty band as in the coverage-of-the-truth panel: the data stay below 0.6 (VI 0.59 at b=0.4, Ours 0.51 at b=7), the HMC lines
    # sit at 0.9 / 0.95
    ax.legend(handles=handles, fontsize=16, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.86))


def make_calibration_figure(records: list[dict], output_path: Path) -> None:
    """The full six-panel calibration_comparison.pdf (CALIBRATION_ALL_PANELS=1 only)."""
    plt.rcParams["mathtext.fontset"] = "cm"
    fig, axes = plt.subplots(3, 2, figsize=(17, 22))
    plot_coverage(axes[0, 0], records, "a")
    plot_residuals(axes[0, 1], records, "b")
    # (c): every corner of the data range holds a marker (HMC at b=0.1 and b=7 sit at the bottom, Ours at
    # b=1-2 and VI at b=7 at the top), so the legend goes above the data at b <= 0.4 (Ours' 2450-2670 there
    # is the highest point under it) with extra head-room on the log axis
    plot_scoring_rule(axes[1, 0], records, nll, r"$\mathrm{(c)\ NLL\ (vs.\ truth)}$", r"$\mathrm{NLL}$",
                      NLL_YTICK_CANDIDATES, dict(loc="upper left"), top_margin=4.0)
    plot_scoring_rule(axes[1, 1], records, crps_gaussian, r"$\mathrm{(d)\ CRPS\ (vs.\ truth)}$", r"$\mathrm{CRPS}$",
                      CRPS_YTICK_CANDIDATES, dict(loc="upper left"))
    plot_truth_coverage_per_b(axes[2, 0], records, "e")
    plot_hmc_coverage_per_b(axes[2, 1], records, "f")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path}")


# ----------------------------------------------------------------------------- table and log


def _n_runs(records: list[dict], beta: float | None) -> int:
    return len(_select(records, beta))


def _b_label(beta: float | None) -> str:
    return "pooled" if beta is None else f"{beta:g}"


def model_level_headline(records: list[dict], method: str, beta: float | None) -> tuple[float, float, float]:
    """(cov@90 in %, cov@95 in %, NLL) of `method` against the truth, over the runs of one beta or all."""
    z = pooled_z(records, method, beta)
    return 100 * coverage(z, 0.9), 100 * coverage(z, 0.95), mean_score(records, method, nll, beta)


def _with_failed_run_lines(records: list[dict], excluded: list[dict]) -> list[str]:
    """Comment lines with the model-level headline numbers of Ours and VI when the runs the HMC
    gate excluded are put back (n then equals the full canonical set at the affected b), and how
    much the exclusion moved Ours' pooled numbers."""
    if not excluded:
        return [r"% (no run was excluded, so the table already holds every run)"]
    everything = records + excluded
    betas = sorted({r["beta"] for r in excluded}) + [None]
    which = ", ".join(f"b={r['beta']:g} idx {r['idx']}" for r in excluded)
    n_text = ", ".join(f"{_n_runs(everything, b)} at b={b:g}" for b in betas if b is not None)
    lines = [f"% For reference, the model-level numbers WITH the excluded run(s) put back ({which}; n = {n_text}, "
             f"{_n_runs(everything, None)} pooled;",
             r"% the excluded run(s) have no usable HMC reference, so only Ours and VI are listed):"]
    for method in ("Ours", "VI"):
        parts = []
        for beta in betas:
            c90, c95, nll_val = model_level_headline(everything, method, beta)
            parts.append(f"{_b_label(beta)} (n={_n_runs(everything, beta)}): cov@90 {c90:.1f}%, cov@95 {c95:.1f}%, NLL {nll_val:.1f}")
        lines.append(f"%   {method:4s} {';  '.join(parts)}")
    c90_in, c95_in, nll_in = model_level_headline(everything, "Ours", None)
    c90_ex, c95_ex, nll_ex = model_level_headline(records, "Ours", None)
    lines.append(rf"% Dropping the run(s) therefore moves Ours' pooled cov@90 / cov@95 by {c90_ex - c90_in:+.2f} / {c95_ex - c95_in:+.2f} pp "
                 rf"and its pooled NLL by {100 * (nll_ex / nll_in - 1):+.0f}%.")
    return lines


def rel_l2_of_mean(m: dict) -> float:
    """Relative L2 error of one method's posterior mean against the truth on one run; ~1 means the
    mean has collapsed to about zero (VI at b=7). Only used in the table comment."""
    return float(np.linalg.norm(m["true_u"] - m["pred_u"]) / np.linalg.norm(m["true_u"]))


def write_calibration_table(records: list[dict], excluded: list[dict], path: Path) -> None:
    """LaTeX table of part 1 (calibration_comparison_table.tex), two blocks. Block 1 (model level, vs truth): per beta and pooled,
    coverage at each nominal level (panels a/e), std of the standardized residuals (panel b),
    mean NLL / CRPS (panels c/d), for Ours, HMC and VI. Block 2 (inference level, vs the HMC
    predictive): coverage at 90%/95% (panel f) and the expected NLL under the HMC predictive,
    for Ours and VI, plus HMC against itself (nominal coverage, entropy floor). `excluded` (the
    runs dropped by the HMC gate) only feeds the block 1 comment."""
    # numbers for the block 1 comment: Ours and HMC share the residuals of the mean, not the variances
    rmse = {m: np.array([rmse_of_mean(r[m]) for r in records]) for m in ("Ours", "HMC")}
    rmse_gap = float(np.max(np.abs(rmse["Ours"] - rmse["HMC"])))
    std_ratio = float(np.median(np.concatenate([np.sqrt(_moments(r["HMC"])[1] / _moments(r["Ours"])[1]).ravel()
                                                for r in records])))
    nll_ours, nll_hmc = mean_score(records, "Ours", nll), mean_score(records, "HMC", nll)
    vi_rel_l2 = ", ".join(f"{np.mean([rel_l2_of_mean(r['VI']) for r in _select(records, b)]):.2f} at b={b:g}" for b in BETAS)
    lines = [
        r"% Auto-generated by figures_onepass/advection/calibration_comparison.py (part 1; the full table of the",
        r"% six-panel calibration_comparison.pdf, whose panel letters are used below; panels (d) and (e) are",
        r"% panels (a) and (b) of the shipped calibration_variance.pdf).",
        r"% Ours = one-pass EP (expectation propagation) posterior predictive; HMC = 4-chain Hamiltonian Monte",
        r"% Carlo reference (3000 pooled draws); VI = mean-field (every weight gets its own independent Gaussian)",
        *[f"% {line}" for line in textwrap.wrap(
            "variational posterior. Per b: the canonical trajectories whose HMC reference converged, out of "
            "the ten canonical trajectories idxs(10) at every b: "
            f"{n_per_beta_text(records)}. A run enters only if at least {HMC_MIN_ADMITTED:.0%} of its HMC "
            f"coordinates have split-R-hat < {RHAT_MAX} (split-R-hat compares the spread between the chain halves "
            f"with the spread within them, 1 = agreement) and bulk ESS > {ESS_MIN:.0f} (bulk ESS = effective sample "
            "size, the number of independent draws the correlated chains are worth); a run that fails is "
            f"dropped for EVERY method. Dropped by the HMC gate: {dropped_runs_text(excluded)}. 'pooled' = every "
            "admitted run of every b listed above (b=7 included). Every statistic uses the full 1024 x 201 "
            "(x,t) grid of each run; sigma = sqrt(max(pred_var, 1e-10)) (the floor never binds). VI at b=7: "
            "its predictive mean collapses; mean relative L2 error of the VI posterior mean "
            f"against the truth = {vi_rel_l2} (about 1 = the mean is about zero), which is why its NLL / CRPS "
            "at b=7 are far larger than the other methods'. The VI numbers are shown in full.", 100)],
        r"%",
        r"% BLOCK 1, model-level calibration (against the TRUE solution). z = (u_true - u_pred)/sigma.",
        r"% Coverage = fraction of grid points with |z| <= z_level (z_level = 0.674 / 1.282 / 1.645 / 1.960 at",
        r"% 50 / 80 / 90 / 95%); std(z) 'all' is the std of z over every grid point, std(z) '|z|<10' the std over",
        r"% the grid points with |z| < 10 only (the points the residual histogram of panel (b) is drawn from);",
        r"% NLL (negative log-likelihood) = mean over grid points of 0.5 ln(2 pi sigma^2) + z^2/2 (Gaussian",
        r"% predictive); CRPS (continuous ranked probability score, a proper scoring rule in the units of u) =",
        r"% mean closed-form Gaussian CRPS. NLL and CRPS are means of the per-run means (what panels c/d plot).",
        r"% Every method, exact HMC included, is far under-covered here: the dominant error is model-form",
        r"% error shared by all three methods (the truncated basis cannot represent the true solution, and",
        r"% that error is not in the noise model beta_ic / beta_pde), so the coverage columns measure the",
        r"% model more than the inference method. The NLL gap between the methods is nevertheless due to the",
        r"% inference variances: the per-run RMSE (root-mean-square error) of the posterior mean against the",
        rf"% truth is the same for Ours and HMC to 2-3 decimals (largest difference {rmse_gap:.4f} over the {len(records)} runs,",
        rf"% RMSE from {rmse['HMC'].min():.4f} to {rmse['HMC'].max():.4f}), while HMC's predictive std is a median {std_ratio:.1f}x Ours' over the",
        rf"% grid, so the pooled NLL of {nll_ours:.0f} (Ours) vs {nll_hmc:.0f} (HMC) comes from the variances alone.",
        *_with_failed_run_lines(records, excluded),
        r"\begin{tabular}{l r r rrrr rr rr}",
        r"\toprule",
        r" & & & \multicolumn{4}{c}{Coverage of the truth at nominal level} & \multicolumn{2}{c}{std$(z)$} & & \\",
        r"\cmidrule(lr){4-7}\cmidrule(lr){8-9}",
        r"Method & $b$ & $n$ & 50\% & 80\% & \textbf{90\%} & \textbf{95\%} & all & $|z|<10$ & \textbf{NLL} & CRPS \\",
        r"\midrule",
    ]
    for method in METHODS:
        for beta in BETAS + [None]:
            z = pooled_z(records, method, beta)
            z_hist = z[np.abs(z) < Z_HIST_MAX]
            cov = " & ".join(f"{100 * coverage(z, lev):.1f}\\%" for lev in NOMINAL_LEVELS)
            lines.append(f"{method} & {_b_label(beta)} & {_n_runs(records, beta)} & {cov} & {z.std():.1f} & "
                         f"{z_hist.std():.2f} & {mean_score(records, method, nll, beta):.1f} & "
                         f"{mean_score(records, method, crps_gaussian, beta):.4f} \\\\")
        if method != METHODS[-1]:
            lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\medskip",
        "",
        r"% BLOCK 2, inference-level calibration (against the EXACT posterior predictive, i.e. HMC's), which",
        r"% isolates the approximation error of the inference method. The HMC predictive at each grid point is",
        r"% taken as Gaussian with HMC's stored mean m_H and variance v_H; with the method's mean m_q and",
        r"% variance v_q = s_q^2 (z = 1.645 / 1.960 at 90 / 95%):",
        r"%   coverage@level = Phi((m_q + z s_q - m_H)/s_H) - Phi((m_q - z s_q - m_H)/s_H), the probability mass",
        r"%                    of the HMC predictive inside the method's interval, averaged over the grid;",
        r"%   E-NLL = 0.5 ln(2 pi v_q) + (v_H + (m_H - m_q)^2)/(2 v_q), the expected negative log-likelihood of",
        r"%           the method's Gaussian predictive under the HMC predictive, averaged over the grid.",
        r"% Both are means of the per-run means. HMC against itself gives the nominal coverage and its own",
        r"% entropy 0.5 ln(2 pi v_H) + 0.5, the floor of E-NLL. Below nominal coverage = the method's",
        r"% intervals are too narrow and/or its mean is off; E-NLL minus the floor = the excess in nats (natural-log units).",
        *[f"% {line}" for line in textwrap.wrap(VERIFICATION_NOTE, 100)],
        r"\begin{tabular}{l r r rr rr}",
        r"\toprule",
        r" & & & \multicolumn{2}{c}{Coverage of the HMC predictive} & \multicolumn{2}{c}{Expected NLL under HMC} \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"Method & $b$ & $n$ & \textbf{90\%} & \textbf{95\%} & \textbf{E-NLL} & E-NLL $-$ floor \\",
        r"\midrule",
    ]
    for method in ("Ours", "VI", "HMC"):
        for beta in BETAS + [None]:
            covs = " & ".join(f"{100 * mean_inference_stat(records, method, hmc_coverage, beta, level=lev):.1f}\\%"
                              for lev in HEADLINE_LEVELS)
            enll = mean_inference_stat(records, method, hmc_expected_nll, beta)
            floor = mean_inference_stat(records, "HMC", hmc_expected_nll, beta)
            label = "HMC (itself)" if method == "HMC" else method
            lines.append(f"{label} & {_b_label(beta)} & {_n_runs(records, beta)} & {covs} & {enll:.2f} & "
                         f"{enll - floor:.2f} \\\\")
        if method != "HMC":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines))
    print(f"Saved table to {path}")


def print_calibration_summary(records: list[dict], excluded: list[dict]) -> None:
    n_floor = sum(int(np.sum(r[m]["pred_var"] < MIN_VAR)) for r in records for m in METHODS)
    print(f"\n{len(records)} runs used ({n_per_beta_text(records)}); grid points at the variance floor: {n_floor}")
    print(f"Dropped by the HMC gate (>= {HMC_MIN_ADMITTED:.0%} of the coordinates with split-R-hat < {RHAT_MAX} and "
          f"bulk ESS > {ESS_MIN:.0f}): {dropped_runs_text(excluded)}")
    print("VI relative L2 error of the posterior mean vs truth (||u_true - u_pred|| / ||u_true||; about 1 = the mean is about zero), per b: "
          + ", ".join(f"{np.mean([rel_l2_of_mean(r['VI']) for r in _select(records, b)]):.2f} at b={b:g}" for b in BETAS))
    print("\nModel level (vs the true solution), per b and pooled:")
    print(f"  {'method':6s} {'b':>6s}  cov@90   cov@95   NLL        CRPS")
    for method in METHODS:
        for beta in BETAS + [None]:
            z = pooled_z(records, method, beta)
            print(f"  {method:6s} {_b_label(beta):>6s}  {100 * coverage(z, 0.9):5.1f}%   {100 * coverage(z, 0.95):5.1f}%   "
                  f"{mean_score(records, method, nll, beta):8.1f}   {mean_score(records, method, crps_gaussian, beta):.4f}")
    rmse = {m: [rmse_of_mean(r[m]) for r in records] for m in ("Ours", "HMC")}
    print(f"\nPer-run RMSE of the posterior mean vs truth, Ours vs HMC: largest |difference| "
          f"{np.max(np.abs(np.subtract(rmse['Ours'], rmse['HMC']))):.4f} (HMC RMSE from {min(rmse['HMC']):.4f} to {max(rmse['HMC']):.4f})")
    if excluded:
        print("\nModel level WITH the excluded run(s) put back (Ours and VI; the excluded run(s) have no usable HMC reference):")
        for line in _with_failed_run_lines(records, excluded)[2:]:
            print("  " + line.lstrip("% "))
    print("\nInference level (vs the HMC predictive), per b and pooled:")
    print(f"  {'method':6s} {'b':>6s}  cov@90   cov@95   E-NLL     floor (HMC entropy)")
    for method in ("Ours", "VI"):
        for beta in BETAS + [None]:
            c90 = mean_inference_stat(records, method, hmc_coverage, beta, level=0.9)
            c95 = mean_inference_stat(records, method, hmc_coverage, beta, level=0.95)
            enll = mean_inference_stat(records, method, hmc_expected_nll, beta)
            floor = mean_inference_stat(records, "HMC", hmc_expected_nll, beta)
            print(f"  {method:6s} {_b_label(beta):>6s}  {100 * c90:5.1f}%   {100 * c95:5.1f}%   {enll:7.2f}   {floor:7.2f}")


# ============================================================================= PART 2: variance consistency
# (the panel letters in the comments refer to variance_consistency.pdf)

VC_MIN_VAR = 1e-14  # floor on the predictive variance of part 2 (part 1 uses MIN_VAR = 1e-10; neither binds)
N_SIGMA_BINS = 14
MIN_BIN_POINTS = 100  # panel (b) bins with fewer pooled grid points are dropped
LOG_PAD = 0.25  # panel (b): each axis pads the plotted bins' range by this fraction of its log-span per side
N_EDGE_BINS = 4  # the VI wave-speed caveat quotes the top and bottom this-many bins
VI_HIGH_BETAS = [1.0, 2.0, 7.0]  # the wave speeds whose share of the edge bins the VI caveat quotes
# (legend label, predictive-std key, absolute-error key) for every series of panel (b)
RELIABILITY_SERIES = (("Ours", "sig_mp", "err"), ("HMC", "sig_hmc", "err_hmc"), ("VI", "sig_vi", "err_vi"))
RELIABILITY_STYLES = {"Ours": dict(color="#4a3aa7", marker="o"),
                      "HMC": dict(color="#eb6834", marker="s"),
                      "VI": dict(color="#008300", marker="^")}
# Font sizes, line widths and marker sizes of the part-2 panels: VC_STYLE is that of the full
# variance_consistency.pdf (7.25 x 5.75 inch panels); MERGED_VC_STYLE matches panels (a)/(b) of the
# shipped figure (8.5 x 7.33 inch panels, the sizes of _style in part 1).
# plain_log_ticks: label the log y axis of the std-ratio panel with plain numbers at the tick font
# size (as panel (a) of the shipped figure) instead of matplotlib's small 'k x 10^-1' minor labels.
# b_fmt: how a wave speed is written in the legends and tick labels ('b=1.0' in the full figure,
# 'b=1' in the shipped figure, as in the b axes of its panels (a)/(b)).
VC_STYLE = dict(label=19, title=20, tick=13, legend=13, legend_rel=14, xticklabel=15, lw=2.0, s=42,
                plain_log_ticks=False, b_fmt="$b={}$")
MERGED_VC_STYLE = dict(label=24, title=26, tick=18, legend=18, legend_rel=18, xticklabel=18, lw=2.5, s=70,
                       plain_log_ticks=True, b_fmt="$b={:g}$")
RATIO_YTICK_CANDIDATES = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.5, 2, 3, 5]  # plain_log_ticks only


def variance_records(records: list[dict], excluded: list[dict]) -> tuple[list[dict], list[dict]]:
    """(records, dropped) of part 2, derived from the loaded records of part 1 (the same runs, the
    same HMC gate): per admitted run the absolute errors of the three predictive means against the
    truth and the three predictive stds (floored at VC_MIN_VAR); per run the gate dropped, its
    reason (admitted share and chain acceptance rates)."""
    vrecords = []
    for r in records:
        mp, hmc, vi = r["Ours"], r["HMC"], r["VI"]
        vrecords.append(dict(beta=r["beta"], idx=r["idx"], t=r["t"],
                             err=np.abs(mp["true_u"] - mp["pred_u"]),
                             sig_mp=np.sqrt(np.maximum(mp["pred_var"], VC_MIN_VAR)),
                             err_hmc=np.abs(hmc["true_u"] - hmc["pred_u"]),
                             sig_hmc=np.sqrt(np.maximum(hmc["pred_var"], VC_MIN_VAR)),
                             err_vi=np.abs(vi["true_u"] - vi["pred_u"]),
                             sig_vi=np.sqrt(np.maximum(vi["pred_var"], VC_MIN_VAR))))
    dropped = [dict(beta=r["beta"], idx=r["idx"],
                    reason=f"{r['hmc_admitted']:.0%} admitted, chain acceptance {r['hmc_accept']}") for r in excluded]
    empty = [b for b in BETAS if not _by_beta(vrecords, b)]
    if empty:
        raise RuntimeError(f"no admitted run at b in {_beta_set_text(empty)} (all dropped: "
                           f"{dropped_text(dropped)}); every wave speed in BETAS needs at least one run")
    print(f"{len(vrecords)} runs used ({', '.join(n_used_text(vrecords, b) for b in BETAS)}); "
          f"dropped by the HMC gate: {dropped_text(dropped)}")
    return vrecords, dropped


def n_used_text(records, beta) -> str:
    """'k of n at b=..': runs admitted at one wave speed out of the trajectories it was run on."""
    return f"{len(_by_beta(records, beta))} of {len(IDXS_BY_BETA[beta])} at b={beta:g}"


def dropped_text(dropped) -> str:
    """The drop list as prose: 'b=.. idx .. (..% admitted, chain acceptance ..)' per run the HMC gate
    rejected, or 'none'."""
    if not dropped:
        return "none"
    return "; ".join(f"b={d['beta']:g} idx {d['idx']} ({d['reason']})" for d in dropped)


def _by_beta(records, beta):
    return [r for r in records if r["beta"] == beta]


def growth_curves(records, beta):
    """(t, curves): per-trajectory spatial-mean predictive std, shape (n_runs, n_t), panel (a)."""
    rs = _by_beta(records, beta)
    return rs[0]["t"], np.stack([r["sig_mp"].mean(axis=0) for r in rs])


def _pooled(records, sig_key, err_key):
    """Every grid point of every record: (predicted std, |error|, wave speed of its run)."""
    sig = np.concatenate([r[sig_key].ravel() for r in records])
    err = np.concatenate([r[err_key].ravel() for r in records])
    beta = np.concatenate([np.full(r[sig_key].size, r["beta"]) for r in records])
    return sig, err, beta


def _bin_index(sig):
    """Per-method quantile bin (0-based) of every pooled grid point, N_SIGMA_BINS equal-count bins."""
    edges = np.quantile(sig, np.linspace(0, 1, N_SIGMA_BINS + 1))
    edges[-1] *= 1.0000001
    return np.digitize(sig, edges) - 1


def reliability_bins(records, sig_key, err_key):
    """Panel (b) points: list of (bin, RMS predicted std, empirical RMSE) over the per-method
    quantile bins of the pooled grid points; bins with fewer than MIN_BIN_POINTS points are dropped."""
    sig, err, _ = _pooled(records, sig_key, err_key)
    which = _bin_index(sig)
    out = []
    for b in range(N_SIGMA_BINS):
        m = which == b
        if m.sum() < MIN_BIN_POINTS:
            continue
        out.append((b, float(np.sqrt(np.mean(sig[m] ** 2))), float(np.sqrt(np.mean(err[m] ** 2)))))
    return out


def bin_beta_shares(records, sig_key, err_key) -> dict[int, dict[float, float]]:
    """For every kept bin of one method, the share of its points that come from each wave speed."""
    sig, _, beta = _pooled(records, sig_key, err_key)
    which = _bin_index(sig)
    return {b: {bb: float(np.mean(beta[which == b] == bb)) for bb in BETAS}
            for b in range(N_SIGMA_BINS) if (which == b).sum() >= MIN_BIN_POINTS}


def edge_bin_share(shares: dict[int, dict[float, float]], betas: list[float]) -> tuple[float, float]:
    """(share in the top N_EDGE_BINS bins, share in the bottom N_EDGE_BINS bins) of the points that
    come from the wave speeds `betas`, each pooled over those bins."""
    keys = sorted(shares)
    top = np.mean([sum(shares[k][bb] for bb in betas) for k in keys[-N_EDGE_BINS:]])
    bottom = np.mean([sum(shares[k][bb] for bb in betas) for k in keys[:N_EDGE_BINS]])
    return float(top), float(bottom)


def rmse_vs_truth(records, err_key, beta=None) -> float:
    """Root mean square of |true - predicted mean| over every grid point of the runs at one wave
    speed (all runs if beta is None)."""
    rs = records if beta is None else _by_beta(records, beta)
    return float(np.sqrt(np.mean(np.concatenate([r[err_key].ravel() ** 2 for r in rs]))))


def ratio_vs_time(records, beta):
    """(t, curve): median over trajectories of the per-trajectory spatial median of
    sigma_Ours/sigma_HMC at each time, panel (c)."""
    rs = _by_beta(records, beta)
    ratios = np.stack([np.median(r["sig_mp"] / r["sig_hmc"], axis=0) for r in rs])  # (n_runs, n_t)
    return rs[0]["t"], np.median(ratios, axis=0)


def ratio_per_traj(records, beta):
    """Per-trajectory median of sigma_Ours/sigma_HMC over the whole grid, panel (d) points."""
    return [float(np.median(r["sig_mp"] / r["sig_hmc"])) for r in _by_beta(records, beta)]


def plot_growth(ax, records, beta_colors, tag: str, style: dict) -> None:
    for beta in BETAS:
        t, curves = growth_curves(records, beta)
        ax.plot(t, curves.mean(axis=0), color=beta_colors[beta], lw=style["lw"], label=style["b_fmt"].format(beta))
        ax.fill_between(t, curves.min(axis=0), curves.max(axis=0), color=beta_colors[beta], alpha=0.18)
    ax.set_xlabel(r"$t$", fontsize=style["label"])
    ax.set_ylabel(r"$\mathrm{predictive\ std}$", fontsize=style["label"])
    ax.set_title(r"$\mathrm{(%s)\ Uncertainty\ vs.\ distance\ from\ data}$" % tag, fontsize=style["title"])
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=style["tick"])
    ax.legend(fontsize=style["legend"], ncol=2)


def _padded_log_range(values) -> tuple[float, float]:
    """(lo, hi) enclosing `values`, widened by LOG_PAD of their log10-span on each side."""
    lo, hi = np.log10(min(values)), np.log10(max(values))
    pad = LOG_PAD * (hi - lo)
    return float(10 ** (lo - pad)), float(10 ** (hi + pad))


def plot_reliability(ax, records, tag: str, style: dict) -> None:
    xs_all, ys_all = [], []
    for label, sig_key, err_key in RELIABILITY_SERIES:
        pts = reliability_bins(records, sig_key, err_key)
        xs = [x for _, x, _ in pts]
        ys = [y for _, _, y in pts]
        ax.plot(xs, ys, lw=style["lw"], markersize=8, label=label, **RELIABILITY_STYLES[label])
        xs_all += xs
        ys_all += ys
    # x limits = the plotted bins' own (padded) std range; the y axis runs from the bottom of that
    # x range up to the padded RMSE range, so the y=x reference stays visible across the whole
    # x range (every bin sits well above it; the offset IS the finding).
    xlim, ylim = _padded_log_range(xs_all), _padded_log_range(ys_all)
    ylim = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
    ax.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], "k--", lw=1.6, alpha=0.6, label=r"$y=x$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(r"$\mathrm{RMS\ predicted\ std\ (bin)}$", fontsize=style["label"])
    ax.set_ylabel(r"$\mathrm{empirical\ RMSE\ (bin)}$", fontsize=style["label"])
    ax.set_title(r"$\mathrm{(%s)\ Predicted\ std\ vs.\ actual\ error}$" % tag, fontsize=style["title"])
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(labelsize=style["tick"])
    ax.legend(fontsize=style["legend_rel"])


def plot_ratio_vs_time(ax, records, beta_colors, tag: str, style: dict) -> None:
    for beta in BETAS:
        t, curve = ratio_vs_time(records, beta)
        ax.plot(t, curve, color=beta_colors[beta], lw=style["lw"], label=style["b_fmt"].format(beta))
    ax.axhline(1.0, color="k", linestyle=":", lw=1.6, alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlabel(r"$t$", fontsize=style["label"])
    ax.set_ylabel(r"$\sigma_{\mathrm{Ours}} / \sigma_{\mathrm{HMC}}$", fontsize=style["label"])
    ax.set_title(r"$\mathrm{(%s)\ Std\ ratio\ vs.\ HMC\ over\ time}$" % tag, fontsize=style["title"])
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=style["tick"])
    ax.legend(fontsize=style["legend"], ncol=2)


def plot_ratio_per_traj(ax, records, beta_colors, tag: str, style: dict) -> None:
    rng = np.random.RandomState(2)
    for i, beta in enumerate(BETAS):
        vals = ratio_per_traj(records, beta)
        ax.scatter(np.full(len(vals), float(i)) + rng.uniform(-0.06, 0.06, len(vals)), vals,
                   s=style["s"], alpha=0.75,
                   facecolors=beta_colors[beta],
                   edgecolors="black", linewidths=0.6)
        ax.plot([i - 0.1, i + 0.1], [np.median(vals)] * 2, color="black", lw=style["lw"])
    ax.axhline(1.0, color="k", linestyle="--", lw=1.6, alpha=0.6)
    ax.set_yscale("log")
    ax.set_xticks(range(len(BETAS)))
    ax.set_xticklabels([style["b_fmt"].format(b) for b in BETAS], fontsize=style["xticklabel"])
    ax.set_ylabel(r"$\sigma_{\mathrm{Ours}} / \sigma_{\mathrm{HMC}}$", fontsize=style["label"])
    ax.set_title(r"$\mathrm{(%s)\ Std\ ratio\ per\ trajectory}$" % tag, fontsize=style["title"])
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(labelsize=style["tick"])
    if style["plain_log_ticks"]:
        lo, hi = ax.get_ylim()
        yticks = [v for v in RATIO_YTICK_CANDIDATES if lo <= v <= hi]
        assert len(yticks) >= 2, f"too few candidate ticks inside {(lo, hi)}: extend RATIO_YTICK_CANDIDATES"
        ax.set_yticks(yticks)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(lo, hi)


def grid_ratio_quantiles(records) -> tuple[float, float, float]:
    """(median, Q1, Q3) of sigma_Ours/sigma_HMC over every grid point of every record."""
    ratio = np.concatenate([(r["sig_mp"] / r["sig_hmc"]).ravel() for r in records])
    return tuple(float(q) for q in np.quantile(ratio, [0.5, 0.25, 0.75]))


def _sci(x: float) -> str:
    m, e = f"{x:.2e}".split("e")
    return rf"${m}{{\times}}10^{{{int(e)}}}$"


N_COLS = 1 + 3 * len(RELIABILITY_SERIES)  # block (b) is the widest: bin index + 3 cells per method


def _row(cells) -> str:
    """One tabular row; shorter blocks are padded with empty cells so every block shares the tabular."""
    cells = list(cells)
    return " & ".join(cells + [""] * (N_COLS - len(cells))) + r" \\"


def _section(text: str) -> str:
    return rf"\multicolumn{{{N_COLS}}}{{l}}{{{text}}}\\"


def _beta_set_text(betas) -> str:
    return "{" + ", ".join(f"{b:g}" for b in betas) + "}"


def vi_caveat_lines(records) -> list[str]:
    """Table comment lines for the wave-speed sorting of the VI reliability bins, with the numbers
    behind the claim: the share of the top / bottom N_EDGE_BINS VI bins that comes from b in
    VI_HIGH_BETAS and every method's RMSE vs. the truth at the largest wave speed."""
    top, bottom = edge_bin_share(bin_beta_shares(records, "sig_vi", "err_vi"), VI_HIGH_BETAS)
    b_max = BETAS[-1]
    rmse = {label: rmse_vs_truth(records, err_key, b_max) for label, _, err_key in RELIABILITY_SERIES}
    per_b = ["; ".join(f"b={b:g}: " + ", ".join(f"{label} {rmse_vs_truth(records, ek, b):.3f}"
                                               for label, _, ek in RELIABILITY_SERIES) for b in bs)
             for bs in (BETAS[:2], BETAS[2:])]
    return [
        r"%     Caveat on the VI series: the pooled quantile bins sort strongly by wave speed for VI, whose",
        r"%     predicted std grows with b while its mean gets worse with b (RMSE vs. the truth at "
        rf"b={b_max:g}: VI {rmse['VI']:.3f},",
        rf"%     Ours {rmse['Ours']:.3f}, HMC {rmse['HMC']:.3f}; at b=7 the VI mean collapses); VI's top "
        rf"{N_EDGE_BINS} bins draw {top:.0%} of their",
        rf"%     points from b in {_beta_set_text(VI_HIGH_BETAS)} and its bottom {N_EDGE_BINS} bins {bottom:.0%}. "
        r"The steep VI rise of RMSE with predicted",
        r"%     std is therefore a between-wave-speed effect, not the predicted std tracking the error within a",
        r"%     run. VI is shown in panel (b) at every wave speed (no bin is left out); the log axes are set from",
        r"%     the plotted bins of all three methods. RMSE vs. the truth per b, over all grid points of the runs",
        rf"%     at that b: {per_b[0]};",
        rf"%     {per_b[1]}.",
    ]


def write_variance_table(records, dropped, path: Path) -> None:
    """LaTeX table of part 2 (variance_consistency_table.tex) mirroring the four panels of the full
    variance_consistency.pdf, one section each, in one 10-column
    tabular: (a) mean-std curve samples and its t=end band, (b) every reliability bin for Ours,
    HMC and VI (the only block that needs all 10 columns; (a), (c), (d) use the first 7 and
    leave the rest empty), (c) ratio-vs-time curve samples and range, (d) per-trajectory and
    grid-point ratio spread. Every number is explained in the leading comment lines, which also
    state the runs used per wave speed and the runs the HMC gate dropped (from `dropped`)."""
    t = records[0]["t"]
    i_samp = [0, len(t) // 2, len(t) - 1]  # t = 0, mid, end: the same three columns in (a) and (c)
    t_heads = [rf"$t={t[i]:g}$" for i in i_samp]
    n_pts = records[0]["sig_mp"].size
    n_series = len(RELIABILITY_SERIES)
    lines = [
        r"% Auto-generated by figures_onepass/advection/calibration_comparison.py (part 2; the full table of the",
        r"%     four-panel variance_consistency.pdf, whose panel letters are",
        r"%     used below; panels (a) and (d) are panels (c) and (d) of the shipped calibration_variance.pdf).",
        rf"% {len(records)} runs used ({', '.join(n_used_text(records, b) for b in BETAS)}): the ten canonical",
        r"%     trajectories idxs(10) (one seeded draw) at every b, b = 7 included (one b = 7 run costs about an",
        rf"%     hour of HMC and two hours of VI). Each predictive grid has {n_pts} points; sigma = sqrt(max(pred_var, {VC_MIN_VAR:g})).",
        r"% Ours = one-pass EP (expectation propagation) posterior; HMC = multi-chain Hamiltonian Monte Carlo",
        r"%     reference (all chains pooled); VI = mean-field (independent-weight) Gaussian variational inference baseline.",
        rf"% Run gate: a run enters only if at least {HMC_MIN_ADMITTED:.0%} of its HMC eigenbasis coordinates (the rotated",
        rf"%     weight coordinates all methods share) have split-R-hat < {RHAT_MAX:g} and bulk ESS > {ESS_MIN:g} (the standard MCMC",
        r"%     convergence diagnostics: chain agreement and effective sample count), the same rule as",
        r"%     calibration_comparison_table.tex (one shared loader). A run that fails it is dropped for EVERY method,",
        r"%     so every block uses the same runs. Dropped by the HMC gate:",
        *(f"%         {line}" for line in dropped_text(dropped).split("; ")),
        r"% (a) Ours' predictive std, spatial mean at each t, then the mean over the trajectories of that b, sampled",
        r"%     at three times; growth = the t=end value over the t=0 value; band min / band max = the smallest /",
        r"%     largest trajectory's spatial-mean std at t=end (the shaded band of panel (a) at its right edge).",
        rf"% (b) bins are per-method quantile bins ({N_SIGMA_BINS} equal-count bins) of the pooled grid points,",
        r"%     so bin k holds the same share of each method's points but not the same points (the edges",
        r"%     differ between Ours, HMC and VI). RMS std = root mean square of the predicted std in the bin,",
        r"%     RMSE = root mean square of |true - predicted mean| in the bin; RMSE/std = 1 means calibrated",
        rf"%     within that bin, > 1 overconfident, < 1 underconfident. Bins with fewer than {MIN_BIN_POINTS} points are '--'.",
        *vi_caveat_lines(records),
        r"% (c) sigma_Ours / sigma_HMC at every grid point; spatial median at each t per trajectory, then the",
        r"%     median over the trajectories of that b, sampled at three times; min / max over t of that curve",
        r"%     and their ratio (1 = the same factor at every time).",
        r"% (d) left: per trajectory, the median of sigma_Ours / sigma_HMC over its whole grid; median / min / max",
        r"%     of those per-trajectory values over the trajectories of that b. right: median and quartiles of",
        r"%     sigma_Ours / sigma_HMC over every grid point of every trajectory of that b. 'all' pools every b.",
        rf"% Blocks (a), (c), (d) use the first 7 of the {N_COLS} columns; their trailing cells are empty.",
        r"\begin{tabular}{l " + " ".join(["rrr"] * n_series) + "}",
        r"\toprule",
        _section(r"\emph{(a) Mean predictive std of Ours vs.\ time}"),
        _row(["$b$", *t_heads, rf"growth $t{{=}}{t[i_samp[-1]]:g}$ / $t{{=}}{t[0]:g}$", "band min", "band max"]),
        r"\midrule",
    ]
    for b in BETAS:
        _, curves = growth_curves(records, b)
        mean = curves.mean(axis=0)
        lines.append(_row([f"{b:g}", *(_sci(mean[i]) for i in i_samp), f"{mean[i_samp[-1]] / mean[0]:.2f}",
                           _sci(curves.min(axis=0)[i_samp[-1]]), _sci(curves.max(axis=0)[i_samp[-1]])]))
    lines += [
        r"\midrule",
        _section(r"\emph{(b) Reliability bins} (all runs pooled)"),
        "bin & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{label}}}" for label, _, _ in RELIABILITY_SERIES) + r" \\",
        "".join(rf"\cmidrule(lr){{{2 + 3 * j}-{4 + 3 * j}}}" for j in range(n_series)),
        _row([""] + ["RMS std", "RMSE", "RMSE/std"] * n_series),
        r"\midrule",
    ]
    bins = {label: {k: (x, y) for k, x, y in reliability_bins(records, sk, ek)}
            for label, sk, ek in RELIABILITY_SERIES}
    for k in range(N_SIGMA_BINS):
        cells = []
        for label, _, _ in RELIABILITY_SERIES:
            if k in bins[label]:
                x, y = bins[label][k]
                cells += [_sci(x), _sci(y), f"{y / x:.2f}"]
            else:
                cells += ["--"] * 3
        lines.append(_row([f"{k + 1}", *cells]))
    lines += [
        r"\midrule",
        _section(r"\emph{(c) Median $\sigma_{\mathrm{Ours}}/\sigma_{\mathrm{HMC}}$ vs.\ time}"),
        _row(["$b$", *t_heads, "min over $t$", "max over $t$", "max/min"]),
        r"\midrule",
    ]
    for b in BETAS:
        _, curve = ratio_vs_time(records, b)
        lines.append(_row([f"{b:g}", *(f"{curve[i]:.3f}" for i in i_samp), f"{curve.min():.3f}",
                           f"{curve.max():.3f}", f"{curve.max() / curve.min():.2f}"]))
    lines += [
        r"\midrule",
        _section(r"\emph{(d) $\sigma_{\mathrm{Ours}}/\sigma_{\mathrm{HMC}}$ per trajectory and per grid point}"),
        _row(["$b$", "median", "min", "max", "grid median", "grid Q1", "grid Q3"]),
        r"\midrule",
    ]
    all_vals = []
    for b in BETAS:
        vals = ratio_per_traj(records, b)
        all_vals += vals
        med, q1, q3 = grid_ratio_quantiles(_by_beta(records, b))
        lines.append(_row([f"{b:g}", f"{np.median(vals):.3f}", f"{min(vals):.3f}", f"{max(vals):.3f}",
                           f"{med:.3f}", f"{q1:.3f}", f"{q3:.3f}"]))
    med, q1, q3 = grid_ratio_quantiles(records)
    lines.append(_row(["all", f"{np.median(all_vals):.3f}", f"{min(all_vals):.3f}", f"{max(all_vals):.3f}",
                       f"{med:.3f}", f"{q1:.3f}", f"{q3:.3f}"]))
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines))
    print(f"Saved table to {path}")


def _beta_colors() -> dict[float, tuple]:
    """One viridis colour per wave speed, the colours of the part-2 panels."""
    cmap = plt.get_cmap("viridis")
    return {b: cmap(i / max(1, len(BETAS) - 1)) for i, b in enumerate(BETAS)}


def make_variance_figure(records, output_path: Path) -> None:
    """The full four-panel variance_consistency.pdf (CALIBRATION_ALL_PANELS=1 only)."""
    plt.rcParams["mathtext.fontset"] = "cm"
    beta_colors = _beta_colors()
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 11.5))
    plot_growth(axes[0, 0], records, beta_colors, "a", VC_STYLE)
    plot_reliability(axes[0, 1], records, "b", VC_STYLE)
    plot_ratio_vs_time(axes[1, 0], records, beta_colors, "c", VC_STYLE)
    plot_ratio_per_traj(axes[1, 1], records, beta_colors, "d", VC_STYLE)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path}")


def print_variance_summary(records) -> None:
    med, q1, q3 = grid_ratio_quantiles(records)
    print(f"Pooled sigma ratio Ours/HMC: median {med:.2f}, IQR [{q1:.2f}, {q3:.2f}]")
    for label, sig_key, err_key in RELIABILITY_SERIES:
        pts = reliability_bins(records, sig_key, err_key)
        ratios = [y / x for _, x, y in pts]
        print(f"Reliability bins {label}: {len(pts)} bins, RMS std {pts[0][1]:.2e}..{pts[-1][1]:.2e}, "
              f"RMSE {min(y for _, _, y in pts):.2e}..{max(y for _, _, y in pts):.2e}, "
              f"RMSE/std {min(ratios):.1f}..{max(ratios):.1f}")
    print("RMSE vs truth per b (all grid points of the runs at that b): "
          + "; ".join(f"b={b:g}: " + ", ".join(f"{label} {rmse_vs_truth(records, ek, b):.3f}"
                                              for label, _, ek in RELIABILITY_SERIES) for b in BETAS))
    for label, sig_key, err_key in RELIABILITY_SERIES:
        shares = bin_beta_shares(records, sig_key, err_key)
        top, bottom = edge_bin_share(shares, VI_HIGH_BETAS)
        print(f"Share of points from b in {_beta_set_text(VI_HIGH_BETAS)} per {label} bin (low to high std): "
              + " ".join(f"{sum(shares[k][bb] for bb in VI_HIGH_BETAS):.0%}" for k in sorted(shares))
              + f" (top {N_EDGE_BINS} bins {top:.0%}, bottom {N_EDGE_BINS} bins {bottom:.0%})")


# ============================================================================= the shipped figure


def make_figure(records: list[dict], vrecords: list[dict], output_path: Path) -> None:
    """calibration_variance.pdf, 2 x 2 with the panel size of the full calibration_comparison.pdf
    (8.5 x 7.33 inch): (a) CRPS per b, (b) coverage of the truth per b (part 1); (c) Ours'
    predictive std vs time, (d) per-trajectory std ratio (part 2, drawn with MERGED_VC_STYLE)."""
    plt.rcParams["mathtext.fontset"] = "cm"
    beta_colors = _beta_colors()
    fig, axes = plt.subplots(2, 2, figsize=(17, 44 / 3))
    plot_scoring_rule(axes[0, 0], records, crps_gaussian, r"$\mathrm{(a)\ CRPS\ (vs.\ truth)}$", r"$\mathrm{CRPS}$",
                      CRPS_YTICK_CANDIDATES, dict(loc="upper left"))
    plot_truth_coverage_per_b(axes[0, 1], records, "b")
    plot_growth(axes[1, 0], vrecords, beta_colors, "c", MERGED_VC_STYLE)
    plot_ratio_per_traj(axes[1, 1], vrecords, beta_colors, "d", MERGED_VC_STYLE)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path}")


def write_merged_table(records: list[dict], excluded: list[dict], vrecords: list[dict], path: Path) -> None:
    """Companion table of calibration_variance.pdf, one block per panel: (a) mean CRPS against the
    truth per b and pooled (Ours, HMC, VI); (b) coverage of the truth by the 90% / 95% intervals per
    b and pooled; (c) Ours' mean predictive std at t = 0 / mid / end, its growth and the t=end band;
    (d) the per-trajectory medians of sigma_Ours/sigma_HMC (median, min, max over the
    trajectories) and the grid-point quantiles. The same numbers, with more context, are in
    calibration_comparison_table.tex (blocks behind a, b) and variance_consistency_table.tex (c, d)."""
    n_cols = 7

    def row(cells) -> str:
        cells = list(cells)
        return " & ".join(cells + [""] * (n_cols - len(cells))) + r" \\"

    def section(text: str) -> str:
        return rf"\multicolumn{{{n_cols}}}{{l}}{{{text}}}\\"

    t = vrecords[0]["t"]
    i_samp = [0, len(t) // 2, len(t) - 1]  # t = 0, mid, end, as in variance_consistency_table.tex
    lines = [
        r"% Auto-generated by figures_onepass/advection/calibration_comparison.py",
        r"% Companion table of calibration_variance.pdf (one block per panel). Ours = one-pass EP; HMC = 4-chain",
        r"% Hamiltonian Monte Carlo reference (3000 pooled draws); VI = mean-field Gaussian variational inference.",
        *[f"% {line}" for line in textwrap.wrap(
            f"Runs: the ten canonical trajectories idxs(10) at every b whose HMC reference converged ({n_per_beta_text(records)}; "
            f"at least {HMC_MIN_ADMITTED:.0%} of the HMC coordinates with split-R-hat < {RHAT_MAX} and bulk ESS > "
            f"{ESS_MIN:.0f}); dropped for every method by that gate: {dropped_runs_text(excluded)}. Every statistic "
            "uses the full 1024 x 201 (x,t) grid of each run and is measured against the TRUE solution "
            "((a)-(b)), shows Ours' predictive std ((c)) or compares Ours' std with HMC's ((d)).", 100)],
        r"% (a) CRPS (continuous ranked probability score) = mean over the runs of the per-run mean closed-form",
        r"%     Gaussian CRPS (sigma = sqrt(max(pred_var, 1e-10))), per b and pooled over all b; = column CRPS of",
        r"%     calibration_comparison_table.tex (block 1).",
        r"% (b) coverage = share of all grid points of the runs whose truth lies inside the nominal 90% / 95% Gaussian",
        r"%     predictive interval; = columns 90% / 95% of block 1 of calibration_comparison_table.tex.",
        r"% (c) Ours' predictive std, spatial mean at each t, then the mean over the trajectories of that b, at three",
        r"%     times; growth = t=end over t=0; band min / max = smallest / largest trajectory's spatial-mean std at",
        r"%     t=end; = block (a) of variance_consistency_table.tex.",
        r"% (d) per trajectory the median of sigma_Ours/sigma_HMC over its grid; median / min / max over the",
        r"%     trajectories of that b; grid median / Q1 / Q3 = over every grid point of those trajectories ('all'",
        r"%     pools every b); = block (d) of variance_consistency_table.tex.",
        r"\begin{tabular}{l rrrrrr}",
        r"\toprule",
        section(r"\emph{(a) CRPS vs.\ truth}"),
        row(["$b$", *METHODS]),
        r"\midrule",
    ]
    for beta in BETAS + [None]:
        lines.append(row([_b_label(beta), *(f"{mean_score(records, m, crps_gaussian, beta):.4f}" for m in METHODS)]))
    lines += [
        r"\midrule",
        section(r"\emph{(b) Coverage of the truth at nominal 90\% / 95\%}"),
        row(["$b$", *(f"{m} {lev:.0%}".replace("%", r"\%") for m in METHODS for lev in HEADLINE_LEVELS)]),
        r"\midrule",
    ]
    for beta in BETAS + [None]:
        lines.append(row([_b_label(beta), *(f"{100 * coverage(pooled_z(records, m, beta), lev):.1f}\\%"
                                            for m in METHODS for lev in HEADLINE_LEVELS)]))
    lines += [
        r"\midrule",
        section(r"\emph{(c) Mean predictive std of Ours vs.\ time}"),
        row(["$b$", *(rf"$t={t[i]:g}$" for i in i_samp), rf"growth $t{{=}}{t[i_samp[-1]]:g}$ / $t{{=}}{t[0]:g}$",
             "band min", "band max"]),
        r"\midrule",
    ]
    for b in BETAS:
        _, curves = growth_curves(vrecords, b)
        mean = curves.mean(axis=0)
        lines.append(row([f"{b:g}", *(_sci(mean[i]) for i in i_samp), f"{mean[i_samp[-1]] / mean[0]:.2f}",
                          _sci(curves.min(axis=0)[i_samp[-1]]), _sci(curves.max(axis=0)[i_samp[-1]])]))
    lines += [
        r"\midrule",
        section(r"\emph{(d) $\sigma_{\mathrm{Ours}}/\sigma_{\mathrm{HMC}}$ per trajectory and per grid point}"),
        row(["$b$", "median", "min", "max", "grid median", "grid Q1", "grid Q3"]),
        r"\midrule",
    ]
    all_vals = []
    for b in BETAS:
        vals = ratio_per_traj(vrecords, b)
        all_vals += vals
        med, q1, q3 = grid_ratio_quantiles(_by_beta(vrecords, b))
        lines.append(row([f"{b:g}", f"{np.median(vals):.3f}", f"{min(vals):.3f}", f"{max(vals):.3f}",
                          f"{med:.3f}", f"{q1:.3f}", f"{q3:.3f}"]))
    med, q1, q3 = grid_ratio_quantiles(vrecords)
    lines.append(row(["all", f"{np.median(all_vals):.3f}", f"{min(all_vals):.3f}", f"{max(all_vals):.3f}",
                      f"{med:.3f}", f"{q1:.3f}", f"{q3:.3f}"]))
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines))
    print(f"Saved table to {path}")


def main() -> None:
    records, excluded = collect_records()
    vrecords, vdropped = variance_records(records, excluded)
    make_figure(records, vrecords, OUTPUT_PDF)
    if ALL_PANELS:
        print(f"NOTE: CALIBRATION_ALL_PANELS=1: also writing the full figures {CAL_PDF.name} and {VC_PDF.name}")
        make_calibration_figure(records, CAL_PDF)
        make_variance_figure(vrecords, VC_PDF)
    write_merged_table(records, excluded, vrecords, OUTPUT_TEX)
    write_calibration_table(records, excluded, CAL_TEX)
    write_variance_table(vrecords, vdropped, VC_TEX)
    print_calibration_summary(records, excluded)
    print()
    print_variance_summary(vrecords)


if __name__ == "__main__":
    main()
