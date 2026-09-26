"""
Does the one-pass EP (expectation propagation) method still work on a slightly wider network, on the
PDEBench Fisher-KPP (reaction-diffusion) dataset u_t = nu u_xx + rho u (1 - u)? The single-hidden-unit
model u = g(w.phi) is compared with u = sum_h a_h g(w_h.phi), H = 2, 3 and 4 hidden units with FIXED
readout weights a (H=2: (1,-1); H=3: (1,-1,0.5); H=4: (1,-1,0.5,-0.5)), trained by the SAME one-pass EP
schedule, priors, noise levels, tolerances and rho-continuation ladder (the reaction strength is raised
in four rungs s*rho, s = 0.25, 0.5, 0.75, 1, each rung a full EP training of at most 300 epochs started
from the previous rung's posterior; runner: pde/diffusion/mp_pinn_reacdiff_wide_run.jl; the H=1
points are the existing single-neuron artifacts of pde/diffusion/mp_pinn_reacdiff_dataset_run.jl).
This is the reaction-diffusion replica of figures_onepass/advection/wider_model.py.

The grid: 16 cells (settings) nu in {0.5, 1, 2, 5} x rho in {1, 2, 5, 10}, five canonical trajectories
per cell (idxs(5), the first five of the canonical seeded draw (pde/advection/trajectory_selection.py)).
Where the advection figure varies the wave speed b, this one varies the reaction strength rho on the
x axis (the parameter that drives the nonlinearity and the failure modes),
and the diffusion coefficient nu is pooled: every rho point aggregates the 4 nu x 5 trajectories = 20
runs that the gate admits, and where a panel shows per-run points the marker shape gives nu (nu = 0.5:
circle, 1: square, 2: diamond, 5: triangle; the H of a point is its colour). The x-axis label is the
physical rho; the model works with the normalized values rho_norm = rho T_basis = 2 rho and
nu_norm = nu T_basis / L^2 = 2 nu. Everything is evaluated against the TRUTH on the full (x,t) grid,
where the truth is the float64 re-solve of the nominal PDE from the trajectory's initial condition
(u_ref, pde/diffusion/reacdiff_reference.py), never the stored PDEBench trajectory (its
late-time freeze is a data artifact).

The shipped figure (wider_model.pdf) has four panels, 2 x 2, as the advection one:

  (a) rel-L2 error of the predictive mean vs truth: one point per run, line through the per-rho medians.
  (b) empirical coverage of the nominal 90% (solid) and 95% (dashed) predictive intervals vs the truth,
      pooled over every grid point of the admitted runs per rho (the formula of
      calibration_comparison.py).
  (c) predictive NLL (negative log-likelihood of the truth under the Gaussian predictive): points per
      run (the run's mean over the grid), line (with markers) through the per-rho mean of those, as in
      calibration_comparison.py.
  (d) training wall time per run: points per run, line through medians.
  The grey series with x markers in (b) and (c) is the multichain HMC posterior of the H=1 model.

Two further panels are not in wider_model.pdf (as in the advection figure): their numbers are in the
companion table, and WIDER_MODEL_ALL_PANELS=1 builds the extended six-panel figure, with its letters
(a)-(f), into wider_model_all_panels.pdf (wider_model.pdf is then left untouched); their plotting
functions are plot_mean_std and plot_neuron_usage:

  - mean predictive std at the last time step, panel (b) of the six-panel figure (the spatial mean of
    sqrt(pred_var) at t=end, then the across-run mean, i.e. the t=end value of calibration_variance.pdf panel (c));
    points = per run. The dashed grey line is the multichain HMC posterior of the H=1 model.
  - neuron usage, panel (f) of the six-panel figure: each neuron's share of the predictive mean,
    rms(a_h E[g(z_h)]) / rms(u_pred) over the grid (rms = root mean square), median across the admitted
    runs of the rho column. It shows whether every neuron contributes or one neuron carries
    everything; the dotted lines mark the equal share 1/H and the small digits at the right end of
    the lines are the neuron indices (1..H, in the order of a).

How to read the coverage and NLL panels (and the mean-std panel of the six-panel figure): the wide predictive variance is the mean-field sum over
independent neurons, Var[u] = sum_h a_h^2 Var[g(z_h)], because the EP beliefs factorize over all
weights of all neurons and carry no covariance between neurons. Wherever the predictive std or the
coverage rises, or the NLL falls, with H, that is mainly an independence effect (more independent
variance terms are added) and must not be read as better-calibrated inference: the true posterior of
the wide model has strongly anti-correlated neuron pairs, whose negative covariance the
mean-field sum leaves out. On this dataset the direction is not even that: most H=2 runs hit the epoch
cap (see the capped/div counts in the table), their coverage is far below H=1 and their NLL far above
it, so the wide-model uncertainty columns also reflect a ladder that did not converge. The HMC series is
grey because it is the posterior of a different model, the single-neuron model (its exact posterior), and
is shown for scale only, not as the reference for the wide models.

The companion table (wider_model_table.tex) holds every number: for EVERY (nu, rho, H) cell and, below
each rho block, the pooled row over the four nu (which says what it pools): the median / min / max
rel-L2, the mean std at t=end, coverage at 90% / 95%, NLL, epochs and training time, n per cell, plus
the per-neuron usage block (share of the mean, pre-activation range, median posterior std) per cell
and pooled per rho. The 'train [s]' entry of the HMC rows is the MAP (maximum a posteriori) warm start
plus the summed wall time of the four chains, which run one after the other; it is not comparable to
EP epochs.

Run set: the same rule as the advection calibration_comparison.py. A run (cell, trajectory) enters only
when its multichain HMC reference converged (at least HMC_MIN_ADMITTED of the eigenbasis (the rotated
weight coordinates all methods share) coordinates pass split-R-hat (the chain-agreement diagnostic)
< 1.05 and bulk ESS (effective sample size) > 100). A run that fails this gate is dropped for EVERY
model, H=1..4 and the HMC line alike, so every statistic here uses one run set. Which runs are dropped
is NOT hard-coded: it is computed from the cached HMC diagnostics at run time, printed in the log
(together with the dropped run's own per-model numbers, for the record) and written into the table
comment with the resulting n per cell, so the list follows the caches. Every set of records compared
in eigen coordinates is checked to share one eigenbasis (assert_same_basis on the Ours, HMC and wide
artifacts of the same trajectory). A run counts as CAPPED when at least one rung of its ladder hit the 300-epoch
limit (so the total can reach 4 x 300 = 1200 epochs; the per-rung counts are in the log), and as
DIVERGED when it is capped and its per-epoch max KL (Kullback-Leibler divergence, in nats) still
exceeded 1 within its last 50 epochs, i.e. the final rung blew up instead of settling. Median epochs
are rounded half up to an integer everywhere (table, log and printed summary). The NLL panel uses a log y
axis while every mean NLL is positive and falls back to symlog (linear near zero, log beyond
+-NLL_LINTHRESH) otherwise, so a negative NLL (small predictive std, smaller error) is never dropped
silently. Reads cached artifacts only; never trains, and raises naming the file if an artifact is
missing. Writes wider_model.log (a copy of everything printed) next to the outputs.

Run: python figures_onepass/diffusion/wider_model.py [--png out.png]
     (six-panel figure: WIDER_MODEL_ALL_PANELS=1 in front of the same command)

Environment variables that restrict the build to a subset of the cells or widths (both are
announced in the log and in the table comment; the shipped figure always uses the defaults):
  REACDIFF_CELLS   comma-separated "nu:rho" pairs (default = all 16 cells), e.g. REACDIFF_CELLS=0.5:1,0.5:2.
                   A missing artifact of a chosen cell still raises; cells are never skipped silently.
  REACDIFF_WIDTHS  comma-separated H values (default 1,2,3,4), e.g. REACDIFF_WIDTHS=1,2.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.legend import Legend
import numpy as np
from scipy import stats

FIGURES_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURES_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "pde" / "diffusion"))
sys.path.insert(0, str(REPO_ROOT / "pde" / "advection"))

import reacdiff_baselines_sweep as sweep
from trajectory_selection import idxs

OUTPUT_PDF = FIGURES_DIR / "wider_model.pdf"
OUTPUT_PDF_ALL_PANELS = FIGURES_DIR / "wider_model_all_panels.pdf"   # WIDER_MODEL_ALL_PANELS=1 only
OUTPUT_TEX = FIGURES_DIR / "wider_model_table.tex"
OUTPUT_LOG = FIGURES_DIR / "wider_model.log"

# Panels in figure order; the letters (a), (b), ... follow the order. "mean_std" and "neuron_usage"
# appear only in the six-panel figure (WIDER_MODEL_ALL_PANELS=1, see the docstring).
PANELS_SHIPPED = ("rel_l2", "coverage", "nll", "train_time")
PANELS_ALL = ("rel_l2", "mean_std", "coverage", "nll", "train_time", "neuron_usage")
ALL_PANELS = os.environ.get("WIDER_MODEL_ALL_PANELS", "").strip() == "1"

NUS_ALL = list(sweep.NUS)      # [0.5, 1.0, 2.0, 5.0]
RHOS_ALL = list(sweep.RHOS)    # [1.0, 2.0, 5.0, 10.0]
CELLS_ALL = [(nu, rho) for rho in RHOS_ALL for nu in NUS_ALL]
IDXS = idxs(5)                 # the five canonical trajectories of every cell
WIDTHS_ALL = [1, 2, 3, 4]


def _cells_from_env() -> list[tuple[float, float]]:
    """REACDIFF_CELLS restricts the cells (a build from the artifacts of a subset of them)."""
    raw = os.environ.get("REACDIFF_CELLS", "").strip()
    if not raw:
        return list(CELLS_ALL)
    cells = []
    for item in raw.split(","):
        if not item.strip():
            continue
        try:
            nu_s, rho_s = item.split(":")
            cell = (float(nu_s), float(rho_s))
        except ValueError:
            raise ValueError(f"REACDIFF_CELLS={raw!r}: expected comma-separated nu:rho pairs") from None
        if cell not in CELLS_ALL:
            raise ValueError(f"REACDIFF_CELLS={raw!r}: unknown cell {item!r}; nu in {NUS_ALL}, rho in {RHOS_ALL}")
        cells.append(cell)
    if not cells:
        raise ValueError(f"REACDIFF_CELLS={raw!r}: no cell given")
    return sorted(set(cells), key=lambda c: (c[1], c[0]))


def _widths_from_env() -> list[int]:
    """REACDIFF_WIDTHS restricts the widths H (a build from the artifacts of a subset of them)."""
    raw = os.environ.get("REACDIFF_WIDTHS", "").strip()
    if not raw:
        return list(WIDTHS_ALL)
    widths = sorted({int(s) for s in raw.split(",") if s.strip()})
    if not widths or any(H not in WIDTHS_ALL for H in widths):
        raise ValueError(f"REACDIFF_WIDTHS={raw!r}: allowed widths are {WIDTHS_ALL}")
    return widths


CELLS = _cells_from_env()
CELLS_RESTRICTED = CELLS != CELLS_ALL
RHOS = sorted({rho for _, rho in CELLS})                       # the x axis
NUS_AT = {rho: sorted(nu for nu, r in CELLS if r == rho) for rho in RHOS}   # the nu pooled into each rho point
WIDTHS = _widths_from_env()
WIDTHS_RESTRICTED = WIDTHS != WIDTHS_ALL
MIN_VAR = 1e-10           # same floor on the predictive variance as calibration_comparison.py
COVERAGE_LEVELS = [0.9, 0.95]
RHAT_MAX, ESS_MIN = 1.05, 100.0  # per-coordinate HMC convergence gate (as in calibration_comparison.py)
HMC_MIN_ADMITTED = 0.9           # a run is used only if at least this fraction of its HMC coordinates pass
RUNG_CAP = 300                   # epoch limit of ONE rung of the ladder (4 rungs -> at most 1200 epochs)
KL_DIVERGED = 1.0                # a capped run whose max KL exceeds this within its last 50 epochs diverged
TRUTH = "u_ref"                  # the float64 nominal-PDE reference is the truth for every metric
COLORS = {1: "#4a3aa7", 2: "#2b8cbe", 3: "#1b9e77", 4: "#c51b7d"}   # H=1 keeps the "Ours" purple
NU_MARKERS = {0.5: "o", 1.0: "s", 2.0: "D", 5.0: "^"}               # per-run points: marker = nu
HMC_STYLE = dict(color="#7f7f7f", linestyle="--", marker="x", markersize=8, lw=1.8)
# Multiplicative offsets of the per-run points on the log-rho axis: a fixed factor is a fixed visual
# distance on a log axis, so the four H groups stay equally separated at every rho; inside an H group
# the four nu are spread by a smaller factor so the 20 points of a group do not sit on one vertical line.
X_JITTER = {1: 0.88, 2: 0.96, 3: 1.04, 4: 1.13}
X_JITTER_NU = {0.5: 0.975, 1.0: 0.992, 2.0: 1.008, 5.0: 1.025}
X_JITTER_F = {2: 0.85, 3: 1.0, 4: 1.18}  # wider offsets in the neuron-usage panel only, so the neuron-index labels fit between the H groups
X_PAD = 1.4           # log-rho axis limits: RHOS[0] / X_PAD .. RHOS[-1] * X_PAD, so the jittered points at rho=1 and rho=10 are not clipped
LABEL_GAP_F = 0.022   # minimum vertical distance (data units of the neuron-usage panel) between two neuron-index labels of one H
F_YMIN_TOP = 0.72     # neuron-usage panel: the y axis reaches at least this high; it is raised further (LEGEND_BAND) when a share is larger
LEGEND_BAND = 0.30    # rel-L2, mean-std, NLL, training-time and neuron-usage panels: fraction of the axes height kept free of data at the top, where the legend sits
LEGEND_BAND_NU = 0.16 # rel-L2, mean-std and training-time panels: fraction of the axes height kept free at the bottom for the nu-marker legend (the top band holds the H (+ HMC) legend, which fills its width with four H)
NLL_LINTHRESH = 10.0  # the NLL panel falls back from log to symlog (linear inside +-NLL_LINTHRESH, log outside) when a mean NLL is <= 0
# Markers on the H lines of the NLL panel: one shape for every H, which the colour tells
# apart. The advection figure uses one shape per H, but here those shapes already mean nu (NU_MARKERS).
NLL_LINE_MARKER = dict(marker="o", markersize=9)


class _Tee:
    """Duplicates everything printed into the .log file next to the outputs."""

    def __init__(self, path: Path):
        self.terminal, self.file = sys.stdout, open(path, "w")

    def write(self, s):
        self.terminal.write(s)
        self.file.write(s)

    def flush(self):
        self.terminal.flush()
        self.file.flush()


def _cell(nu: float, rho: float) -> str:
    return f"nu={nu:g} rho={rho:g}"


def load_one(nu: float, rho: float, idx: int, H: int) -> dict:
    """Cached artifact only (no training is ever triggered here); H=1 = the single-neuron artifact.
    Every record gets `rel_l2` = the error against the truth (u_ref) and `capped`."""
    if H == 1:
        path = sweep.cache_path("Ours", nu, rho, idx)
        if not path.exists():
            raise FileNotFoundError(f"missing single-neuron artifact {path} ({_cell(nu, rho)}, idx={idx})")
        r = sweep.load_cached("Ours", nu, rho, idx)
        r.update(H=1, a=np.array([1.0]), neuron_out_rms=np.array([float(np.sqrt(np.mean(r["pred_u"] ** 2)))]))
    else:
        r = sweep.load_wide(nu, rho, idx, H)   # raises FileNotFoundError naming the file
    r["rel_l2"] = r["rel_l2_ref"]
    return r


def load_hmc(nu: float, rho: float, idx: int) -> dict:
    path = sweep.cache_path("HMC", nu, rho, idx)
    if not path.exists():
        raise FileNotFoundError(f"missing multichain HMC artifact {path} ({_cell(nu, rho)}, idx={idx})")
    r = sweep.load_cached("HMC", nu, rho, idx)
    r["rel_l2"] = r["rel_l2_ref"]
    return r


def collect_hmc() -> tuple[dict, dict, list[dict]]:
    """Multichain HMC posterior of the H=1 model per cell, the trajectories it admits, and the runs the
    gate dropped. A run whose HMC reference did not converge (fewer than HMC_MIN_ADMITTED of its
    eigenbasis coordinates pass the R-hat / ESS gate) is dropped for EVERY model of this figure,
    exactly as calibration_comparison.py does, so the reference line and all H rows share one run
    set. Each dropped run is returned as a dict (nu, rho, idx, admitted, accept_rates)."""
    if CELLS_RESTRICTED:
        print(f"NOTE: REACDIFF_CELLS restricts this build to the cells {[_cell(*c) for c in CELLS]} "
              f"(default: all {len(CELLS_ALL)} cells).")
    if WIDTHS_RESTRICTED:
        print(f"NOTE: REACDIFF_WIDTHS restricts this build to H in {WIDTHS} (default: {WIDTHS_ALL}).")
    hmc, used, dropped = {}, {}, []
    for nu, rho in CELLS:
        hmc[(nu, rho)], used[(nu, rho)] = [], []
        for idx in IDXS:
            r = load_hmc(nu, rho, idx)
            admitted = float(np.mean((r["rhat"] < RHAT_MAX) & (r["ess"] > ESS_MIN)))
            if admitted < HMC_MIN_ADMITTED:
                rates = np.round(r["accept_rates"], 3).tolist()
                print(f"[{_cell(nu, rho)} idx={idx}] EXCLUDED for every model: only {admitted:.0%} of the HMC coordinates "
                      f"converged (max split-R-hat {np.nanmax(r['rhat']):.3g}, min ESS {np.nanmin(r['ess']):.3g}, "
                      f"chain acceptance rates {rates}), so there is no usable reference for this run.")
                _report_excluded_run(nu, rho, idx, r)
                dropped.append(dict(nu=nu, rho=rho, idx=idx, admitted=admitted, accept_rates=rates))
                continue
            hmc[(nu, rho)].append(r)
            used[(nu, rho)].append(idx)
        print(f"[{_cell(nu, rho)} HMC H=1 reference] loaded {len(hmc[(nu, rho)])} of {len(IDXS)} canonical runs "
              f"(n = {len(used[(nu, rho)])}; admitted trajectories: {used[(nu, rho)]})")
        if not used[(nu, rho)]:
            raise RuntimeError(f"{_cell(nu, rho)}: the HMC gate admitted no trajectory, nothing to plot")
    print("Dropped by the HMC gate: " + (_dropped_text(dropped) if dropped else "none"))
    print("n per cell: " + "; ".join(f"{len(used[c])} at {_cell(*c)} (of {len(IDXS)} canonical)" for c in CELLS))
    print("n per rho (pooled over nu): " + ", ".join(
        f"{sum(len(used[(nu, rho)]) for nu in NUS_AT[rho])} at rho={rho:g} (nu in {NUS_AT[rho]})" for rho in RHOS))
    return hmc, used, dropped


def _dropped_text(dropped: list[dict]) -> str:
    return "; ".join(f"{_cell(d['nu'], d['rho'])} idx {d['idx']} ({d['admitted']:.0%} admitted, chain acceptance {d['accept_rates']})"
                     for d in dropped)


def _report_excluded_run(nu: float, rho: float, idx: int, hmc_run: dict) -> None:
    """For the record only: the single-run numbers (vs truth) of every model on a dropped run."""
    for H in WIDTHS:
        try:
            r = load_one(nu, rho, idx, H)
        except FileNotFoundError as e:   # the record is informational; a dropped run never enters the figure
            print(f"    for the record, H={H} on this run alone: artifact missing ({e})")
            continue
        print(f"    for the record, H={H} on this run alone: rel-L2 {r['rel_l2']:.4f}, mean std t=end "
              f"{mean_std_end(r):.3e}, cov@90 {100 * coverage([r], 0.9):.1f}%, cov@95 {100 * coverage([r], 0.95):.1f}%, "
              f"NLL {nll(r):.1f}, epochs {r['epochs']} (rungs {'+'.join(map(str, r['rung_epochs']))})"
              + (" (capped)" if capped(r) else "") + (" (diverged)" if diverged(r) else ""))
    print(f"    for the record, HMC (H=1 model) on this run alone: mean std t=end {mean_std_end(hmc_run):.3e}, "
          f"cov@90 {100 * coverage([hmc_run], 0.9):.1f}%, NLL {nll(hmc_run):.1f}")


def collect(hmc: dict, used: dict) -> dict[tuple[float, float, int], list[dict]]:
    """One list of runs per (nu, rho, H), restricted to the trajectories `used[(nu, rho)]` admitted by
    collect_hmc. The Ours, wide and HMC records of a cell are checked to share one eigenbasis."""
    runs = {}
    for nu, rho in CELLS:
        for H in WIDTHS:
            rs = []
            for idx in used[(nu, rho)]:
                try:
                    rs.append(load_one(nu, rho, idx, H))
                except FileNotFoundError as e:
                    if H == 1:
                        raise
                    # H = 3 and 4 are run on a subset of the trajectories only (run_experiments.DIFF_WIDE_RUNS:
                    # 1-3 per cell, every cell covered; each run takes 10-30 min and none converges): a
                    # missing wide artifact is recorded, printed and left out; the (cell, H) group then has fewer than n runs and
                    # its own n column says so. Nothing is ever trained here.
                    MISSING.append((nu, rho, H, idx))
                    print(f"[{_cell(nu, rho)} H={H}] idx {idx}: wide artifact missing, left out ({e})")
            runs[(nu, rho, H)] = rs
            rungs = ", ".join("+".join(map(str, r["rung_epochs"])) for r in rs)
            flags = "".join(("C" if capped(r) else "-") + ("D" if diverged(r) else "-") + " " for r in rs)
            print(f"[{_cell(nu, rho)} H={H}] loaded {len(rs)} of {len(used[(nu, rho)])} runs; rung epochs per run: {rungs}; "
                  f"capped/diverged flags: {flags.strip()}")
        # One eigenbasis per trajectory (its Kx, the number of spatial modes, follows the spectrum of
        # the trajectory's initial condition, so it differs between trajectories): the Ours, wide and
        # HMC records of the SAME trajectory must share it (checked over the widths that exist for it).
        for k, idx in enumerate(used[(nu, rho)]):
            recs = [r for H in WIDTHS for r in runs[(nu, rho, H)] if r["idx"] == idx]
            sweep.assert_same_basis(hmc[(nu, rho)][k], *recs)
            assert hmc[(nu, rho)][k]["idx"] == idx
    if MISSING:
        print(f"Wide artifacts missing ({len(MISSING)}; nu, rho, H, idx): " + ", ".join(f"({nu:g}, {rho:g}, {H}, {i})" for nu, rho, H, i in MISSING))
    return runs


MISSING: list[tuple[float, float, int, int]] = []   # (nu, rho, H, idx) of wide artifacts that do not exist (filled by collect)


def pooled(runs: dict, rho: float, H: int) -> list[dict]:
    """The runs of one rho point: all admitted runs of the nu in NUS_AT[rho]."""
    return [r for nu in NUS_AT[rho] for r in runs[(nu, rho, H)]]


def pooled_hmc(hmc: dict, rho: float) -> list[dict]:
    return [r for nu in NUS_AT[rho] for r in hmc[(nu, rho)]]


# --- per-run / per-group statistics (formulas shared with calibration_comparison.py) -------------

def mean_std_end(r: dict) -> float:
    """Spatial mean of the predictive std at the last time step."""
    return float(np.sqrt(np.maximum(r["pred_var"][:, -1], 0.0)).mean())


def standardized_residuals(r: dict) -> np.ndarray:
    std = np.sqrt(np.maximum(r["pred_var"], MIN_VAR))
    return ((r[TRUTH] - r["pred_u"]) / std).ravel()


def coverage(rs: list[dict], level: float) -> float:
    if not rs:
        return float("nan")
    z = np.concatenate([standardized_residuals(r) for r in rs])
    return float(np.mean(np.abs(z) <= stats.norm.ppf(0.5 + level / 2)))


def nll(r: dict) -> float:
    var = np.maximum(r["pred_var"], MIN_VAR)
    return float(np.mean(0.5 * np.log(2 * np.pi * var) + (r[TRUTH] - r["pred_u"]) ** 2 / (2 * var)))


def mean_nll(rs: list[dict]) -> float:
    return float(np.mean([nll(r) for r in rs])) if rs else float("nan")


def capped(r: dict) -> bool:
    """At least one rung of the ladder hit the RUNG_CAP-epoch limit without reaching the KL tolerance."""
    return max(r["rung_epochs"]) >= RUNG_CAP


def diverged(r: dict) -> bool:
    """A capped run whose per-epoch max KL still exceeded KL_DIVERGED within its last 50 epochs: the EP
    iteration of the final rung blew up instead of settling. (Runs that converged are never counted,
    even though their first few epochs routinely have KL spikes above 1.) `loss_y` is the per-epoch
    max-KL trace of the four rungs concatenated, so the window is the tail of the final rung when
    that rung ran at least 50 epochs (a capped final rung always did) and otherwise reaches back
    into the end of the previous rung."""
    return capped(r) and bool(np.max(r["loss_y"][-50:]) > KL_DIVERGED)


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


def _reserve_bottom(ax, frac: float) -> None:
    """The mirror image of _reserve_top: the bottom `frac` of the axes height holds no data (for a
    second legend, in a panel whose top band is taken by the first one). Uses the scale (log, linear
    or symlog) the axis already has."""
    _, hi = ax.get_ylim()
    ymin = ax.dataLim.y0
    if ax.get_yscale() == "log":
        ax.set_ylim(bottom=10 ** ((np.log10(ymin) - frac * np.log10(hi)) / (1 - frac)))
    elif ax.get_yscale() == "linear":
        ax.set_ylim(bottom=(ymin - frac * hi) / (1 - frac))
    else:   # symlog etc.: the same construction in the axis' own transformed coordinates
        tr = ax.yaxis.get_transform()
        ymin_t, hi_t = tr.transform([ymin, hi])
        ax.set_ylim(bottom=float(tr.inverted().transform([(ymin_t - frac * hi_t) / (1 - frac)])[0]))


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


def _points_and_median_line(ax, runs, key_fn, agg=np.median, line_kw=None):
    """Per-run points (marker = nu, colour = H) and one line per H through the per-rho aggregate
    (`line_kw`: extra style of those lines, e.g. vertex markers)."""
    for H in WIDTHS:
        for rho in RHOS:
            for nu in NUS_AT[rho]:
                vals = [key_fn(r) for r in runs[(nu, rho, H)]]
                ax.scatter(np.full(len(vals), rho * X_JITTER[H] * X_JITTER_NU[nu]), vals, s=30, alpha=0.5,
                           color=COLORS[H], marker=NU_MARKERS[nu], edgecolors="none")
        ax.plot(RHOS, [agg([key_fn(r) for r in pooled(runs, rho, H)]) for rho in RHOS], color=COLORS[H],
                lw=2.4, label=rf"$H={H}$", **(line_kw or {}))


def _nu_legend(ax, loc="lower right"):
    """The small marker legend (marker shape = nu) of the panels with per-run points; a separate
    artist, so the panel's H legend (ax.legend) stays. It sits in the empty bottom band of the axes
    (_reserve_bottom), the H legend in the top band."""
    handles = [plt.Line2D([], [], linestyle="none", marker=NU_MARKERS[nu], color="k", alpha=0.6, markersize=7,
                          label=rf"$\nu={nu:g}$") for nu in NUS_ALL if any(nu in NUS_AT[rho] for rho in RHOS)]
    leg = Legend(ax, handles, [h.get_label() for h in handles], fontsize=11, loc=loc, ncol=len(handles),
                 handletextpad=0.2, columnspacing=0.8, title=r"$\mathrm{points:\ marker} = \nu$", title_fontsize=11)
    ax.add_artist(leg)
    return leg


def _finish(ax, title, ylabel, ylog=True, legend=True, legend_kw=None):
    ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    ax.set_xticks(RHOS)
    ax.set_xticklabels([f"{rho:g}" for rho in RHOS])
    ax.minorticks_off()
    ax.set_xlim(RHOS[0] / X_PAD, RHOS[-1] * X_PAD)   # fixed margins on the log-rho axis (the jittered points fit)
    ax.set_xlabel(r"$\rho$", fontsize=19)
    ax.set_ylabel(ylabel, fontsize=19)
    ax.set_title(title, fontsize=20)
    ax.grid(True, which="major", alpha=0.3)
    ax.tick_params(labelsize=13)
    if legend:
        ax.legend(**{"fontsize": 13, **(legend_kw or {})})


def plot_rel_l2(ax, runs, tag):
    _points_and_median_line(ax, runs, lambda r: r["rel_l2"])
    _finish(ax, r"$\mathrm{(%s)\ Relative\ } L_2 \mathrm{\ error\ vs.\ truth}$" % tag,
            r"$\mathrm{rel.}\ L_2\ \mathrm{error}$", legend=False)
    _reserve_top(ax, LEGEND_BAND)   # the H legend sits in the empty band above the data ...
    ax.legend(fontsize=13, ncol=4, loc="upper left")
    _reserve_bottom(ax, LEGEND_BAND_NU)   # ... and the nu-marker legend in an empty band below it
    _nu_legend(ax, loc="lower right")


def plot_mean_std(ax, runs, hmc, tag):
    _points_and_median_line(ax, runs, mean_std_end, agg=np.mean)
    ax.plot(RHOS, [np.mean([mean_std_end(r) for r in pooled_hmc(hmc, rho)]) for rho in RHOS],
            label=r"$\mathrm{HMC}\ (H{=}1\ \mathrm{model})$", **HMC_STYLE)
    _finish(ax, r"$\mathrm{(%s)\ Mean\ predictive\ std\ at\ } t{=}t_{\mathrm{end}}$" % tag, r"$\mathrm{predictive\ std}$",
            legend=False)
    _reserve_top(ax, LEGEND_BAND)   # the H / HMC legend sits in the empty band above the data ...
    ax.legend(fontsize=13, ncol=3, loc="upper left")
    _reserve_bottom(ax, LEGEND_BAND_NU)   # ... and the nu-marker legend in an empty band below it
    _nu_legend(ax, loc="lower right")


def plot_coverage(ax, runs, hmc, tag):
    for H in WIDTHS:
        for level, ls in zip(COVERAGE_LEVELS, ("-", "--")):
            ax.plot(RHOS, [coverage(pooled(runs, rho, H), level) for rho in RHOS], color=COLORS[H],
                    lw=2.4, linestyle=ls, label=rf"$H={H}$" if level == COVERAGE_LEVELS[0] else None)
    for level, ls in zip(COVERAGE_LEVELS, ("-", "--")):
        ax.plot(RHOS, [coverage(pooled_hmc(hmc, rho), level) for rho in RHOS], **{**HMC_STYLE, "linestyle": ls},
                label=r"$\mathrm{HMC}\ (H{=}1\ \mathrm{model})$" if level == COVERAGE_LEVELS[0] else None)
        ax.axhline(level, color="k", linestyle=ls, lw=1.0, alpha=0.45)
    ax.plot([], [], color="k", linestyle="-", lw=2, label=r"$90\%\ \mathrm{interval}$")
    ax.plot([], [], color="k", linestyle="--", lw=2, label=r"$95\%\ \mathrm{interval}$")
    ax.set_ylim(-0.02, 1.02)   # symmetric margins: a coverage of exactly 0 (it happens) stays visible above the spine
    _finish(ax, r"$\mathrm{(%s)\ Empirical\ coverage\ vs.\ truth}$" % tag, r"$\mathrm{empirical\ coverage\ of\ the\ truth}$", ylog=False)
    ax.legend(fontsize=11, ncol=2, loc="upper left", bbox_to_anchor=(0.01, 0.86))


def plot_nll(ax, runs, hmc, tag):
    # per-run points (marker = nu, colour = H) and one line per H through the per-rho means, as in the
    # rel-L2 and training-time panels; the line values are mean_nll of the pooled runs
    _points_and_median_line(ax, runs, nll, agg=np.mean, line_kw=NLL_LINE_MARKER)
    vals = {H: [mean_nll(pooled(runs, rho, H)) for rho in RHOS] for H in WIDTHS}
    vals["hmc"] = [mean_nll(pooled_hmc(hmc, rho)) for rho in RHOS]
    ax.plot(RHOS, vals["hmc"], label=r"$\mathrm{HMC}\ (H{=}1\ \mathrm{model})$", **HMC_STYLE)
    # A Gaussian NLL is negative when the predictive std is small and the error smaller still; a log
    # axis would drop such a point silently, so the axis is log only while every value is positive
    # (every mean and every per-run point).
    lowest = min(min(min(v) for v in vals.values()),
                 min(nll(r) for H in WIDTHS for rho in RHOS for r in pooled(runs, rho, H)))
    all_positive = lowest > 0
    _finish(ax, r"$\mathrm{(%s)\ Predictive\ NLL\ vs.\ truth}$" % tag, r"$\mathrm{NLL}$",
            ylog=all_positive, legend=False)
    if not all_positive:
        print(f"[NLL panel] an NLL is <= 0 (min {lowest:.3g}): symlog y axis "
              f"(linear inside +-{NLL_LINTHRESH:g}, log outside) instead of log")
        ax.set_yscale("symlog", linthresh=NLL_LINTHRESH)
    _reserve_top(ax, LEGEND_BAND)   # the H / HMC legend sits in the empty band above the data ...
    ax.legend(fontsize=13, ncol=3, loc="upper left")
    _reserve_bottom(ax, LEGEND_BAND_NU)   # ... and the nu-marker legend in an empty band below it
    _nu_legend(ax, loc="lower right")


def plot_train_time(ax, runs, tag):
    _points_and_median_line(ax, runs, lambda r: r["train_time_seconds"])
    _finish(ax, r"$\mathrm{(%s)\ Training\ time}$" % tag, r"$\mathrm{time\ (s)}$", legend=False)
    _reserve_top(ax, LEGEND_BAND)   # the H legend sits in the empty band above the data ...
    ax.legend(fontsize=13, ncol=4, loc="upper left")
    _reserve_bottom(ax, LEGEND_BAND_NU)   # ... and the nu-marker legend in an empty band below it
    _nu_legend(ax, loc="lower right")


def plot_neuron_usage(ax, runs, tag):
    """One line per (H, neuron); the dotted line per H is the equal share 1/H (in the legend next
    to its H); the neuron index is written at the right end of each line, pushed apart vertically
    where the lines of one H end close together."""
    handles = []   # (artist, label) pairs in legend order: each H followed by its 1/H reference line
    wide = [H for H in WIDTHS if H > 1]
    for H in wide:
        xs = np.array(RHOS) * X_JITTER_F[H]
        end_shares = []
        for h in range(H):
            shares = [np.median([neuron_shares(r)[h] for r in pooled(runs, rho, H)]) for rho in RHOS]
            line, = ax.plot(xs, shares, color=COLORS[H], marker="o", markersize=7, lw=1.2,
                            alpha=0.5 + 0.5 * (h == 0))
            if h == 0:
                handles.append((line, rf"$H={H}$"))
            end_shares.append(shares[-1])
        ref = ax.axhline(1.0 / H, color=COLORS[H], linestyle=":", lw=1.2, alpha=0.7)
        handles.append((ref, rf"$1/H = 1/{H}$"))
        for h, y in enumerate(_spread(end_shares, LABEL_GAP_F)):
            ax.text(xs[-1] * 1.06, y, f"{h + 1}", fontsize=9, color=COLORS[H], ha="left", va="center")
    ax.set_ylim(0, F_YMIN_TOP)
    _finish(ax, r"$\mathrm{(%s)\ Neuron\ share\ of\ the\ predictive\ mean}$" % tag,
            r"$\mathrm{neuron\ share}$", ylog=False, legend=False)
    if wide:
        ax.set_xlim(RHOS[0] * X_JITTER_F[wide[0]] / X_PAD, RHOS[-1] * X_JITTER_F[wide[-1]] * 1.3)   # room for the neuron-index labels
        # the legend sits in a band above the highest share (the shares can exceed 1/2 when one neuron dominates)
        ax.set_ylim(0, max(F_YMIN_TOP, ax.dataLim.y1 / (1 - LEGEND_BAND)))
        ax.legend([a for a, _ in handles], [l for _, l in handles], fontsize=13, ncol=3, loc="upper center")
    else:
        ax.text(0.5, 0.5, "no wide model in this build\n(REACDIFF_WIDTHS)", ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="gray")


# --- table ----------------------------------------------------------------------------------------

def _sci(x: float) -> str:
    m, e = f"{x:.2e}".split("e")
    return rf"${m}{{\times}}10^{{{int(e)}}}$"


def _model_rows(rho_label: str, nu_label: str, runs_by_H: dict[int, list[dict]], hs: list[dict]) -> list[str]:
    """The block-1 rows (one per H, then the HMC row) of one cell or of one pooled rho point."""
    lines = []
    for H in WIDTHS:
        rs = runs_by_H[H]
        if not rs:
            lines.append(f"{rho_label} & {nu_label} & $H={H}$ & \\multicolumn{{7}}{{l}}{{not run}} & 0 \\\\")
            continue
        rl = np.array([r["rel_l2"] for r in rs])
        ms = np.array([mean_std_end(r) for r in rs])
        ep = np.array([r["epochs"] for r in rs])
        tt = np.array([r["train_time_seconds"] for r in rs])
        lines.append(
            f"{rho_label} & {nu_label} & $H={H}$ & {np.median(rl):.4f} [{rl.min():.4f}, {rl.max():.4f}] & "
            f"{_sci(ms.mean())} [{_sci(ms.min())}, {_sci(ms.max())}] & "
            f"{100 * coverage(rs, 0.9):.1f}\\% & {100 * coverage(rs, 0.95):.1f}\\% & {mean_nll(rs):.0f} & "
            f"{_round_half_up(np.median(ep))} [{ep.min()}, {ep.max()}] ({sum(capped(r) for r in rs)}/{sum(diverged(r) for r in rs)}) & "
            f"{np.median(tt):.1f} [{tt.min():.1f}, {tt.max():.1f}] & {len(rs)} \\\\")
    rl = np.array([r["rel_l2"] for r in hs])
    ms = np.array([mean_std_end(r) for r in hs])
    tt = np.array([r["train_time_seconds"] for r in hs])
    lines.append(
        f"{rho_label} & {nu_label} & HMC ($H{{=}}1$) & {np.median(rl):.4f} [{rl.min():.4f}, {rl.max():.4f}] & "
        f"{_sci(ms.mean())} [{_sci(ms.min())}, {_sci(ms.max())}] & "
        f"{100 * coverage(hs, 0.9):.1f}\\% & {100 * coverage(hs, 0.95):.1f}\\% & {mean_nll(hs):.0f} & "
        f"-- & {np.median(tt):.0f} [{tt.min():.0f}, {tt.max():.0f}] & {len(hs)} \\\\")
    return lines


def _neuron_rows(rho_label: str, nu_label: str, runs_by_H: dict[int, list[dict]]) -> list[str]:
    """The block-2 rows (one per (H, neuron)) of one cell or of one pooled rho point."""
    lines = []
    for H in WIDTHS:
        if H == 1:
            continue
        rs = runs_by_H[H]
        if not rs:
            continue
        for h in range(H):
            sh = np.array([neuron_shares(r)[h] for r in rs])
            zmin = min(float(r["neuron_z_min"][h]) for r in rs)
            zmax = max(float(r["neuron_z_max"][h]) for r in rs)
            sig = np.median(np.concatenate([r["w_eigen_std"][:, h] for r in rs]))
            lines.append(f"{rho_label} & {nu_label} & $H={H}$ & {h + 1} & {rs[0]['a'][h]:+.1f} & {np.median(sh):.3f} & "
                         f"[{sh.min():.3f}, {sh.max():.3f}] & [{zmin:.2f}, {zmax:.2f}] & {_sci(sig)} & & \\\\")
    return lines


def _pooled_label(rho: float) -> str:
    return "pooled: $\\nu \\in \\{" + ", ".join(f"{nu:g}" for nu in NUS_AT[rho]) + "\\}$"


def write_table(runs, hmc, used, dropped, path: Path) -> None:
    n_cell = {c: len(used[c]) for c in CELLS}
    lines = [
        r"% Auto-generated by figures_onepass/diffusion/wider_model.py",
        r"% PDEBench Fisher-KPP (reaction-diffusion) dataset, u_t = nu u_xx + rho u (1 - u), 16 cells nu in {0.5, 1, 2, 5} x",
        r"%   rho in {1, 2, 5, 10}, five canonical trajectories per cell (idxs(5), the first five of the canonical seeded",
        r"%   draw, pde/advection/trajectory_selection.py). The model works with rho_norm = 2 rho and nu_norm = 2 nu.",
        r"% One-pass EP (expectation propagation) on u = sum_h a_h g(w_h.phi): H=1 is the single-neuron model (existing",
        r"%   artifacts), H=2 uses a=(1,-1), H=3 a=(1,-1,0.5), H=4 a=(1,-1,0.5,-0.5); same schedule, priors, noise levels and",
        r"%   rho-continuation ladder (four rungs s*rho, s = 0.25, 0.5, 0.75, 1, each a full EP training of at most",
        rf"%   {RUNG_CAP} epochs started from the previous rung's posterior) for all H; 'epochs' is the sum over the four rungs.",
        rf"% 'capped' = number of the n runs in which at least one rung hit the {RUNG_CAP}-epoch limit without reaching the KL",
        r"%   (Kullback-Leibler divergence, in nats) tolerance 1e-5 (the same limit applies to H=1); 'div' = number of those",
        rf"%   capped runs whose per-epoch max KL still exceeded {KL_DIVERGED:g} within the last 50 epochs, i.e. the EP iteration of",
        r"%   the final rung diverged instead of converging slowly. The per-rung epoch counts of every run are in wider_model.log.",
        rf"% n per cell = the canonical trajectories whose multichain HMC (Hamiltonian Monte Carlo) reference converged (at least",
        rf"%   {HMC_MIN_ADMITTED:.0%} of the eigenbasis (the rotated weight coordinates all methods share) coordinates with",
        rf"%   split-R-hat (the chain-agreement diagnostic) < {RHAT_MAX} and bulk ESS (effective sample size) > {ESS_MIN:g}, the rule",
        r"%   of calibration_comparison.py); computed from the cached HMC diagnostics when this table was written:",
        rf"%   n = {'; '.join(f'{n_cell[c]} at {_cell(*c)} (of {len(IDXS)} canonical)' for c in CELLS)}.",
        rf"%   Dropped by the HMC gate, for EVERY model (all H and the HMC rows): {_dropped_text(dropped) if dropped else 'none'}.",
        r"%   The dropped runs' own per-model numbers are in wider_model.log. Because the run set is shared, the H=1 rows",
        r"%   of a cell equal the 'Ours' rows of calibration_comparison_table.tex.",
    ]
    if CELLS_RESTRICTED:
        lines.append(rf"% NOTE: built with REACDIFF_CELLS={os.environ['REACDIFF_CELLS']!r}, i.e. only the cells "
                     rf"{[_cell(*c) for c in CELLS]} (default: all {len(CELLS_ALL)}); rebuild without the variable for the shipped table.")
    if WIDTHS_RESTRICTED:
        lines.append(rf"% NOTE: built with REACDIFF_WIDTHS={os.environ['REACDIFF_WIDTHS']!r}, i.e. H in {WIDTHS} only "
                     rf"(default: {WIDTHS_ALL}); rebuild without the variable for the shipped table.")
    if MISSING:
        by_H = {H: sum(1 for m in MISSING if m[2] == H) for H in WIDTHS if H > 1}
        lines.append(r"% Wide artifacts missing (H = 3 and 4 are run on the subset listed in run_experiments.DIFF_WIDE_RUNS; a (cell, H) row then has fewer than 5 runs, "
                     r"its own n column says how many): " + ", ".join(f"H={H}: {n} of 80" for H, n in by_H.items() if n) + ".")
    else:
        lines.append(r"% Every (cell, H, trajectory) wide artifact exists.")
    lines += [
        r"% Everything is measured against the TRUTH on the full 1024 x 101 (x,t) grid, where the truth is the float64",
        r"%   re-solve of the nominal PDE from the trajectory's initial condition (u_ref, reacdiff_reference.py), never the",
        r"%   stored PDEBench trajectory (its late-time freeze is a data artifact).",
        r"% Block 1, per (rho, nu, H) cell and, after the cells of a rho, the row 'pooled' over the listed nu (the rho point",
        r"%   of the figure; its n is the sum over those cells): rel-L2 of the predictive mean (median [min, max] over runs);",
        r"%   mean std at t=end = across-run mean of the spatial mean of sqrt(pred_var) at the last time step (the t=end",
        r"%   value of calibration_variance.pdf panel (c)), with its min-max over runs; cov90 / cov95 = empirical coverage of the",
        r"%   nominal 90% / 95% Gaussian predictive interval, pooled over every grid point of the n runs; NLL = mean over",
        rf"%   runs of the per-run mean Gaussian NLL (variance floored at {MIN_VAR:g}); epochs and training time = median",
        r"%   [min, max]; the median epochs are rounded half up to an integer (the same rule as in wider_model.log and the",
        r"%   printed summary).",
        r"%   The 'HMC (H=1)' rows are the multichain HMC posterior of the single-neuron model (its exact posterior). Their",
        r"%   'train [s]' = MAP (maximum a posteriori) warm start plus the summed wall time of the four chains, which run one",
        r"%   after the other (the step-size search is not counted); this is not comparable to EP epochs, so their epochs",
        r"%   cell is empty ('--').",
        r"% Reading the uncertainty columns (mean std, cov90, cov95, NLL) across H: the wide predictive variance is the",
        r"%   mean-field sum over independent neurons, Var[u] = sum_h a_h^2 Var[g(z_h)] (the EP beliefs factorize over all",
        r"%   weights of all neurons and carry no covariance between neurons). Wherever the mean std or the coverage rises,",
        r"%   or the NLL falls, with H, that is mainly an independence effect (more independent variance terms are added) and",
        r"%   must NOT be read as better-calibrated inference: the true posterior of the wide model has strongly",
        r"%   anti-correlated neuron pairs, whose negative covariance the mean-field sum leaves out. On this",
        r"%   dataset most H=2 runs hit the epoch cap (capped/div column), their coverage is far below H=1 and their NLL far",
        r"%   above it, so the wide-model uncertainty columns also reflect a ladder that did not converge. The 'HMC (H=1)'",
        r"%   rows are a scale reference from a different model, not the reference posterior of the wide models.",
        r"% Block 2 (after the 'Neuron usage' heading), per (rho, nu, H, neuron) cell and pooled per rho: share =",
        r"%   rms(a_h E[g(z_h)]) / rms(u_pred) over the grid (median over runs, then its [min, max] over runs); z range =",
        r"%   [min over runs of min z_h, max over runs of max z_h] of the pre-activation z_h = w_h.phi at the posterior mean;",
        r"%   sigma_w = median over coordinates and runs of the posterior std of neuron h's eigenbasis weights (scientific",
        r"%   notation: the values are 1e-5 .. 1e-4 here).",
        r"\begin{tabular}{l l l r r r r r r r r}",
        r"\toprule",
        r"$\rho$ & $\nu$ & model & rel-$L_2$ median [min, max] & mean std $t{=}t_\mathrm{end}$ [min, max] & cov90 & cov95 & NLL & epochs (capped/div) & train [s] & $n$ \\",
        r"\midrule",
    ]
    for rho in RHOS:
        for nu in NUS_AT[rho]:
            lines += _model_rows(f"{rho:g}", f"{nu:g}", {H: runs[(nu, rho, H)] for H in WIDTHS}, hmc[(nu, rho)])
        lines += _model_rows(f"{rho:g}", _pooled_label(rho), {H: pooled(runs, rho, H) for H in WIDTHS}, pooled_hmc(hmc, rho))
        if rho != RHOS[-1]:
            lines.append(r"\midrule")
    lines += [
        r"\midrule",
        r"\multicolumn{11}{l}{\emph{Neuron usage}, $H = 2, 3, 4$}\\",
        r"$\rho$ & $\nu$ & model & neuron & $a_h$ & share (median) & share [min, max] & $z_h$ range & $\sigma_w$ median & & \\",
        r"\midrule",
    ]
    for rho in RHOS:
        for nu in NUS_AT[rho]:
            lines += _neuron_rows(f"{rho:g}", f"{nu:g}", {H: runs[(nu, rho, H)] for H in WIDTHS})
        lines += _neuron_rows(f"{rho:g}", _pooled_label(rho), {H: pooled(runs, rho, H) for H in WIDTHS})
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines))
    print(f"Saved table to {path}")


def print_summary(runs, hmc, used, dropped) -> None:
    print(f"\n{sum(len(v) for v in used.values())} runs used "
          f"({'; '.join(f'{len(used[c])} at {_cell(*c)} (of {len(IDXS)} canonical)' for c in CELLS)}); "
          "the same trajectories for every H and for the HMC reference. "
          "Dropped by the HMC gate: " + (_dropped_text(dropped) if dropped else "none") + ".")
    if CELLS_RESTRICTED:
        print(f"NOTE: REACDIFF_CELLS restricted this build to the cells {[_cell(*c) for c in CELLS]} (default: all {len(CELLS_ALL)}).")
    if WIDTHS_RESTRICTED:
        print(f"NOTE: REACDIFF_WIDTHS restricted this build to H in {WIDTHS} (default: {WIDTHS_ALL}).")

    def line(label: str, rs: list[dict]) -> str:
        return (f"  {label}: {np.median([r['rel_l2'] for r in rs]):.4f} | "
                f"{np.mean([mean_std_end(r) for r in rs]):.3e} | {coverage(rs, 0.9):.3f} | {coverage(rs, 0.95):.3f} | "
                f"{mean_nll(rs):8.1f} | {_round_half_up(np.median([r['epochs'] for r in rs]))} ({sum(capped(r) for r in rs)}/{sum(diverged(r) for r in rs)}) | "
                f"{np.median([r['train_time_seconds'] for r in rs]):.1f} | n={len(rs)}")

    def hmc_line(label: str, hs: list[dict]) -> str:
        return (f"  {label}: {np.median([r['rel_l2'] for r in hs]):.4f} | "
                f"{np.mean([mean_std_end(r) for r in hs]):.3e} | {coverage(hs, 0.9):.3f} | {coverage(hs, 0.95):.3f} | {mean_nll(hs):8.1f} | n={len(hs)}")

    print("\nSummary per rho point (pooled over nu) and per cell: rel-L2 median | mean std t=end | cov90 | cov95 | NLL | "
          f"epochs median (capped: a rung at {RUNG_CAP} / diverged) | train s median | n")
    for rho in RHOS:
        for H in WIDTHS:
            print(line(f"rho={rho:<3g} pooled nu in {NUS_AT[rho]} H={H}", pooled(runs, rho, H)))
        print(hmc_line(f"rho={rho:<3g} pooled nu in {NUS_AT[rho]} HMC(H=1)", pooled_hmc(hmc, rho)))
        for nu in NUS_AT[rho]:
            for H in WIDTHS:
                print(line(f"    rho={rho:<3g} nu={nu:<4g} H={H}", runs[(nu, rho, H)]))
            print(hmc_line(f"    rho={rho:<3g} nu={nu:<4g} HMC(H=1)", hmc[(nu, rho)]))
    print("\nNeuron usage (median share of the predictive mean per neuron over the runs of the rho point; z range over runs):")
    for rho in RHOS:
        for H in WIDTHS:
            if H == 1:
                continue
            rs = pooled(runs, rho, H)
            parts = []
            for h in range(H):
                sh = np.median([neuron_shares(r)[h] for r in rs])
                zmin = min(float(r["neuron_z_min"][h]) for r in rs)
                zmax = max(float(r["neuron_z_max"][h]) for r in rs)
                parts.append(f"n{h + 1}(a={rs[0]['a'][h]:+.1f}): share {sh:.3f}, z[{zmin:.2f},{zmax:.2f}]")
            print(f"  rho={rho:<3g} H={H}: " + "; ".join(parts))


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
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--png", default="", help="also save a PNG copy of the figure here (visual check)")
    args = parser.parse_args()
    sys.stdout = _Tee(OUTPUT_LOG)
    hmc, used, dropped = collect_hmc()
    runs = collect(hmc, used)
    panels, output_pdf = (PANELS_ALL, OUTPUT_PDF_ALL_PANELS) if ALL_PANELS else (PANELS_SHIPPED, OUTPUT_PDF)
    if ALL_PANELS:
        print(f"NOTE: WIDER_MODEL_ALL_PANELS is set: the six-panel figure goes to {output_pdf}; "
              f"{OUTPUT_PDF} is not written.")
    fig = make_figure(runs, hmc, panels)
    fig.savefig(output_pdf, bbox_inches="tight")
    if args.png:
        fig.savefig(args.png, dpi=110, bbox_inches="tight")
        print(f"Saved PNG copy to {args.png}")
    plt.close(fig)
    print(f"Saved figure to {output_pdf}")
    write_table(runs, hmc, used, dropped, OUTPUT_TEX)
    print_summary(runs, hmc, used, dropped)
    print(f"Log written to {OUTPUT_LOG}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
