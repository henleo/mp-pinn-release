"""
Calibration and variance consistency of the one-pass EP (expectation propagation) posterior
predictive ("Ours") against the multi-chain HMC (Hamiltonian Monte Carlo) reference and the
mean-field VI (variational inference) baseline on the PDEBench 1D Fisher-KPP grid (16 cells
nu x rho, the five canonical trajectories idxs(5) per cell, 80 runs), against the float64 re-solve
u_ref. This script combines the Fisher-KPP calibration analysis (part 1) and the variance-consistency
analysis (part 2), the reaction-diffusion counterpart of
figures_onepass/advection/calibration_comparison.py: the cached artifacts are loaded once, and both
parts use one run set (the HMC convergence gate below).

SHIPPED FIGURE, calibration_variance.pdf (2 x 2), with its companion table calibration_variance_table.tex:
  (a) CRPS (continuous ranked probability score) against the truth per rho (nu pooled), Ours / HMC / VI
      = panel (d) of the full figure calibration_comparison.pdf (part 1).
  (b) empirical coverage of the truth by the nominal 90% (solid) / 95% (dashed) intervals per rho
      = panel (e) of the full figure calibration_comparison.pdf (part 1).
  (c) Ours' mean predictive std vs time per rho (nu pooled; band = min to max over the runs)
      = panel (a) of the full figure variance_consistency.pdf (part 2).
  (d) per-run median of sigma_Ours / sigma_HMC, one point per (rho, nu, trajectory), nu as marker
      = panel (d) of the full figure variance_consistency.pdf (part 2).
Panels (c) and (d) are drawn on the panel size (8.5 x 7.33 inch) and with the font sizes of panels
(a) and (b) (the full variance_consistency.pdf uses 7.25 x 5.75 inch panels, smaller fonts,
thinner lines and smaller markers; MERGED_VC_STYLE vs VC_STYLE), the log axis of (d) is labelled
with plain numbers like that of (a), and the rho legend of (c) is one row in a band kept free of
data at the top of the panel. The plotted data are those of the corresponding panels of the full
figures.

THE FULL FIGURES. CALIBRATION_ALL_PANELS=1 additionally writes calibration_comparison.pdf (six
panels a-f) and variance_consistency.pdf (four panels a-d), the full figures of part 1 and part 2;
the shipped figure is written in both modes. The two full companion tables,
calibration_comparison_table.tex and variance_consistency_table.tex, are written on every run,
since they hold the numbers of every panel and of every cell. The panel letters in PART 1 and
PART 2 below refer to those two full figures.

Both parts share one loader and one log (calibration_comparison.log); --png renders the shipped
figure; a missing artifact is an error naming the first missing file. The trajectory, u_ref and
time-grid checks run on every loaded run, gate-excluded ones included. A rho whose every run fails
the HMC gate stops the whole script (part 2 needs at least one run per rho), so the 'dropped' rows
of the part-1 table can only appear for single cells, not for a whole rho.

PART 1: calibration (full figure calibration_comparison.pdf)

Calibration (does the stated uncertainty match the actual error?) of the one-pass EP (expectation
propagation) posterior predictive ("Ours") against the multi-chain HMC (Hamiltonian Monte Carlo)
reference and the mean-field VI (variational inference; mean-field = every weight gets its own
independent Gaussian) baseline, over the full (x,t) grid of the five canonical trajectories
IDXS = idxs(5) in every cell of the PDEBench Fisher-KPP grid (nu in NUS x rho in RHOS, 16 cells,
up to 80 runs; the Fisher-KPP equation is u_t = nu u_xx + rho u (1 - u), nu the diffusion
coefficient, rho the reaction strength). This is the reaction-diffusion port of
figures_onepass/advection/calibration_comparison.py; the methods, the gate, the metrics and the
table format are inherited unchanged, only the axis changes: the wave speed b of the
advection figure becomes the reaction strength rho (log axis, 4 values), and the diffusion
coefficient nu is pooled into every rho point (each rho point aggregates the admitted runs of the
four nu values, up to 4 x 5 = 20 runs). The companion table lists all 16 (nu, rho) cells
explicitly, then the per-rho rows pooled over nu, then the grand pooled row.

The truth is the float64 re-solve of the nominal PDE from the dataset's own initial condition
(`u_ref`, pde/diffusion/reacdiff_reference.py), never the stored PDEBench trajectory,
whose late-time freeze is a data artifact. The model uses the normalized coefficients
rho_norm = rho T_basis = 2 rho and nu_norm = nu T_basis / L^2 = 2 nu; the axis shows the physical rho.

Two levels of calibration are shown, because they answer two different questions:

  * MODEL-LEVEL, against the TRUE solution (panels a-e, table block 1): are the nominal posterior
    intervals honest about the distance to the true trajectory? Every method, including exact
    HMC, shares the model-form error (the truncated Fourier basis cannot represent the true
    solution exactly, and that representation error is not part of the noise model beta_ic =
    0.02, beta_pde = 0.05), so the coverage of the truth mostly measures the model, not the
    inference method; the numbers are reported in full (table block 1, log). The NLL (negative
    log-likelihood) gap between the methods is nevertheless an inference effect whenever the
    per-run residuals of the posterior mean against the truth agree between Ours and HMC (the
    table comment gives the largest per-run RMSE difference and the median ratio of the
    predictive stds, so the reader can check this on the numbers of this dataset). Block 1 is
    reported in full, with 90%/95% coverage and NLL kept prominent and CRPS (continuous ranked
    probability score, a proper scoring rule in the units of u) alongside.
  * INFERENCE-LEVEL, against the EXACT posterior (panel f, table block 2): how well does each
    approximate posterior reproduce the exact posterior predictive of the same model, which HMC
    samples? This isolates the approximation error of the inference method itself. It is computed
    in closed form from the stored predictive moments, treating the HMC predictive at each grid
    point as Gaussian with HMC's own mean m_H and variance v_H. With the method's mean m_q and
    std s_q, and z the two-sided standard-normal quantile of the level:
      coverage of the HMC predictive by the interval m_q +/- z s_q
          = Phi((m_q + z s_q - m_H)/s_H) - Phi((m_q - z s_q - m_H)/s_H),
      expected NLL of the method's Gaussian predictive under the HMC predictive
          = 0.5 ln(2 pi v_q) + (v_H + (m_H - m_q)^2) / (2 v_q),
    both averaged over the grid. HMC against itself gives nominal coverage and its own entropy
    (0.5 ln(2 pi v_H) + 0.5), which is the floor of the expected NLL; the excess over that floor is in
    nats (natural-log units of NLL).
    CAVEAT (stated in the table comment and the log as well): these closed forms assume a
    Gaussian HMC predictive at every grid point (see CAPTION_NOTE and VERIFICATION_NOTE below);
    block 2 is exact only insofar as that assumption holds.

Panels of calibration_comparison.pdf: (a) pooled coverage curve at nominal 50/80/90/95%; (b) standardized residuals (N(0,1) if
calibrated); (c) NLL per rho (log axis); (d) CRPS per rho (log axis); (e) 90%/95% coverage of the
true solution per rho; (f) inference-level 90%/95% coverage of the HMC predictive per rho. Panels
(a)/(b) pool every admitted run of every cell; panels (c)-(f) show every rho in RHOS on a log-rho
axis, each rho point pooling the admitted runs of all four nu values (the per-cell numbers are in
the table). The companion table (calibration_comparison_table.tex) holds every number per cell,
per rho (pooled over nu) and pooled, with n per row; the y-axes of (c)/(d) are chosen from the
plotted values at run time.

A run enters only when its HMC reference converged: at least HMC_MIN_ADMITTED of the eigenbasis
coordinates (the rotated weight coordinates all methods share) pass split-R-hat < 1.05 and bulk
ESS > 100 (split-R-hat compares the spread between the chain halves with the spread within them,
1 means they agree; bulk ESS is the effective sample size, the number of independent draws the
correlated chains are worth). A run that fails the gate is dropped for EVERY method, so all
comparisons share one run set; which runs are dropped is computed from the cached HMC diagnostics
at run time and is printed in the log and in the table comment, and the table comment also lists
Ours' and VI's model-level numbers with the dropped runs put back. Reads cached artifacts only;
never trains: a missing artifact raises FileNotFoundError naming the file.

Cell override: the environment variable REACDIFF_CELLS, a comma-separated list of "nu:rho" pairs
(for example REACDIFF_CELLS="0.5:1.0,0.5:2.0"), restricts the run to those cells (the outputs are
then marked PARTIAL); the default (unset or empty) is all 16 cells and never skips a cell. In
override mode the log, the table comment and a title line on the figure say which cells were used,
and the other cells appear in the table as "not run".
The override applies to both parts and to the shipped figure.
Within a cell every one of the five canonical trajectories must exist in both modes.


PART 2: variance consistency (full figure variance_consistency.pdf)

Is the predictive variance MEANINGFUL? Four questions, one figure (variance_consistency.pdf), on the PDEBench 1D
reaction-diffusion (Fisher-KPP) dataset, u_t = nu u_xx + rho u (1 - u), over the 4 x 4 grid of
settings nu in {0.5, 1, 2, 5} (diffusion coefficient) x rho in {1, 2, 5, 10} (reaction strength)
and the five canonical trajectories `idxs(5)` per setting (the same seeded draw the advection
figures use, see `pde/advection/trajectory_selection.py`). This is the reaction-diffusion port of
part 2 of `figures_onepass/advection/calibration_comparison.py`: what the wave speed b is
there, the reaction strength rho is here (it drives the nonlinearity, the failure modes and the
cost), and nu is the second dimension. Line panels are "vs rho" with the four nu values pooled into
every rho point (one rho point = 4 x 5 = 20 runs); the per-run panel shows nu by marker shape
(nu = 0.5 circle, 1 square, 2 diamond, 5 triangle). The model sees the normalized values
rho_norm = 2 rho and nu_norm = 2 nu (the axes show the physical rho). "Ours" is the one-pass EP
(expectation propagation: a Gaussian posterior built by one sweep of local moment matching)
posterior predictive; the reference is the multi-chain HMC (Hamiltonian Monte Carlo, a sampler
that is exact in the limit of many samples) posterior of the same model; VI (variational
inference) is the mean-field Gaussian baseline (mean-field: one independent Gaussian per weight
coordinate). All three share the model and its noise model.
The truth for every error is the float64 re-solve of the nominal PDE (`u_ref`), never the stored
PDEBench trajectory (its late-time freeze is a data artifact).

  (a) Does uncertainty grow with distance from the observations? Predictive std of Ours vs. time
      (observations exist only at t=0), one curve per rho: solid = mean over the 20 runs (5
      trajectories x 4 nu) of the spatial-mean predictive std, band = min to max over those runs.
  (b) Does the predicted std track the actual error? Binned reliability: grid points pooled
      across all runs are binned by predicted std (per-method quantile bins, so every bin holds
      the same share of that method's points); each bin shows the empirical RMSE of those points
      vs. the bin's RMS predicted std. A calibrated model follows y=x; a constant multiplicative
      miscalibration is a parallel offset. Shown for Ours, HMC and VI: HMC's own offset separates
      "EP approximation error" from "shared modeling assumptions", and VI shows where a cheaper
      Gaussian approximation lands. The x limits are the padded std range of the plotted bins
      (LOG_PAD of its log-span on each side); the y axis runs from the bottom of that x range up
      to the padded RMSE range, so the y=x reference line stays visible across the whole x range
      (RMSE/std per bin is in table block (b)). Caveat on the VI series: pooled quantile bins can
      sort by SETTING rather than by the error within a run when a method's predicted std and its
      accuracy both move with rho or nu (in the advection set the VI bins were wave-speed bins).
      Whether that happens here is recomputed from the data at run time: the share of the top and
      bottom N_EDGE_BINS VI bins that comes from each rho and from each nu (and from each cell),
      and every method's RMSE vs. the truth per rho and per cell, are written to the log and the
      table. VI is never dropped from the panel; the log axes are set from the plotted bins of all
      three methods.
  (c) Is the miscalibration a consistent factor? The ratio σ_Ours/σ_HMC on the predictive
      grid vs. time (median over space and over the 20 runs, per rho): a flat curve means Ours
      differs from the exact posterior by a roughly constant factor everywhere.
  (d) The same ratio's per-run median, one point per (rho, nu, trajectory) with nu as the marker
      shape: consistency across runs and settings rather than across the domain.

Run gate (the same rule as the advection set, and the same loader as part 1, so both parts share
one run set): a run enters only when its HMC reference
converged, i.e. at least HMC_MIN_ADMITTED (90%) of the eigenbasis coordinates (the rotated
weight coordinates all methods share) have split-R-hat < 1.05 (R-hat: agreement between chains,
1 = perfect) and bulk ESS > 100 (effective sample size: how many independent draws the chains are
worth). A run that fails the gate is dropped for EVERY method, so every panel and table block uses
the same run set. Which runs are dropped is decided from the cached HMC diagnostics at run time
(nothing is hard-coded): the log and the table comment list them with their admitted share and
chain acceptance rates, and state the number of runs used per cell. Reads cached artifacts only;
never trains; a missing artifact raises FileNotFoundError naming the file.

Cell selection: by default all 16 cells (every artifact must exist). REACDIFF_CELLS (see part 1)
restricts the whole script to the named cells; the log then states which cells were used and the
part-2 table marks the others as not selected (n/s). The default run is the one to report. Part 2
lists and pools the cells in its own order (rho outer, nu inner, VC_ALL_CELLS).

The companion table (`variance_consistency_table.tex`) holds the numbers behind each panel, for
ALL 16 cells explicitly (n = admitted runs per cell) plus the pooled rows the panels show:
(a) the mean-std curve at t=0 / t=0.5 / t=1 and its min-max band at t=1, (b) every reliability
bin for Ours, HMC and VI, then per cell each method's RMSE vs. the truth and the share of the VI
edge bins that the cell fills, (c) the ratio-vs-time curve at the same three times plus its range
over t, (d) the per-run medians' spread and the grid-point ratio quantiles. Panels (a), (c), (d)
are about Ours vs. HMC only; VI appears in panel (b) alone.

Run: python figures_onepass/diffusion/calibration_comparison.py [--png out.png]
     (+ the full figures: CALIBRATION_ALL_PANELS=1 in front of the same command; --png renders
     the shipped figure). The console output is also written to calibration_comparison.log.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogFormatterSciNotation, LogLocator, NullFormatter
from scipy import stats

FIGURES_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURES_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "pde" / "diffusion"))
sys.path.insert(0, str(REPO_ROOT / "pde" / "advection"))

import reacdiff_baselines_sweep as sweep  # noqa: E402  (load_cached / cache_path / assert_same_basis)
from trajectory_selection import idxs  # noqa: E402

OUTPUT_PDF = FIGURES_DIR / "calibration_variance.pdf"          # the shipped (merged) figure
OUTPUT_TEX = FIGURES_DIR / "calibration_variance_table.tex"    # its companion table
OUTPUT_LOG = FIGURES_DIR / "calibration_comparison.log"        # the log of the whole script
CAL_PDF = FIGURES_DIR / "calibration_comparison.pdf"           # full figure of part 1 (CALIBRATION_ALL_PANELS=1 only)
CAL_TEX = FIGURES_DIR / "calibration_comparison_table.tex"     # full table of part 1 (every run)
VC_PDF = FIGURES_DIR / "variance_consistency.pdf"              # full figure of part 2 (CALIBRATION_ALL_PANELS=1 only)
VC_TEX = FIGURES_DIR / "variance_consistency_table.tex"        # full table of part 2 (every run)
ALL_PANELS = os.environ.get("CALIBRATION_ALL_PANELS", "").strip() == "1"

NUS = list(sweep.NUS)  # diffusion coefficients of the dataset grid (0.5, 1, 2, 5)
RHOS = list(sweep.RHOS)  # reaction strengths of the dataset grid (1, 2, 5, 10): the x-axis of the per-rho panels
ALL_CELLS = [(nu, rho) for nu in NUS for rho in RHOS]  # nu outer: the order of part 1 (part 2: VC_ALL_CELLS)
IDXS = idxs(5)  # the five canonical trajectories of every cell (the same seeded draw as the advection figures)
CELLS_ENV = "REACDIFF_CELLS"
METHODS = ("Ours", "HMC", "VI")
NOMINAL_LEVELS = [0.5, 0.8, 0.9, 0.95]
HEADLINE_LEVELS = [0.9, 0.95]  # the headline levels; calibration_comparison.pdf panels (e)/(f) (panel (b) of the shipped figure) and both table blocks
MIN_VAR = 1e-10  # floor on predictive variance, avoids divide-by-zero (whether it binds is printed)
Z_HIST_MAX = 10  # the residual histogram (calibration_comparison.pdf panel (b)) drops |z| >= this
RHAT_MAX, ESS_MIN = 1.05, 100.0  # per-coordinate HMC convergence gate (as in the advection set)
HMC_MIN_ADMITTED = 0.9  # a run is used only if at least this fraction of its HMC coordinates pass the gate
COLORS = {"Ours": "#4a3aa7", "HMC": "#eb6834", "VI": "#008300"}
MARKERS = {"Ours": "o", "HMC": "s", "VI": "^"}
LEVEL_STYLES = {0.9: "-", 0.95: "--"}  # solid = 90% interval, dashed = 95% interval, everywhere
# Candidate labelled ticks of the log y-axes of the NLL and CRPS panels (calibration_comparison.pdf
# panels (c), (d); the CRPS panel is (a) of the shipped figure), written as plain numbers; the
# ticks and limits actually used are chosen at run time from the plotted values (see _log_y_axis),
# because the range depends on which runs the HMC gate admits.
NLL_YTICK_CANDIDATES = [t * 10 ** k for k in range(-1, 6) for t in (1, 2, 5)]
CRPS_YTICK_CANDIDATES = [t * 10 ** k for k in range(-5, 2) for t in (1, 2, 3, 5)]
LOG_Y_MARGIN = 1.35  # the axis limits sit this factor beyond the smallest / largest plotted value
# The assumption behind the closed forms of block 2 (written into the table comment; the same text
# as in figures_onepass/advection/calibration_comparison.py).
VERIFICATION_NOTE = (
    "Assumption: the closed forms of block 2 assume a Gaussian HMC predictive at every grid point, "
    "with HMC's stored mean m_H and variance v_H; they use these two moments, not the HMC draws."
)
CAPTION_NOTE = (
    "The closed forms of block 2 / panel (f) of the full figure calibration_comparison.pdf (not part of the "
    "shipped calibration_variance.pdf) assume a Gaussian HMC predictive at every grid point; the block-2 "
    "numbers are exact only insofar as that assumption holds."
)


# ============================================================================= PART 1: calibration
# (the panel letters in the comments refer to the full figure calibration_comparison.pdf)


# ----------------------------------------------------------------------------- log file


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


# ----------------------------------------------------------------------------- cells


def selected_cells() -> tuple[list[tuple[float, float]], bool]:
    """(cells to run, override?) from the REACDIFF_CELLS environment variable; default = all 16."""
    raw = os.environ.get(CELLS_ENV, "").strip()
    if not raw:
        return list(ALL_CELLS), False
    cells = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            nu_s, rho_s = item.split(":")
            nu, rho = float(nu_s), float(rho_s)
        except ValueError as e:
            raise ValueError(f"{CELLS_ENV} entry {item!r} is not of the form nu:rho") from e
        if (nu, rho) not in ALL_CELLS:
            raise ValueError(f"{CELLS_ENV} entry {item!r} is not a grid cell; nu in {NUS}, rho in {RHOS}")
        if (nu, rho) not in cells:
            cells.append((nu, rho))
    return cells, True


def cell_label(nu: float, rho: float) -> str:
    return f"nu={nu:g} rho={rho:g}"


# ----------------------------------------------------------------------------- data


def _slim(r: dict, truth: np.ndarray) -> dict:
    """Keep only the (n_x, n_t) predictive arrays a calibration statistic needs. `true_u` is the
    float64 reference solution u_ref (shared, not copied), never the stored PDEBench trajectory."""
    return dict(true_u=truth, pred_u=r["pred_u"], pred_var=r["pred_var"])


def _load(method: str, nu: float, rho: float, idx: int) -> dict:
    path = sweep.cache_path(method, nu, rho, idx)
    if not path.exists():
        raise FileNotFoundError(f"missing {method} artifact {path} ({cell_label(nu, rho)} idx={idx}); this script "
                                f"never trains, generate it with {sweep.RUNNERS[method].name}")
    return sweep.load_cached(method, nu, rho, idx)


def collect_records(cells: list[tuple[float, float]]) -> tuple[list[dict], list[dict]]:
    """One record per (nu, rho, idx) holding the three predictives, split into (admitted, excluded).
    Runs whose HMC reference did not converge are excluded for every method, so all statistics
    share one run set; the excluded records are kept only for the table comment that shows how
    much the exclusion changes the model-level numbers of Ours and VI. Also the loader of part 2
    (variance_records derives its arrays from these records)."""
    records, excluded = [], []
    for nu, rho in cells:
        for idx in IDXS:
            print(f"[{cell_label(nu, rho)} idx={idx}] loading Ours (one-pass EP) + multichain HMC + VI from cache...")
            hmc = _load("HMC", nu, rho, idx)
            ours = _load("Ours", nu, rho, idx)
            vi = _load("VI", nu, rho, idx)
            sweep.assert_same_basis(ours, hmc, vi)  # same problem, graph and eigenbasis V
            for other in (hmc, vi):
                assert np.array_equal(other["true_u"], ours["true_u"]), "methods must share the trajectory"
                assert np.array_equal(other["u_ref"], ours["u_ref"]), "methods must share the reference solution"
                assert np.allclose(other["t"], ours["t"]), "methods must share the time grid"
            truth = ours["u_ref"]
            admitted = float(np.mean((hmc["rhat"] < RHAT_MAX) & (hmc["ess"] > ESS_MIN)))
            rec = dict(nu=nu, rho=rho, idx=idx, hmc_admitted=admitted, hmc_accept=np.round(hmc["accept_rates"], 3).tolist(),
                       grid_shape=tuple(truth.shape), t=ours["t"],
                       Ours=_slim(ours, truth), HMC=_slim(hmc, truth), VI=_slim(vi, truth))
            if admitted < HMC_MIN_ADMITTED:
                print(f"[{cell_label(nu, rho)} idx={idx}] EXCLUDED from every method: only {admitted:.0%} of the HMC "
                      f"coordinates converged (chain acceptance rates "
                      f"{np.round(hmc['accept_rates'], 3).tolist()}), so there is no usable reference. "
                      "For the record, on this run alone (vs truth): "
                      + ", ".join(f"{name} NLL {nll(m['true_u'], *_moments(m)):.1f}, "
                                  f"cov@90 {100 * coverage(standardized_residuals(m), 0.9):.1f}%"
                                  for name, m in (("Ours", rec["Ours"]), ("HMC", rec["HMC"]), ("VI", rec["VI"]))))
                excluded.append(rec)
                continue
            records.append(rec)
    shapes = {r["grid_shape"] for r in records + excluded}
    assert len(shapes) <= 1, f"every run must share one (x,t) grid (the means of per-run means assume it): {shapes}"
    return records, excluded


def _select(records: list[dict], rho: float | None = None, nu: float | None = None) -> list[dict]:
    """Runs of one cell (nu, rho), of one rho pooled over nu (nu=None), or of everything."""
    return [r for r in records if (rho is None or r["rho"] == rho) and (nu is None or r["nu"] == nu)]


def dropped_runs_text(excluded: list[dict]) -> str:
    """One line naming the runs the HMC gate dropped (with what the gate saw), or 'none'."""
    if not excluded:
        return "none"
    return "; ".join(f"{cell_label(r['nu'], r['rho'])} idx {r['idx']} ({r['hmc_admitted']:.0%} of the HMC coordinates "
                     f"admitted, chain acceptance {r['hmc_accept']})" for r in excluded)


def n_per_rho_text(records: list[dict], cells: list[tuple[float, float]]) -> str:
    """'n = 20 of 20 at rho=1, ...': admitted runs out of the canonical trajectories of the cells run
    (only the rho values with a cell run are listed, so override mode lists fewer)."""
    return "n = " + ", ".join(f"{len(_select(records, rho))} of {len(IDXS) * sum(1 for _, r in cells if r == rho)} "
                              f"at rho={rho:g}" for rho in RHOS if any(r == rho for _, r in cells))


def _moments(m: dict) -> tuple[np.ndarray, np.ndarray]:
    """(predictive mean, floored predictive variance) of one method on one run."""
    return m["pred_u"], np.maximum(m["pred_var"], MIN_VAR)


# ----------------------------------------------------------------------------- model level (vs truth)


def standardized_residuals(m: dict) -> np.ndarray:
    mean, var = _moments(m)
    return ((m["true_u"] - mean) / np.sqrt(var)).ravel()


def pooled_z(records: list[dict], method: str, rho: float | None = None, nu: float | None = None) -> np.ndarray:
    """Standardized residuals of `method`, pooled over the runs of one cell, one rho (all nu) or all."""
    return np.concatenate([standardized_residuals(r[method]) for r in _select(records, rho, nu)])


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


def mean_score(records: list[dict], method: str, score_fn, rho: float | None = None, nu: float | None = None) -> float:
    """Mean of the per-run scores of `method` (every run has the same grid, so this equals the
    pooled grid-point mean), over one cell, one rho (all nu) or all."""
    return float(np.mean([score_fn(r[method]["true_u"], r[method]["pred_u"], r[method]["pred_var"])
                          for r in _select(records, rho, nu)]))


def rmse_of_mean(m: dict) -> float:
    """RMSE (root-mean-square error) of one method's posterior mean against the truth on one run;
    only used in the table comment, to show whether Ours and HMC share the same residuals."""
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


def mean_inference_stat(records: list[dict], method: str, fn, rho: float | None = None,
                        nu: float | None = None, **kw) -> float:
    return float(np.mean([fn(r, method, **kw) for r in _select(records, rho, nu)]))


# ----------------------------------------------------------------------------- panels


def _style(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel, fontsize=24)
    ax.set_ylabel(ylabel, fontsize=24)
    ax.set_title(title, fontsize=26)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=18)


def _plotted_rhos(records: list[dict]) -> list[float]:
    """The rho values with at least one admitted run (all four in the default mode; in override
    mode only the rho values of the cells run; a rho whose every run failed the gate is absent)."""
    return [rho for rho in RHOS if _select(records, rho)]


def _rho_axis(ax) -> None:
    ax.set_xscale("log")
    ax.set_xticks(RHOS)
    ax.set_xticklabels([f"${r:g}$" for r in RHOS])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(RHOS[0] / 1.4, RHOS[-1] * 1.4)


def _log_y_axis(ax, tick_candidates: list[float], values: list[float]) -> tuple[float, float]:
    """Log y-axis whose limits bracket every plotted value by the factor LOG_Y_MARGIN, with the
    candidate ticks inside those limits labelled as plain numbers (no 10^k) and no minor tick
    labels. Returns the limits. A mean NLL can be negative in principle (a run whose predictive
    is both sharp and right), so the log axis needs every plotted mean to be positive; if one is
    not, the axis falls back to a symmetric-log scale (linear near zero, logarithmic beyond 1)
    and says so on the axis."""
    lo, hi = min(values), max(values)
    if lo <= 0:
        print(f"WARNING: a plotted mean score is {lo:.3g} <= 0; using a symlog y-axis instead of a log axis")
        ax.set_yscale("symlog", linthresh=1.0)
        ax.set_title(ax.get_title() + r"$\ \mathrm{(symlog\ axis)}$", fontsize=26)
        return ax.get_ylim()
    ylim = (lo / LOG_Y_MARGIN, hi * LOG_Y_MARGIN)
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


def _coverage_legend(ax, handles: list[Line2D], data_max: float) -> None:
    """Legend of the coverage-per-rho panels (calibration_comparison.pdf panels (e)/(f)): in the band between the data and the nominal lines (0.9 / 0.95)
    when that band is empty, else wherever matplotlib finds room; which was used is printed."""
    if data_max < 0.62:
        ax.legend(handles=handles, fontsize=16, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 0.86))
        print(f"  legend placed in the empty band above the data (data max {data_max:.2f})")
    else:
        ax.legend(handles=handles, fontsize=16, ncol=2, loc="best")
        print(f"  legend placed by matplotlib ('best'): the data reach {data_max:.2f}, so the band below the nominal lines is not empty")


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
                      tick_candidates: list[float], legend_kw: dict) -> None:
    rhos = _plotted_rhos(records)
    values = {method: [mean_score(records, method, score_fn, rho) for rho in rhos] for method in METHODS}
    for method in METHODS:
        ax.plot(rhos, values[method], marker=MARKERS[method], color=COLORS[method],
                markersize=14, linewidth=2.5, label=method)
    _rho_axis(ax)
    _style(ax, r"$\rho$", ylabel, title)
    ylim = _log_y_axis(ax, tick_candidates, [v for vals in values.values() for v in vals])
    print(f"{title}: y-axis {ylim[0]:.4g} to {ylim[1]:.4g}; per-rho means (pooled over nu) at rho={rhos}: "
          + "; ".join(f"{m} {np.round(values[m], 4).tolist()}" for m in METHODS))
    ax.legend(fontsize=18, **legend_kw)


def plot_truth_coverage_per_rho(ax, records: list[dict], tag: str) -> None:
    rhos = _plotted_rhos(records)
    data_max = 0.0
    for method in METHODS:
        for level in HEADLINE_LEVELS:
            vals = [coverage(pooled_z(records, method, rho), level) for rho in rhos]
            data_max = max(data_max, max(vals))
            ax.plot(rhos, vals, ls=LEVEL_STYLES[level], marker=MARKERS[method], color=COLORS[method],
                    markersize=12, linewidth=2.5)
    for level in HEADLINE_LEVELS:
        ax.axhline(level, color="k", ls=LEVEL_STYLES[level], lw=1.5, alpha=0.5)
    ax.text(RHOS[0] / 1.3, 0.925, r"$\mathrm{nominal}$", fontsize=16, color="k", alpha=0.7, va="center")
    _rho_axis(ax)
    ax.set_ylim(0, 1.0)
    _style(ax, r"$\rho$", r"$\mathrm{empirical\ coverage\ of\ the\ truth}$",
           r"$\mathrm{(%s)\ 90\%%/95\%%\ coverage\ of\ the\ truth}$" % tag)
    print(f"({tag}) coverage of the truth per rho (pooled over nu):")
    _coverage_legend(ax, [_method_handle(m) for m in METHODS] + _level_handles(), data_max)


def plot_hmc_coverage_per_rho(ax, records: list[dict], tag: str) -> None:
    rhos = _plotted_rhos(records)
    data_max = 0.0
    for method in ("Ours", "VI"):
        for level in HEADLINE_LEVELS:
            vals = [mean_inference_stat(records, method, hmc_coverage, rho, level=level) for rho in rhos]
            data_max = max(data_max, max(vals))
            ax.plot(rhos, vals, ls=LEVEL_STYLES[level], marker=MARKERS[method], color=COLORS[method],
                    markersize=12, linewidth=2.5)
    for level in HEADLINE_LEVELS:  # HMC against itself is nominal by construction
        ax.axhline(level, color=COLORS["HMC"], ls=LEVEL_STYLES[level], lw=2.5, alpha=0.9)
    _rho_axis(ax)
    ax.set_ylim(0, 1.0)
    _style(ax, r"$\rho$", r"$\mathrm{coverage\ of\ the\ HMC\ predictive}$",
           r"$\mathrm{(%s)\ 90\%%/95\%%\ coverage\ of\ the\ HMC\ predictive}$" % tag)
    handles = [_method_handle("Ours"), _method_handle("VI"),
               Line2D([], [], color=COLORS["HMC"], lw=2.5, label=r"$\mathrm{HMC\ (nominal)}$")] + _level_handles()
    print(f"({tag}) coverage of the HMC predictive per rho (pooled over nu):")
    _coverage_legend(ax, handles, data_max)


def make_calibration_figure(records: list[dict], output_path: Path, png_path: Path | None = None,
                            cells_note: str | None = None) -> None:
    """The full six-panel calibration_comparison.pdf (CALIBRATION_ALL_PANELS=1 only)."""
    plt.rcParams["mathtext.fontset"] = "cm"
    fig, axes = plt.subplots(3, 2, figsize=(17, 22))
    plot_coverage(axes[0, 0], records, "a")
    plot_residuals(axes[0, 1], records, "b")
    plot_scoring_rule(axes[1, 0], records, nll, r"$\mathrm{(c)\ NLL\ (vs.\ truth)}$", r"$\mathrm{NLL}$",
                      NLL_YTICK_CANDIDATES, dict(loc="best"))
    plot_scoring_rule(axes[1, 1], records, crps_gaussian, r"$\mathrm{(d)\ CRPS\ (vs.\ truth)}$", r"$\mathrm{CRPS}$",
                      CRPS_YTICK_CANDIDATES, dict(loc="best"))
    plot_truth_coverage_per_rho(axes[2, 0], records, "e")
    plot_hmc_coverage_per_rho(axes[2, 1], records, "f")
    fig.tight_layout()
    if cells_note:  # override mode: say on the figure itself that it is partial
        fig.suptitle(cells_note, fontsize=20, color="#b00020", y=1.01)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved figure to {output_path}")
    if png_path is not None:
        fig.savefig(png_path, bbox_inches="tight", dpi=60)
        print(f"Saved PNG preview to {png_path}")
    plt.close(fig)


# ----------------------------------------------------------------------------- table and log


def _n_runs(records: list[dict], rho: float | None, nu: float | None = None) -> int:
    return len(_select(records, rho, nu))


def _row_labels(nu: float | None, rho: float | None) -> str:
    """'nu & rho' cells of a table row: a number, or 'all' = pooled over that variable."""
    return f"{'all' if nu is None else f'{nu:g}'} & {'all' if rho is None else f'{rho:g}'}"


def table_rows() -> list[tuple[float | None, float | None]]:
    """(nu, rho) of every table row per method: all 16 cells, the 4 per-rho rows pooled over nu,
    the grand pooled row."""
    return ALL_CELLS + [(None, rho) for rho in RHOS] + [(None, None)]


def model_level_headline(records: list[dict], method: str, rho: float | None, nu: float | None = None) -> tuple[float, float, float]:
    """(cov@90 in %, cov@95 in %, NLL) of `method` against the truth, over the runs of one cell,
    one rho (all nu) or all."""
    z = pooled_z(records, method, rho, nu)
    return 100 * coverage(z, 0.9), 100 * coverage(z, 0.95), mean_score(records, method, nll, rho, nu)


def _with_failed_run_lines(records: list[dict], excluded: list[dict]) -> list[str]:
    """Comment lines with the model-level headline numbers of Ours and VI when the runs the HMC
    gate excluded are put back (n then equals the full canonical set in the affected cells), and
    how much the exclusion moved Ours' pooled numbers."""
    if not excluded:
        return [r"% (no run was excluded, so the table already holds every run)"]
    everything = records + excluded
    groups = sorted({(r["nu"], r["rho"]) for r in excluded}) + [(None, None)]
    which = ", ".join(f"{cell_label(r['nu'], r['rho'])} idx {r['idx']}" for r in excluded)
    n_text = ", ".join(f"{_n_runs(everything, rho, nu)} at {cell_label(nu, rho)}" for nu, rho in groups if nu is not None)
    lines = [f"% For reference, the model-level numbers WITH the excluded run(s) put back ({which}; n = {n_text}, "
             f"{_n_runs(everything, None)} pooled;",
             r"% the excluded run(s) have no usable HMC reference, so only Ours and VI are listed):"]
    for method in ("Ours", "VI"):
        parts = []
        for nu, rho in groups:
            c90, c95, nll_val = model_level_headline(everything, method, rho, nu)
            label = "pooled" if nu is None else cell_label(nu, rho)
            parts.append(f"{label} (n={_n_runs(everything, rho, nu)}): cov@90 {c90:.1f}%, cov@95 {c95:.1f}%, NLL {nll_val:.1f}")
        lines.append(f"%   {method:4s} {';  '.join(parts)}")
    c90_in, c95_in, nll_in = model_level_headline(everything, "Ours", None)
    c90_ex, c95_ex, nll_ex = model_level_headline(records, "Ours", None)
    lines.append(rf"% Dropping the run(s) therefore moves Ours' pooled cov@90 / cov@95 by {c90_ex - c90_in:+.2f} / {c95_ex - c95_in:+.2f} pp "
                 rf"and its pooled NLL from {nll_in:.2f} to {nll_ex:.2f}.")
    return lines


def rel_l2_of_mean(m: dict) -> float:
    """Relative L2 error of one method's posterior mean against the truth on one run; ~1 means the
    mean has collapsed to about zero. Only used in the table comment and the log."""
    return float(np.linalg.norm(m["true_u"] - m["pred_u"]) / np.linalg.norm(m["true_u"]))


def _rel_l2_text(records: list[dict], method: str) -> str:
    return ", ".join(f"{np.mean([rel_l2_of_mean(r[method]) for r in _select(records, rho)]):.2f} at rho={rho:g}"
                     for rho in _plotted_rhos(records))


def write_calibration_table(records: list[dict], excluded: list[dict], cells: list[tuple[float, float]],
                            override: bool, path: Path) -> None:
    """LaTeX table of part 1 (calibration_comparison_table.tex), two blocks. Block 1 (model level, vs truth): per cell (nu, rho), per
    rho pooled over nu, and pooled: coverage at each nominal level (panels a/e), std of the
    standardized residuals (panel b), mean NLL / CRPS (panels c/d), for Ours, HMC and VI. Block 2
    (inference level, vs the HMC predictive): coverage at 90%/95% (panel f) and the expected NLL
    under the HMC predictive, for Ours and VI, plus HMC against itself (nominal coverage, entropy
    floor). `excluded` (the runs dropped by the HMC gate) only feeds the block 1 comment. A row
    with no admitted run shows 'not run' in every column when none of its cells was run (override
    mode only) and 'dropped' when its cells were run but every run failed the HMC gate."""
    # numbers for the block 1 comment: do Ours and HMC share the residuals of the mean, and how do the variances compare?
    rmse = {m: np.array([rmse_of_mean(r[m]) for r in records]) for m in ("Ours", "HMC")}
    rmse_gap = float(np.max(np.abs(rmse["Ours"] - rmse["HMC"])))
    std_ratio = float(np.median(np.concatenate([np.sqrt(_moments(r["HMC"])[1] / _moments(r["Ours"])[1]).ravel()
                                                for r in records])))
    nll_ours, nll_hmc = mean_score(records, "Ours", nll), mean_score(records, "HMC", nll)
    n_floor = sum(int(np.sum(r[m]["pred_var"] < MIN_VAR)) for r in records for m in METHODS)
    n_x, n_t = records[0]["grid_shape"]
    cells_text = (f"OVERRIDE MODE ({CELLS_ENV}): only the cells {', '.join(cell_label(*c) for c in cells)} were run; "
                  "the other cells are marked 'not run'. This is NOT the final table." if override
                  else "All 16 cells of the grid were run.")

    def empty_row(nu: float | None, rho: float | None, n_cols: int) -> str:
        """Data columns of a row with no admitted run: 'dropped' if a cell of the row was run (so every
        run of it failed the HMC gate), 'not run' if none was (override mode)."""
        was_run = any((nu is None or c[0] == nu) and (rho is None or c[1] == rho) for c in cells)
        return " & ".join(["dropped" if was_run else "not run"] * n_cols)

    lines = [
        r"% Auto-generated by figures_onepass/diffusion/calibration_comparison.py (part 1; the full table of the",
        r"% full six-panel calibration_comparison.pdf, whose panel letters are used below; panels (d) and (e) are",
        r"% panels (a) and (b) of the shipped calibration_variance.pdf).",
        r"% Ours = one-pass EP (expectation propagation) posterior predictive; HMC = 4-chain Hamiltonian Monte",
        r"% Carlo reference (3000 pooled draws); VI = mean-field (every weight gets its own independent Gaussian)",
        *[f"% {line}" for line in textwrap.wrap(
            "variational posterior. Dataset: PDEBench 1D Fisher-KPP, u_t = nu u_xx + rho u (1 - u), 16 cells "
            f"nu in {NUS} x rho in {RHOS}; the model uses rho_norm = 2 rho and nu_norm = 2 nu. Per cell: the "
            f"five canonical trajectories idxs(5) = {IDXS} whose HMC reference converged ({n_per_rho_text(records, cells)}; "
            f"{len(records)} pooled). {cells_text} A run enters only if at least {HMC_MIN_ADMITTED:.0%} of its HMC "
            f"coordinates have split-R-hat < {RHAT_MAX} (split-R-hat compares the spread between the chain halves "
            f"with the spread within them, 1 = agreement) and bulk ESS > {ESS_MIN:.0f} (bulk ESS = effective sample "
            "size, the number of independent draws the correlated chains are worth); a run that fails is "
            f"dropped for EVERY method. Dropped by the HMC gate: {dropped_runs_text(excluded)} (a row whose every "
            "run failed the gate reads 'dropped'). Rows: every (nu, rho) cell, then 'nu = all' rows = pooled over the four nu values at that rho (what panels c-f "
            "plot, up to 4 x 5 = 20 runs), then 'all, all' = every admitted run of every cell (panels a/b). The "
            "truth is the float64 re-solve of the nominal PDE from the dataset's initial condition (u_ref, "
            "pde/diffusion/reacdiff_reference.py), not the stored PDEBench trajectory (its late-time freeze is a "
            f"data artifact). Every statistic uses the full {n_x} x {n_t} (x,t) grid of each run; sigma = "
            f"sqrt(max(pred_var, {MIN_VAR:g})) (the floor binds at {n_floor} grid points over all runs and methods). "
            "Mean relative L2 error of the posterior mean against the truth, per rho (pooled over nu; about 1 = the "
            f"mean is about zero): Ours {_rel_l2_text(records, 'Ours')}; HMC {_rel_l2_text(records, 'HMC')}; "
            f"VI {_rel_l2_text(records, 'VI')}.", 100)],
        r"%",
        r"% BLOCK 1, model-level calibration (against the TRUE solution). z = (u_true - u_pred)/sigma.",
        r"% Coverage = fraction of grid points with |z| <= z_level (z_level = 0.674 / 1.282 / 1.645 / 1.960 at",
        r"% 50 / 80 / 90 / 95%); std(z) 'all' is the std of z over every grid point, std(z) '|z|<10' the std over",
        r"% the grid points with |z| < 10 only (the points the residual histogram of panel (b) is drawn from);",
        r"% NLL (negative log-likelihood) = mean over grid points of 0.5 ln(2 pi sigma^2) + z^2/2 (Gaussian",
        r"% predictive); CRPS (continuous ranked probability score, a proper scoring rule in the units of u) =",
        r"% mean closed-form Gaussian CRPS. NLL and CRPS are means of the per-run means (what panels c/d plot).",
        r"% The coverage of the truth is shared by all three methods to the extent that the dominant error is",
        r"% model-form error (the truncated basis cannot represent the true solution, and that error is not in",
        r"% the noise model beta_ic / beta_pde), so the coverage columns measure the model more than the",
        r"% inference method; the NLL gap between the methods is an inference effect to the extent that the",
        r"% per-run RMSE (root-mean-square error) of the posterior mean against the truth agrees between Ours",
        rf"% and HMC: largest per-run difference {rmse_gap:.4f} over the {len(records)} runs (HMC RMSE from",
        rf"% {rmse['HMC'].min():.4f} to {rmse['HMC'].max():.4f}); HMC's predictive std is a median {std_ratio:.2f}x Ours' over the grid;",
        rf"% pooled NLL {nll_ours:.2f} (Ours) vs {nll_hmc:.2f} (HMC).",
        *_with_failed_run_lines(records, excluded),
        r"\begin{tabular}{l r r r rrrr rr rr}",
        r"\toprule",
        r" & & & & \multicolumn{4}{c}{Coverage of the truth at nominal level} & \multicolumn{2}{c}{std$(z)$} & & \\",
        r"\cmidrule(lr){5-8}\cmidrule(lr){9-10}",
        r"Method & $\nu$ & $\rho$ & $n$ & 50\% & 80\% & \textbf{90\%} & \textbf{95\%} & all & $|z|<10$ & \textbf{NLL} & CRPS \\",
        r"\midrule",
    ]
    for method in METHODS:
        for nu, rho in table_rows():
            n = _n_runs(records, rho, nu)
            if n == 0:
                lines.append(f"{method} & {_row_labels(nu, rho)} & 0 & {empty_row(nu, rho, 8)} \\\\")
                continue
            z = pooled_z(records, method, rho, nu)
            z_hist = z[np.abs(z) < Z_HIST_MAX]
            cov = " & ".join(f"{100 * coverage(z, lev):.1f}\\%" for lev in NOMINAL_LEVELS)
            lines.append(f"{method} & {_row_labels(nu, rho)} & {n} & {cov} & {z.std():.1f} & "
                         f"{z_hist.std():.2f} & {mean_score(records, method, nll, rho, nu):.1f} & "
                         f"{mean_score(records, method, crps_gaussian, rho, nu):.4f} \\\\")
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
        r"% intervals are too narrow and/or its mean is off; E-NLL minus the floor = the excess in nats (natural-log",
        r"% units of NLL).",
        r"% CAVEAT FOR THE CAPTION:",
        *[f"% {line}" for line in textwrap.wrap(CAPTION_NOTE, 100)],
        *[f"% {line}" for line in textwrap.wrap(VERIFICATION_NOTE, 100)],
        r"\begin{tabular}{l r r r rr rr}",
        r"\toprule",
        r" & & & & \multicolumn{2}{c}{Coverage of the HMC predictive} & \multicolumn{2}{c}{Expected NLL under HMC} \\",
        r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        r"Method & $\nu$ & $\rho$ & $n$ & \textbf{90\%} & \textbf{95\%} & \textbf{E-NLL} & E-NLL $-$ floor \\",
        r"\midrule",
    ]
    for method in ("Ours", "VI", "HMC"):
        label = "HMC (itself)" if method == "HMC" else method
        for nu, rho in table_rows():
            n = _n_runs(records, rho, nu)
            if n == 0:
                lines.append(f"{label} & {_row_labels(nu, rho)} & 0 & {empty_row(nu, rho, 4)} \\\\")
                continue
            covs = " & ".join(f"{100 * mean_inference_stat(records, method, hmc_coverage, rho, nu, level=lev):.1f}\\%"
                              for lev in HEADLINE_LEVELS)
            enll = mean_inference_stat(records, method, hmc_expected_nll, rho, nu)
            floor = mean_inference_stat(records, "HMC", hmc_expected_nll, rho, nu)
            lines.append(f"{label} & {_row_labels(nu, rho)} & {n} & {covs} & {enll:.2f} & {enll - floor:.2f} \\\\")
        if method != "HMC":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines))
    print(f"Saved table to {path}")


def print_calibration_summary(records: list[dict], excluded: list[dict], cells: list[tuple[float, float]]) -> None:
    n_floor = sum(int(np.sum(r[m]["pred_var"] < MIN_VAR)) for r in records for m in METHODS)
    n_x, n_t = records[0]["grid_shape"]
    print(f"\n{len(records)} runs used ({n_per_rho_text(records, cells)}); grid {n_x} x {n_t} (x,t) points per run; "
          f"grid points at the variance floor: {n_floor}")
    print(f"Dropped by the HMC gate (>= {HMC_MIN_ADMITTED:.0%} of the coordinates with split-R-hat < {RHAT_MAX} and "
          f"bulk ESS > {ESS_MIN:.0f}): {dropped_runs_text(excluded)}")
    print("HMC gate per run (fraction of coordinates admitted, chain acceptance rates):")
    for r in records + excluded:
        print(f"  {cell_label(r['nu'], r['rho']):18s} idx {r['idx']:5d}: {r['hmc_admitted']:.1%}  {r['hmc_accept']}"
              + ("   EXCLUDED" if r in excluded else ""))
    for method in METHODS:
        print(f"{method} relative L2 error of the posterior mean vs truth (||u_true - u_pred|| / ||u_true||; about 1 = "
              f"the mean is about zero), per rho (pooled over nu): {_rel_l2_text(records, method)}")
    print("\nModel level (vs the true solution), per cell, per rho (nu = all) and pooled:")
    print(f"  {'method':6s} {'nu':>4s} {'rho':>4s} {'n':>3s}  cov@90   cov@95   NLL        CRPS")
    for method in METHODS:
        for nu, rho in table_rows():
            n = _n_runs(records, rho, nu)
            if n == 0:
                continue
            z = pooled_z(records, method, rho, nu)
            print(f"  {method:6s} {_row_labels(nu, rho).replace(' & ', ' '):>9s} {n:3d}  {100 * coverage(z, 0.9):5.1f}%   "
                  f"{100 * coverage(z, 0.95):5.1f}%   {mean_score(records, method, nll, rho, nu):8.2f}   "
                  f"{mean_score(records, method, crps_gaussian, rho, nu):.4f}")
    rmse = {m: [rmse_of_mean(r[m]) for r in records] for m in ("Ours", "HMC")}
    print(f"\nPer-run RMSE of the posterior mean vs truth, Ours vs HMC: largest |difference| "
          f"{np.max(np.abs(np.subtract(rmse['Ours'], rmse['HMC']))):.4f} (HMC RMSE from {min(rmse['HMC']):.4f} to {max(rmse['HMC']):.4f})")
    if excluded:
        print("\nModel level WITH the excluded run(s) put back (Ours and VI; the excluded run(s) have no usable HMC reference):")
        for line in _with_failed_run_lines(records, excluded)[2:]:
            print("  " + line.lstrip("% "))
    print("\nInference level (vs the HMC predictive), per cell, per rho (nu = all) and pooled:")
    print(f"  {'method':6s} {'nu':>4s} {'rho':>4s} {'n':>3s}  cov@90   cov@95   E-NLL     floor (HMC entropy)")
    for method in ("Ours", "VI"):
        for nu, rho in table_rows():
            n = _n_runs(records, rho, nu)
            if n == 0:
                continue
            c90 = mean_inference_stat(records, method, hmc_coverage, rho, nu, level=0.9)
            c95 = mean_inference_stat(records, method, hmc_coverage, rho, nu, level=0.95)
            enll = mean_inference_stat(records, method, hmc_expected_nll, rho, nu)
            floor = mean_inference_stat(records, "HMC", hmc_expected_nll, rho, nu)
            print(f"  {method:6s} {_row_labels(nu, rho).replace(' & ', ' '):>9s} {n:3d}  {100 * c90:5.1f}%   {100 * c95:5.1f}%   "
                  f"{enll:7.2f}   {floor:7.2f}")
    print("\nCAVEAT: " + CAPTION_NOTE)


# ============================================================================= PART 2: variance consistency
# (the panel letters in the comments refer to the full figure variance_consistency.pdf)

VC_ALL_CELLS = [(nu, rho) for rho in RHOS for nu in NUS]  # rho outer, nu inner: the order of part 2 (its tables, pooling)
NU_MARKERS = {0.5: "o", 1.0: "s", 2.0: "D", 5.0: "^"}  # nu by marker shape
VC_MIN_VAR = 1e-14  # floor on the predictive variance of part 2 (part 1 uses MIN_VAR = 1e-10)
N_SIGMA_BINS = 14
MIN_BIN_POINTS = 100  # panel (b) bins with fewer pooled grid points are dropped
LOG_PAD = 0.25  # panel (b): each axis pads the plotted bins' range by this fraction of its log-span per side
N_EDGE_BINS = 4  # the VI setting caveat quotes the top and bottom this-many bins
LEGEND_HEADROOM = 0.3  # panel (d): extra log-span above the data reserved for the nu-marker legend
# (legend label, predictive-std key, absolute-error key) for every series of panel (b)
RELIABILITY_SERIES = (("Ours", "sig_mp", "err"), ("HMC", "sig_hmc", "err_hmc"), ("VI", "sig_vi", "err_vi"))
RELIABILITY_STYLES = {"Ours": dict(color="#4a3aa7", marker="o"),
                      "HMC": dict(color="#eb6834", marker="s"),
                      "VI": dict(color="#008300", marker="^")}
# Font sizes, line widths and marker sizes of the part-2 panels: VC_STYLE is the full figure
# variance_consistency.pdf (7.25 x 5.75 inch panels); MERGED_VC_STYLE matches panels (a)/(b) of the
# shipped figure (8.5 x 7.33 inch panels, the sizes of _style in part 1). plain_log_ticks: label the
# log axes with plain numbers (as panel (a) of the shipped figure) instead of 'k x 10^-1'.
# growth_ncol / growth_headroom: columns of the rho legend of the std-vs-time panel, and the share of
# that panel's height kept free of data at the top for it (None: no reserved band); the larger fonts
# of the shipped figure need the one-row legend and the band, or the legend covers the rho=1 band.
VC_STYLE = dict(label=19, title=20, tick=13, legend=13, legend_title=12, legend_rel=14, xticklabel=15, lw=2.0,
                s=46, nu_marker=8, nu_legend=12, plain_log_ticks=False, growth_ncol=2, growth_headroom=None)
MERGED_VC_STYLE = dict(label=24, title=26, tick=18, legend=18, legend_title=17, legend_rel=18, xticklabel=18, lw=2.5,
                       s=76, nu_marker=11, nu_legend=16, plain_log_ticks=True, growth_ncol=4, growth_headroom=0.22)


def cell_text(nu: float, rho: float) -> str:
    return f"nu={nu:g} rho={rho:g}"


def variance_records(records: list[dict], excluded: list[dict], cells) -> tuple[list[dict], list[dict]]:
    """(records, dropped) of part 2, derived from the loaded records of part 1 (the same runs, the
    same HMC gate) and put in the order of part 2 (VC_ALL_CELLS, then IDXS): per admitted run the
    absolute errors of the three predictive means against the truth u_ref and the three predictive
    stds (floored at VC_MIN_VAR); per run the gate dropped, its reason (admitted share and chain
    acceptance rates). `cells` = the selected cells in the order of VC_ALL_CELLS."""
    order = {c: i for i, c in enumerate(VC_ALL_CELLS)}

    def key(r):
        return order[(r["nu"], r["rho"])], IDXS.index(r["idx"])

    vrecords = []
    for r in sorted(records, key=key):
        mp, hmc, vi = r["Ours"], r["HMC"], r["VI"]
        vrecords.append(dict(nu=r["nu"], rho=r["rho"], idx=r["idx"], t=r["t"],
                             err=np.abs(mp["true_u"] - mp["pred_u"]),
                             sig_mp=np.sqrt(np.maximum(mp["pred_var"], VC_MIN_VAR)),
                             err_hmc=np.abs(hmc["true_u"] - hmc["pred_u"]),
                             sig_hmc=np.sqrt(np.maximum(hmc["pred_var"], VC_MIN_VAR)),
                             err_vi=np.abs(vi["true_u"] - vi["pred_u"]),
                             sig_vi=np.sqrt(np.maximum(vi["pred_var"], VC_MIN_VAR))))
    dropped = [dict(nu=r["nu"], rho=r["rho"], idx=r["idx"],
                    reason=f"{r['hmc_admitted']:.0%} admitted, chain acceptance {r['hmc_accept']}")
               for r in sorted(excluded, key=key)]
    empty = [rho for rho in rhos_of(cells) if not _at(vrecords, rho=rho)]
    if empty:
        raise RuntimeError(f"no admitted run at rho in {_set_text(empty)} (dropped: {dropped_text(dropped)}); "
                           "every rho of the selected cells needs at least one run")
    print(f"{len(vrecords)} runs used ({', '.join(n_used_text(vrecords, cells, rho) for rho in rhos_of(cells))}); "
          f"dropped by the HMC gate: {dropped_text(dropped)}")
    return vrecords, dropped


def rhos_of(cells) -> list[float]:
    return [rho for rho in RHOS if any(c[1] == rho for c in cells)]


def nus_of(cells) -> list[float]:
    return [nu for nu in NUS if any(c[0] == nu for c in cells)]


def _at(records, rho=None, nu=None):
    return [r for r in records if (rho is None or r["rho"] == rho) and (nu is None or r["nu"] == nu)]


def n_used_text(records, cells, rho) -> str:
    """'k of n at rho=.. (nu in {..})': runs admitted at one rho out of the runs of its selected cells."""
    nus = [c[0] for c in cells if c[1] == rho]
    return f"{len(_at(records, rho=rho))} of {len(nus) * len(IDXS)} at rho={rho:g} (nu in {_set_text(nus)})"


def dropped_text(dropped) -> str:
    """The drop list as prose: 'nu=.. rho=.. idx .. (..% admitted, chain acceptance ..)' per run the HMC
    gate rejected, or 'none'."""
    if not dropped:
        return "none"
    return "; ".join(f"{cell_text(d['nu'], d['rho'])} idx {d['idx']} ({d['reason']})" for d in dropped)


def _set_text(values) -> str:
    return "{" + ", ".join(f"{v:g}" for v in values) + "}"


# ----------------------------------------------------------------------------- metrics
def growth_curves(records, rho, nu=None):
    """(t, curves): per-run spatial-mean predictive std, shape (n_runs, n_t), panel (a); the runs
    at one rho (all nu pooled) or at one cell (nu given)."""
    rs = _at(records, rho=rho, nu=nu)
    return rs[0]["t"], np.stack([r["sig_mp"].mean(axis=0) for r in rs])


def _pooled(records, sig_key, err_key):
    """Every grid point of every record: (predicted std, |error|, index of its cell in VC_ALL_CELLS)."""
    sig = np.concatenate([r[sig_key].ravel() for r in records])
    err = np.concatenate([r[err_key].ravel() for r in records])
    cell = np.concatenate([np.full(r[sig_key].size, VC_ALL_CELLS.index((r["nu"], r["rho"]))) for r in records])
    return sig, err, cell


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


def bin_cell_shares(records, sig_key, err_key) -> dict[int, np.ndarray]:
    """For every kept bin of one method, the share of its points that come from each cell of
    VC_ALL_CELLS (a length-16 vector per bin)."""
    sig, _, cell = _pooled(records, sig_key, err_key)
    which = _bin_index(sig)
    return {b: np.bincount(cell[which == b], minlength=len(VC_ALL_CELLS)) / (which == b).sum()
            for b in range(N_SIGMA_BINS) if (which == b).sum() >= MIN_BIN_POINTS}


def edge_bin_share(shares: dict[int, np.ndarray], cells) -> tuple[float, float]:
    """(share in the top N_EDGE_BINS bins, share in the bottom N_EDGE_BINS bins) of the points that
    come from the cells `cells`, each averaged over those bins (the advection script's rule)."""
    sel = [VC_ALL_CELLS.index(c) for c in cells]
    keys = sorted(shares)
    top = np.mean([shares[k][sel].sum() for k in keys[-N_EDGE_BINS:]])
    bottom = np.mean([shares[k][sel].sum() for k in keys[:N_EDGE_BINS]])
    return float(top), float(bottom)


def rmse_vs_truth(records, err_key, rho=None, nu=None) -> float:
    """Root mean square of |u_ref - predicted mean| over every grid point of the runs at one rho
    / one nu / one cell (all runs if both are None)."""
    rs = _at(records, rho=rho, nu=nu)
    return float(np.sqrt(np.mean(np.concatenate([r[err_key].ravel() ** 2 for r in rs]))))


def ratio_vs_time(records, rho, nu=None):
    """(t, curve): median over runs of the per-run spatial median of sigma_Ours/sigma_HMC at each
    time, panel (c); the runs at one rho (all nu pooled) or at one cell (nu given)."""
    rs = _at(records, rho=rho, nu=nu)
    ratios = np.stack([np.median(r["sig_mp"] / r["sig_hmc"], axis=0) for r in rs])  # (n_runs, n_t)
    return rs[0]["t"], np.median(ratios, axis=0)


def ratio_per_traj(records, rho, nu=None):
    """Per-run median of sigma_Ours/sigma_HMC over the whole grid, panel (d) points."""
    return [float(np.median(r["sig_mp"] / r["sig_hmc"])) for r in _at(records, rho=rho, nu=nu)]


def grid_ratio_quantiles(records) -> tuple[float, float, float]:
    """(median, Q1, Q3) of sigma_Ours/sigma_HMC over every grid point of every record."""
    ratio = np.concatenate([(r["sig_mp"] / r["sig_hmc"]).ravel() for r in records])
    return tuple(float(q) for q in np.quantile(ratio, [0.5, 0.25, 0.75]))


# ----------------------------------------------------------------------------- panels
def _rho_label(rho) -> str:
    return rf"$\rho={rho:g}$"


def plot_growth(ax, records, rhos, rho_colors, tag: str, style: dict) -> None:
    for rho in rhos:
        t, curves = growth_curves(records, rho)
        ax.plot(t, curves.mean(axis=0), color=rho_colors[rho], lw=style["lw"], label=_rho_label(rho))
        ax.fill_between(t, curves.min(axis=0), curves.max(axis=0), color=rho_colors[rho], alpha=0.18)
    ax.set_xlabel(r"$t$", fontsize=style["label"])
    ax.set_ylabel(r"$\mathrm{predictive\ std}$", fontsize=style["label"])
    ax.set_title(r"$\mathrm{(%s)\ Uncertainty\ vs.\ distance\ from\ data}$" % tag, fontsize=style["title"])
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=style["tick"])
    if style["growth_headroom"]:  # keep the top share of the panel free of data, for the legend
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, lo + (hi - lo) / (1 - style["growth_headroom"]))
    ax.legend(fontsize=style["legend"], ncol=style["growth_ncol"], title=r"$\mathrm{all}\ \nu\ \mathrm{pooled}$",
              title_fontsize=style["legend_title"])


def _label_log_axis(axis, subs=(1.0, 2.0, 5.0), plain: bool = False) -> None:
    """Labelled major ticks at `subs` x 10^k: the log axes here span one to two decades (the ratio
    axes of (c)/(d), the std axis of (b)), where matplotlib's default labels only the decade and
    the values cannot be read off. The wide x labels of (b) take three per decade (four overlap
    once the x range spans the ~1.5 decades of the full grid), the y axes five. plain: the labels
    are plain numbers (0.3, 1, 2) instead of 'k x 10^n' (the shipped figure)."""
    axis.set_major_locator(LogLocator(base=10.0, subs=subs, numticks=30))
    if plain:
        axis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
    else:
        # minor_thresholds=(inf, inf): label EVERY major tick (the default labels only some sub-decade ticks)
        axis.set_major_formatter(LogFormatterSciNotation(base=10.0, labelOnlyBase=False,
                                                         minor_thresholds=(np.inf, np.inf)))
    axis.set_minor_formatter(plt.NullFormatter())


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
    # x range (the offset from it IS the finding).
    xlim, ylim = _padded_log_range(xs_all), _padded_log_range(ys_all)
    ylim = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
    ax.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], "k--", lw=1.6, alpha=0.6, label=r"$y=x$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    _label_log_axis(ax.xaxis, subs=(1.0, 2.0, 5.0), plain=style["plain_log_ticks"])
    ax.set_xlabel(r"$\mathrm{RMS\ predicted\ std\ (bin)}$", fontsize=style["label"])
    ax.set_ylabel(r"$\mathrm{empirical\ RMSE\ (bin)}$", fontsize=style["label"])
    ax.set_title(r"$\mathrm{(%s)\ Predicted\ std\ vs.\ actual\ error}$" % tag, fontsize=style["title"])
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(labelsize=style["tick"])
    ax.legend(fontsize=style["legend_rel"])


def plot_ratio_vs_time(ax, records, rhos, rho_colors, tag: str, style: dict) -> None:
    for rho in rhos:
        t, curve = ratio_vs_time(records, rho)
        ax.plot(t, curve, color=rho_colors[rho], lw=style["lw"], label=_rho_label(rho))
    ax.axhline(1.0, color="k", linestyle=":", lw=1.6, alpha=0.6)
    ax.set_yscale("log")
    _label_log_axis(ax.yaxis, subs=(1.0, 2.0, 3.0, 5.0, 7.0), plain=style["plain_log_ticks"])
    ax.set_xlabel(r"$t$", fontsize=style["label"])
    ax.set_ylabel(r"$\sigma_{\mathrm{Ours}} / \sigma_{\mathrm{HMC}}$", fontsize=style["label"])
    ax.set_title(r"$\mathrm{(%s)\ Std\ ratio\ vs.\ HMC\ over\ time}$" % tag, fontsize=style["title"])
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=style["tick"])
    ax.legend(fontsize=style["legend"], ncol=2, title=r"$\mathrm{all}\ \nu\ \mathrm{pooled}$",
              title_fontsize=style["legend_title"])


def plot_ratio_per_traj(ax, records, rhos, nus, rho_colors, tag: str, style: dict) -> None:
    rng = np.random.RandomState(2)
    for i, rho in enumerate(rhos):
        for nu in nus:
            vals = ratio_per_traj(records, rho, nu)
            if not vals:
                continue
            ax.scatter(np.full(len(vals), float(i)) + rng.uniform(-0.12, 0.12, len(vals)), vals,
                       s=style["s"], alpha=0.75, marker=NU_MARKERS[nu],
                       facecolors=rho_colors[rho], edgecolors="black", linewidths=0.6)
        ax.plot([i - 0.18, i + 0.18], [np.median(ratio_per_traj(records, rho))] * 2, color="black", lw=style["lw"])
    ax.axhline(1.0, color="k", linestyle="--", lw=1.6, alpha=0.6)
    ax.set_yscale("log")
    _label_log_axis(ax.yaxis, subs=(1.0, 2.0, 3.0, 5.0, 7.0), plain=style["plain_log_ticks"])
    ax.set_xlim(-0.5, len(rhos) - 0.5)
    ax.set_xticks(range(len(rhos)))
    ax.set_xticklabels([_rho_label(rho) for rho in rhos], fontsize=style["xticklabel"])
    ax.set_ylabel(r"$\sigma_{\mathrm{Ours}} / \sigma_{\mathrm{HMC}}$", fontsize=style["label"])
    ax.set_title(r"$\mathrm{(%s)\ Std\ ratio\ per\ run}$" % tag, fontsize=style["title"])
    ax.grid(True, axis="y", alpha=0.3)
    ax.tick_params(labelsize=style["tick"])
    # The nu-marker legend gets its own headroom above the data (and above the y=1 line), so it can
    # never cover a point: the y range is widened upwards by LEGEND_HEADROOM of its log-span.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, 10 ** (np.log10(hi) + LEGEND_HEADROOM * (np.log10(hi) - np.log10(lo))))
    handles = [plt.Line2D([], [], linestyle="none", marker=NU_MARKERS[nu], markersize=style["nu_marker"],
                          markerfacecolor="0.75", markeredgecolor="black", label=rf"$\nu={nu:g}$")
               for nu in nus]
    ax.legend(handles=handles, fontsize=style["nu_legend"], ncol=len(nus), loc="upper center",
              title=r"$\mathrm{marker}=\nu$", title_fontsize=style["nu_legend"])


def _rho_colors() -> dict[float, tuple]:
    """One viridis colour per rho, fixed per rho (the colours of the part-2 panels)."""
    cmap = plt.get_cmap("viridis")
    return {rho: cmap(i / max(1, len(RHOS) - 1)) for i, rho in enumerate(RHOS)}


def make_variance_figure(records, cells, output_paths: list[Path]) -> None:
    """The full four-panel variance_consistency.pdf (CALIBRATION_ALL_PANELS=1 only)."""
    plt.rcParams["mathtext.fontset"] = "cm"
    rho_colors = _rho_colors()
    rhos, nus = rhos_of(cells), nus_of(cells)
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 11.5))
    plot_growth(axes[0, 0], records, rhos, rho_colors, "a", VC_STYLE)
    plot_reliability(axes[0, 1], records, "b", VC_STYLE)
    plot_ratio_vs_time(axes[1, 0], records, rhos, rho_colors, "c", VC_STYLE)
    plot_ratio_per_traj(axes[1, 1], records, rhos, nus, rho_colors, "d", VC_STYLE)
    fig.tight_layout()
    for path in output_paths:
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"Saved figure to {path}")
    plt.close(fig)


# ----------------------------------------------------------------------------- table
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


def _cell_rows(records, cells, n_values: int, fn) -> list[str]:
    """Rows for ALL 16 cells in VC_ALL_CELLS order: 'nu & rho & n & fn(records of the cell)...'; a cell
    with no admitted run shows n = 0 and '--' in every value cell; a cell left out by the
    REACDIFF_CELLS override shows n = n/s (not selected)."""
    rows = []
    for nu, rho in VC_ALL_CELLS:
        rs = _at(records, rho=rho, nu=nu)
        if rs:
            rows.append(_row([f"{nu:g}", f"{rho:g}", f"{len(rs)}", *fn(rs)]))
        else:  # n = 0: every run dropped by the gate; n = n/s: cell not selected (REACDIFF_CELLS)
            n_text = "0" if (nu, rho) in cells else "n/s"
            rows.append(_row([f"{nu:g}", f"{rho:g}", n_text, *(["--"] * n_values)]))
    return rows


def _pooled_rows(records, cells, fn, tail: bool = True) -> list[str]:
    """Rows 'all & rho & n & ...' pooling every nu at that rho (what the panels show), plus 'all & all'."""
    rows = [_row([r"all $\nu$", f"{rho:g}", f"{len(_at(records, rho=rho))}", *fn(_at(records, rho=rho))])
            for rho in rhos_of(cells)]
    if tail:
        rows.append(_row([r"all $\nu$", r"all $\rho$", f"{len(records)}", *fn(records)]))
    return rows


def vi_caveat_lines(records, cells) -> list[str]:
    """Table comment lines for the setting-sorting of the VI reliability bins, recomputed from the
    data: the share of the top / bottom N_EDGE_BINS VI bins that comes from each rho, from each nu,
    the largest of them, and every method's RMSE vs. the truth per rho (all nu pooled)."""
    shares = bin_cell_shares(records, "sig_vi", "err_vi")
    rhos, nus = rhos_of(cells), nus_of(cells)
    by_rho = {rho: edge_bin_share(shares, [c for c in cells if c[1] == rho]) for rho in rhos}
    by_nu = {nu: edge_bin_share(shares, [c for c in cells if c[0] == nu]) for nu in nus}
    top_rho = max(rhos, key=lambda r: by_rho[r][0])
    bot_rho = max(rhos, key=lambda r: by_rho[r][1])
    top_nu = max(nus, key=lambda n: by_nu[n][0])
    bot_nu = max(nus, key=lambda n: by_nu[n][1])
    per_rho = "; ".join(f"rho={rho:g}: " + ", ".join(f"{label} {rmse_vs_truth(records, ek, rho=rho):.3f}"
                                                    for label, _, ek in RELIABILITY_SERIES) for rho in rhos)
    return [
        r"%     Caveat on the VI series: pooled quantile bins can sort by setting (rho, nu) rather than by the error",
        r"%     within a run when a method's predicted std and its accuracy both move with the setting (in the",
        rf"%     advection set the VI bins were wave-speed bins). Recomputed here: VI's top {N_EDGE_BINS} bins draw",
        rf"%     {by_rho[top_rho][0]:.0%} of their points from rho={top_rho:g} and {by_nu[top_nu][0]:.0%} from nu={top_nu:g}; "
        rf"its bottom {N_EDGE_BINS} bins draw {by_rho[bot_rho][1]:.0%} from rho={bot_rho:g}",
        rf"%     and {by_nu[bot_nu][1]:.0%} from nu={bot_nu:g}. Per rho (top / bottom): "
        + "; ".join(f"rho={rho:g} {by_rho[rho][0]:.0%} / {by_rho[rho][1]:.0%}" for rho in rhos) + ".",
        r"%     Per nu (top / bottom): "
        + "; ".join(f"nu={nu:g} {by_nu[nu][0]:.0%} / {by_nu[nu][1]:.0%}" for nu in nus) + ".",
        r"%     A share near 100% for one setting means the steep part of the VI curve is a between-setting effect,",
        r"%     not the predicted std tracking the error within a run (the per-cell shares are in the block below).",
        r"%     VI is shown in panel (b) for every cell (no bin is left out); the log axes are set from the plotted",
        r"%     bins of all three methods. RMSE vs. the truth (u_ref) per rho, all nu pooled, over every grid point:",
        rf"%     {per_rho}.",
    ]


def write_variance_table(records, dropped, cells, override: bool, path: Path) -> None:
    """LaTeX table of part 2 (variance_consistency_table.tex) mirroring the four panels of the full figure
    variance_consistency.pdf, one section each, in one 10-column
    tabular: (a) mean-std curve samples and its t=end band, (b) every reliability bin for Ours,
    HMC and VI (the only block that needs all 10 columns) followed by the per-cell RMSE vs. the
    truth and the VI edge-bin shares, (c) ratio-vs-time curve samples and range, (d) per-run and
    grid-point ratio spread. Blocks (a), (c), (d) list all 16 cells (nu, rho; n = admitted runs)
    and then the pooled rows the panels show. Every number is explained in the leading comment
    lines, which also state the runs used per rho and the runs the HMC gate dropped."""
    t = records[0]["t"]
    i_samp = [0, len(t) // 2, len(t) - 1]  # t = 0, mid, end: the same three columns in (a) and (c)
    t_heads = [rf"$t={t[i]:g}$" for i in i_samp]
    n_pts = records[0]["sig_mp"].size
    n_series = len(RELIABILITY_SERIES)
    rhos = rhos_of(cells)
    lines = [
        r"% Auto-generated by figures_onepass/diffusion/calibration_comparison.py (part 2;",
        r"%     the full table of the four-panel variance_consistency.pdf, whose panel letters are",
        r"%     used below; panels (a) and (d) are panels (c) and (d) of the shipped calibration_variance.pdf).",
        rf"% {len(records)} runs used ({'; '.join(n_used_text(records, cells, rho) for rho in rhos)}):",
        rf"%     the five canonical trajectories idxs(5) = {IDXS} in every cell (nu, rho) of the PDEBench 1D",
        r"%     reaction-diffusion grid nu in {0.5, 1, 2, 5} x rho in {1, 2, 5, 10} (model values rho_norm = 2 rho,",
        rf"%     nu_norm = 2 nu). Each predictive grid has {n_pts} points; sigma = sqrt(max(pred_var, {VC_MIN_VAR:g})).",
        *([rf"%     REACDIFF_CELLS override: only the cells {', '.join(cell_text(*c) for c in cells)} were used;",
           r"%     the other cells show n = n/s (not selected) and '--'. NOT the default run."] if override else []),
        r"% Truth for every error = the float64 re-solve of the nominal PDE (u_ref), not the stored PDEBench trajectory.",
        r"% Ours = one-pass EP (expectation propagation) posterior; HMC = multi-chain Hamiltonian Monte Carlo",
        r"%     reference (all chains pooled); VI = mean-field Gaussian variational inference baseline.",
        rf"% Run gate: a run enters only if at least {HMC_MIN_ADMITTED:.0%} of its HMC eigenbasis coordinates (the rotated",
        rf"%     weight coordinates all methods share) have split-R-hat < {RHAT_MAX:g} and bulk ESS > {ESS_MIN:g}, the same rule",
        r"%     as the advection set and calibration_comparison_table.tex (one shared loader). A run that fails it",
        r"%     is dropped for EVERY method, so every block uses the",
        r"%     same runs. Dropped by the HMC gate (decided from the cached diagnostics at run time):",
        *(f"%         {line}" for line in dropped_text(dropped).split("; ")),
        r"% n = admitted runs in the row (per cell out of 5; pooled rows pool every nu at that rho, 'all rho' pools all).",
        r"% (a) Ours' predictive std, spatial mean at each t, then the mean over the runs of the row, sampled at",
        r"%     three times; growth = the t=end value over the t=0 value; band min / band max = the smallest /",
        r"%     largest run's spatial-mean std at t=end (the shaded band of panel (a) at its right edge, per rho).",
        rf"% (b) bins are per-method quantile bins ({N_SIGMA_BINS} equal-count bins) of the pooled grid points of all",
        r"%     runs, so bin k holds the same share of each method's points but not the same points (the edges",
        r"%     differ between Ours, HMC and VI). RMS std = root mean square of the predicted std in the bin,",
        r"%     RMSE = root mean square of |u_ref - predicted mean| in the bin; RMSE/std = 1 means calibrated",
        rf"%     within that bin, > 1 overconfident, < 1 underconfident. Bins with fewer than {MIN_BIN_POINTS} points are '--'.",
        *vi_caveat_lines(records, cells),
        rf"%     The per-cell block gives each method's RMSE vs. u_ref over the cell's runs and the share of VI's top /",
        rf"%     bottom {N_EDGE_BINS} bins (averaged over those bins) that the cell's points fill.",
        r"% (c) sigma_Ours / sigma_HMC at every grid point; spatial median at each t per run, then the median",
        r"%     over the runs of the row, sampled at three times; min / max over t of that curve and their",
        r"%     ratio (1 = the same factor at every time).",
        r"% (d) left: per run, the median of sigma_Ours / sigma_HMC over its whole grid; median / min / max of",
        r"%     those per-run values over the runs of the row. right: median and quartiles of sigma_Ours /",
        r"%     sigma_HMC over every grid point of every run of the row.",
        rf"% Blocks (a), (c), (d) use the first 9 of the {N_COLS} columns; their trailing cells are empty.",
        r"\begin{tabular}{l " + " ".join(["rrr"] * n_series) + "}",
        r"\toprule",
        _section(r"\emph{(a) Mean predictive std of Ours vs.\ time}"),
        _row([r"$\nu$", r"$\rho$", "$n$", *t_heads, rf"growth $t{{=}}{t[i_samp[-1]]:g}$ / $t{{=}}{t[0]:g}$",
              "band min", "band max"]),
        r"\midrule",
    ]

    def growth_cells(rs):
        curves = np.stack([r["sig_mp"].mean(axis=0) for r in rs])
        mean = curves.mean(axis=0)
        return [*(_sci(mean[i]) for i in i_samp), f"{mean[i_samp[-1]] / mean[0]:.2f}",
                _sci(curves.min(axis=0)[i_samp[-1]]), _sci(curves.max(axis=0)[i_samp[-1]])]

    lines += _cell_rows(records, cells, 6, growth_cells)
    lines.append(r"\midrule")
    lines += _pooled_rows(records, cells, growth_cells, tail=False)
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
        row_cells = []
        for label, _, _ in RELIABILITY_SERIES:
            if k in bins[label]:
                x, y = bins[label][k]
                row_cells += [_sci(x), _sci(y), f"{y / x:.2f}"]
            else:
                row_cells += ["--"] * 3
        lines.append(_row([f"{k + 1}", *row_cells]))
    lines += [
        r"\midrule",
        _section(r"\emph{(b) RMSE vs.\ truth per cell and share of the VI edge bins}"),
        _row([r"$\nu$", r"$\rho$", "$n$", *(f"RMSE {label}" for label, _, _ in RELIABILITY_SERIES),
              rf"VI top {N_EDGE_BINS}", rf"VI bottom {N_EDGE_BINS}"]),
        r"\midrule",
    ]
    vi_shares = bin_cell_shares(records, "sig_vi", "err_vi")

    def rmse_share_cells(rs):
        sel = sorted({(r["nu"], r["rho"]) for r in rs})
        top, bottom = edge_bin_share(vi_shares, sel)
        return [*(f"{float(np.sqrt(np.mean(np.concatenate([r[ek].ravel() ** 2 for r in rs])))):.3f}"
                  for _, _, ek in RELIABILITY_SERIES), f"{top:.0%}".replace("%", r"\%"),
                f"{bottom:.0%}".replace("%", r"\%")]

    lines += _cell_rows(records, cells, 5, rmse_share_cells)
    lines.append(r"\midrule")
    lines += _pooled_rows(records, cells, rmse_share_cells)
    lines += [
        r"\midrule",
        _section(r"\emph{(c) Median $\sigma_{\mathrm{Ours}}/\sigma_{\mathrm{HMC}}$ vs.\ time}"),
        _row([r"$\nu$", r"$\rho$", "$n$", *t_heads, "min over $t$", "max over $t$", "max/min"]),
        r"\midrule",
    ]

    def ratio_time_cells(rs):
        curve = np.median(np.stack([np.median(r["sig_mp"] / r["sig_hmc"], axis=0) for r in rs]), axis=0)
        return [*(f"{curve[i]:.3f}" for i in i_samp), f"{curve.min():.3f}", f"{curve.max():.3f}",
                f"{curve.max() / curve.min():.2f}"]

    lines += _cell_rows(records, cells, 6, ratio_time_cells)
    lines.append(r"\midrule")
    lines += _pooled_rows(records, cells, ratio_time_cells, tail=False)
    lines += [
        r"\midrule",
        _section(r"\emph{(d) $\sigma_{\mathrm{Ours}}/\sigma_{\mathrm{HMC}}$ per run and per grid point}"),
        _row([r"$\nu$", r"$\rho$", "$n$", "median", "min", "max", "grid median", "grid Q1", "grid Q3"]),
        r"\midrule",
    ]

    def ratio_traj_cells(rs):
        vals = [float(np.median(r["sig_mp"] / r["sig_hmc"])) for r in rs]
        med, q1, q3 = grid_ratio_quantiles(rs)
        return [f"{np.median(vals):.3f}", f"{min(vals):.3f}", f"{max(vals):.3f}", f"{med:.3f}", f"{q1:.3f}", f"{q3:.3f}"]

    lines += _cell_rows(records, cells, 6, ratio_traj_cells)
    lines.append(r"\midrule")
    lines += _pooled_rows(records, cells, ratio_traj_cells)
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines))
    print(f"Saved table to {path}")


# ----------------------------------------------------------------------------- summary
def print_variance_summary(records, cells) -> None:
    rhos, nus = rhos_of(cells), nus_of(cells)
    med, q1, q3 = grid_ratio_quantiles(records)
    print(f"Pooled sigma ratio Ours/HMC: median {med:.2f}, IQR (interquartile range, Q1 to Q3) [{q1:.2f}, {q3:.2f}]")
    for rho in rhos:
        med, q1, q3 = grid_ratio_quantiles(_at(records, rho=rho))
        per_nu = ", ".join(f"nu={nu:g} {np.median(ratio_per_traj(records, rho, nu)):.2f}"
                           for nu in nus if _at(records, rho=rho, nu=nu))
        print(f"  rho={rho:g}: grid median {med:.2f}, IQR [{q1:.2f}, {q3:.2f}]; per-run median by nu: {per_nu}")
    for label, sig_key, err_key in RELIABILITY_SERIES:
        pts = reliability_bins(records, sig_key, err_key)
        ratios = [y / x for _, x, y in pts]
        print(f"Reliability bins {label}: {len(pts)} bins, RMS std {pts[0][1]:.2e}..{pts[-1][1]:.2e}, "
              f"RMSE {min(y for _, _, y in pts):.2e}..{max(y for _, _, y in pts):.2e}, "
              f"RMSE/std {min(ratios):.1f}..{max(ratios):.1f}")
    print("RMSE vs truth (u_ref) per rho, all nu pooled: "
          + "; ".join(f"rho={rho:g}: " + ", ".join(f"{label} {rmse_vs_truth(records, ek, rho=rho):.3f}"
                                                  for label, _, ek in RELIABILITY_SERIES) for rho in rhos))
    print("RMSE vs truth (u_ref) per cell: "
          + "; ".join(f"{cell_text(nu, rho)}: " + ", ".join(f"{label} {rmse_vs_truth(records, ek, rho=rho, nu=nu):.3f}"
                                                            for label, _, ek in RELIABILITY_SERIES)
                      for nu, rho in cells if _at(records, rho=rho, nu=nu)))
    for label, sig_key, err_key in RELIABILITY_SERIES:
        shares = bin_cell_shares(records, sig_key, err_key)
        for name, values, pick in (("rho", rhos, 1), ("nu", nus, 0)):
            per_value = {v: edge_bin_share(shares, [c for c in cells if c[pick] == v]) for v in values}
            top_v = max(values, key=lambda v: per_value[v][0])
            bot_v = max(values, key=lambda v: per_value[v][1])
            print(f"Share of {label} bin points by {name} (top {N_EDGE_BINS} / bottom {N_EDGE_BINS} bins): "
                  + "; ".join(f"{name}={v:g} {per_value[v][0]:.0%} / {per_value[v][1]:.0%}" for v in values)
                  + f" -> top bins mostly {name}={top_v:g} ({per_value[top_v][0]:.0%}), "
                  f"bottom bins mostly {name}={bot_v:g} ({per_value[bot_v][1]:.0%})")


# ============================================================================= the shipped figure


def make_figure(records: list[dict], vrecords: list[dict], vcells, output_path: Path, png_path: Path | None = None,
                cells_note: str | None = None) -> None:
    """calibration_variance.pdf, 2 x 2 with the panel size of the full figure calibration_comparison.pdf
    (8.5 x 7.33 inch): (a) CRPS per rho, (b) coverage of the truth per rho (part 1); (c) Ours'
    predictive std vs time per rho, (d) per-run std ratio (part 2, drawn with MERGED_VC_STYLE)."""
    plt.rcParams["mathtext.fontset"] = "cm"
    rho_colors = _rho_colors()
    fig, axes = plt.subplots(2, 2, figsize=(17, 44 / 3))
    plot_scoring_rule(axes[0, 0], records, crps_gaussian, r"$\mathrm{(a)\ CRPS\ (vs.\ truth)}$", r"$\mathrm{CRPS}$",
                      CRPS_YTICK_CANDIDATES, dict(loc="best"))
    plot_truth_coverage_per_rho(axes[0, 1], records, "b")
    plot_growth(axes[1, 0], vrecords, rhos_of(vcells), rho_colors, "c", MERGED_VC_STYLE)
    plot_ratio_per_traj(axes[1, 1], vrecords, rhos_of(vcells), nus_of(vcells), rho_colors, "d", MERGED_VC_STYLE)
    fig.tight_layout()
    if cells_note:  # override mode: say on the figure itself that it is partial
        fig.suptitle(cells_note, fontsize=20, color="#b00020", y=1.02)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved figure to {output_path}")
    if png_path is not None:
        fig.savefig(png_path, bbox_inches="tight", dpi=60)
        print(f"Saved PNG preview to {png_path}")
    plt.close(fig)


def write_merged_table(records: list[dict], excluded: list[dict], vrecords: list[dict], cells, vcells,
                       override: bool, path: Path) -> None:
    """Companion table of calibration_variance.pdf, one block per panel, one row per rho (the nu
    values of the selected cells pooled, as the panels show them; all four in the default run) plus,
    except in block (c) (as in the full table), the row pooled over every run: (a) mean CRPS against the truth
    (Ours, HMC, VI); (b) coverage of the truth by the 90% / 95% intervals; (c) Ours' mean
    predictive std at t = 0 / mid / end, its growth and the t=end band; (d) the per-run medians of
    sigma_Ours/sigma_HMC (median, min, max over the runs) and the grid-point quantiles. The per-cell
    numbers, with more context, are in calibration_comparison_table.tex (blocks behind a, b) and
    variance_consistency_table.tex (c, d)."""
    n_cols = 7

    def row(cells_) -> str:
        cells_ = list(cells_)
        return " & ".join(cells_ + [""] * (n_cols - len(cells_))) + r" \\"

    def section(text: str) -> str:
        return rf"\multicolumn{{{n_cols}}}{{l}}{{{text}}}\\"

    def rho_label(rho: float | None) -> str:
        return "all" if rho is None else f"{rho:g}"

    rhos = _plotted_rhos(records)
    t = vrecords[0]["t"]
    i_samp = [0, len(t) // 2, len(t) - 1]  # t = 0, mid, end, as in variance_consistency_table.tex
    n_x, n_t = records[0]["grid_shape"]
    lines = [
        r"% Auto-generated by figures_onepass/diffusion/calibration_comparison.py",
        r"% Companion table of calibration_variance.pdf (one block per panel). Ours = one-pass EP; HMC = 4-chain",
        r"% Hamiltonian Monte Carlo reference (3000 pooled draws); VI = mean-field Gaussian variational inference.",
        *[f"% {line}" for line in textwrap.wrap(
            "Dataset: PDEBench 1D Fisher-KPP, u_t = nu u_xx + rho u (1 - u), 16 cells nu in "
            f"{NUS} x rho in {RHOS}, the five canonical trajectories idxs(5) = {IDXS} per cell whose HMC reference "
            f"converged ({n_per_rho_text(records, cells)}; {len(records)} pooled; at least {HMC_MIN_ADMITTED:.0%} of "
            f"the HMC coordinates with split-R-hat < {RHAT_MAX} and bulk ESS > {ESS_MIN:.0f}); dropped for every "
            f"method by that gate: {dropped_runs_text(excluded)}. "
            + (f"OVERRIDE MODE ({CELLS_ENV}): only the cells {', '.join(cell_label(*c) for c in cells)} were run; "
               "NOT the final table. " if override else "All 16 cells were run. ")
            + "Every row pools the nu values of the selected cells at that rho (all four in the default run; what "
            "the panels show); 'all' pools every run. "
            f"Every statistic uses the full {n_x} x {n_t} (x,t) grid of each run and is measured against the truth "
            "u_ref, the float64 re-solve of the nominal PDE ((a)-(b)), shows Ours' predictive std ((c)) or compares "
            "Ours' std with HMC's ((d)). The per-cell numbers are in calibration_comparison_table.tex and "
            "variance_consistency_table.tex.", 100)],
        r"% (a) CRPS (continuous ranked probability score) = mean over the runs of the per-run mean closed-form",
        rf"%     Gaussian CRPS (sigma = sqrt(max(pred_var, {MIN_VAR:g}))); = column CRPS of block 1 of",
        r"%     calibration_comparison_table.tex (its 'nu = all' rows).",
        r"% (b) coverage = share of all grid points of the runs whose truth lies inside the nominal 90% / 95% Gaussian",
        r"%     predictive interval; = columns 90% / 95% of block 1 of calibration_comparison_table.tex.",
        r"% (c) Ours' predictive std, spatial mean at each t, then the mean over the runs of that rho, at three times;",
        r"%     growth = t=end over t=0; band min / max = smallest / largest run's spatial-mean std at t=end; = the",
        r"%     pooled rows of block (a) of variance_consistency_table.tex.",
        r"% (d) per run the median of sigma_Ours/sigma_HMC over its grid; median / min / max over the runs of that rho;",
        r"%     grid median / Q1 / Q3 = over every grid point of those runs; = the pooled rows of block (d) of",
        r"%     variance_consistency_table.tex.",
        r"\begin{tabular}{l rrrrrr}",
        r"\toprule",
        section(r"\emph{(a) CRPS vs.\ truth}"),
        row([r"$\rho$", *METHODS]),
        r"\midrule",
    ]
    for rho in rhos + [None]:
        lines.append(row([rho_label(rho), *(f"{mean_score(records, m, crps_gaussian, rho):.4f}" for m in METHODS)]))
    lines += [
        r"\midrule",
        section(r"\emph{(b) Coverage of the truth at nominal 90\% / 95\%}"),
        row([r"$\rho$", *(f"{m} {lev:.0%}".replace("%", r"\%") for m in METHODS for lev in HEADLINE_LEVELS)]),
        r"\midrule",
    ]
    for rho in rhos + [None]:
        lines.append(row([rho_label(rho), *(f"{100 * coverage(pooled_z(records, m, rho), lev):.1f}\\%"
                                            for m in METHODS for lev in HEADLINE_LEVELS)]))
    lines += [
        r"\midrule",
        section(r"\emph{(c) Mean predictive std of Ours vs.\ time}"),
        row([r"$\rho$", *(rf"$t={t[i]:g}$" for i in i_samp), rf"growth $t{{=}}{t[i_samp[-1]]:g}$ / $t{{=}}{t[0]:g}$",
             "band min", "band max"]),
        r"\midrule",
    ]
    for rho in rhos_of(vcells):
        _, curves = growth_curves(vrecords, rho)
        mean = curves.mean(axis=0)
        lines.append(row([rho_label(rho), *(_sci(mean[i]) for i in i_samp), f"{mean[i_samp[-1]] / mean[0]:.2f}",
                          _sci(curves.min(axis=0)[i_samp[-1]]), _sci(curves.max(axis=0)[i_samp[-1]])]))
    lines += [
        r"\midrule",
        section(r"\emph{(d) $\sigma_{\mathrm{Ours}}/\sigma_{\mathrm{HMC}}$ per run and per grid point}"),
        row([r"$\rho$", "median", "min", "max", "grid median", "grid Q1", "grid Q3"]),
        r"\midrule",
    ]
    for rho in rhos_of(vcells) + [None]:
        rs = vrecords if rho is None else _at(vrecords, rho=rho)
        vals = [float(np.median(r["sig_mp"] / r["sig_hmc"])) for r in rs]
        med, q1, q3 = grid_ratio_quantiles(rs)
        lines.append(row([rho_label(rho), f"{np.median(vals):.3f}", f"{min(vals):.3f}", f"{max(vals):.3f}",
                          f"{med:.3f}", f"{q1:.3f}", f"{q3:.3f}"]))
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines))
    print(f"Saved table to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--png", type=Path, default=None, help="also render the shipped figure to this PNG (for inspection)")
    args = parser.parse_args()
    tee = _Tee(OUTPUT_LOG)
    sys.stdout = tee
    try:
        cells, override = selected_cells()
        if override:
            print(f"OVERRIDE MODE: {CELLS_ENV}={os.environ[CELLS_ENV]!r} -> running only the cells "
                  f"{', '.join(cell_label(*c) for c in cells)} ({len(cells)} of {len(ALL_CELLS)}). "
                  "The outputs are PARTIAL and marked as such; unset the variable for the full figure.")
        else:
            print(f"Running all {len(ALL_CELLS)} cells (nu in {NUS} x rho in {RHOS}), trajectories {IDXS}")
        records, excluded = collect_records(cells)
        if not records:
            raise RuntimeError("no run passed the HMC gate; nothing to plot")
        vcells = [c for c in VC_ALL_CELLS if c in cells]  # part 2 lists and pools the cells in its own order
        vrecords, vdropped = variance_records(records, excluded, vcells)
        note = (f"PARTIAL ({CELLS_ENV}): cells {', '.join(cell_label(*c) for c in cells)} only" if override else None)
        make_figure(records, vrecords, vcells, OUTPUT_PDF, args.png, note)
        if ALL_PANELS:
            print(f"NOTE: CALIBRATION_ALL_PANELS=1: also writing the full figures {CAL_PDF.name} and {VC_PDF.name}")
            make_calibration_figure(records, CAL_PDF, None, note)
            make_variance_figure(vrecords, vcells, [VC_PDF])
        write_merged_table(records, excluded, vrecords, cells, vcells, override, OUTPUT_TEX)
        write_calibration_table(records, excluded, cells, override, CAL_TEX)
        write_variance_table(vrecords, vdropped, vcells, override, VC_TEX)
        print_calibration_summary(records, excluded, cells)
        print()
        print_variance_summary(vrecords, vcells)
        print(f"Log written to {OUTPUT_LOG}")
    except BaseException as e:  # the traceback goes to stderr; the log must say why it stops short
        print(f"\nFAILED, no outputs written past this point: {type(e).__name__}: {e}")
        raise
    finally:
        tee.close()


if __name__ == "__main__":
    main()
