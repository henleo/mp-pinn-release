"""
How well do the one-pass EP (expectation propagation) posterior ("Ours") and the mean-field VI
(variational inference) posterior each fit the multi-chain HMC (Hamiltonian Monte Carlo)
reference posterior, on the PDEBench 1D reaction-diffusion (Fisher-KPP) dataset,
u_t = nu u_xx + rho u (1 - u)? Mean-field means that the posterior is approximated by
independent coordinates: both approximations are diagonal Gaussians in the shared eigenbasis
(the rotated weight coordinates all methods share; every artifact carries a fingerprint of that
basis and the three artifacts of a run are checked against each other before anything is
compared); HMC gives 4 chains x 750 kept draws = 3000 pooled draws per run. The experiment grid
is 16 cells, diffusion coefficient nu in {0.5, 1, 2, 5} x reaction strength rho in {1, 2, 5,
10}, with the FIVE canonical trajectories idxs(5) (the prefix of the seeded draw the advection
figures use) in every cell: 80 runs per method. All from cached artifacts (this script never
trains; a missing artifact raises an error naming the file). Every number is a KL
(Kullback-Leibler divergence, in nats; a nat is the unit of a KL computed with natural
logarithms). Nothing here is compared to the exact solution: the reference is the HMC
posterior of the same model, so the float64 reference solution u_ref is not needed.

How the wave-speed axis of the advection figure maps onto the grid: rho plays the role of b
(it drives the nonlinearity, the failure modes and the cost), so the bars of panels (a),(b) are
grouped by rho and each group pools the 4 nu values, i.e. 4 x 5 = 20 runs per rho; in the
per-run panels (c),(d) nu is shown by the marker shape (nu = 0.5: circle, 1: square, 2: diamond,
5: triangle) and the method by the colour. The x-axis label is the physical rho; the model works
with the normalized values rho_norm = 2 rho and nu_norm = 2 nu (stated in the table comment from
the artifacts). The companion table lists every cell (nu, rho) explicitly (median over its
5 runs), the per-rho rows (all nu pooled) and the pooled rows (everything).

Weight space, the "mean-field excess KL" D_MF (nats per admitted coordinate):
  D_MF(q) = (1/K) sum_k KL( N(mu_hat_k, s_hat_k^2) || N(mu_q_k, s_q_k^2) ),
  mu_hat_k, s_hat_k = HMC marginal mean and std of eigen-coordinate k (pooled draws),
  mu_q_k, s_q_k = the method's marginal. Per coordinate this splits exactly into a
  mean term (mu_hat - mu_q)^2 / (2 s_q^2) and a variance term 0.5 (r - 1 - ln r), r = s_hat^2/s_q^2.
  Why this is the right number: for ANY posterior p and any diagonal Gaussian q, log q is a
  quadratic, so E_p[log q] depends on p only through its marginal means and variances; hence
  KL(p||q) minus the best achievable KL(p||q') over all diagonal Gaussians q' equals exactly
  the sum above (the minimizer q' is the moment-matched diagonal Gaussian). D_MF is the number
  of nats the method loses relative to the best possible mean-field approximation of the HMC
  posterior, needs no Gaussian assumption on p, and is 0 iff every marginal mean and variance
  is matched. Coordinates enter only when HMC converged there: split-R-hat < 1.05 (split-R-hat
  compares the spread between the split half-chains with the spread within them and is 1 at
  convergence) and bulk ESS > 100 (bulk ESS is the effective number of independent draws for
  the central part of the distribution, after discounting autocorrelation); a run with fewer
  than half its coordinates admitted is dropped from every statistic (for every method). Which
  runs this drops is decided from the cached HMC chains at run time and is printed in the log,
  written into the table comment and stored in the summary JSON, together with the number of
  runs per rho n (also under every figure tick) and per cell (a cell whose five runs are all
  dropped stays in the table with n = 0 and '--' entries; a rho group with no admitted run at
  all is an error, the figure cannot be drawn without it); a dropped run is one whose HMC
  has not converged (typically visible as chain acceptance rates near zero), and a run enters
  as soon as its cached HMC reference passes the gate, without any edit here.
  Two reference lines: (i) the Monte-Carlo noise floor "HMC half vs half" (chains {1,2} give q,
  chains {3,4} give the target moments, same formula), which is what perfect agreement looks
  like given HMC sampling noise (a conservative floor: both halves are noisy and each has only
  1500 draws, so it overstates the noise of the 3000-draw reference by about 4x); and (ii) the
  mean-field gap of the target itself, -0.5 ln det Corr_p / K over the admitted coordinates
  (ln det is the log-determinant, the natural log of the determinant of the matrix; Corr_p is
  the HMC sample correlation matrix), which is the KL a PERFECT mean-field fit would still lose
  if p were Gaussian. Its raw value is biased upward (a sample log-determinant is biased
  downward); the reported corrected value applies the Wishart bias correction (a correction
  for the downward bias of a sample log-determinant) of a sample correlation matrix,
  sum_{i=1..K} [digamma((n-i)/2) - digamma((n-1)/2)] with n = 3000 (digamma is the derivative
  of the log of the gamma function). Both gap values are approximate (autocorrelated draws,
  Gaussian assumption). If the sample correlation matrix of a run is not positive definite
  (its log-determinant is then undefined), that run's gap is recorded as not available and
  the gap medians are taken over the runs where it is; the table comment says how many.

Predictive space (nats per grid point): the same Gaussian KL per (x,t) grid point between the
  stored HMC predictive N(m_H, v_H) and the method's N(m_q, v_q), with the same
  mean/variance split, averaged over the full grid (1024 x 101 points here; the size is read
  from the artifacts). HMC's own Monte-Carlo error puts a floor under these numbers, drawn in
  the figure as the band "MC floor (approx.)" from 3e-4 to 2e-3 nats per point: 1/n_eff with
  n_eff of 2000 to 3000 draws gives 3e-4 to 5e-4 for the mean term alone, but the weight-space
  half-vs-half floor shows that the variance term of HMC's own noise is about 4 to 5 times
  larger than 1/n would give (autocorrelated draws make the second moments noisier than the
  means), so the band is widened up to 2e-3.

Because the per-coordinate contributions are heavy-tailed (a few coordinates carry most of the
sum), the table also reports the median contribution over all admitted coordinates ("typical
coordinate") and the share of the sum carried by the top 5% of coordinates.

Outputs, all next to this script: posterior_fit_metrics.pdf, posterior_fit_metrics_table.tex,
posterior_fit_metrics_summary.json and posterior_fit_metrics.log (a copy of everything printed).

Run: python figures_onepass/diffusion/posterior_fit_metrics.py [--png preview.png]
Partial grid: REACDIFF_CELLS="0.5:1,0.5:2" (a
comma-separated list of nu:rho pairs) restricts the script to those cells; the cells used are
printed, written into the table comment and the JSON, and stamped on the figure. Without the
variable all 16 cells are required and a missing artifact is an error, never a skipped cell.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import digamma

FIGURES_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURES_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "pde" / "diffusion"))
sys.path.insert(0, str(REPO_ROOT / "pde" / "advection"))

import reacdiff_baselines_sweep as rd  # noqa: E402  (load_cached / cache_path / assert_same_basis / NUS / RHOS)
from trajectory_selection import idxs  # noqa: E402

OUTPUT_PDF = FIGURES_DIR / "posterior_fit_metrics.pdf"
OUTPUT_TABLE = FIGURES_DIR / "posterior_fit_metrics_table.tex"
OUTPUT_SUMMARY = FIGURES_DIR / "posterior_fit_metrics_summary.json"
OUTPUT_LOG = FIGURES_DIR / "posterior_fit_metrics.log"

NUS = list(rd.NUS)      # [0.5, 1, 2, 5]
RHOS = list(rd.RHOS)    # [1, 2, 5, 10]
IDXS = idxs(5)          # the five canonical trajectories, in every cell
RHAT_MAX = 1.05
ESS_MIN = 100.0
MIN_ADMITTED_FRACTION = 0.5   # runs with fewer admitted HMC coordinates are dropped everywhere
MIN_VAR = 1e-14               # floor on predictive variances, as in the other scripts
N_EFF_RANGE = (2000.0, 3000.0)     # HMC effective draws behind the 1/n_eff part of the predictive-space floor
PRED_MC_FLOOR_BAND = (3e-4, 2e-3)  # predictive-space MC floor band [nats per point], see the docstring
TOP_SHARE = 0.05              # "top 5% of coordinates" share of the sum
METHODS = ("Ours", "VI")
COLORS = {"Ours": "#4a3aa7", "HMC": "#eb6834", "VI": "#008300"}
NU_MARKERS = {0.5: "o", 1.0: "s", 2.0: "D", 5.0: "^"}   # marker shape = nu, in the per-run panels
FLOOR_COLOR = "#7a7a7a"


# ----------------------------------------------------------------------------- which cells
def _parse_cells() -> tuple[list[tuple[float, float]], bool]:
    """The (nu, rho) cells this run covers: all 16 by default, or the REACDIFF_CELLS override
    ("nu:rho,nu:rho,..."; every pair must be a grid cell)."""
    spec = os.environ.get("REACDIFF_CELLS", "").strip()
    if not spec:
        return [(nu, rho) for nu in NUS for rho in RHOS], False
    cells = []
    for item in spec.split(","):
        try:
            nu_s, rho_s = item.strip().split(":")
            nu, rho = float(nu_s), float(rho_s)
        except ValueError:
            raise ValueError(f"REACDIFF_CELLS entry {item!r} is not of the form nu:rho") from None
        if nu not in NUS or rho not in RHOS:
            raise ValueError(f"REACDIFF_CELLS entry {item!r}: nu must be one of {NUS} and rho one of {RHOS}")
        if (nu, rho) not in cells:
            cells.append((nu, rho))
    return sorted(cells, key=lambda c: (RHOS.index(c[1]), NUS.index(c[0]))), True


ACTIVE_CELLS, OVERRIDE = _parse_cells()
ACTIVE_RHOS = [rho for rho in RHOS if any(c[1] == rho for c in ACTIVE_CELLS)]
ACTIVE_NUS = [nu for nu in NUS if any(c[0] == nu for c in ACTIVE_CELLS)]


def _nlabel(nu: float) -> str:
    return f"{nu:g}"


def _rlabel(rho: float) -> str:
    """The rho label used for the figure ticks and the table: the physical rho, 1, 2, 5, 10."""
    return f"{rho:g}"


def _rkey(rho: float) -> str:
    return f"rho={_rlabel(rho)}"


def _ckey(nu: float, rho: float) -> str:
    return f"nu={_nlabel(nu)},rho={_rlabel(rho)}"


def _groups() -> list[tuple[str, float | None, float | None]]:
    """Every aggregation group as (key, nu, rho): the cells (nu, rho), the per-rho groups
    (nu = None: all nu pooled) and 'pooled' (everything)."""
    return ([(_ckey(nu, rho), nu, rho) for (nu, rho) in ACTIVE_CELLS]
            + [(_rkey(rho), None, rho) for rho in ACTIVE_RHOS] + [("pooled", None, None)])


def cells_description() -> str:
    return ", ".join(f"(nu={_nlabel(nu)}, rho={_rlabel(rho)})" for nu, rho in ACTIVE_CELLS)


# ----------------------------------------------------------------------------- metrics
def kl_gauss_terms(mu_p, var_p, mu_q, var_q):
    """KL( N(mu_p, var_p) || N(mu_q, var_q) ) split into its mean term and variance term,
    elementwise. Both terms are >= 0 by construction."""
    r = var_p / var_q
    return (mu_p - mu_q) ** 2 / (2 * var_q), 0.5 * (r - 1 - np.log(r))


def mean_field_gap(draws: np.ndarray) -> dict:
    """-0.5 ln det Corr / K from (n, K) draws: raw, and corrected for the Wishart bias of a
    sample log-determinant. 'corrected' uses the exact correction for a sample CORRELATION
    matrix; 'corrected_cov_formula' uses the covariance-matrix formula
    sum digamma((n-i)/2) + K ln(2/(n-1)); the two differ by about K/(n-1) nats in total."""
    n, k = draws.shape
    sign, logdet = np.linalg.slogdet(np.corrcoef(draws.T))
    if sign <= 0:  # log-determinant undefined: record the gap of this run as not available
        print(f"    -> sample correlation matrix ({k}x{k} from {n} draws) is not positive definite; "
              f"mean-field gap of this run recorded as not available")
        return dict(raw=float("nan"), corrected=float("nan"), corrected_cov_formula=float("nan"),
                    n_draws=int(n), K=int(k), available=False)
    i = np.arange(1, k + 1)
    bias_corr = float(np.sum(digamma((n - i) / 2) - digamma((n - 1) / 2)))
    bias_cov = float(np.sum(digamma((n - i) / 2)) + k * np.log(2 / (n - 1)))
    return dict(raw=-0.5 * logdet / k,
                corrected=-0.5 * (logdet - bias_corr) / k,
                corrected_cov_formula=-0.5 * (logdet - bias_cov) / k,
                n_draws=int(n), K=int(k), available=True)


def top_share(x: np.ndarray, frac: float = TOP_SHARE) -> float:
    """Share of sum(x) carried by the largest `frac` of its entries."""
    n_top = max(1, int(round(frac * x.size)))
    return float(np.sort(x.ravel())[-n_top:].sum() / x.sum())


def _stats(mean_term: np.ndarray, var_term: np.ndarray) -> dict:
    """Per-run summary of elementwise KL contributions: averages (the KL proper), medians
    (the typical element) and the top-5% share of the total."""
    total = mean_term + var_term
    return dict(mean=float(mean_term.mean()), var=float(var_term.mean()), total=float(total.mean()),
                median_mean=float(np.median(mean_term)), median_var=float(np.median(var_term)),
                median_total=float(np.median(total)), top_share=top_share(total), n=int(total.size))


def weight_space(rec: dict) -> dict:
    """Weight-space D_MF of both methods, the half-vs-half floor and the mean-field gap, over
    the admitted coordinates of one run. Also keeps the per-coordinate arrays (for pooling)."""
    hmc, ok = rec["hmc"], rec["ok"]
    chains = hmc["chains"]                       # (4, 750, K)
    pooled = chains.reshape(-1, chains.shape[-1])  # (3000, K)
    mu_h, s2_h = pooled[:, ok].mean(0), pooled[:, ok].var(0, ddof=1)
    # the artifact fields are exactly these pooled moments (ddof=1); check, then use them
    assert np.allclose(mu_h, hmc["w_eigen_mean"][ok], atol=1e-9)
    assert np.allclose(np.sqrt(s2_h), hmc["w_eigen_std"][ok], atol=1e-9)
    out = {"K_admitted": int(ok.sum()), "K": int(ok.size), "per_coord": {}}
    for name in METHODS:
        q = rec[name]
        mt, vt = kl_gauss_terms(mu_h, s2_h, q["w_eigen_mean"][ok], q["w_eigen_std"][ok] ** 2)
        out[name] = _stats(mt, vt)
        # standardized mean offset at the HMC scale, (dmu/s_hat)^2/2: the mean term a method
        # with the right variances would have had; separates "mean off" from "variance off"
        out[name]["mean_term_at_hmc_scale"] = float(np.mean((q["w_eigen_mean"][ok] - mu_h) ** 2 / (2 * s2_h)))
        out[name]["median_std_ratio"] = float(np.median(q["w_eigen_std"][ok] / np.sqrt(s2_h)))
        out["per_coord"][name] = (mt, vt)
    half_a = chains[:2].reshape(-1, chains.shape[-1])[:, ok]
    half_b = chains[2:].reshape(-1, chains.shape[-1])[:, ok]
    mt, vt = kl_gauss_terms(half_b.mean(0), half_b.var(0, ddof=1), half_a.mean(0), half_a.var(0, ddof=1))
    out["floor"] = _stats(mt, vt)
    out["per_coord"]["floor"] = (mt, vt)
    mt, vt = kl_gauss_terms(half_a.mean(0), half_a.var(0, ddof=1), half_b.mean(0), half_b.var(0, ddof=1))
    out["floor_reverse"] = _stats(mt, vt)
    out["gap"] = mean_field_gap(pooled[:, ok])
    return out


def predictive_space(rec: dict) -> dict:
    """Predictive-space KL of both methods against the HMC predictive, over the full grid."""
    hmc = rec["hmc"]
    v_h = np.maximum(hmc["pred_var"], MIN_VAR)
    out = {"per_point": {}, "grid_shape": [int(s) for s in hmc["pred_u"].shape]}
    for name in METHODS:
        q = rec[name]
        mt, vt = kl_gauss_terms(hmc["pred_u"], v_h, q["pred_u"], np.maximum(q["pred_var"], MIN_VAR))
        out[name] = _stats(mt, vt)
        out[name]["median_std_ratio"] = float(np.median(np.sqrt(np.maximum(q["pred_var"], MIN_VAR) / v_h)))
        out[name]["rmse_vs_hmc_mean"] = float(np.sqrt(np.mean((q["pred_u"] - hmc["pred_u"]) ** 2)))
        out["per_point"][name] = (mt, vt)
    return out


# ----------------------------------------------------------------------------- data
def _load(method: str, nu: float, rho: float, idx: int) -> dict:
    """Cached artifact only. A missing file is an error that names it; nothing is trained."""
    path = rd.cache_path(method, nu, rho, idx)
    if not path.exists():
        raise FileNotFoundError(f"missing {method} artifact {path} (nu={nu:g}, rho={rho:g}, idx={idx}); "
                                f"generate it with {rd.RUNNERS[method].relative_to(REPO_ROOT)} "
                                "(this figure script never trains)")
    return rd.load_cached(method, nu, rho, idx)


def collect_records() -> list[dict]:
    if OVERRIDE:
        print(f"REACDIFF_CELLS override: using {len(ACTIVE_CELLS)} of {len(NUS) * len(RHOS)} cells: {cells_description()}")
    else:
        print(f"all {len(ACTIVE_CELLS)} cells of the grid (nu in {NUS}, rho in {RHOS}), {len(IDXS)} trajectories each")
    records = []
    for nu, rho in ACTIVE_CELLS:
        for idx in IDXS:
            print(f"[nu={_nlabel(nu)} rho={_rlabel(rho)} idx={idx}] loading Ours + multichain HMC + VI (cached)...")
            hmc = _load("HMC", nu, rho, idx)
            ok = (hmc["rhat"] < RHAT_MAX) & (hmc["ess"] > ESS_MIN)
            rec = dict(nu=nu, rho=rho, idx=idx, hmc=hmc, ok=ok, admitted_fraction=float(ok.mean()),
                       accept_rates=[float(a) for a in np.atleast_1d(hmc["accept_rates"])],
                       Ours=_load("Ours", nu, rho, idx), VI=_load("VI", nu, rho, idx))
            rd.assert_same_basis(rec["Ours"], hmc, rec["VI"])   # same eigenbasis V in all three artifacts
            rec["nu_norm"], rec["rho_norm"] = float(hmc["nu_norm"]), float(hmc["rho_norm"])
            rec["included"] = rec["admitted_fraction"] >= MIN_ADMITTED_FRACTION
            if rec["included"]:
                rec["weight"] = weight_space(rec)
                rec["pred"] = predictive_space(rec)
            else:
                print(f"    -> only {100 * rec['admitted_fraction']:.0f}% of HMC coordinates converged "
                      f"(chain acceptance {_accept_str(rec)}); run dropped from every statistic")
            records.append(rec)
    print("\nHMC gate (a run enters only if at least "
          f"{100 * MIN_ADMITTED_FRACTION:.0f}% of its coordinates have split-R-hat < {RHAT_MAX} and bulk ESS > {ESS_MIN:g}):")
    for nu, rho in ACTIVE_CELLS:
        rs = _included(records, nu, rho)
        print(f"  nu={_nlabel(nu)} rho={_rlabel(rho)}: n = {len(rs)} of {len(IDXS)} canonical trajectories admitted "
              f"(idx {', '.join(str(r['idx']) for r in rs)})")
    for rho in ACTIVE_RHOS:
        n_all = len(IDXS) * sum(1 for c in ACTIVE_CELLS if c[1] == rho)
        print(f"  rho={_rlabel(rho)} (all nu pooled): n = {len(_included(records, rho=rho))} of {n_all} runs admitted")
    print(f"  dropped by the HMC gate: {dropped_description(records)}")
    print(f"  {len(_included(records))} runs in total enter every statistic")
    return records


def _accept_str(rec: dict) -> str:
    return "/".join(f"{a:.2f}" for a in rec["accept_rates"])


def dropped_runs(records) -> list[dict]:
    return [r for r in records if not r["included"]]


def dropped_description(records) -> str:
    """'nu=.. rho=.. idx .. (..% admitted, chain acceptance ../../../..); ...' or 'none'."""
    return "; ".join(f"nu={_nlabel(r['nu'])} rho={_rlabel(r['rho'])} idx {r['idx']} ({100 * r['admitted_fraction']:.0f}% admitted, "
                     f"chain acceptance {_accept_str(r)})" for r in dropped_runs(records)) or "none"


def _included(records, nu=None, rho=None):
    return [r for r in records if r["included"] and (nu is None or r["nu"] == nu) and (rho is None or r["rho"] == rho)]


def pooled_description(records) -> str:
    """Which runs the per-rho and 'pooled' rows pool: every admitted run of every cell, listed per rho and nu."""
    parts = []
    for rho in ACTIVE_RHOS:
        per_nu = "; ".join(f"nu={_nlabel(nu)}: idx {', '.join(str(r['idx']) for r in _included(records, nu, rho))}"
                           for nu in ACTIVE_NUS if (nu, rho) in ACTIVE_CELLS)
        parts.append(f"rho={_rlabel(rho)}: {len(_included(records, rho=rho))} runs ({per_nu})")
    return " | ".join(parts)


def aggregate(records, space: str, key: str) -> dict:
    """Per-cell, per-rho and pooled summaries of one series (`key` in Ours / VI / floor) of one
    space. Per group: median over the included runs of the per-run averages, with min-max; plus
    the median contribution over ALL admitted elements of that group (typical element) and the
    element-weighted mean (the KL per element over everything in that group)."""
    arrays_key = "per_coord" if space == "weight" else "per_point"
    out = {}
    for gkey, nu, rho in _groups():
        rs = _included(records, nu, rho)
        if not rs and nu is None:
            # a whole rho group (or everything) empty: the figure cannot be drawn without it
            raise RuntimeError(f"no run in group {gkey} passed the HMC gate; nothing to aggregate there")
        if not rs:
            # a whole CELL dropped by the HMC gate: keep the row (n = 0, every entry '--' / NaN) so that
            # the table still lists all 16 cells; the rho group and the pooled rows are unaffected
            nan = float("nan")
            d = {term: dict(median=nan, min=nan, max=nan, per_run=[]) for term in ("mean", "var", "total")}
            d["elementwise"] = dict(mean=nan, mean_term=nan, var_term=nan, median_total=nan, median_mean_term=nan,
                                    median_var_term=nan, top_share=nan, n_elements=0)
            d["n_runs"], d["runs"] = 0, []
            out[gkey] = d
            continue
        per_run = [r[space][key] for r in rs]
        mt = np.concatenate([r[space][arrays_key][key][0].ravel() for r in rs])
        vt = np.concatenate([r[space][arrays_key][key][1].ravel() for r in rs])
        d = {}
        for term in ("mean", "var", "total"):
            vals = np.array([p[term] for p in per_run])
            d[term] = dict(median=float(np.median(vals)), min=float(vals.min()), max=float(vals.max()),
                           per_run=[float(v) for v in vals])
        d["elementwise"] = dict(mean=float((mt + vt).mean()), mean_term=float(mt.mean()), var_term=float(vt.mean()),
                                median_total=float(np.median(mt + vt)), median_mean_term=float(np.median(mt)),
                                median_var_term=float(np.median(vt)), top_share=top_share(mt + vt),
                                n_elements=int(mt.size))
        d["n_runs"] = len(rs)
        d["runs"] = [dict(nu=r["nu"], rho=r["rho"], idx=r["idx"]) for r in rs]   # same order as per_run
        out[gkey] = d
    return out


def aggregate_gap(records) -> dict:
    """Medians over the runs whose gap is available (a run whose sample correlation matrix is
    not positive definite has no gap); 'n_available' of 'n_runs' says how many that is."""
    out = {}
    for gkey, nu, rho in _groups():
        rs = _included(records, nu, rho)
        d = {}
        for k in ("raw", "corrected", "corrected_cov_formula"):
            vals = np.array([r["weight"]["gap"][k] for r in rs])
            fin = vals[np.isfinite(vals)]
            d[k] = dict(median=float(np.median(fin)) if fin.size else float("nan"),
                        min=float(fin.min()) if fin.size else float("nan"),
                        max=float(fin.max()) if fin.size else float("nan"),
                        per_run=[float(v) for v in vals])
        d["n_runs"] = len(rs)
        d["n_available"] = int(sum(r["weight"]["gap"]["available"] for r in rs))
        d["not_available"] = [f"nu={_nlabel(r['nu'])} rho={_rlabel(r['rho'])} idx {r['idx']}"
                              for r in rs if not r["weight"]["gap"]["available"]]
        out[gkey] = d
    return out


# ----------------------------------------------------------------------------- figure
def _bar_panel(ax, agg: dict, title: str, ylabel: str, floor_band=None, gap=None, ylim=None) -> None:
    """Grouped bars (log y) of the median mean term and median variance term per method and rho
    (each rho group pools all nu), a short black line at the median of the per-run sums (the KL
    itself; not the sum of the two bars, since medians do not add), the MC floor and (weight
    space) the mean-field gap."""
    width, offset = 0.17, 0.2
    for i, rho in enumerate(ACTIVE_RHOS):
        for j, name in enumerate(METHODS):
            a = agg[name][_rkey(rho)]
            xc = i + (j - 0.5) * 2 * offset
            ax.bar(xc - width / 2, a["mean"]["median"], width, color=COLORS[name], edgecolor=COLORS[name])
            ax.bar(xc + width / 2, a["var"]["median"], width, facecolor="white", edgecolor=COLORS[name],
                   hatch="////", linewidth=1.2)
            ax.plot([xc - width, xc + width], [a["total"]["median"]] * 2, color="black", lw=2.2, solid_capstyle="butt")
        if "floor" in agg:
            f = agg["floor"][_rkey(rho)]["total"]["median"]
            ax.plot(i, f, marker="D", color=FLOOR_COLOR, markersize=9, linestyle="none", zorder=5)
        if gap is not None:
            ax.plot(i, gap[_rkey(rho)]["corrected"]["median"], marker="o", markersize=11, markerfacecolor="white",
                    markeredgecolor="black", markeredgewidth=1.8, linestyle="none", zorder=5)
    if floor_band is not None:
        ax.axhspan(floor_band[0], floor_band[1], color=FLOOR_COLOR, alpha=0.35, zorder=0)
    ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    _xaxis(ax, agg)
    ax.set_ylabel(ylabel, fontsize=19)
    ax.set_title(title, fontsize=20)
    ax.grid(True, axis="y", which="major", alpha=0.3)
    ax.tick_params(axis="y", labelsize=13)


def _xaxis(ax, agg: dict) -> None:
    """One tick per rho, labelled with the physical rho and the number of admitted runs n behind
    the group (all nu pooled)."""
    ax.set_xticks(range(len(ACTIVE_RHOS)))
    ax.set_xticklabels([rf"$\rho={_rlabel(rho)}$" + "\n" + rf"$(n={agg[METHODS[0]][_rkey(rho)]['n_runs']})$"
                        for rho in ACTIVE_RHOS], fontsize=14)
    ax.set_xlim(-0.6, len(ACTIVE_RHOS) - 0.4)


def _points_panel(ax, agg: dict, title: str, ylabel: str, floor_band=None, ylim=None) -> None:
    """One point per run (cell x trajectory): the per-run KL total, colour = method, marker
    shape = nu; black bar = median over the rho group (all nu)."""
    rng = np.random.RandomState(3)
    nu_shift = {nu: (ACTIVE_NUS.index(nu) - (len(ACTIVE_NUS) - 1) / 2) * 0.05 for nu in ACTIVE_NUS}
    for i, rho in enumerate(ACTIVE_RHOS):
        for j, name in enumerate(METHODS):
            a = agg[name][_rkey(rho)]
            vals, runs = np.array(a["total"]["per_run"]), a["runs"]
            xc = i + (j - 0.5) * 0.36
            for nu in ACTIVE_NUS:
                sel = np.array([r["nu"] == nu for r in runs])
                if not sel.any():
                    continue
                x = xc + nu_shift[nu] + rng.uniform(-0.02, 0.02, int(sel.sum()))
                ax.scatter(x, vals[sel], s=46, alpha=0.75, marker=NU_MARKERS[nu],
                           facecolors=COLORS[name], edgecolors="black", linewidths=0.6)
            ax.plot([xc - 0.14, xc + 0.14], [np.median(vals)] * 2, color="black", lw=2.2)
        if "floor" in agg:
            a = agg["floor"][_rkey(rho)]
            vals, runs = np.array(a["total"]["per_run"]), a["runs"]
            for nu in ACTIVE_NUS:
                sel = np.array([r["nu"] == nu for r in runs])
                if not sel.any():
                    continue
                x = float(i) + nu_shift[nu] + rng.uniform(-0.015, 0.015, int(sel.sum()))
                ax.scatter(x, vals[sel], s=30, marker=NU_MARKERS[nu], facecolors=FLOOR_COLOR, edgecolors="none", alpha=0.8)
    if floor_band is not None:
        ax.axhspan(floor_band[0], floor_band[1], color=FLOOR_COLOR, alpha=0.35, zorder=0)
    ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    _xaxis(ax, agg)
    ax.set_xlabel(r"$\rho$", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=19)
    ax.set_title(title, fontsize=20)
    ax.grid(True, axis="y", which="major", alpha=0.3)
    ax.tick_params(axis="y", labelsize=13)


def _ylim(agg: dict, lo: float, gap=None) -> tuple[float, float]:
    """Log-axis limits: fixed lower edge (below every floor value), upper edge one third of a decade
    above the largest plotted value, so that extreme per-run values (a collapsed VI run) do not clip
    anything."""
    top = max(v for name in agg for rho in ACTIVE_RHOS for t in ("mean", "var", "total")
              for v in [agg[name][_rkey(rho)][t]["max"]] if np.isfinite(v))
    if gap is not None:
        top = max(top, max(gap[_rkey(rho)]["corrected"]["median"] for rho in ACTIVE_RHOS
                           if np.isfinite(gap[_rkey(rho)]["corrected"]["median"])))
    return lo, 10 ** (np.ceil(3 * np.log10(top)) / 3 + 1 / 3)


def make_figure(agg_w: dict, agg_p: dict, gap: dict, output_path: Path) -> None:
    plt.rcParams["mathtext.fontset"] = "cm"
    pred_floor = PRED_MC_FLOOR_BAND
    ylim_w, ylim_p = _ylim(agg_w, 5e-4, gap=gap), _ylim(agg_p, 1e-4)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12.5))
    _bar_panel(axes[0, 0], agg_w, r"$\mathrm{(a)\ Weight\ space:\ mean}$-$\mathrm{field\ excess\ KL}$",
               r"$\mathrm{excess\ KL\ (nats\ per\ coordinate)}$", gap=gap, ylim=ylim_w)
    _bar_panel(axes[0, 1], agg_p, r"$\mathrm{(b)\ Predictive\ space:\ KL(HMC\,\|\,method)}$",
               r"$\mathrm{KL\ (nats\ per\ grid\ point)}$", floor_band=pred_floor, ylim=ylim_p)
    _points_panel(axes[1, 0], agg_w, r"$\mathrm{(c)\ Weight\ space,\ per\ run}$",
                  r"$\mathrm{excess\ KL\ (nats\ per\ coordinate)}$", ylim=ylim_w)
    _points_panel(axes[1, 1], agg_p, r"$\mathrm{(d)\ Predictive\ space,\ per\ run}$",
                  r"$\mathrm{KL\ (nats\ per\ grid\ point)}$", floor_band=pred_floor, ylim=ylim_p)

    from matplotlib.patches import Patch
    handles = []
    for name in METHODS:
        handles.append(Patch(facecolor=COLORS[name], edgecolor=COLORS[name], label=rf"$\mathrm{{{name}:\ mean\ term}}$"))
        handles.append(Patch(facecolor="white", edgecolor=COLORS[name], hatch="////",
                             label=rf"$\mathrm{{{name}:\ variance\ term}}$"))
    handles += [
        plt.Line2D([], [], color="black", lw=2.2, label=r"$\mathrm{median\ of\ per}$-$\mathrm{run\ sums}$"),
        plt.Line2D([], [], marker="D", color=FLOOR_COLOR, linestyle="none", markersize=9,
                   label=r"$\mathrm{HMC\ half\ vs.\ half\ (MC\ floor)}$"),
        Patch(facecolor=FLOOR_COLOR, alpha=0.35, label=r"$\mathrm{MC\ floor\ (approx.)}$"),
        plt.Line2D([], [], marker="o", markersize=11, markerfacecolor="white", markeredgecolor="black",
                   markeredgewidth=1.8, linestyle="none", label=r"$\mathrm{mean}$-$\mathrm{field\ gap\ of\ HMC\ (corrected)}$"),
        plt.Line2D([], [], marker="o", color=COLORS["Ours"], linestyle="none", markersize=9,
                   markeredgecolor="black", label=r"$\mathrm{Ours,\ one\ run\ (c,\,d)}$"),
        plt.Line2D([], [], marker="o", color=COLORS["VI"], linestyle="none", markersize=9,
                   markeredgecolor="black", label=r"$\mathrm{VI,\ one\ run\ (c,\,d)}$"),
        plt.Line2D([], [], marker="o", color=FLOOR_COLOR, linestyle="none", markersize=7,
                   label=r"$\mathrm{HMC\ half\ vs.\ half,\ one\ run\ (c)}$"),
    ]
    # An invisible spacer entry so that, with 4 columns filled top to bottom, the four nu marker
    # entries below sit together in the last column instead of being split over two columns.
    handles.append(plt.Line2D([], [], linestyle="none", marker="none", label=" "))
    handles += [plt.Line2D([], [], marker=NU_MARKERS[nu], markerfacecolor="white", markeredgecolor="black",
                           markeredgewidth=1.2, linestyle="none", markersize=9,
                           label=rf"$\nu={_nlabel(nu)}\ \mathrm{{(marker\ shape\ in\ c,\,d)}}$") for nu in NUS]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=13.5, bbox_to_anchor=(0.5, -0.005),
               frameon=True)
    if OVERRIDE:
        fig.text(0.5, 0.995, f"REACDIFF_CELLS override: only {cells_description()}", ha="center", va="top",
                 fontsize=13, color="red")
    fig.tight_layout(rect=(0, 0.1, 1, 0.985 if OVERRIDE else 1))
    fig.savefig(output_path, bbox_inches="tight")
    if PNG_PREVIEW is not None:  # same figure object, for checking the PDF without poppler
        fig.savefig(PNG_PREVIEW, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path}")


PNG_PREVIEW: Path | None = None  # set by `--png <path>`: also write a PNG preview there (not shipped)


# ----------------------------------------------------------------------------- table
def _f(x: float) -> str:
    if not np.isfinite(x):
        return "--"
    if abs(x) >= 1000:
        return f"{x:.0f}"
    return f"{x:.3g}"


def _plain_sci(x: float) -> str:
    """3e-4 style, for the comment lines."""
    m, e = f"{x:.0e}".split("e")
    return f"{m}e{int(e)}"


def _cell(d: dict) -> str:
    """median [min--max] over the runs of the group; '--' for a cell without admitted runs."""
    if not np.isfinite(d["median"]):
        return "--"
    return rf"{_f(d['median'])} [{_f(d['min'])}--{_f(d['max'])}]"


def _median(vals) -> float:
    """Median of a list, NaN (printed as '--') when the list is empty (a cell without admitted runs)."""
    return float(np.median(vals)) if len(vals) else float("nan")


def _row_head(gkey: str, nu, rho, n: int) -> str:
    """The three label columns nu & rho & n of a table row."""
    if nu is not None:
        return f"{_nlabel(nu)} & {_rlabel(rho)} & {n}"
    if rho is not None:
        return rf"\multicolumn{{2}}{{l}}{{all $\nu$, $\rho={_rlabel(rho)}$}} & {n}"
    return rf"\multicolumn{{2}}{{l}}{{pooled}} & {n}"


def write_table(agg_w: dict, agg_p: dict, gap: dict, records, path: Path) -> None:
    n_all = agg_w["Ours"]["pooled"]["n_runs"]
    k_adm = {g: [(r["weight"]["K_admitted"], r["weight"]["K"]) for r in _included(records, nu, rho)]
             for g, nu, rho in _groups() if nu is not None}
    k_all = [r["weight"]["K"] for r in _included(records)]
    grid = _included(records)[0]["pred"]["grid_shape"]
    gap_na = [k for k in gap if gap[k]["n_available"] < gap[k]["n_runs"]]
    vi_nonfinite = [f"nu={_nlabel(r['nu'])} rho={_rlabel(r['rho'])} idx {r['idx']}" for r in _included(records)
                    if not (np.isfinite(r["weight"]["VI"]["total"]) and np.isfinite(r["pred"]["VI"]["total"]))]
    norm = sorted({(r["nu"], r["nu_norm"], r["rho"], r["rho_norm"]) for r in records})
    lo, hi = (_plain_sci(v) for v in PRED_MC_FLOOR_BAND)
    lo_neff, hi_neff = (_plain_sci(1 / n) for n in (N_EFF_RANGE[1], N_EFF_RANGE[0]))
    pct = f"{100 * TOP_SHARE:.0f}"
    lines = [
        r"% Auto-generated by figures_onepass/diffusion/posterior_fit_metrics.py",
        r"% Fit of each mean-field posterior (Ours = one-pass EP (expectation propagation), and VI (variational inference))",
        r"% to the multichain HMC (Hamiltonian Monte Carlo) reference, on the PDEBench 1D reaction-diffusion (Fisher-KPP)",
        r"% dataset u_t = nu u_xx + rho u (1 - u). Mean-field = independent coordinates, here a diagonal Gaussian in the",
        r"% shared eigenbasis (the rotated weight coordinates all methods share; the three artifacts of every run were",
        r"% checked to carry the same basis). Every entry is a KL (Kullback-Leibler divergence) in nats.",
        rf"% Grid: nu (diffusion coefficient) in {{{', '.join(_nlabel(n) for n in NUS)}}} x rho (reaction strength) in {{{', '.join(_rlabel(r) for r in RHOS)}}},",
        rf"% {len(IDXS)} canonical trajectories per cell (idx {', '.join(str(i) for i in IDXS)}). The rho and nu labels are the physical values;",
        r"% the model uses the normalized values rho_norm = rho T_basis and nu_norm = nu T_basis / L^2, read from the artifacts:",
        r"% " + "; ".join(f"nu={_nlabel(a)} -> nu_norm={b:g}, rho={_rlabel(c)} -> rho_norm={d:g}" for a, b, c, d in norm) + ".",
    ]
    if OVERRIDE:
        lines.append(rf"% REACDIFF_CELLS OVERRIDE (partial grid, not the shipped table): only the cells {cells_description()} were used.")
    lines += [
        rf"% HMC gate: a run enters only if at least {100 * MIN_ADMITTED_FRACTION:.0f}% of its HMC coordinates converged (split-R-hat < {RHAT_MAX}, bulk ESS > {ESS_MIN:g});",
        rf"% decided from the cached chains when this table was made. Runs entering: the n column of every row ({n_all} in total);",
        rf"% dropped by the HMC gate: {dropped_description(records)}.",
        rf"% Admitted coordinates (HMC converged there) per run, as 'admitted of K' (K = the number of eigen-coordinates of the",
        rf"% run's basis, {min(k_all)} to {max(k_all)} here; it varies with the trajectory): "
        + "; ".join(f"{g}: " + (", ".join(f"{a} of {k}" for a, k in k_adm[g]) if k_adm[g] else "no run admitted")
                    for g in k_adm) + ".",
        r"% Rows: one per cell (nu, rho): median over its runs [min--max]; 'all nu, rho=..' pools the runs of the four nu",
        r"% at that rho (the rho groups of the figure); 'pooled' pools every run that entered, i.e.",
        rf"% {pooled_description(records)}.",
        rf"% VI runs whose KL is not finite (shown as '--'): {', '.join(vi_nonfinite) or 'none'}.",
        rf"% Mean-field gap not available (sample correlation matrix not positive definite): "
        + ("; ".join(f"{k}: {', '.join(gap[k]['not_available'])} ({gap[k]['n_available']} of {gap[k]['n_runs']} runs available)" for k in gap_na) or "none") + ".",
        r"% (A) Weight space. Every entry is in nats per admitted coordinate. mean term = (mu_HMC - mu_q)^2 / (2 s_q^2),",
        r"%     variance term = 0.5 (r - 1 - ln r) with r = s_HMC^2 / s_q^2, both averaged over the admitted",
        r"%     coordinates of a run; sum = D_MF, the KL(HMC marginals || method) in excess of the best mean-field fit.",
        r"%     Cells: median over the runs of the row [min--max] of the per-run values. The 'sum' column is the median over",
        r"%     runs of the per-run totals (mean term + variance term of the same run), so it is NOT the sum of",
        r"%     the two neighbouring median cells (medians do not add). 'HMC half vs half' = the same D_MF between the",
        r"%     moment-matched Gaussian of chains 1,2 (as q) and the moments of chains 3,4 (as target): the Monte-Carlo",
        r"%     noise floor. 'mean-field gap' = -0.5 ln det Corr_HMC / K over the admitted coordinates (ln det = the",
        r"%     log-determinant, Corr_HMC = the HMC sample correlation matrix): the KL a PERFECT mean-field fit would still",
        r"%     lose if the HMC posterior were Gaussian; 'raw', and 'corrected' with the Wishart bias correction (a",
        r"%     correction for the downward bias of a sample log-determinant; n = 3000 draws treated as independent;",
        r"%     approximate). The 'HMC half vs half', 'raw' and 'corrected' columns are medians over runs only",
        r"%     (no min--max). Pooled rows: 'median' = median over all runs; 'mean' = coordinate-weighted average over",
        r"%     all admitted coordinates of all runs (the KL per coordinate over everything).",
        r"% (B) Weight space, robustness. The per-coordinate contributions are heavy-tailed (a few coordinates carry most",
        r"%     of the sum), so this block reports, per cell, per rho and pooled: 'median mean term' and 'median var. term'",
        r"%     = the median contribution over ALL admitted coordinates of all included runs of the row (the typical",
        rf"%     coordinate, not the average); 'top-{pct}% share' = the share of the summed contribution (mean term +",
        rf"%     variance term over the same coordinates) carried by the {pct}% largest per-coordinate contributions;",
        r"%     '(dmu/s_HMC)^2/2' = (mu_HMC - mu_q)^2 / (2 s_HMC^2) averaged over the admitted coordinates of a run, then",
        r"%     the median over the runs of the row: the mean term the method would have if its variances were right, which",
        r"%     separates 'mean off' from 'variance off'.",
        r"% (C) Predictive space. KL(HMC predictive || method predictive) per grid point, in nats: the same Gaussian KL",
        r"%     and mean/variance split as in (A), but between the stored HMC predictive N(m_H, v_H) and the method's",
        rf"%     N(m_q, v_q) at each (x,t) grid point, averaged over the full {grid[0]} x {grid[1]} grid (variances floored at",
        rf"%     {_plain_sci(MIN_VAR)}). Cells: median over runs [min--max] of the per-run averages; 'sum' is again the median",
        r"%     of the per-run totals, not the sum of the neighbouring cells. 'median std ratio to HMC' = the median over",
        r"%     grid points of sqrt(v_q / v_H) per run, then the median over the runs of the row. The HMC predictive's own",
        r"%     Monte-Carlo error puts a floor under these numbers, drawn in the figure as the band 'MC floor (approx.)'",
        rf"%     from {lo} to {hi} nats per point: 1/n_eff with n_eff = {N_EFF_RANGE[0]:g} to {N_EFF_RANGE[1]:g} draws gives {lo_neff} to {hi_neff} for the",
        r"%     mean term alone, but the weight-space 'HMC half vs half' floor shows that the variance term of HMC's own",
        rf"%     noise is about 4 to 5 times larger than 1/n would give, so the band is widened up to {hi}.",
        r"%     Pooled rows as in (A), with grid points in place of coordinates.",
        r"\begin{tabular}{l l r rrr rrr r rr}",
        r"\toprule",
        r"\multicolumn{12}{l}{\emph{(A) Weight space: mean-field excess KL $D_{\mathrm{MF}}$ [nats per coordinate]}}\\",
        r" & & & \multicolumn{3}{c}{Ours (one-pass EP)} & \multicolumn{3}{c}{VI} & HMC half & \multicolumn{2}{c}{mean-field gap} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-10}\cmidrule(lr){11-12}",
        r"$\nu$ & $\rho$ & $n$ & mean term & variance term & sum & mean term & variance term & sum & vs.\ half & raw & corrected \\",
        r"\midrule",
    ]
    groups = _groups()
    for gkey, nu, rho in groups:
        if nu is None:
            continue   # cells first, then the per-rho rows, then pooled
        cells = [_cell(agg_w["Ours"][gkey][t]) for t in ("mean", "var", "total")]
        cells += [_cell(agg_w["VI"][gkey][t]) for t in ("mean", "var", "total")]
        cells += [_f(agg_w["floor"][gkey]["total"]["median"]), _f(gap[gkey]["raw"]["median"]), _f(gap[gkey]["corrected"]["median"])]
        lines.append(_row_head(gkey, nu, rho, agg_w["Ours"][gkey]["n_runs"]) + " & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    for gkey, nu, rho in groups:
        if nu is not None or rho is None:
            continue
        cells = [_cell(agg_w["Ours"][gkey][t]) for t in ("mean", "var", "total")]
        cells += [_cell(agg_w["VI"][gkey][t]) for t in ("mean", "var", "total")]
        cells += [_f(agg_w["floor"][gkey]["total"]["median"]), _f(gap[gkey]["raw"]["median"]), _f(gap[gkey]["corrected"]["median"])]
        lines.append(_row_head(gkey, nu, rho, agg_w["Ours"][gkey]["n_runs"]) + " & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    k = "pooled"
    cells = [_f(agg_w[m][k][t]["median"]) for m in METHODS for t in ("mean", "var", "total")]
    cells += [_f(agg_w["floor"][k]["total"]["median"]), _f(gap[k]["raw"]["median"]), _f(gap[k]["corrected"]["median"])]
    lines.append(rf"\multicolumn{{2}}{{l}}{{pooled (median)}} & {n_all} & " + " & ".join(cells) + r" \\")
    cells = [_f(agg_w[m][k]["elementwise"][t]) for m in METHODS for t in ("mean_term", "var_term", "mean")]
    cells += [_f(agg_w["floor"][k]["elementwise"]["mean"]), "--", "--"]
    lines.append(rf"\multicolumn{{2}}{{l}}{{pooled (mean)}} & {n_all} & " + " & ".join(cells) + r" \\")
    lines += [
        r"\midrule",
        rf"\multicolumn{{12}}{{l}}{{\emph{{(B) Weight space, robustness: typical coordinate, top-{pct}\% share, mean offset at the HMC scale}}}}\\",
        r" & & & \multicolumn{4}{c}{Ours (one-pass EP)} & \multicolumn{4}{c}{VI} & \\",
        r"\cmidrule(lr){4-7}\cmidrule(lr){8-11}",
        rf"$\nu$ & $\rho$ & $n$ & median mean term & median var.\ term & top-{pct}\% share & $(\Delta\mu/\sigma_{{\mathrm{{HMC}}}})^2/2$ & "
        rf"median mean term & median var.\ term & top-{pct}\% share & $(\Delta\mu/\sigma_{{\mathrm{{HMC}}}})^2/2$ & \\",
        r"\midrule",
    ]
    last_kind = None
    for gkey, nu, rho in groups:
        kind = "cell" if nu is not None else ("rho" if rho is not None else "pooled")
        if last_kind is not None and kind != last_kind:
            lines.append(r"\midrule")
        last_kind = kind
        cells = []
        for m in METHODS:
            e = agg_w[m][gkey]["elementwise"]
            z2 = _median([r["weight"][m]["mean_term_at_hmc_scale"] for r in _included(records, nu, rho)])
            cells += [_f(e["median_mean_term"]), _f(e["median_var_term"]),
                      f"{e['top_share']:.2f}" if np.isfinite(e["top_share"]) else "--", _f(z2)]
        lines.append(_row_head(gkey, nu, rho, agg_w["Ours"][gkey]["n_runs"]) + " & " + " & ".join(cells) + r" & \\")
    lines += [
        r"\midrule",
        r"\multicolumn{12}{l}{\emph{(C) Predictive space: KL(HMC predictive $\|$ method predictive) [nats per grid point]}}\\",
        r" & & & \multicolumn{3}{c}{Ours (one-pass EP)} & \multicolumn{3}{c}{VI} & & \multicolumn{2}{c}{median std ratio to HMC} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){11-12}",
        r"$\nu$ & $\rho$ & $n$ & mean term & variance term & sum & mean term & variance term & sum & & Ours & VI \\",
        r"\midrule",
    ]
    last_kind = None
    for gkey, nu, rho in groups:
        if nu is None and rho is None:
            continue
        kind = "cell" if nu is not None else "rho"
        if last_kind is not None and kind != last_kind:
            lines.append(r"\midrule")
        last_kind = kind
        cells = [_cell(agg_p[m][gkey][t]) for m in METHODS for t in ("mean", "var", "total")]
        ratios = [_median([r["pred"][m]["median_std_ratio"] for r in _included(records, nu, rho)]) for m in METHODS]
        lines.append(_row_head(gkey, nu, rho, agg_p["Ours"][gkey]["n_runs"]) + " & " + " & ".join(cells) + " & & "
                     + " & ".join(f"{v:.2f}" if np.isfinite(v) else "--" for v in ratios) + r" \\")
    lines.append(r"\midrule")
    k = "pooled"
    cells = [_f(agg_p[m][k][t]["median"]) for m in METHODS for t in ("mean", "var", "total")]
    ratios = [np.median([r["pred"][m]["median_std_ratio"] for r in _included(records)]) for m in METHODS]
    lines.append(rf"\multicolumn{{2}}{{l}}{{pooled (median)}} & {n_all} & " + " & ".join(cells) + " & & "
                 + " & ".join(f"{v:.2f}" for v in ratios) + r" \\")
    cells = [_f(agg_p[m][k]["elementwise"][t]) for m in METHODS for t in ("mean_term", "var_term", "mean")]
    lines.append(rf"\multicolumn{{2}}{{l}}{{pooled (mean)}} & {n_all} & " + " & ".join(cells) + r" & & & \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines))
    print(f"Saved table to {path}")


# ----------------------------------------------------------------------------- checks
def sanity_checks(records, agg_w, agg_p) -> dict:
    """Printed and stored: non-negativity, and closed-form consistency of the variance and
    mean terms at the median std ratio / median standardized mean offset, per rho group."""
    checks = {}
    min_val = min(min(r[space][arr][k][i].min() for space, arr in (("weight", "per_coord"), ("pred", "per_point"))
                      for k in r[space][arr] for i in (0, 1)) for r in _included(records))
    checks["min_contribution"] = float(min_val)
    print(f"\nSanity: smallest per-element KL contribution over everything = {min_val:.3e} (must be >= 0)")
    for rho in ACTIVE_RHOS:
        rs = _included(records, rho=rho)
        if not rs:
            print(f"  rho={_rlabel(rho)}: no run admitted by the HMC gate, skipped")
            continue
        for m in METHODS:
            ratio = float(np.median([r["weight"][m]["median_std_ratio"] for r in rs]))
            r2 = 1 / ratio ** 2
            closed = 0.5 * (r2 - 1 - np.log(r2))
            med_var = agg_w[m][_rkey(rho)]["elementwise"]["median_var_term"]
            checks[f"rho={_rlabel(rho)} {m} var-term"] = dict(
                median_std_ratio=ratio, closed_form_at_median_ratio=float(closed),
                median_var_term=med_var, mean_var_term=agg_w[m][_rkey(rho)]["var"]["median"])
            print(f"  rho={_rlabel(rho)} {m}: median s_q/s_HMC = {ratio:.3f} -> closed-form variance term {closed:.2f}; "
                  f"median per-coordinate variance term {med_var:.2f}; per-run AVERAGE {agg_w[m][_rkey(rho)]['var']['median']:.2f}")
    return checks


class _Tee:
    """Duplicate everything printed into the .log next to the outputs."""

    def __init__(self, path: Path):
        self.file, self.stdout = open(path, "w"), sys.stdout

    def write(self, s: str) -> None:
        self.stdout.write(s)
        self.file.write(s)

    def flush(self) -> None:
        self.stdout.flush()
        self.file.flush()


def main() -> None:
    sys.stdout = _Tee(OUTPUT_LOG)
    records = collect_records()
    agg_w = {k: aggregate(records, "weight", k) for k in METHODS + ("floor",)}
    agg_p = {k: aggregate(records, "pred", k) for k in METHODS}
    gap = aggregate_gap(records)
    make_figure(agg_w, agg_p, gap, OUTPUT_PDF)
    write_table(agg_w, agg_p, gap, records, OUTPUT_TABLE)
    checks = sanity_checks(records, agg_w, agg_p)

    per_run = [dict(nu=r["nu"], rho=r["rho"], idx=r["idx"], nu_norm=r["nu_norm"], rho_norm=r["rho_norm"],
                    included=r["included"], admitted_fraction=r["admitted_fraction"], accept_rates=r["accept_rates"],
                    weight={k: v for k, v in r["weight"].items() if k != "per_coord"} if r["included"] else None,
                    pred={k: v for k, v in r["pred"].items() if k != "per_point"} if r["included"] else None)
               for r in records]
    summary = dict(
        description="Fit of the one-pass EP posterior (Ours) and the VI posterior to the multichain HMC reference on "
                    "the PDEBench reaction-diffusion (Fisher-KPP) grid nu x rho. weight: mean-field excess KL per "
                    "admitted eigen-coordinate (mean term, variance term, total), the half-vs-half Monte-Carlo floor "
                    "and the mean-field gap of the HMC posterior; pred: Gaussian KL per grid point between the HMC "
                    "predictive and the method predictive. Groups: 'nu=..,rho=..' = one cell (its trajectories), "
                    "'rho=..' = all nu pooled at that rho (the figure's rho groups), 'pooled' = everything. Per-group "
                    "entries are medians over runs [min, max] of the per-run averages ('runs' lists the runs in the "
                    "order of 'per_run'); 'elementwise' pools all coordinates / grid points; 'median_*' are medians of "
                    "the per-element contributions (typical element). pred_mc_floor_band: the approximate Monte-Carlo "
                    "floor band [nats per grid point] drawn in the figure (1/n_eff for n_eff in n_eff_range covers the "
                    "mean term; the upper edge allows for the variance term of HMC's own noise, about 4 to 5 times 1/n). "
                    "cells: the (nu, rho) cells this run covered (override = REACDIFF_CELLS was set, partial grid); "
                    "idxs: the canonical trajectories of every cell; n_runs_by_cell / n_runs_by_rho and "
                    "dropped_by_hmc_gate: what the HMC gate (min_admitted_fraction) admitted and dropped at run time; "
                    "pooled_runs: the runs the per-rho and 'pooled' entries pool. A mean-field gap of NaN means the "
                    "sample correlation matrix was not positive definite (gap not available).",
        rhat_max=RHAT_MAX, ess_min=ESS_MIN, min_admitted_fraction=MIN_ADMITTED_FRACTION, min_var=MIN_VAR,
        pred_mc_floor_band=list(PRED_MC_FLOOR_BAND), n_eff_range=list(N_EFF_RANGE),
        nus=NUS, rhos=RHOS, cells=[dict(nu=nu, rho=rho) for nu, rho in ACTIVE_CELLS], override=OVERRIDE, idxs=IDXS,
        n_runs_by_cell={_ckey(nu, rho): len(_included(records, nu, rho)) for nu, rho in ACTIVE_CELLS},
        n_runs_by_rho={_rkey(rho): len(_included(records, rho=rho)) for rho in ACTIVE_RHOS},
        dropped_by_hmc_gate=[dict(nu=r["nu"], rho=r["rho"], idx=r["idx"], admitted_fraction=r["admitted_fraction"],
                                  accept_rates=r["accept_rates"]) for r in dropped_runs(records)],
        pooled_runs={_ckey(nu, rho): [r["idx"] for r in _included(records, nu, rho)] for nu, rho in ACTIVE_CELLS},
        weight=agg_w, weight_mean_field_gap=gap, pred=agg_p, sanity=checks, per_run=per_run,
    )
    OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2))
    print(f"Saved summary to {OUTPUT_SUMMARY}")

    print("\nWeight space, nats per admitted coordinate (median over runs of the per-run average):")
    for k, nu, rho in _groups():
        s = " | ".join(f"{m}: mean {agg_w[m][k]['mean']['median']:.3g} + var {agg_w[m][k]['var']['median']:.3g} "
                       f"= {agg_w[m][k]['total']['median']:.3g}" for m in METHODS)
        print(f"  {k:>16} (n={agg_w['Ours'][k]['n_runs']}): {s} | floor {agg_w['floor'][k]['total']['median']:.2e} | "
              f"gap raw {gap[k]['raw']['median']:.3f} corrected {gap[k]['corrected']['median']:.3f} "
              f"({gap[k]['n_available']} of {gap[k]['n_runs']} runs available)")
    print("Predictive space, nats per grid point (median over runs of the per-run average):")
    for k, nu, rho in _groups():
        s = " | ".join(f"{m}: mean {agg_p[m][k]['mean']['median']:.3g} + var {agg_p[m][k]['var']['median']:.3g} "
                       f"= {agg_p[m][k]['total']['median']:.3g}" for m in METHODS)
        print(f"  {k:>16} (n={agg_p['Ours'][k]['n_runs']}): {s}")
    print(f"Cells used: {cells_description()}" + (" (REACDIFF_CELLS override, partial grid)" if OVERRIDE else " (full grid)"))
    print(f"Pooled rows pool: {pooled_description(records)}")
    print(f"Dropped by the HMC gate: {dropped_description(records)}")
    print(f"Log written to {OUTPUT_LOG}")


if __name__ == "__main__":
    if "--png" in sys.argv:
        PNG_PREVIEW = Path(sys.argv[sys.argv.index("--png") + 1])
    try:
        main()
    except BaseException:
        import traceback
        print("\nFAILED (the .pdf/.tex/.json next to this log are from an earlier run, not this one):\n"
              + traceback.format_exc())
        raise
