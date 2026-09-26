"""
How well do the one-pass EP (expectation propagation) posterior ("Ours") and the mean-field VI
(variational inference) posterior each fit the multi-chain HMC (Hamiltonian Monte Carlo)
reference posterior? Mean-field means that the posterior is approximated by independent
coordinates: both approximations are diagonal Gaussians in the shared eigenbasis (the rotated
weight coordinates all methods share); HMC gives 4 chains x 750 kept draws = 3000 pooled draws
per run. Wave speeds b in {0.1, 0.4, 1.0, 2.0, 7.0}, the ten canonical trajectories idxs(10) at
every b (one b = 7 run costs about an hour of HMC and two hours of VI). All from cached
artifacts. Every number is a KL
(Kullback-Leibler divergence, in nats; a nat is the unit of a KL computed with natural
logarithms).

Weight space, the "mean-field excess KL" D_MF (nats per admitted coordinate):
  D_MF(q) = (1/K) sum_k KL( N(mu_hat_k, s_hat_k^2) || N(mu_q_k, s_q_k^2) ),
  mu_hat_k, s_hat_k = HMC marginal mean and std of eigen-coordinate k (pooled draws),
  mu_q_k, s_q_k = the method's marginal. Per coordinate this splits exactly into a
  mean term (mu_hat - mu_q)^2 / (2 s_q^2) and a variance term 0.5 (r - 1 - ln r), r = s_hat^2/s_q^2.
  Why this is the right number: for ANY posterior p and any diagonal Gaussian q, log q is a
  quadratic, so E_p[log q] depends on p only through its marginal means and variances; hence
  KL(p||q) minus the best achievable KL(p||q') over all diagonal Gaussians q' equals exactly
  the sum above (the minimizer q' is the moment-matched diagonal Gaussian, the one with the same
  means and variances as p). D_MF is the number of nats the method loses relative to the best
  possible mean-field approximation of the HMC posterior, needs no Gaussian assumption on p,
  and is 0 iff every marginal mean and variance is matched. Coordinates enter only when HMC
  converged there: split-R-hat < 1.05 (split-R-hat compares the spread between the split
  half-chains with the spread within them and is 1 at convergence) and bulk ESS > 100 (bulk ESS
  is the effective number of independent draws for the central part of the distribution, after
  discounting autocorrelation); a run with fewer
  than half its coordinates admitted is dropped from every statistic. Which runs this drops is
  decided from the cached HMC chains at run time and is printed in the log, written into the
  table comment and stored in the summary JSON, together with the number of runs per wave
  speed n (also under every figure tick); a dropped run is one whose HMC has not converged
  (typically visible as chain acceptance rates near zero), and re-running this script on a
  regenerated HMC artifact picks it up without any edit.
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
  mean/variance split, averaged over the full 1024 x 201 grid. HMC's own Monte-Carlo error
  puts a floor under these numbers, drawn in the figure as the band "MC floor (approx.)" from
  3e-4 to 2e-3 nats per point: 1/n_eff with n_eff of 2000 to 3000 draws gives 3e-4 to 5e-4 for
  the mean term alone, but the weight-space half-vs-half floor shows that the variance term of
  HMC's own noise is about 4 to 5 times larger than 1/n would give (autocorrelated draws make
  the second moments noisier than the means), so the band is widened up to 2e-3.

Because the per-coordinate contributions are heavy-tailed (a few coordinates carry most of the
sum), the table also reports the median contribution over all admitted coordinates ("typical
coordinate") and the share of the sum carried by the top 5% of coordinates.

Outputs, all next to this script: posterior_fit_metrics.pdf, posterior_fit_metrics_table.tex,
posterior_fit_metrics_summary.json.

Run: python figures_onepass/advection/posterior_fit_metrics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import digamma

FIGURES_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURES_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "pde" / "advection"))

import mp_pinn_beta_sweep as mp_sweep
import bpinn_multichain_sweep as hmc_sweep
import vi_baseline_sweep as vi_sweep
from trajectory_selection import idxs

OUTPUT_PDF = FIGURES_DIR / "posterior_fit_metrics.pdf"
OUTPUT_TABLE = FIGURES_DIR / "posterior_fit_metrics_table.tex"
OUTPUT_SUMMARY = FIGURES_DIR / "posterior_fit_metrics_summary.json"

BETAS = [0.1, 0.4, 1.0, 2.0, 7.0]
# The same ten canonical trajectories (one seeded draw) at every wave speed.
IDXS_BY_BETA = {b: idxs(10) for b in (0.1, 0.4, 1.0, 2.0, 7.0)}  # the same ten trajectories at every b
RHAT_MAX = 1.05
ESS_MIN = 100.0
MIN_ADMITTED_FRACTION = 0.5   # runs with fewer admitted HMC coordinates are dropped everywhere
MIN_VAR = 1e-14               # floor on predictive variances, as in the other scripts
N_EFF_RANGE = (2000.0, 3000.0)     # HMC effective draws behind the 1/n_eff part of the predictive-space floor
PRED_MC_FLOOR_BAND = (3e-4, 2e-3)  # predictive-space MC floor band [nats per point], see the docstring
TOP_SHARE = 0.05              # "top 5% of coordinates" share of the sum
METHODS = ("Ours", "VI")
COLORS = {"Ours": "#4a3aa7", "HMC": "#eb6834", "VI": "#008300"}
MARKERS = {"Ours": "o", "HMC": "s", "VI": "^"}
FLOOR_COLOR = "#7a7a7a"


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
    out = {"per_point": {}}
    for name in METHODS:
        q = rec[name]
        mt, vt = kl_gauss_terms(hmc["pred_u"], v_h, q["pred_u"], np.maximum(q["pred_var"], MIN_VAR))
        out[name] = _stats(mt, vt)
        out[name]["median_std_ratio"] = float(np.median(np.sqrt(np.maximum(q["pred_var"], MIN_VAR) / v_h)))
        out[name]["rmse_vs_hmc_mean"] = float(np.sqrt(np.mean((q["pred_u"] - hmc["pred_u"]) ** 2)))
        out["per_point"][name] = (mt, vt)
    return out


# ----------------------------------------------------------------------------- data
def collect_records() -> list[dict]:
    records = []
    for beta in BETAS:
        for idx in IDXS_BY_BETA[beta]:
            print(f"[beta={beta} idx={idx}] loading Ours + multichain HMC + VI (cached)...")
            hmc = hmc_sweep.load_cached(beta, idx)
            ok = (hmc["rhat"] < RHAT_MAX) & (hmc["ess"] > ESS_MIN)
            rec = dict(beta=beta, idx=idx, hmc=hmc, ok=ok, admitted_fraction=float(ok.mean()),
                       accept_rates=[float(a) for a in np.atleast_1d(hmc["accept_rates"])],
                       Ours=mp_sweep.load_cached(beta, idx), VI=vi_sweep.load_cached(beta, idx))
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
    for beta in BETAS:
        n_in, n_all = len(_included(records, beta)), len(IDXS_BY_BETA[beta])
        print(f"  b={beta:g}: n = {n_in} of {n_all} canonical trajectories admitted "
              f"(idx {', '.join(str(r['idx']) for r in _included(records, beta))})")
    print(f"  dropped by the HMC gate: {dropped_description(records)}")
    print(f"  {len(_included(records))} runs in total enter every statistic")
    return records


def _accept_str(rec: dict) -> str:
    return "/".join(f"{a:.2f}" for a in rec["accept_rates"])


def dropped_runs(records) -> list[dict]:
    return [r for r in records if not r["included"]]


def dropped_description(records) -> str:
    """'b=.. idx .. (..% admitted, chain acceptance ../../../..); ...' or 'none'."""
    return "; ".join(f"b={_blabel(r['beta'])} idx {r['idx']} ({100 * r['admitted_fraction']:.0f}% admitted, "
                     f"chain acceptance {_accept_str(r)})" for r in dropped_runs(records)) or "none"


def _included(records, beta=None):
    return [r for r in records if r["included"] and (beta is None or r["beta"] == beta)]


def pooled_description(records) -> str:
    """Which runs the 'pooled' rows pool: every admitted run of every wave speed, listed per b."""
    return "; ".join(f"b={_blabel(b)}: {len(_included(records, b))} runs (idx "
                     f"{', '.join(str(r['idx']) for r in _included(records, b))})" for b in BETAS)


def aggregate(records, space: str, key: str) -> dict:
    """Per-b and pooled summaries of one series (`key` in Ours / VI / floor) of one space.
    Per b: median over the included trajectories of the per-run averages, with min-max; plus
    the median contribution over ALL admitted elements of that b (typical element) and the
    element-weighted mean (the KL per element over everything at that b)."""
    arrays_key = "per_coord" if space == "weight" else "per_point"
    out = {}
    for beta in BETAS + [None]:
        rs = _included(records, beta)
        if not rs:
            raise RuntimeError(f"no run at b={beta} passed the HMC gate; nothing to aggregate for that wave speed")
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
        out["pooled" if beta is None else str(beta)] = d
    return out


def aggregate_gap(records) -> dict:
    """Medians over the runs whose gap is available (a run whose sample correlation matrix is
    not positive definite has no gap); 'n_available' of 'n_runs' says how many that is."""
    out = {}
    for beta in BETAS + [None]:
        rs = _included(records, beta)
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
        d["not_available"] = [f"b={_blabel(r['beta'])} idx {r['idx']}" for r in rs if not r["weight"]["gap"]["available"]]
        out["pooled" if beta is None else str(beta)] = d
    return out


# ----------------------------------------------------------------------------- figure
def _bar_panel(ax, agg: dict, title: str, ylabel: str, floor_band=None, gap=None, ylim=None) -> None:
    """Grouped bars (log y) of the median mean term and median variance term per method and b, a
    short black line at the median of the per-run sums (the KL itself; not the sum of the two bars,
    since medians do not add), the MC floor and (weight space) the mean-field gap."""
    width, offset = 0.17, 0.2
    for i, beta in enumerate(BETAS):
        for j, name in enumerate(METHODS):
            a = agg[name][str(beta)]
            xc = i + (j - 0.5) * 2 * offset
            ax.bar(xc - width / 2, a["mean"]["median"], width, color=COLORS[name], edgecolor=COLORS[name])
            ax.bar(xc + width / 2, a["var"]["median"], width, facecolor="white", edgecolor=COLORS[name],
                   hatch="////", linewidth=1.2)
            ax.plot([xc - width, xc + width], [a["total"]["median"]] * 2, color="black", lw=2.2, solid_capstyle="butt")
        if "floor" in agg:
            f = agg["floor"][str(beta)]["total"]["median"]
            ax.plot(i, f, marker="D", color=FLOOR_COLOR, markersize=9, linestyle="none", zorder=5)
        if gap is not None:
            ax.plot(i, gap[str(beta)]["corrected"]["median"], marker="o", markersize=11, markerfacecolor="white",
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
    """One tick per wave speed, labelled with b and the number of admitted runs n behind the group."""
    ax.set_xticks(range(len(BETAS)))
    ax.set_xticklabels([rf"$b={b}$" + "\n" + rf"$(n={agg[METHODS[0]][str(b)]['n_runs']})$" for b in BETAS], fontsize=14)
    ax.set_xlim(-0.6, len(BETAS) - 0.4)


def _points_panel(ax, agg: dict, title: str, ylabel: str, floor_band=None, ylim=None) -> None:
    """One point per (b, trajectory): the per-run KL total, per method; black bar = median."""
    rng = np.random.RandomState(3)
    for i, beta in enumerate(BETAS):
        for j, name in enumerate(METHODS):
            vals = np.array(agg[name][str(beta)]["total"]["per_run"])
            xc = i + (j - 0.5) * 0.36
            ax.scatter(xc + rng.uniform(-0.07, 0.07, len(vals)), vals, s=46, alpha=0.75, marker=MARKERS[name],
                       facecolors=COLORS[name], edgecolors="black", linewidths=0.6)
            ax.plot([xc - 0.12, xc + 0.12], [np.median(vals)] * 2, color="black", lw=2.2)
        if "floor" in agg:
            vals = np.array(agg["floor"][str(beta)]["total"]["per_run"])
            ax.scatter(np.full(len(vals), float(i)) + rng.uniform(-0.05, 0.05, len(vals)), vals, s=30,
                       marker="D", facecolors=FLOOR_COLOR, edgecolors="none", alpha=0.8)
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


def _ylim(agg: dict, lo: float, gap=None) -> tuple[float, float]:
    """Log-axis limits: fixed lower edge (below every floor value), upper edge one third of a decade
    above the largest plotted value, so that the extreme VI values at b=7 do not clip anything."""
    top = max(v for name in agg for b in BETAS for t in ("mean", "var", "total")
              for v in [agg[name][str(b)][t]["max"]] if np.isfinite(v))
    if gap is not None:
        top = max(top, max(gap[str(b)]["corrected"]["median"] for b in BETAS if np.isfinite(gap[str(b)]["corrected"]["median"])))
    return lo, 10 ** (np.ceil(3 * np.log10(top)) / 3 + 1 / 3)


def make_figure(agg_w: dict, agg_p: dict, gap: dict, output_path: Path) -> None:
    plt.rcParams["mathtext.fontset"] = "cm"
    pred_floor = PRED_MC_FLOOR_BAND
    ylim_w, ylim_p = _ylim(agg_w, 5e-4, gap=gap), _ylim(agg_p, 1e-4)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    _bar_panel(axes[0, 0], agg_w, r"$\mathrm{(a)\ Weight\ space:\ mean}$-$\mathrm{field\ excess\ KL}$",
               r"$\mathrm{excess\ KL\ (nats\ per\ coordinate)}$", gap=gap, ylim=ylim_w)
    _bar_panel(axes[0, 1], agg_p, r"$\mathrm{(b)\ Predictive\ space:\ KL(HMC\,\|\,method)}$",
               r"$\mathrm{KL\ (nats\ per\ grid\ point)}$", floor_band=pred_floor, ylim=ylim_p)
    _points_panel(axes[1, 0], agg_w, r"$\mathrm{(c)\ Weight\ space,\ per\ trajectory}$",
                  r"$\mathrm{excess\ KL\ (nats\ per\ coordinate)}$", ylim=ylim_w)
    _points_panel(axes[1, 1], agg_p, r"$\mathrm{(d)\ Predictive\ space,\ per\ trajectory}$",
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
        plt.Line2D([], [], marker=MARKERS["Ours"], color=COLORS["Ours"], linestyle="none", markersize=9,
                   markeredgecolor="black", label=r"$\mathrm{Ours,\ one\ trajectory}$"),
        plt.Line2D([], [], marker=MARKERS["VI"], color=COLORS["VI"], linestyle="none", markersize=9,
                   markeredgecolor="black", label=r"$\mathrm{VI,\ one\ trajectory}$"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=13.5, bbox_to_anchor=(0.5, -0.005),
               frameon=True)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(output_path, bbox_inches="tight")
    if PNG_PREVIEW is not None:  # same figure object, for checking the PDF without poppler
        fig.savefig(PNG_PREVIEW, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {output_path}")


PNG_PREVIEW: Path | None = None  # set by `--png <path>`: also write a PNG preview there (not one of the outputs above)


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


def _blabel(b: float) -> str:
    """The b label used for the figure ticks ($b=0.1$, ..., $b=2.0$, $b=7.0$): the float printed as is."""
    return f"{b}"


def _cell(d: dict) -> str:
    """median [min--max] over trajectories."""
    return rf"{_f(d['median'])} [{_f(d['min'])}--{_f(d['max'])}]"


def write_table(agg_w: dict, agg_p: dict, gap: dict, records, path: Path) -> None:
    n_runs = {b: agg_w["Ours"][str(b)]["n_runs"] for b in BETAS}
    n_all = agg_w["Ours"]["pooled"]["n_runs"]
    k_adm = {b: [r["weight"]["K_admitted"] for r in _included(records, b)] for b in BETAS}
    gap_na = [k for k in gap if gap[k]["n_available"] < gap[k]["n_runs"]]
    vi_nonfinite = [f"b={_blabel(r['beta'])} idx {r['idx']}" for r in _included(records)
                    if not (np.isfinite(r["weight"]["VI"]["total"]) and np.isfinite(r["pred"]["VI"]["total"]))]
    lo, hi = (_plain_sci(v) for v in PRED_MC_FLOOR_BAND)
    lo_neff, hi_neff = (_plain_sci(1 / n) for n in (N_EFF_RANGE[1], N_EFF_RANGE[0]))
    pct = f"{100 * TOP_SHARE:.0f}"
    lines = [
        r"% Auto-generated by figures_onepass/advection/posterior_fit_metrics.py",
        r"% Fit of each mean-field posterior (Ours = one-pass EP (expectation propagation), and VI (variational inference))",
        r"% to the multichain HMC (Hamiltonian Monte Carlo) reference. Mean-field = independent coordinates, here a diagonal",
        r"% Gaussian in the shared eigenbasis (the rotated weight coordinates all methods share). Every entry is a KL",
        rf"% (Kullback-Leibler divergence) in nats. The b labels are the ones used for the figure ticks ({', '.join('b=' + _blabel(b) for b in BETAS)}).",
        rf"% Canonical trajectories per wave speed: {', '.join(f'{len(IDXS_BY_BETA[b])} at b={_blabel(b)}' for b in BETAS)} (the same seeded draw at every b).",
        rf"% HMC gate: a run enters only if at least {100 * MIN_ADMITTED_FRACTION:.0f}% of its HMC coordinates converged (split-R-hat < {RHAT_MAX}, bulk ESS > {ESS_MIN:g});",
        rf"% decided from the cached chains when this table was made. Runs entering (n per wave speed): "
        + ", ".join(f"n={n_runs[b]} at b={_blabel(b)}" for b in BETAS) + f" ({n_all} in total);",
        rf"% dropped by the HMC gate: {dropped_description(records)}.",
        rf"% Admitted coordinates (HMC converged there) per run: "
        + ", ".join(f"b={_blabel(b)}: {min(k_adm[b])} to {max(k_adm[b])} of {_included(records, b)[0]['weight']['K']}" for b in BETAS) + ".",
        rf"% The 'pooled' rows pool every run that entered, i.e. {pooled_description(records)}.",
        rf"% VI at b=7.0 fits nothing (its predictive mean collapses: rel-L2 about 1, i.e. the relative L2 error of the predictive",
        rf"% mean, its error divided by the size of the true solution, is about 1), so its entries there are extreme but shown;",
        rf"% VI runs whose KL is not finite (shown as '--'): {', '.join(vi_nonfinite) or 'none'}.",
        rf"% Mean-field gap not available (sample correlation matrix not positive definite): "
        + ("; ".join(f"{k}: {', '.join(gap[k]['not_available'])} ({gap[k]['n_available']} of {gap[k]['n_runs']} runs available)" for k in gap_na) or "none") + ".",
        r"% (A) Weight space. Every entry is in nats per admitted coordinate. mean term = (mu_HMC - mu_q)^2 / (2 s_q^2),",
        r"%     variance term = 0.5 (r - 1 - ln r) with r = s_HMC^2 / s_q^2, both averaged over the admitted",
        r"%     coordinates of a run; sum = D_MF, the KL(HMC marginals || method) in excess of the best mean-field fit.",
        r"%     Cells: median over trajectories [min--max] of the per-run values. The 'sum' column is the median over",
        r"%     trajectories of the per-run totals (mean term + variance term of the same run), so it is NOT the sum of",
        r"%     the two neighbouring median cells (medians do not add). 'HMC half vs half' = the same D_MF between the",
        r"%     moment-matched Gaussian of chains 1,2 (as q; moment-matched = same means and variances as those chains) and",
        r"%     the moments of chains 3,4 (as target): the Monte-Carlo",
        r"%     noise floor. 'mean-field gap' = -0.5 ln det Corr_HMC / K over the admitted coordinates (ln det = the",
        r"%     log-determinant, Corr_HMC = the HMC sample correlation matrix): the KL a PERFECT mean-field fit would still",
        r"%     lose if the HMC posterior were Gaussian; 'raw', and 'corrected' with the Wishart bias correction (a",
        r"%     correction for the downward bias of a sample log-determinant; n = 3000 draws treated as independent;",
        r"%     approximate). The 'HMC half vs half', 'raw' and 'corrected' columns are medians over trajectories only",
        r"%     (no min--max). Pooled rows: 'median' = median over all runs; 'mean' = coordinate-weighted average over",
        r"%     all admitted coordinates of all runs (the KL per coordinate over everything).",
        r"% (B) Weight space, robustness. The per-coordinate contributions are heavy-tailed (a few coordinates carry most",
        r"%     of the sum), so this block reports, per b and pooled over all b: 'median mean term' and 'median var. term'",
        r"%     = the median contribution over ALL admitted coordinates of all included runs at that b (the typical",
        rf"%     coordinate, not the average); 'top-{pct}% share' = the share of the summed contribution (mean term +",
        rf"%     variance term over the same coordinates) carried by the {pct}% largest per-coordinate contributions;",
        r"%     '(dmu/s_HMC)^2/2' = (mu_HMC - mu_q)^2 / (2 s_HMC^2) averaged over the admitted coordinates of a run, then",
        r"%     the median over trajectories: the mean term the method would have if its variances were right, which",
        r"%     separates 'mean off' from 'variance off'. The 'pooled' row of (B) pools all admitted coordinates of all",
        r"%     runs (median terms, top share) and takes the median over all runs for (dmu/s_HMC)^2/2.",
        r"% (C) Predictive space. KL(HMC predictive || method predictive) per grid point, in nats: the same Gaussian KL",
        r"%     and mean/variance split as in (A), but between the stored HMC predictive N(m_H, v_H) and the method's",
        rf"%     N(m_q, v_q) at each (x,t) grid point, averaged over the full 1024 x 201 grid (variances floored at",
        rf"%     {_plain_sci(MIN_VAR)}). Cells: median over trajectories [min--max] of the per-run averages; 'sum' is again the median",
        r"%     of the per-run totals, not the sum of the neighbouring cells. 'median std ratio to HMC' = the median over",
        r"%     grid points of sqrt(v_q / v_H) per run, then the median over trajectories. The HMC predictive's own",
        r"%     Monte-Carlo error puts a floor under these numbers, drawn in the figure as the band 'MC floor (approx.)'",
        rf"%     from {lo} to {hi} nats per point: 1/n_eff with n_eff = {N_EFF_RANGE[0]:g} to {N_EFF_RANGE[1]:g} draws gives {lo_neff} to {hi_neff} for the",
        r"%     mean term alone, but the weight-space 'HMC half vs half' floor shows that the variance term of HMC's own",
        rf"%     noise is about 4 to 5 times larger than 1/n would give, so the band is widened up to {hi}.",
        r"%     Pooled rows as in (A), with grid points in place of coordinates.",
        r"\begin{tabular}{l rrr rrr r rr}",
        r"\toprule",
        r"\multicolumn{10}{l}{\emph{(A) Weight space: mean-field excess KL $D_{\mathrm{MF}}$ [nats per coordinate]}}\\",
        r" & \multicolumn{3}{c}{Ours (one-pass EP)} & \multicolumn{3}{c}{VI} & HMC half & \multicolumn{2}{c}{mean-field gap} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-8}\cmidrule(lr){9-10}",
        r"$b$ & mean term & variance term & sum & mean term & variance term & sum & vs.\ half & raw & corrected \\",
        r"\midrule",
    ]
    for b in BETAS:
        k = str(b)
        cells = [_cell(agg_w["Ours"][k][t]) for t in ("mean", "var", "total")]
        cells += [_cell(agg_w["VI"][k][t]) for t in ("mean", "var", "total")]
        cells += [_f(agg_w["floor"][k]["total"]["median"]), _f(gap[k]["raw"]["median"]), _f(gap[k]["corrected"]["median"])]
        lines.append(f"{_blabel(b)} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    k = "pooled"
    cells = [_f(agg_w[m][k][t]["median"]) for m in METHODS for t in ("mean", "var", "total")]
    cells += [_f(agg_w["floor"][k]["total"]["median"]), _f(gap[k]["raw"]["median"]), _f(gap[k]["corrected"]["median"])]
    lines.append("pooled (median) & " + " & ".join(cells) + r" \\")
    cells = [_f(agg_w[m][k]["elementwise"][t]) for m in METHODS for t in ("mean_term", "var_term", "mean")]
    cells += [_f(agg_w["floor"][k]["elementwise"]["mean"]), "--", "--"]
    lines.append("pooled (mean) & " + " & ".join(cells) + r" \\")
    lines += [
        r"\midrule",
        rf"\multicolumn{{10}}{{l}}{{\emph{{(B) Weight space, robustness: typical coordinate, top-{pct}\% share, mean offset at the HMC scale}}}}\\",
        r" & \multicolumn{4}{c}{Ours (one-pass EP)} & \multicolumn{4}{c}{VI} & \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
        rf"$b$ & median mean term & median var.\ term & top-{pct}\% share & $(\Delta\mu/\sigma_{{\mathrm{{HMC}}}})^2/2$ & "
        rf"median mean term & median var.\ term & top-{pct}\% share & $(\Delta\mu/\sigma_{{\mathrm{{HMC}}}})^2/2$ & \\",
        r"\midrule",
    ]
    for b in BETAS + [None]:
        k = "pooled" if b is None else str(b)
        cells = []
        for m in METHODS:
            e = agg_w[m][k]["elementwise"]
            z2 = np.median([r["weight"][m]["mean_term_at_hmc_scale"] for r in _included(records, b)])
            cells += [_f(e["median_mean_term"]), _f(e["median_var_term"]), f"{e['top_share']:.2f}", _f(z2)]
        lines.append(f"{'pooled' if b is None else _blabel(b)} & " + " & ".join(cells) + r" & \\")
    lines += [
        r"\midrule",
        r"\multicolumn{10}{l}{\emph{(C) Predictive space: KL(HMC predictive $\|$ method predictive) [nats per grid point]}}\\",
        r" & \multicolumn{3}{c}{Ours (one-pass EP)} & \multicolumn{3}{c}{VI} & & \multicolumn{2}{c}{median std ratio to HMC} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){9-10}",
        r"$b$ & mean term & variance term & sum & mean term & variance term & sum & & Ours & VI \\",
        r"\midrule",
    ]
    for b in BETAS:
        k = str(b)
        cells = [_cell(agg_p[m][k][t]) for m in METHODS for t in ("mean", "var", "total")]
        ratios = [np.median([r["pred"][m]["median_std_ratio"] for r in _included(records, b)]) for m in METHODS]
        lines.append(f"{_blabel(b)} & " + " & ".join(cells) + " & & " + " & ".join(f"{v:.2f}" for v in ratios) + r" \\")
    lines.append(r"\midrule")
    k = "pooled"
    cells = [_f(agg_p[m][k][t]["median"]) for m in METHODS for t in ("mean", "var", "total")]
    ratios = [np.median([r["pred"][m]["median_std_ratio"] for r in _included(records)]) for m in METHODS]
    lines.append("pooled (median) & " + " & ".join(cells) + " & & " + " & ".join(f"{v:.2f}" for v in ratios) + r" \\")
    cells = [_f(agg_p[m][k]["elementwise"][t]) for m in METHODS for t in ("mean_term", "var_term", "mean")]
    lines.append("pooled (mean) & " + " & ".join(cells) + r" & & & \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines))
    print(f"Saved table to {path}")


# ----------------------------------------------------------------------------- checks
def sanity_checks(records, agg_w, agg_p) -> dict:
    """Printed and stored: non-negativity, and closed-form consistency of the variance and
    mean terms at the median std ratio / median standardized mean offset."""
    checks = {}
    min_val = min(min(r[space][arr][k][i].min() for space, arr in (("weight", "per_coord"), ("pred", "per_point"))
                      for k in r[space][arr] for i in (0, 1)) for r in _included(records))
    checks["min_contribution"] = float(min_val)
    print(f"\nSanity: smallest per-element KL contribution over everything = {min_val:.3e} (must be >= 0)")
    for b in BETAS:
        rs = _included(records, b)
        if not rs:
            print(f"  b={b:g}: no run admitted by the HMC gate, skipped")
            continue
        for m in METHODS:
            ratio = float(np.median([r["weight"][m]["median_std_ratio"] for r in rs]))
            r2 = 1 / ratio ** 2
            closed = 0.5 * (r2 - 1 - np.log(r2))
            med_var = agg_w[m][str(b)]["elementwise"]["median_var_term"]
            checks[f"b={b:g} {m} var-term"] = dict(median_std_ratio=ratio, closed_form_at_median_ratio=float(closed),
                                                   median_var_term=med_var, mean_var_term=agg_w[m][str(b)]["var"]["median"])
            print(f"  b={b:g} {m}: median s_q/s_HMC = {ratio:.3f} -> closed-form variance term {closed:.2f}; "
                  f"median per-coordinate variance term {med_var:.2f}; per-run AVERAGE {agg_w[m][str(b)]['var']['median']:.2f}")
    return checks


def main() -> None:
    records = collect_records()
    agg_w = {k: aggregate(records, "weight", k) for k in METHODS + ("floor",)}
    agg_p = {k: aggregate(records, "pred", k) for k in METHODS}
    gap = aggregate_gap(records)
    make_figure(agg_w, agg_p, gap, OUTPUT_PDF)
    write_table(agg_w, agg_p, gap, records, OUTPUT_TABLE)
    checks = sanity_checks(records, agg_w, agg_p)

    per_run = [dict(beta=r["beta"], idx=r["idx"], included=r["included"], admitted_fraction=r["admitted_fraction"],
                    accept_rates=r["accept_rates"],
                    weight={k: v for k, v in r["weight"].items() if k != "per_coord"} if r["included"] else None,
                    pred={k: v for k, v in r["pred"].items() if k != "per_point"} if r["included"] else None)
               for r in records]
    summary = dict(
        description="Fit of the one-pass EP posterior (Ours) and the VI posterior to the multichain HMC reference. "
                    "weight: mean-field excess KL per admitted eigen-coordinate (mean term, variance term, total), "
                    "the half-vs-half Monte-Carlo floor and the mean-field gap of the HMC posterior; pred: Gaussian KL "
                    "per grid point between the HMC predictive and the method predictive. Per-b entries are medians "
                    "over trajectories [min, max] of the per-run averages; 'elementwise' pools all coordinates / grid "
                    "points; 'median_*' are medians of the per-element contributions (typical element). "
                    "pred_mc_floor_band: the approximate Monte-Carlo floor band [nats per grid point] drawn in the "
                    "figure (1/n_eff for n_eff in n_eff_range covers the mean term; the upper edge allows for the "
                    "variance term of HMC's own noise, about 4 to 5 times 1/n). idxs_by_beta: the canonical "
                    "trajectories per wave speed (the same ten at every b); n_runs_by_beta and "
                    "dropped_by_hmc_gate: what the HMC gate (min_admitted_fraction) admitted and dropped at run "
                    "time; pooled_runs: the runs the 'pooled' entries pool. A mean-field gap of NaN means the "
                    "sample correlation matrix was not positive definite (gap not available).",
        rhat_max=RHAT_MAX, ess_min=ESS_MIN, min_admitted_fraction=MIN_ADMITTED_FRACTION, min_var=MIN_VAR,
        pred_mc_floor_band=list(PRED_MC_FLOOR_BAND), n_eff_range=list(N_EFF_RANGE),
        idxs_by_beta={str(b): IDXS_BY_BETA[b] for b in BETAS},
        n_runs_by_beta={str(b): len(_included(records, b)) for b in BETAS},
        dropped_by_hmc_gate=[dict(beta=r["beta"], idx=r["idx"], admitted_fraction=r["admitted_fraction"],
                                  accept_rates=r["accept_rates"]) for r in dropped_runs(records)],
        pooled_runs={str(b): [r["idx"] for r in _included(records, b)] for b in BETAS},
        weight=agg_w, weight_mean_field_gap=gap, pred=agg_p, sanity=checks, per_run=per_run,
    )
    OUTPUT_SUMMARY.write_text(json.dumps(summary, indent=2))
    print(f"Saved summary to {OUTPUT_SUMMARY}")

    print("\nWeight space, nats per admitted coordinate (median over trajectories of the per-run average):")
    for b in BETAS + [None]:
        k = "pooled" if b is None else str(b)
        s = " | ".join(f"{m}: mean {agg_w[m][k]['mean']['median']:.3g} + var {agg_w[m][k]['var']['median']:.3g} "
                       f"= {agg_w[m][k]['total']['median']:.3g}" for m in METHODS)
        print(f"  {k:>6} (n={agg_w['Ours'][k]['n_runs']}): {s} | floor {agg_w['floor'][k]['total']['median']:.2e} | "
              f"gap raw {gap[k]['raw']['median']:.3f} corrected {gap[k]['corrected']['median']:.3f} "
              f"({gap[k]['n_available']} of {gap[k]['n_runs']} runs available)")
    print("Predictive space, nats per grid point (median over trajectories of the per-run average):")
    for b in BETAS + [None]:
        k = "pooled" if b is None else str(b)
        s = " | ".join(f"{m}: mean {agg_p[m][k]['mean']['median']:.3g} + var {agg_p[m][k]['var']['median']:.3g} "
                       f"= {agg_p[m][k]['total']['median']:.3g}" for m in METHODS)
        print(f"  {k:>6} (n={agg_p['Ours'][k]['n_runs']}): {s}")
    print(f"Pooled rows pool: {pooled_description(records)}")
    print(f"Dropped by the HMC gate: {dropped_description(records)}")


if __name__ == "__main__":
    if "--png" in sys.argv:
        PNG_PREVIEW = Path(sys.argv[sys.argv.index("--png") + 1])
    main()
