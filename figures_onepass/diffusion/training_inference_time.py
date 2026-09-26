"""
Training-time AND inference-time comparison of the one-pass EP (expectation propagation, "Ours")
posterior against the multi-chain HMC (Hamiltonian Monte Carlo) reference and the mean-field VI
(variational inference; mean-field = every weight gets its own independent Gaussian) baseline,
over the five canonical trajectories IDXS = idxs(5) of every cell of the PDEBench Fisher-KPP grid
(nu in NUS x rho in RHOS, 16 cells, up to 80 runs; the Fisher-KPP equation is
u_t = nu u_xx + rho u (1 - u), nu the diffusion coefficient, rho the reaction strength). This is
the reaction-diffusion port of figures_onepass/advection/training_inference_time.py, reading the
same three cached artifacts per run as the other figures of this directory; the methods,
the gate, the metrics, the colours and the table format are inherited unchanged, only the axis
changes: the wave speed b of the advection figure becomes the reaction strength rho (log axis, 4
values), and the diffusion coefficient nu is pooled into every rho point (each rho point
aggregates the admitted runs of the four nu values, up to 4 x 5 = 20 runs) and is shown by the
marker shape of the per-run points (nu = 0.5: circle, 1: square, 2: diamond, 5: triangle). One
point per (method, nu, rho, trajectory); lines join the per-rho medians (pooled over nu). The
companion table (training_inference_time_table.tex) lists every (nu, rho) cell explicitly, then
the per-rho rows pooled over nu, then the grand pooled row, each with n, the median times and the
Ours-relative speedups (ratios of the medians of the same row).

Training time is the wall-clock time of the whole fit. For Ours it is the one-pass EP sweep
through the rho-continuation ladder (the model is fitted at a sequence of increasing reaction
strengths, each fit starting from the previous one); for VI the same ladder with 30k iterations
in total; for HMC it is the ladder MAP warm start (MAP = maximum a posteriori, the most probable
weights under the posterior) plus the damped Gauss-Newton polish of that MAP (a second-order
refinement) -- both inside the artifact attribute map_time_seconds -- plus the sampling time of
all chains, run one after another in one process (train_time_seconds = map_time_seconds +
sum(chain_times)). The table also gives the minimal single-chain HMC variant (map_time_seconds +
the mean chain time), for the text.

Inference time is the full posterior-predictive grid (mean AND variance on the complete (x,t)
grid of the artifact, 1024 x 101 for this dataset; the shape is read from the artifacts and
printed). All three methods use their FAIR inference path: message passing and the VI baseline
both read the closed-form predictive off the diagonal eigenbasis posterior (the posterior over the
rotated weight coordinates all methods share) deterministically (`predict_moments_grid` / `predict_vi_stats_analytic`, VI's analytic path) -- the VI
posterior is a diagonal Gaussian over the same coordinates, so sampling it would be an
implementation choice, not a method property. HMC has no closed form (its samples ARE its
posterior), so it evaluates the network across all pooled draws (n_chains x (num_samples - burn)
per artifact, 4 x 750 = 3000 in the default runner configuration; the stored count is printed).
Training/sampling is excluded from every method's inference time alike.

Timing protocol. Panel (a), training: cold-start wall-clock in a fresh Julia process,
so it includes JIT compilation (just-in-time: Julia compiles a code path the first time it runs)
of the method's own code path -- identical protocol for all three methods; training runs once per
problem, so end-to-end time is the relevant cost, and the compile share is largest for the fastest
method (our training speedups are lower bounds). Panel (b), inference: STEADY-STATE (compiled)
times, since inference is the step that is repeated and its cold-start numbers are dominated by
compilation, from the warm re-timing CSV `training_inference_time_steady_state.csv`
(produced by pde/diffusion/diagnostics/steady_state_timing/warm_retime_reacdiff.jl on an idle
machine): the same timed block re-timed from the saved posteriors/chains, minimum over 7 repeated
calls for the closed-form methods (measurement order of Ours/VI alternated per trajectory) and
over 3 for HMC, after one untimed warm-up call; every re-timed path reproduces the artifact's
saved pred_var to machine precision. The CSV is required and must cover every admitted run (an
error otherwise). The companion table holds both the cold-start times (block 1, with speedups in
block 2) and the steady-state inference times (block 3, panel (b)). Sub-second single-shot timings are noise-dominated
(up to +-50 % for an identical function), which is why the min-of-k protocol is used.

A run enters only when its HMC reference converged (the gate of the whole figure set):
at least HMC_MIN_ADMITTED of the eigenbasis coordinates (the rotated weight coordinates all
methods share) pass split-R-hat < 1.05 and bulk ESS > 100 (split-R-hat compares the spread
between the chain halves with the spread within them, 1 means they agree; bulk ESS is the
effective sample size, the number of independent draws the correlated chains are worth). A run
that fails the gate is dropped for EVERY method, so this figure shares its run set with the other
figures; which runs are dropped is computed from the cached HMC diagnostics at run time and is
printed in the log and in the table comment. Because a wall-clock time does not depend on whether
the chains converged, the table comment and the log also give the per-rho medians with the
dropped runs put back. Reads cached artifacts only; never trains: a missing artifact raises
FileNotFoundError naming the file.

Cell override: the environment variable REACDIFF_CELLS, a comma-separated list of "nu:rho" pairs
(for example REACDIFF_CELLS="0.5:1.0,0.5:2.0"), restricts the run to those cells (the outputs are
then marked PARTIAL); the default (unset or empty) is all 16 cells and never skips a cell. In
override mode the log, the table comment and a title line on the figure say which cells were used,
and the other cells appear in the table as "not run".
Within a cell every one of the five canonical trajectories must exist in both modes.

Run: python figures_onepass/diffusion/training_inference_time.py [--png out.png]
The console output is also written to training_inference_time.log next to the outputs (only when the run
succeeds; a failed run leaves the previous log untouched and writes training_inference_time.failed.log).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

FIGURES_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURES_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "pde" / "diffusion"))
sys.path.insert(0, str(REPO_ROOT / "pde" / "advection"))

import reacdiff_baselines_sweep as sweep  # noqa: E402  (load_cached / cache_path / assert_same_basis)
from trajectory_selection import idxs  # noqa: E402

OUTPUT_PDF = FIGURES_DIR / "training_inference_time.pdf"
OUTPUT_TEX = FIGURES_DIR / "training_inference_time_table.tex"
OUTPUT_LOG = FIGURES_DIR / "training_inference_time.log"
STEADY_CSV = FIGURES_DIR / "training_inference_time_steady_state.csv"
STEADY_FAMILY = {"onepass": "Ours", "hmc": "HMC", "vi": "VI"}  # CSV `family` column -> method name

NUS = list(sweep.NUS)  # diffusion coefficients of the dataset grid (0.5, 1, 2, 5): pooled, shown by marker
RHOS = list(sweep.RHOS)  # reaction strengths of the dataset grid (1, 2, 5, 10): the x-axis
ALL_CELLS = [(nu, rho) for nu in NUS for rho in RHOS]
IDXS = idxs(5)  # the five canonical trajectories of every cell (the same seeded draw as the advection figures)
CELLS_ENV = "REACDIFF_CELLS"
METHODS = ("Ours", "HMC", "VI")
KEYS = {"train": "train_time_seconds", "predict": "predict_time_seconds"}  # panel key -> artifact attribute
RHAT_MAX, ESS_MIN = 1.05, 100.0  # per-coordinate HMC convergence gate (as in the advection set)
HMC_MIN_ADMITTED = 0.9  # a run is used only if at least this fraction of its HMC coordinates pass the gate
COLORS = {"Ours": "#4a3aa7", "HMC": "#eb6834", "VI": "#008300"}
MARKERS = {"Ours": "o", "HMC": "s", "VI": "^"}  # markers of the per-rho median lines (as in the advection figure)
NU_MARKERS = {0.5: "o", 1.0: "s", 2.0: "D", 5.0: "^"}  # markers of the per-run points
# The per-run points of one rho are spread sideways by nu (a fixed factor per nu on the log axis, in
# place of the random jitter of the advection figure), so the four nu values do not sit on top of each other.
NU_OFFSETS = {nu: 1.06 ** (k - (len(NUS) - 1) / 2) for k, nu in enumerate(NUS)}


# ----------------------------------------------------------------------------- log file


class _Tee:
    """Write everything printed to stdout to the log file as well (the log ships with the figure).
    The text goes to a temporary file first and replaces the log only when the run succeeded, so a
    failed run (missing artifact, bad REACDIFF_CELLS) cannot leave an empty or foreign log next to
    the pdf/table of the previous successful run; the failed run's text (with the error) is kept
    as <name>.failed.log instead."""

    def __init__(self, path: Path):
        self._path = path
        self._tmp = path.with_suffix(".log.tmp")
        self._stdout = sys.stdout
        self._file = open(self._tmp, "w")

    def write(self, s: str) -> None:
        self._stdout.write(s)
        self._file.write(s)

    def flush(self) -> None:
        self._stdout.flush()
        self._file.flush()

    def close(self, error: BaseException | None = None) -> None:
        sys.stdout = self._stdout
        if error is not None:
            self._file.write(f"\nRUN FAILED: {type(error).__name__}: {error}\n")
        self._file.close()
        if error is None:
            os.replace(self._tmp, self._path)
        else:
            failed = self._path.with_suffix(".failed.log")
            os.replace(self._tmp, failed)
            self._stdout.write(f"Run failed; its log is {failed} (the previous {self._path.name} is untouched)\n")


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


def _load(method: str, nu: float, rho: float, idx: int) -> dict:
    path = sweep.cache_path(method, nu, rho, idx)
    if not path.exists():
        raise FileNotFoundError(f"missing {method} artifact {path} ({cell_label(nu, rho)} idx={idx}); this script "
                                f"never trains, generate it with {sweep.RUNNERS[method].name}")
    return sweep.load_cached(method, nu, rho, idx)


def collect_records(cells: list[tuple[float, float]]) -> tuple[list[dict], list[dict]]:
    """One record per (nu, rho, idx) holding the cold-start train / inference times of the three
    methods (seconds, from the artifact attributes) and the minimal single-chain HMC training
    time, split into (admitted, excluded) by the HMC gate. A run whose HMC reference did not
    converge is excluded for every method, so this figure shares its run set with the other
    figures; the excluded records feed only the 'with the dropped runs put back' lines."""
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
                assert other["pred_u"].shape == ours["pred_u"].shape, "methods must share the predictive grid"
            admitted = float(np.mean((hmc["rhat"] < RHAT_MAX) & (hmc["ess"] > ESS_MIN)))
            rec = dict(
                nu=nu, rho=rho, idx=idx, hmc_admitted=admitted,
                hmc_accept=np.round(hmc["accept_rates"], 3).tolist(),
                grid_shape=tuple(ours["pred_u"].shape),
                hmc_draws=hmc["n_chains"] * (hmc["num_samples"] - hmc["burn"]),  # pooled draws HMC's inference averages over
                hmc_singlechain=hmc["map_time_seconds"] + float(np.mean(hmc["chain_times"])),
                **{m: {key: float(r[attr]) for key, attr in KEYS.items()} for m, r in (("Ours", ours), ("HMC", hmc), ("VI", vi))},
            )
            # Consistency of the HMC bookkeeping stated in the docstring (train = MAP + polish + all chains).
            assert abs(hmc["map_time_seconds"] + float(np.sum(hmc["chain_times"])) - rec["HMC"]["train"]) < 1e-6 * rec["HMC"]["train"], \
                "HMC train_time_seconds != map_time_seconds + sum(chain_times)"
            if admitted < HMC_MIN_ADMITTED:
                print(f"[{cell_label(nu, rho)} idx={idx}] EXCLUDED from every method: only {admitted:.0%} of the HMC "
                      f"coordinates converged (chain acceptance rates {rec['hmc_accept']}), so there is no usable "
                      "reference. For the record, its cold-start times (train / infer, s): "
                      + ", ".join(f"{m} {rec[m]['train']:.1f} / {rec[m]['predict']:.2f}" for m in METHODS))
                excluded.append(rec)
                continue
            records.append(rec)
    return records, excluded


def _select(records: list[dict], rho: float | None = None, nu: float | None = None) -> list[dict]:
    """Runs of one cell (nu, rho), of one rho pooled over nu (nu=None), or of everything."""
    return [r for r in records if (rho is None or r["rho"] == rho) and (nu is None or r["nu"] == nu)]


def _times(records: list[dict], method: str, key: str, rho: float | None = None, nu: float | None = None) -> np.ndarray:
    return np.array([r[method][key] for r in _select(records, rho, nu)])


def _med(vals) -> float:
    return float(np.median(vals)) if len(vals) else float("nan")


def dropped_runs_text(excluded: list[dict]) -> str:
    """One line naming the runs the HMC gate dropped (with what the gate saw), or 'none'."""
    if not excluded:
        return "none"
    return "; ".join(f"{cell_label(r['nu'], r['rho'])} idx {r['idx']} ({r['hmc_admitted']:.0%} of the HMC coordinates "
                     f"admitted, chain acceptance {r['hmc_accept']})" for r in excluded)


def n_per_rho_text(records: list[dict], cells: list[tuple[float, float]]) -> str:
    """'n = 20 of 20 at rho=1, ...': admitted runs out of the canonical trajectories of the cells run."""
    return "n = " + ", ".join(f"{len(_select(records, rho))} of {len(IDXS) * sum(1 for _, r in cells if r == rho)} "
                              f"at rho={rho:g}" for rho in RHOS if any(r == rho for _, r in cells))


# ----------------------------------------------------------------------------- steady state


def load_steady_state(records: list[dict]) -> list[dict] | None:
    """Steady-state (compiled) inference times from the warm re-timing CSV, as records
    {nu, rho, idx, Ours, HMC, VI} (seconds), restricted to the admitted runs of this figure and to
    the (nu, rho, idx) that all three families were re-timed on (strict parity with the cold
    block, which uses one run set for all three methods). None when the CSV does not exist."""
    if not STEADY_CSV.exists():
        return None
    admitted = {(r["nu"], r["rho"], r["idx"]) for r in records}
    seconds: dict[tuple, dict] = {}
    n_rows = 0
    with STEADY_CSV.open() as fh:
        for row in csv.DictReader(fh):
            n_rows += 1
            method = STEADY_FAMILY.get(row["family"])
            key = (float(row["nu"]), float(row["rho"]), int(row["idx"]))
            if method is None or key not in admitted:
                continue
            seconds.setdefault(key, {})[method] = float(row["warm_predict_seconds"])
    out = [dict(nu=nu, rho=rho, idx=idx, **s) for (nu, rho, idx), s in sorted(seconds.items())
           if all(m in s for m in METHODS)]
    print(f"Steady-state CSV {STEADY_CSV.name}: {n_rows} rows; {len(out)} admitted runs re-timed for all three methods "
          f"(of {len(records)} admitted runs)")
    return out


def _steady(steady: list[dict], method: str, rho: float | None = None, nu: float | None = None) -> np.ndarray:
    return np.array([s[method] for s in _select(steady, rho, nu)])


# ----------------------------------------------------------------------------- figure


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


def plot_panel(ax, records: list[dict], key: str, title: str, name: str) -> None:
    """One per-run point per (method, nu, rho, trajectory), the nu marker on a fixed sideways
    offset per nu, and one line per method through its per-rho medians pooled over nu."""
    rhos = _plotted_rhos(records)
    for method in METHODS:
        for nu in NUS:
            runs = _select(records, nu=nu)
            if runs:
                ax.scatter([r["rho"] * NU_OFFSETS[nu] for r in runs], [r[method][key] for r in runs], s=34, alpha=0.45,
                           color=COLORS[method], marker=NU_MARKERS[nu], edgecolors="none")
        med = [_med(_times(records, method, key, rho)) for rho in rhos]
        ax.plot(rhos, med, marker=MARKERS[method], color=COLORS[method], markersize=11, linewidth=2.2,
                markeredgecolor="white", markeredgewidth=0.8, label=method)
        print(f"  {name}: {method} per-rho medians (pooled over nu) at rho={rhos}: {np.round(med, 3).tolist()}")
    _rho_axis(ax)
    ax.set_xlabel(r"$\rho$", fontsize=20)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))  # labelled at 1, 2, 5 x 10^k
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.set_ylabel(r"$\mathrm{time\ (s)}$", fontsize=20)
    ax.set_title(title, fontsize=22)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.tick_params(axis="both", labelsize=14)


def make_figure(records: list[dict], steady: list[dict], output_path: Path, png_path: Path | None = None,
                cells_note: str | None = None) -> None:
    """Panel (a): cold-start training times; panel (b): steady-state inference times, which must
    exist for every admitted run (attached to the records as the key 'predict_steady')."""
    by_run = {(s["nu"], s["rho"], s["idx"]): s for s in steady}
    missing = [(r["nu"], r["rho"], r["idx"]) for r in records if (r["nu"], r["rho"], r["idx"]) not in by_run]
    if missing:
        raise ValueError(f"{STEADY_CSV.name} lacks steady-state times of {len(missing)} admitted runs: {missing}")
    for r in records:
        for m in METHODS:
            r[m]["predict_steady"] = by_run[(r["nu"], r["rho"], r["idx"])][m]
    plt.rcParams["mathtext.fontset"] = "cm"
    fig, (ax_train, ax_pred) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    plot_panel(ax_train, records, "train", r"$\mathrm{(a)\ Training\ time}$", "(a) training time (cold start)")
    plot_panel(ax_pred, records, "predict_steady", r"$\mathrm{(b)\ Inference\ time}$", "(b) inference time (steady state)")
    handles = [Line2D([], [], color=COLORS[m], marker=MARKERS[m], markersize=11, lw=2.2, label=m) for m in METHODS]
    handles += [Line2D([], [], color="0.35", marker=NU_MARKERS[nu], markersize=8, ls="none", label=rf"$\nu={nu:g}$")
                for nu in NUS]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=15, bbox_to_anchor=(0.5, -0.06),
               columnspacing=1.4, handletextpad=0.5)
    fig.tight_layout()
    if cells_note:  # override mode: say on the figure itself that it is partial
        fig.suptitle(cells_note, fontsize=16, color="#b00020", y=1.02)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved figure to {output_path}")
    if png_path is not None:
        fig.savefig(png_path, bbox_inches="tight", dpi=80)
        print(f"Saved PNG preview to {png_path}")
    plt.close(fig)


# ----------------------------------------------------------------------------- table and log


def table_rows() -> list[tuple[float | None, float | None]]:
    """(nu, rho) of every table row: all 16 cells, the 4 per-rho rows pooled over nu, the grand pooled row."""
    return ALL_CELLS + [(None, rho) for rho in RHOS] + [(None, None)]


def _row_labels(nu: float | None, rho: float | None) -> str:
    """'nu & rho' cells of a table row: a number, or 'all' = pooled over that variable."""
    return f"{'all' if nu is None else f'{nu:g}'} & {'all' if rho is None else f'{rho:g}'}"


def _row_medians(records: list[dict], rho: float | None, nu: float | None) -> dict:
    """{'train': (Ours, HMC, VI), 'predict': (...), 'singlechain': x} medians over the runs of the row."""
    out = {key: tuple(_med(_times(records, m, key, rho, nu)) for m in METHODS) for key in KEYS}
    out["singlechain"] = _med([r["hmc_singlechain"] for r in _select(records, rho, nu)])
    return out


def _speedups(med: dict) -> dict:
    """{'train': (HMC/Ours, VI/Ours), 'predict': (...)} from the medians of one row."""
    return {key: (med[key][1] / med[key][0], med[key][2] / med[key][0]) for key in KEYS}


def _put_back_lines(records: list[dict], excluded: list[dict]) -> list[str]:
    """Table-comment lines with the per-rho medians recomputed with the gate-dropped runs put back
    (a wall-clock time does not depend on whether the chains converged); empty when none was dropped."""
    if not excluded:
        return []
    both = records + excluded
    lines = [r"% With the dropped runs put back (per rho, pooled over nu; median train s Ours/HMC/VI, infer s Ours/HMC/VI, n):"]
    for rho in RHOS:
        if not _select(both, rho):
            continue
        med = _row_medians(both, rho, None)
        lines.append(f"%   rho={rho:g}: train {med['train'][0]:.1f}/{med['train'][1]:.1f}/{med['train'][2]:.1f}, "
                     f"infer {med['predict'][0]:.2f}/{med['predict'][1]:.2f}/{med['predict'][2]:.2f}, "
                     f"n={len(_select(both, rho))} (admitted only: n={len(_select(records, rho))})")
    return lines


def write_table(records: list[dict], excluded: list[dict], cells: list[tuple[float, float]], override: bool,
                steady: list[dict] | None, path: Path) -> None:
    """LaTeX companion table, three blocks, every block with a row per (nu, rho) cell, per rho
    pooled over nu and pooled over everything. Block 1: median cold-start training and inference
    times (s) of the three methods (panel (a) plots the training times) plus the single-chain HMC
    variant.
    Block 2: Ours-relative speedups (ratios of the medians of block 1). Block 3: steady-state
    (compiled) inference times and their speedups when the warm re-timing CSV exists; otherwise a
    statement that the re-timing was not run. A cell not run (override mode only) shows 'not run'."""
    n_x, n_t = records[0]["grid_shape"]
    draws = "/".join(str(d) for d in sorted({r["hmc_draws"] for r in records}))
    cells_text = (f"OVERRIDE MODE ({CELLS_ENV}): only the cells {', '.join(cell_label(*c) for c in cells)} were run; "
                  "the other cells are marked 'not run'. This is NOT the final table." if override
                  else "All 16 cells of the grid were run.")
    lines = [
        r"% Auto-generated by figures_onepass/diffusion/training_inference_time.py",
        *[f"% {line}" for line in textwrap.wrap(
            "Ours = one-pass EP (expectation propagation); HMC = multi-chain Hamiltonian Monte Carlo reference of the "
            "same model; VI = mean-field (every weight gets its own independent Gaussian) variational inference. "
            f"Dataset: PDEBench 1D Fisher-KPP, u_t = nu u_xx + rho u (1 - u), 16 cells nu in {NUS} x rho in {RHOS}; "
            "the model uses rho_norm = 2 rho and nu_norm = 2 nu. Per cell: the five canonical trajectories "
            f"idxs(5) = {IDXS} whose HMC reference converged ({n_per_rho_text(records, cells)}; {len(records)} pooled). "
            f"{cells_text} A run enters only if at least {HMC_MIN_ADMITTED:.0%} of its HMC coordinates have "
            f"split-R-hat < {RHAT_MAX} (split-R-hat compares the spread between the chain halves with the spread "
            f"within them, 1 = agreement) and bulk ESS > {ESS_MIN:.0f} (bulk ESS = effective sample size, the number "
            "of independent draws the correlated chains are worth); a run that fails is dropped for EVERY method. "
            f"Dropped by the HMC gate: {dropped_runs_text(excluded)}. Rows: every (nu, rho) cell (median over its "
            "n runs), then 'nu = all' rows = pooled over the four nu values at that rho (what the figure's median "
            "lines show, up to 4 x 5 = 20 runs), then 'all, all' = every admitted run of every cell. Every time is "
            "cold-start wall-clock in a fresh Julia process, JIT compilation (just-in-time: Julia compiles a code path the "
            "first time it runs) of the method's own code path included "
            "(the compile share is largest for the fastest method, so the speedups are lower bounds). "
            "Training: Ours = one-pass EP through the rho-continuation ladder (a sequence of fits at increasing reaction "
            "strength, each starting from the previous one); VI = the same ladder, 30k iterations "
            "in total; HMC = ladder MAP (maximum a posteriori) warm start + damped Gauss-Newton polish of the MAP (a "
            "second-order refinement) "
            "(both in the artifact attribute map_time_seconds) + the sampling time of all chains run one after "
            "another (train_time_seconds = map_time_seconds + sum(chain_times)); 'HMC 1 chain' = map_time_seconds + "
            f"the mean chain time. Inference: mean AND variance on the full {n_x} x {n_t} (x,t) grid; Ours and VI "
            "read the closed-form predictive off the diagonal eigenbasis posterior (the posterior over the rotated weight "
            "coordinates all methods share) deterministically, HMC averages "
            f"the network over all its pooled draws (n_chains x (num_samples - burn) = {draws} per run here).", 100)],
        *_put_back_lines(records, excluded),
        r"% Figure: panel (a) = the cold-start training times of block 1; panel (b) = the STEADY-STATE inference",
        r"% times of block 3; the cold-start inference times of block 1 are kept for completeness (JIT included).",
        r"% BLOCK 1: median cold-start times (s).",
        r"\begin{tabular}{r r r rrr r rrr}",
        r"\toprule",
        r" & & & \multicolumn{3}{c}{Training time (s)} & HMC 1 chain & \multicolumn{3}{c}{Inference time (s)} \\",
        r"\cmidrule(lr){4-6}\cmidrule(lr){7-7}\cmidrule(lr){8-10}",
        r"$\nu$ & $\rho$ & $n$ & Ours & HMC & VI & train (s) & Ours & HMC & VI \\",
        r"\midrule",
    ]
    not_run = " & ".join(["not run"] * 7)
    for nu, rho in table_rows():
        n = len(_select(records, rho, nu))
        if n == 0:
            lines.append(f"{_row_labels(nu, rho)} & 0 & {not_run} \\\\")
            continue
        med = _row_medians(records, rho, nu)
        (ot, ht, vt), (op, hp, vp) = med["train"], med["predict"]
        lines.append(f"{_row_labels(nu, rho)} & {n} & {ot:.1f} & {ht:.1f} & {vt:.1f} & {med['singlechain']:.1f} & "
                     f"{op:.2f} & {hp:.2f} & {vp:.2f} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\medskip",
        "",
        r"% BLOCK 2: median speedup of Ours = ratio of the medians of the same row of block 1 (HMC/Ours, VI/Ours).",
        r"\begin{tabular}{r r r rr rr}",
        r"\toprule",
        r" & & & \multicolumn{2}{c}{Training speedup of Ours} & \multicolumn{2}{c}{Inference speedup of Ours} \\",
        r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
        r"$\nu$ & $\rho$ & $n$ & vs HMC & vs VI & vs HMC & vs VI \\",
        r"\midrule",
    ]
    not_run = " & ".join(["not run"] * 4)
    for nu, rho in table_rows():
        n = len(_select(records, rho, nu))
        if n == 0:
            lines.append(f"{_row_labels(nu, rho)} & 0 & {not_run} \\\\")
            continue
        sp = _speedups(_row_medians(records, rho, nu))
        lines.append(f"{_row_labels(nu, rho)} & {n} & {sp['train'][0]:.1f}$\\times$ & {sp['train'][1]:.1f}$\\times$ & "
                     f"{sp['predict'][0]:.1f}$\\times$ & {sp['predict'][1]:.1f}$\\times$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", "", r"\medskip", ""]
    # BLOCK 3: steady-state (compiled) inference, same timed block re-timed after a warm-up call.
    if steady is None:
        lines += [
            f"% BLOCK 3 omitted: {STEADY_CSV.name} not found, the steady-state (compiled) re-timing was not run for",
            r"% this dataset (pde/diffusion/diagnostics/steady_state_timing/warm_retime_reacdiff.jl). Not fabricated.",
            r"\noindent\emph{Steady-state (compiled) inference times: the warm re-timing was not run for this dataset"
            r" (no \texttt{training\_inference\_time\_steady\_state.csv}); the inference times above are cold-start,"
            r" JIT compilation included.}",
            "",
        ]
    else:
        lines += [
            r"% BLOCK 3: steady-state (compiled) inference time (s): the same timed block re-timed from the saved",
            r"% posterior/chains on an idle machine, min over 7 repeated calls (Ours, VI; measurement order alternated)",
            r"% / 3 (HMC) after a warm-up call (" + STEADY_CSV.name + r"; cold-start above includes JIT).",
            r"% Only the admitted runs re-timed for all three methods enter (n per row); speedup = ratio of medians.",
            r"\begin{tabular}{r r r rrr rr}",
            r"\toprule",
            r" & & & \multicolumn{3}{c}{Steady-state inference time (s)} & \multicolumn{2}{c}{Speedup of Ours} \\",
            r"\cmidrule(lr){4-6}\cmidrule(lr){7-8}",
            r"$\nu$ & $\rho$ & $n$ & Ours & HMC & VI & vs HMC & vs VI \\",
            r"\midrule",
        ]
        not_run = " & ".join(["not run"] * 5)
        for nu, rho in table_rows():
            n = len(_select(steady, rho, nu))
            if n == 0:
                lines.append(f"{_row_labels(nu, rho)} & 0 & {not_run} \\\\")
                continue
            so, sh, sv = (_med(_steady(steady, m, rho, nu)) for m in METHODS)
            lines.append(f"{_row_labels(nu, rho)} & {n} & {so:.2f} & {sh:.2f} & {sv:.2f} & "
                         f"{sh / so:.1f}$\\times$ & {sv / so:.1f}$\\times$ \\\\")
        lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines))
    print(f"Saved table to {path}")


def print_summary(records: list[dict], excluded: list[dict], cells: list[tuple[float, float]],
                  steady: list[dict] | None) -> None:
    n_x, n_t = records[0]["grid_shape"]
    print(f"\n{len(records)} runs used ({n_per_rho_text(records, cells)}); predictive grid {n_x} x {n_t} (x,t) per run; "
          f"HMC pooled draws per run: {'/'.join(str(d) for d in sorted({r['hmc_draws'] for r in records}))}")
    print(f"Dropped by the HMC gate (>= {HMC_MIN_ADMITTED:.0%} of the coordinates with split-R-hat < {RHAT_MAX} and "
          f"bulk ESS > {ESS_MIN:.0f}): {dropped_runs_text(excluded)}")
    print("HMC gate per run (fraction of coordinates admitted, chain acceptance rates):")
    for r in sorted(records + excluded, key=lambda r: (r["nu"], r["rho"], IDXS.index(r["idx"]))):
        flag = "" if r in records else "  <- DROPPED"
        print(f"  {cell_label(r['nu'], r['rho']):<18} idx {r['idx']:>5}: {r['hmc_admitted']:6.1%}  {r['hmc_accept']}{flag}")

    print("\nMedian cold-start times per cell, per rho (nu = all) and pooled "
          "(train s Ours/HMC/VI, HMC 1 chain, infer s Ours/HMC/VI):")
    print(f"  {'nu':>4} {'rho':>4} {'n':>3}  {'train Ours':>10} {'HMC':>8} {'VI':>8}  {'HMC1ch':>8}  {'inf Ours':>8} {'HMC':>8} {'VI':>8}")
    for nu, rho in table_rows():
        n = len(_select(records, rho, nu))
        if n == 0:
            continue
        med = _row_medians(records, rho, nu)
        (ot, ht, vt), (op, hp, vp) = med["train"], med["predict"]
        print(f"  {'all' if nu is None else f'{nu:g}':>4} {'all' if rho is None else f'{rho:g}':>4} {n:>3}  "
              f"{ot:>10.1f} {ht:>8.1f} {vt:>8.1f}  {med['singlechain']:>8.1f}  {op:>8.2f} {hp:>8.2f} {vp:>8.2f}")

    rhos = _plotted_rhos(records)
    for key, name in (("train", "train"), ("predict", "inference")):
        print(f"\nMedian {name} time speedups of Ours, per rho pooled over nu (n per rho = "
              f"{ {rho: len(_select(records, rho)) for rho in rhos} }):")
        for rho in rhos:
            sp = _speedups(_row_medians(records, rho, None))[key]
            print(f"  rho={rho:g}: {sp[0]:.1f}x vs HMC, {sp[1]:.1f}x vs VI")
    if excluded:
        print("\nWith the dropped runs put back (a wall-clock time does not depend on convergence), per rho pooled over nu:")
        for line in _put_back_lines(records, excluded)[1:]:
            print("  " + line.lstrip("% "))

    print("\nMedian HMC single-chain (MAP + polish + 1 average chain) train time per rho (pooled over nu):")
    for rho in rhos:
        print(f"  rho={rho:g}: {_med([r['hmc_singlechain'] for r in _select(records, rho)]):.1f}s "
              f"(vs Ours {_med(_times(records, 'Ours', 'train', rho)):.1f}s)")

    if steady is None:
        print(f"\n(no {STEADY_CSV.name}: the steady-state (compiled) re-timing was not run; block 3 of the table says so)")
    else:
        print("\nMedian STEADY-STATE (compiled) inference time [s] Ours / HMC / VI, per rho (pooled over nu), n:")
        for rho in rhos:
            so, sh, sv = (_med(_steady(steady, m, rho)) for m in METHODS)
            print(f"  rho={rho:g}: {so:.2f} / {sh:.2f} / {sv:.2f}, n={len(_select(steady, rho))}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--png", type=Path, default=None, help="also render the figure to this PNG (for inspection)")
    args = parser.parse_args()
    tee = _Tee(OUTPUT_LOG)
    sys.stdout = tee
    error = None
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
        note = (f"PARTIAL ({CELLS_ENV}): cells {', '.join(cell_label(*c) for c in cells)} only" if override else None)
        steady = load_steady_state(records)
        if steady is None:
            raise FileNotFoundError(f"{STEADY_CSV} is required: panel (b) plots steady-state inference times")
        make_figure(records, steady, OUTPUT_PDF, args.png, note)
        write_table(records, excluded, cells, override, steady, OUTPUT_TEX)
        print_summary(records, excluded, cells, steady)
        print(f"Log written to {OUTPUT_LOG}")
    except BaseException as e:
        error = e
        raise
    finally:
        tee.close(error)


if __name__ == "__main__":
    main()
