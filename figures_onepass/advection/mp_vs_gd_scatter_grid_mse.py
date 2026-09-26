"""
MSE over all data points of panel (a) of mp_vs_gd_scatter_grid.pdf, from the cached artifacts.

Each point of that panel is one (b, trajectory) run of one method: the MSE of its prediction (Ours:
the predictive mean; Baseline: its point prediction) against the true trajectory on the full 1024 x 201 (x,t) grid, mean((true_u - pred_u)^2). The
panel holds 50 "Ours" points (one-pass EP, artifacts mp_pinn_beta{b}_idx{i}.h5) and 50 "Baseline"
points (the controlled gradient-descent baseline, gd_pinn_beta{b}_idx{i}.h5): the five wave speeds
of the other figures, b = 0.1, 0.4, 1, 2, 7, times the ten canonical trajectories idxs(10).

Only the runs in the plot are read. The run list is taken from the figure script itself
(mp_vs_gd_scatter_grid.BETAS and ._idxs_for, the same calls its collect_results makes), so it
follows the figure if the figure's trajectory count changes; the artifact directory is never
globbed for loading. Other cached runs of the two families (e.g. the solution-grid trajectory 6369
of Ours) are listed at the end of the output as NOT used and are never opened.
As in the figure script, a missing artifact is an error.

Checks, each fatal on failure: every artifact's stored attributes 'beta' and 'idx' match the run
it is loaded for; both methods of a run hold the same ground truth true_u; the grid (the x and t
coordinates) is the same for every run. Then the figure is rebuilt from exactly these runs with the figure script's own
make_scatter_grid into a temporary file and compared with mp_vs_gd_scatter_grid.pdf (written by the
figure script, which must run first), CreationDate aside: equal bytes mean the numbers below are the
numbers in the plot (a mismatch is reported, not fatal, because it can also mean that the PDF is out
of date).

Reported per method: the MSE over all its points, i.e. the squared error summed over every grid
point of its runs divided by the number of grid points. Because every run has the same grid,
this equals the mean of the plotted per-run MSE values (both are computed and must agree).
Also reported: the same for both methods pooled, per wave speed, and per wave-speed
group b <= 1 / b > 1, printed x 1e-3 for comparison with the companion table
mp_vs_gd_scatter_grid_table.tex (they agree to its two-digit rounding; not checked in code),
and the per-run values.

Run: MPLBACKEND=Agg python figures_onepass/advection/mp_vs_gd_scatter_grid_mse.py
(it only prints; `python run_experiments.py figures` saves the output as mp_vs_gd_scatter_grid_mse.stdout.log).
"""

from __future__ import annotations

import contextlib
import io
import re
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np

FIGURES_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURES_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "pde" / "advection"))
sys.path.insert(0, str(FIGURES_DIR))

import mp_pinn_beta_sweep as mp_sweep
import gd_pinn_baseline_sweep as gd_sweep
import mp_vs_gd_scatter_grid as figure

METHODS = (("Ours", mp_sweep), ("Baseline", gd_sweep))   # the figure's labels and loaders
GROUPS = ((r"b <= 1", lambda b: b <= 1), (r"b > 1", lambda b: b > 1))   # the companion table's groups


def plotted_runs() -> list[tuple[float, int]]:
    """The (b, trajectory) pairs of the figure, in the figure's order (collect_results)."""
    return [(beta, idx) for beta in figure.BETAS for idx in figure._idxs_for(beta)]


def load(sweep, beta: float, idx: int) -> dict:
    """The figure's loader; also checks the stored run labels."""
    path = sweep._cache_path(beta, idx)
    if not path.exists():
        raise FileNotFoundError(f"missing artifact {path} (b={beta:g}, idx={idx}); the figure uses this run")
    r = sweep.load_cached(beta, idx)
    with h5py.File(path, "r") as f:
        stored_beta, stored_idx = float(f.attrs["beta"]), int(f.attrs["idx"])
    if stored_beta != beta or stored_idx != idx:
        raise ValueError(f"{path.name} stores beta={stored_beta:g}, idx={stored_idx}; expected b={beta:g}, idx={idx}")
    r["path"] = path
    return r


def mse(r: dict) -> float:
    """The plotted value: the formula of make_scatter_grid's scatter_one (metric 'mse')."""
    return float(np.mean((r["true_u"] - r["pred_u"]) ** 2))


def check_figure(results: dict[str, list[dict]]) -> None:
    """Rebuild the scatter figure from exactly these runs and compare with the figure script's PDF."""
    shipped = FIGURES_DIR / "mp_vs_gd_scatter_grid.pdf"
    strip = lambda b: re.sub(rb"/CreationDate \(D:[^)]*\)", b"", b)
    with tempfile.TemporaryDirectory() as tmp:
        rebuilt = Path(tmp) / "mp_vs_gd_scatter_grid.pdf"
        with contextlib.redirect_stdout(io.StringIO()):   # its 'Saved grid to <temp path>' line
            figure.make_scatter_grid(results["Ours"], results["Baseline"], rebuilt)
        same = strip(rebuilt.read_bytes()) == strip(shipped.read_bytes())
    if same:
        print(f"Figure check: the scatter figure rebuilt from exactly these {sum(map(len, results.values()))} runs "
              f"is byte-identical to {shipped.name} (CreationDate aside), so the numbers below are the plotted ones.")
    else:
        print(f"WARNING: the scatter figure rebuilt from these runs differs from {shipped.name}. Either that "
              "PDF is out of date (artifacts or figure script changed since it was made) or the run list differs; "
              "the numbers below are those of the CURRENT artifacts of the figure's run list.")


def unused_artifacts(used_paths: set[Path]) -> list[str]:
    """Names of cached advection runs of both families that exist but are not in the figure (for the record)."""
    names = []
    for prefix in ("mp_pinn_beta", "gd_pinn_beta"):
        for p in sorted(mp_sweep.ARTIFACTS_DIR.glob(f"{prefix}*_idx*.h5")):
            if p not in used_paths:
                names.append(p.name)
    return names


def main() -> None:
    runs = plotted_runs()
    results = {label: [] for label, _ in METHODS}
    for beta, idx in runs:
        for label, sweep in METHODS:
            results[label].append(load(sweep, beta, idx))
        ours, base = results["Ours"][-1], results["Baseline"][-1]
        if ours["true_u"].shape != base["true_u"].shape or not np.array_equal(ours["true_u"], base["true_u"]):
            raise ValueError(f"b={beta:g} idx={idx}: the two artifacts hold different ground truths")
    shapes = {r["true_u"].shape for rs in results.values() for r in rs}
    first = results["Ours"][0]
    if len(shapes) != 1 or not all(np.array_equal(r["x"], first["x"]) and np.array_equal(r["t"], first["t"])
                                   for rs in results.values() for r in rs):
        raise ValueError(f"grids differ between runs (shapes {shapes} or the x/t coordinates)")
    (n_x, n_t), = shapes
    n_grid = n_x * n_t
    print(f"{len(runs)} (b, trajectory) runs per method, as in the figure: b in {[f'{b:g}' for b in figure.BETAS]} "
          f"x trajectories {figure._idxs_for(figure.BETAS[0])}"
          + ("" if all(figure._idxs_for(b) == figure._idxs_for(figure.BETAS[0]) for b in figure.BETAS)
             else " (per-b lists differ, see the per-run block)")
          + f"; grid {n_x} x {n_t} = {n_grid} points per run.")
    check_figure(results)

    per_run = {label: np.array([mse(r) for r in rs]) for label, rs in results.items()}
    betas = np.array([beta for beta, _ in runs])

    def pooled(rs: list[dict]) -> float:
        """Squared error summed over every grid point of the runs / number of grid points."""
        sse = sum(float(np.sum((r["true_u"] - r["pred_u"]) ** 2)) for r in rs)
        return sse / sum(r["true_u"].size for r in rs)

    print(f"\nMSE over all data points of panel (a) (squared error pooled over all grid points of all points' runs; "
          f"equal to the mean of the plotted per-run MSEs because every run has the same grid):")
    for label, rs in results.items():
        pool, mean_of_points = pooled(rs), float(np.mean(per_run[label]))
        if not np.isclose(pool, mean_of_points, rtol=1e-12, atol=0.0):
            raise AssertionError(f"{label}: pooled MSE {pool!r} != mean of per-run MSEs {mean_of_points!r}")
        print(f"  {label:<8} ({len(rs)} points): MSE = {pool:.6e}   (median of the points {np.median(per_run[label]):.4e}, "
              f"min {per_run[label].min():.4e}, max {per_run[label].max():.4e})")
    both = [r for rs in results.values() for r in rs]
    print(f"  {'Both':<8} ({len(both)} points, Ours and Baseline pooled): MSE = {pooled(both):.6e}")
    diff = per_run["Ours"] - per_run["Baseline"]
    print(f"  Ours - Baseline, paired by run: mean {diff.mean():+.4e}; Ours lower on {int((diff < 0).sum())} of {len(diff)} runs.")

    print("\nPer wave-speed group (mean +- population std of the per-run MSEs; x 1e-3 as in mp_vs_gd_scatter_grid_table.tex):")
    for name, in_group in GROUPS:
        mask = np.array([in_group(b) for b in betas])
        cells = [f"{label} {1e3 * per_run[label][mask].mean():.2f} +- {1e3 * per_run[label][mask].std():.2f}"
                 for label, _ in METHODS]
        print(f"  {name:<7} ({int(mask.sum())} runs per method): " + "; ".join(cells))

    print("\nPer wave speed (MSE over the runs at that b = mean of their per-run MSEs):")
    for beta in figure.BETAS:
        mask = betas == beta
        print(f"  b={beta:<4g} ({int(mask.sum())} runs): " + "; ".join(
            f"{label} {per_run[label][mask].mean():.4e}" for label, _ in METHODS))

    print("\nPer run (b, idx: MSE Ours | MSE Baseline):")
    for k, (beta, idx) in enumerate(runs):
        print(f"  b={beta:<4g} idx={idx:<5d}: {per_run['Ours'][k]:.6e} | {per_run['Baseline'][k]:.6e}")

    unused = unused_artifacts({r["path"] for rs in results.values() for r in rs})
    print(f"\nNOT used (cached advection runs of the two families that are not in the figure; never opened): "
          f"{len(unused)} files: " + ", ".join(unused))


if __name__ == "__main__":
    main()
