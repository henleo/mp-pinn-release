"""
Loader and driver for the subgraph ablation of the one-pass MP-PINN on advection
(`mp_pinn_subgraph_ablation.jl`): per (b, trajectory), the predictive mean/std on the full grid
of four regimes of the SAME factor graph (same basis, collocation points, prior, schedule and
seeds as every one-pass run):

  prior  untrained (the prior belief)
  ic     trained on the data subgraph only (the labeled initial-condition factors)
  res    trained on the PDE subgraph only (the residual factors)
  full   trained on both = the one-pass posterior of every other figure

Grid: wave speeds BETAS x the ten canonical trajectories `idxs(10)`. The runner is called with
regimes prior,ic,res (`artifacts/subgraph_ablation_beta{b}_idx{i}.h5`); the full regime is read
from the canonical one-pass artifact `mp_pinn_beta{b}_idx{i}.h5`, which the runner's own full
regime reproduces bit for bit (checked on b=0.1 idx 9394 and b=2 idx 898: posterior,
predictive mean/std and KL history identical). `load` checks that both files describe the same problem
(truth, grids, number of kept eigen-directions).

Run `python pde/advection/mp_pinn_subgraph_ablation_sweep.py [--jobs N] [--big-jobs M]` to run
every missing ablation (dispatcher `julia_jobs.run_jobs`, single-threaded BLAS).

No plotting here; the figure is made by figures_onepass/advection/subgraph_ablation_plot.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import mp_pinn_beta_sweep as mp_sweep  # noqa: E402
from julia_jobs import run_jobs  # noqa: E402
from trajectory_selection import idxs  # noqa: E402

REPO_ROOT = SCRIPT_DIR.parent.parent
JULIA_RUNNER = SCRIPT_DIR / "mp_pinn_subgraph_ablation.jl"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

BETAS = [0.1, 0.4, 1.0, 2.0, 7.0]
IDXS = idxs(10)
RUN_REGIMES = ("prior", "ic", "res")     # what the runner computes; "full" is the canonical artifact
REGIMES = RUN_REGIMES + ("full",)
BIG_BETA = 7.0


def cache_path(beta: float, idx: int) -> Path:
    return ARTIFACTS_DIR / f"subgraph_ablation_beta{beta}_idx{idx}.h5"


def load(beta: float, idx: int) -> dict:
    """Mean/std grids (n_x, n_t) of all four regimes, the truth, x, t, epochs per regime, the
    largest |posterior mean| and smallest posterior std over the eigen-coordinates per regime,
    and the number of kept eigen-directions; raises if the ablation and the canonical artifact disagree
    on the problem."""
    path = cache_path(beta, idx)
    out = dict(beta=beta, idx=idx, mean={}, std={}, epochs={}, w_absmean_max={}, w_std_min={})
    with h5py.File(path, "r") as f:
        a = f.attrs
        stored = str(a["regimes"].decode() if isinstance(a["regimes"], bytes) else a["regimes"])
        if float(a["beta"]) != beta or int(a["idx"]) != idx or not set(RUN_REGIMES) <= set(stored.split(",")):
            raise ValueError(f"{path}: stored beta={a['beta']}, idx={a['idx']}, regimes={stored} do not match")
        out.update(x=f["x"][:], t=f["t"][:], true_u=f["true_u"][:].T, n_kept=int(a["K_kept"]))
        for r in RUN_REGIMES:
            out["mean"][r] = f[f"mean_{r}"][:].T
            out["std"][r] = f[f"std_{r}"][:].T
            out["epochs"][r] = int(a[f"epochs_{r}"])
            out["w_absmean_max"][r] = float(abs(f[f"w_eigen_mean_{r}"][:]).max())
            out["w_std_min"][r] = float(f[f"w_eigen_std_{r}"][:].min())
    with h5py.File(mp_sweep._cache_path(beta, idx), "r") as f:
        same = (np.array_equal(f["true_u"][:].T, out["true_u"]) and np.array_equal(f["x"][:], out["x"])
                and np.array_equal(f["t"][:], out["t"]) and f["w_eigen_mean"].shape[0] == out["n_kept"]
                and float(f.attrs["beta_pde"]) == 0.05 and float(f.attrs["beta_ic"]) == 0.02
                and float(f.attrs["sigma_w"]) == 1.0)
        if not same:
            raise ValueError(f"b={beta:g} idx={idx}: the ablation and the one-pass artifact describe different problems")
        out["mean"]["full"] = f["pred_u"][:].T
        out["std"]["full"] = np.sqrt(f["pred_var"][:].T)
        out["epochs"]["full"] = int(f.attrs["epochs"])
        out["w_absmean_max"]["full"] = float(abs(f["w_eigen_mean"][:]).max())
        out["w_std_min"]["full"] = float(f["w_eigen_std"][:].min())
    return out


def missing_runs() -> list[tuple[float, int]]:
    return [(b, i) for b in BETAS for i in IDXS if not cache_path(b, i).exists()]


def _command(run: tuple[float, int], out: Path) -> list[str]:
    beta, idx = run
    return ["julia", f"--project={REPO_ROOT}", str(JULIA_RUNNER), str(beta), str(idx), str(out),
            ",".join(RUN_REGIMES)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--jobs", type=int, default=2, help="concurrent Julia processes (default 2)")
    ap.add_argument("--big-jobs", type=int, default=1, help=f"of which at most this many at b={BIG_BETA:g} (default 1)")
    args = ap.parse_args()
    todo = sorted(missing_runs(), key=lambda r: (r[0] != BIG_BETA, -r[0], IDXS.index(r[1])))
    failed = run_jobs(todo, path_for=lambda r: cache_path(*r), command_for=_command,
                      is_big=lambda r: r[0] == BIG_BETA, label_for=lambda r: f"b={r[0]:g} idx={r[1]}",
                      cwd=REPO_ROOT, jobs=args.jobs, big_jobs=args.big_jobs)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
