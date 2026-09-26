"""
Loader and driver for the PDE-tolerance sweep of the one-pass MP-PINN on advection: the same
runner as every one-pass artifact (`mp_pinn_run_beta.jl`), with only the residual noise scale
epsilon (the runner's `beta_pde`, the paper's epsilon_C in R ~ N(0, epsilon_C^2)) changed.

Grid: wave speeds BETAS x the ten canonical trajectories `idxs(10)` (the set every
figures_onepass advection figure uses) x EPSILONS. EPSILONS is a 1-2-5 series over three decades
that contains the default epsilon = 0.05 of every other run. That point is NOT re-run: it is the
canonical one-pass artifact `mp_pinn_beta{b}_idx{i}.h5` itself (same runner, same arguments; a
run with an explicit 0.05 reproduces it bit for bit, checked on b=0.1 idx 9394). Every other
point is cached as `artifacts/mp_pinn_epsilon_beta{b}_eps{eps}_idx{i}.h5`, written by the
same runner.

Run `python pde/advection/mp_pinn_epsilon_sweep.py [--jobs N] [--big-jobs M]` to train every
missing run: at most N Julia processes at once, at most M of them at b = 7 (1.4-1.7 GiB peak and
~10 min each; the limits can be changed during the run through
`artifacts/mp_pinn_epsilon_sweep.limits.json`), dispatched by `julia_jobs.run_jobs` (single-threaded BLAS as for every sweep,
since the eigenvector signs of the basis can depend on BLAS threading; each run writes to a
temporary file that is renamed only on success, its console output goes to a `.runlog` next to
the artifact).

No plotting here; the figure is made by figures_onepass/advection/mp_pinn_epsilon_dependence.py.
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
JULIA_RUNNER = SCRIPT_DIR / "mp_pinn_run_beta.jl"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"

BETAS = [0.1, 0.4, 1.0, 2.0, 7.0]
EPSILONS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
DEFAULT_EPS = 0.05          # the runner's default beta_pde, used by every other one-pass run
SIGMA_W = 1.0               # runner defaults, passed explicitly because beta_pde is the 6th argument
BETA_IC = 0.02
IDXS = idxs(10)
BIG_BETA = 7.0              # the memory-heavy wave speed (separate concurrency limit)


def cache_path(beta: float, eps: float, idx: int) -> Path:
    if eps == DEFAULT_EPS:
        return mp_sweep._cache_path(beta, idx)
    return ARTIFACTS_DIR / f"mp_pinn_epsilon_beta{beta}_eps{eps}_idx{idx}.h5"


def load_cached(beta: float, eps: float, idx: int) -> dict:
    """The fields the epsilon figure needs, in the (n_x, n_t) convention of
    `mp_pinn_beta_sweep.load_cached`; the stored beta, idx and beta_pde are checked against the
    requested run."""
    path = cache_path(beta, eps, idx)
    with h5py.File(path, "r") as f:
        a = f.attrs
        if not (float(a["beta"]) == beta and int(a["idx"]) == idx and float(a["beta_pde"]) == eps
                and float(a["beta_ic"]) == BETA_IC and float(a["sigma_w"]) == SIGMA_W):
            raise ValueError(f"{path}: stored beta={a['beta']}, idx={a['idx']}, beta_pde={a['beta_pde']}, "
                             f"beta_ic={a['beta_ic']}, sigma_w={a['sigma_w']} do not match the requested run")
        x, tn = f["x"][:], f["tn"][:]
        residual = mp_sweep.compute_residual(
            x, tn, f["w"][:],
            Kx=int(a["Kx"]), Kt=int(a["Kt"]), c_norm=float(a["c_norm"]),
            alpha=float(a["alpha"]), piecewise_beta=float(a["piecewise_beta"]),
        )
        return dict(
            beta=beta, eps=eps, idx=idx, path=path,
            epochs=int(a["epochs"]), n_kept=int(f["w_eigen_mean"].shape[0]), n_raw=int(f["w"].shape[0]),
            x=x, t=f["t"][:],
            true_u=f["true_u"][:].T, pred_u=f["pred_u"][:].T, pred_var=f["pred_var"][:].T,
            residual=residual,
        )


def missing_runs() -> list[tuple[float, float, int]]:
    return [(b, e, i) for b in BETAS for e in EPSILONS for i in IDXS if not cache_path(b, e, i).exists()]


def _command(run: tuple[float, float, int], out: Path) -> list[str]:
    beta, eps, idx = run
    return ["julia", f"--project={REPO_ROOT}", str(JULIA_RUNNER), str(beta), str(idx), str(out),
            str(SIGMA_W), str(BETA_IC), str(eps)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--jobs", type=int, default=4, help="concurrent Julia processes (default 4)")
    ap.add_argument("--big-jobs", type=int, default=2, help=f"of which at most this many at b={BIG_BETA:g} (default 2)")
    ap.add_argument("--limits-file", type=Path, default=ARTIFACTS_DIR / "mp_pinn_epsilon_sweep.limits.json",
                    help="JSON {\"jobs\": N, \"big_jobs\": M} re-read while running (default %(default)s)")
    args = ap.parse_args()
    todo = missing_runs()
    # longest first: all b=7 runs (by epsilon, then trajectory), then b=2, 1, 0.4, 0.1
    todo.sort(key=lambda r: (r[0] != BIG_BETA, -r[0], r[1], IDXS.index(r[2])))
    failed = run_jobs(todo, path_for=lambda r: cache_path(*r), command_for=_command,
                      is_big=lambda r: r[0] == BIG_BETA, label_for=lambda r: f"b={r[0]:g} eps={r[1]:g} idx={r[2]}",
                      cwd=REPO_ROOT, jobs=args.jobs, big_jobs=args.big_jobs, limits_file=args.limits_file)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
