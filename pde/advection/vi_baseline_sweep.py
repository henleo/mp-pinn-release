"""
Library for running the Variational Inference baseline (`VIPINN`, `vi_model.jl`, Algorithm 2 of
Yang, Meng & Karniadakis 2021 (B-PINNs, J. Comput. Phys.)) once per advection wave speed, by
shelling out to Julia per beta (`vi_run_beta.jl`, cached in its own per-beta HDF5 file); it mirrors
`mp_pinn_beta_sweep.py`, since this baseline shares the same architecture/basis/data
(`mp_pinn_model.jl` via `bpinn_model.jl`). Has `pred_var` too: `pred_u`/`pred_var` are the
mean-field Gaussian predictive in closed form (`predict_vi_stats_analytic`, which evaluates the
same `predict_moments_grid` activation moments Ours uses; no Monte Carlo), so they are directly
calibration-comparable to MP-PINN and the HMC baseline. The 600-draw sampler of Algorithm 2's own
final step (`predict_vi_stats`) is provided for validation only and is not what the cache holds.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import h5py

import mp_pinn_beta_sweep as mp_sweep

BETAS = [0.1, 0.4, 1.0, 2.0, 7.0]
TRAJ_IDX = -1

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
JULIA_RUNNER = SCRIPT_DIR / "vi_run_beta.jl"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


def _cache_path(beta: float, idx: int = TRAJ_IDX) -> Path:
    return ARTIFACTS_DIR / f"vi_beta{beta}_idx{idx}.h5"


def train_one_beta(beta: float, idx: int = TRAJ_IDX, *, extra_args: list[str] | None = None) -> None:
    """Invoke the Julia runner for one (beta, idx); it writes its own HDF5 cache file.

    `extra_args` forwards optional hyperparameter overrides positionally (`N_z M`), matching
    `vi_run_beta.jl`'s CLI convention.
    """
    args = ["julia", f"--project={REPO_ROOT}", str(JULIA_RUNNER), str(beta), str(idx), str(_cache_path(beta, idx))]
    if extra_args:
        args += [str(a) for a in extra_args]
    subprocess.run(args, check=True)


def load_cached(beta: float, idx: int = TRAJ_IDX) -> dict:
    with h5py.File(_cache_path(beta, idx), "r") as f:
        x, tn = f["x"][:], f["tn"][:]
        residual = mp_sweep.compute_residual(
            x, tn, f["w"][:],
            Kx=int(f.attrs["Kx"]), Kt=int(f.attrs["Kt"]), c_norm=float(f.attrs["c_norm"]),
            alpha=float(f.attrs["alpha"]), piecewise_beta=float(f.attrs["piecewise_beta"]),
        )
        return dict(
            beta=beta,
            idx=int(f.attrs["idx"]),
            N_z=int(f.attrs["N_z"]),
            M_predict=int(f.attrs["M_predict"]),
            iters=int(f.attrs["iters"]),
            train_time_seconds=float(f.attrs["train_time_seconds"]),
            predict_time_seconds=float(f.attrs["predict_time_seconds"]),
            x=x,
            t=f["t"][:],
            # see mp_pinn_beta_sweep.load_cached: HDF5.jl (column-major) wrote these
            # (n_x, n_t)-shaped; h5py (row-major) reads the same bytes back transposed.
            true_u=f["true_u"][:].T,
            pred_u=f["pred_u"][:].T,
            pred_var=f["pred_var"][:].T,
            residual=residual,
            loss_x=f["loss_x"][:],
            loss_y=f["loss_y"][:],
            w_eigen_mean=f["w_eigen_mean"][:],
            w_eigen_std=f["w_eigen_std"][:],
        )
