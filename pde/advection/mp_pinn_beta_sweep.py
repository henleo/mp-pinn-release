"""
Library for running the message-passing MP-PINN (`MPPINN`, `mp_pinn_model.jl`) once per
advection wave speed, by shelling out to Julia per beta (`mp_pinn_run_beta.jl`, cached in its
own per-beta HDF5 file).

The Julia runner leaves the trained posterior (raw trig-basis weights `w`, plus the
basis/normalization it was built with) in the HDF5 cache rather than a precomputed residual
grid; `compute_residual` below rebuilds the PDE residual `g'(z) * (w_t + c_norm*w_x)` from that
posterior with the same analytic derivative formulas `predict_residual` uses in
`mp_pinn_model.jl`, so callers never need to shell back out to Julia just to get the residual.

Each beta's HDF5 result is left in place by the Julia runner, so a re-run resumes rather than
re-trains betas already done.

No plotting here -- see figures_onepass/ for scripts that render this data (`load_cached`/
`train_one_beta`) into a figure.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import h5py
import numpy as np

BETAS = [0.1, 0.4, 1.0, 2.0, 7.0]  # the advection wave speeds of the experiments
TRAJ_IDX = -1  # last trajectory in each beta file (each holds 10,000 independent ICs)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
JULIA_RUNNER = SCRIPT_DIR / "mp_pinn_run_beta.jl"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


def _cache_path(beta: float, idx: int = TRAJ_IDX) -> Path:
    return ARTIFACTS_DIR / f"mp_pinn_beta{beta}_idx{idx}.h5"


def _trig_basis_matrix(K: int, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Matches Julia's `trig_vals(K, u)` (mp_pinn_model.jl), vectorized over `u`:
    row i holds the length-(2K+1) basis vector (and its u-derivative) at `u[i]`."""
    n = u.shape[0]
    v = np.empty((n, 2 * K + 1))
    dv = np.empty((n, 2 * K + 1))
    v[:, 0] = 1.0
    dv[:, 0] = 0.0
    for k in range(1, K + 1):
        s, c = np.sin(2 * np.pi * k * u), np.cos(2 * np.pi * k * u)
        v[:, 2 * k - 1] = s
        v[:, 2 * k] = c
        dv[:, 2 * k - 1] = 2 * np.pi * k * c
        dv[:, 2 * k] = -2 * np.pi * k * s
    return v, dv


def compute_residual(
    x: np.ndarray, tn: np.ndarray, w: np.ndarray, *,
    Kx: int, Kt: int, c_norm: float, alpha: float, piecewise_beta: float,
) -> np.ndarray:
    """PDE residual `g'(z) * (w_t + c_norm*w_x)` of the trained network, at the
    posterior mean weights `w` (raw trig-basis coefficients) -- the same formula
    `predict_residual` computes in mp_pinn_model.jl, rebuilt here from the posterior
    so plotting doesn't need Julia to have precomputed a residual grid. Returns an
    `n_x x n_t` matrix, `x` rows by `tn` columns.
    """
    Xmat, dXmat = _trig_basis_matrix(Kx, x)
    Tmat, dTmat = _trig_basis_matrix(Kt, tn)
    W = w.reshape(2 * Kx + 1, 2 * Kt + 1, order="F")  # matches Julia's column-major reshape
    z = Xmat @ W @ Tmat.T
    dzdx = dXmat @ W @ Tmat.T
    dzdt = Xmat @ W @ dTmat.T
    r_grid = dzdt + c_norm * dzdx
    g_prime = np.where(
        z > 0,
        piecewise_beta * np.exp(piecewise_beta * z),
        alpha * piecewise_beta * np.exp(-piecewise_beta * z),
    )
    return g_prime * r_grid


def train_one_beta(beta: float, idx: int = TRAJ_IDX) -> None:
    """Invoke the Julia runner for one (beta, idx); it writes its own HDF5 cache file."""
    subprocess.run(
        ["julia", f"--project={REPO_ROOT}", str(JULIA_RUNNER), str(beta), str(idx), str(_cache_path(beta, idx))],
        check=True,
    )


def load_cached(beta: float, idx: int = TRAJ_IDX) -> dict:
    with h5py.File(_cache_path(beta, idx), "r") as f:
        x, tn = f["x"][:], f["tn"][:]
        residual = compute_residual(
            x, tn, f["w"][:],
            Kx=int(f.attrs["Kx"]), Kt=int(f.attrs["Kt"]), c_norm=float(f.attrs["c_norm"]),
            alpha=float(f.attrs["alpha"]), piecewise_beta=float(f.attrs["piecewise_beta"]),
        )
        return dict(
            beta=beta,
            idx=int(f.attrs["idx"]),
            epochs=int(f.attrs["epochs"]),
            train_time_seconds=float(f.attrs["train_time_seconds"]),
            predict_time_seconds=float(f.attrs["predict_time_seconds"]),
            x=x,
            t=f["t"][:],
            # HDF5.jl (column-major) wrote these (n_x, n_t)-shaped; h5py (row-major) reads the
            # same bytes back with the dimension tuple reversed, i.e. (n_t, n_x) -- not
            # scrambled, just transposed relative to the (n_x, n_t) convention callers expect.
            true_u=f["true_u"][:].T,
            pred_u=f["pred_u"][:].T,
            # already (n_x, n_t): built directly in that convention by compute_residual, not
            # read back from Julia-written bytes.
            residual=residual,
            pred_var=f["pred_var"][:].T,
            # 1D, so no column-/row-major transpose issue.
            loss_x=f["loss_x"][:],
            loss_y=f["loss_y"][:],
            # Posterior marginal mean/std per EIGENBASIS coordinate (not rotated back out via V,
            # unlike `w` above) -- directly comparable index-for-index against
            # bpinn_multichain_sweep.load_cached's same-named fields for the same (beta, idx), since
            # both runners build `graph` with the identical rng seed.
            w_eigen_mean=f["w_eigen_mean"][:],
            w_eigen_std=f["w_eigen_std"][:],
        )
