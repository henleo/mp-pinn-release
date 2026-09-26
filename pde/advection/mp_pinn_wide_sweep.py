"""
Loader for the WIDE one-pass MP-PINN artifacts (`mp_pinn_run_beta_wide.jl`: u = sum_h a_h g(w_h.phi)
with H hidden units and fixed readout weights a, same EP schedule as the single-neuron runner),
mirroring `mp_pinn_beta_sweep.py`. Cache file per (beta, idx, H):
`artifacts/mp_pinn_wideH{H}_beta{beta}_idx{idx}.h5`.

`load_cached` returns the same keys as `mp_pinn_beta_sweep.load_cached` (true_u, pred_u, pred_var
as (n_x, n_t) arrays, t, x, epochs, timings, loss curve) plus the width-specific ones: `H`, `a`,
per-neuron posteriors `w_eigen_mean` / `w_eigen_std` as (K, H) arrays (column h = neuron h, same
eigenbasis coordinates as the single-neuron and HMC artifacts), raw weights `w` (K_raw, H), the
wide residual, the per-neuron pre-activation grids `z_neuron` (n_x, n_t, H) at the posterior mean
(rebuilt from `w` like the residual), and the neuron-usage summaries the runner saved.

No plotting here; the figure is made by figures_onepass/advection/wider_model.py.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import h5py
import numpy as np

import mp_pinn_beta_sweep as mp_sweep

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
JULIA_RUNNER = SCRIPT_DIR / "mp_pinn_run_beta_wide.jl"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


def _cache_path(beta: float, idx: int, H: int) -> Path:
    return ARTIFACTS_DIR / f"mp_pinn_wideH{H}_beta{beta}_idx{idx}.h5"


def preactivation_grid(x: np.ndarray, tn: np.ndarray, w: np.ndarray, *, Kx: int, Kt: int) -> np.ndarray:
    """`z = w.phi` over the (x, tn) grid for ONE raw weight vector `w`; the same as Julia's `predict_grid`."""
    Xmat, _ = mp_sweep._trig_basis_matrix(Kx, x)
    Tmat, _ = mp_sweep._trig_basis_matrix(Kt, tn)
    W = w.reshape(2 * Kx + 1, 2 * Kt + 1, order="F")
    return Xmat @ W @ Tmat.T


def train_one(beta: float, idx: int, H: int) -> None:
    """Invoke the Julia runner for one (beta, idx, H); it writes its own HDF5 cache file."""
    subprocess.run(
        ["julia", f"--project={REPO_ROOT}", str(JULIA_RUNNER), str(beta), str(idx),
         str(_cache_path(beta, idx, H)), str(H)],
        check=True,
    )


def load_cached(beta: float, idx: int, H: int) -> dict:
    with h5py.File(_cache_path(beta, idx, H), "r") as f:
        x, tn = f["x"][:], f["tn"][:]
        a = f["a"][:]
        # Julia wrote (K_raw, H) / (K, H); h5py reads the same bytes back as (H, K_raw) / (H, K).
        w = f["w"][:].T
        w_eigen_mean = f["w_eigen_mean"][:].T
        w_eigen_std = f["w_eigen_std"][:].T
        Kx, Kt = int(f.attrs["Kx"]), int(f.attrs["Kt"])
        common = dict(Kx=Kx, Kt=Kt, c_norm=float(f.attrs["c_norm"]),
                      alpha=float(f.attrs["alpha"]), piecewise_beta=float(f.attrs["piecewise_beta"]))
        residual = sum(a[h] * mp_sweep.compute_residual(x, tn, w[:, h], **common) for h in range(len(a)))
        z_neuron = np.stack([preactivation_grid(x, tn, w[:, h], Kx=Kx, Kt=Kt) for h in range(len(a))], axis=-1)
        return dict(
            beta=beta,
            idx=int(f.attrs["idx"]),
            H=int(f.attrs["H"]),
            a=a,
            epochs=int(f.attrs["epochs"]),
            rel_l2=float(f.attrs["rel_l2"]),
            train_time_seconds=float(f.attrs["train_time_seconds"]),
            predict_time_seconds=float(f.attrs["predict_time_seconds"]),
            x=x,
            t=f["t"][:],
            true_u=f["true_u"][:].T,
            pred_u=f["pred_u"][:].T,
            pred_var=f["pred_var"][:].T,
            residual=residual,
            z_neuron=z_neuron,
            loss_x=f["loss_x"][:],
            loss_y=f["loss_y"][:],
            w=w,
            w_eigen_mean=w_eigen_mean,
            w_eigen_std=w_eigen_std,
            neuron_z_min=f["neuron_z_min"][:],
            neuron_z_max=f["neuron_z_max"][:],
            neuron_z_rms=f["neuron_z_rms"][:],
            neuron_out_rms=f["neuron_out_rms"][:],
            neuron_var_mean=f["neuron_var_mean"][:],
        )


def run_or_load(beta: float, idx: int, H: int) -> dict:
    if not _cache_path(beta, idx, H).exists():
        train_one(beta, idx, H)
    return load_cached(beta, idx, H)
