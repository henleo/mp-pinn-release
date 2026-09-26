"""
Library for the PDEBench-dataset Fisher-KPP runs (`mp_pinn_reacdiff_dataset_run.jl`):
cache-or-train per (nu, rho, traj_idx) and load results together with the float64 reference
solution of the nominal PDE (`reacdiff_reference.py`) -- all error metrics returned here are
computed against that reference, NOT against the stored PDEBench trajectory, because the
stored trajectories freeze at non-physical values once diffusion homogenizes the field (see
`reacdiff_reference.py`'s module docstring; the stored trajectory and its own rel_l2 are still
returned for transparency).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from reacdiff_reference import reference_solution  # noqa: E402

JULIA_RUNNER = SCRIPT_DIR / "mp_pinn_reacdiff_dataset_run.jl"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

NUS = [0.5, 1.0, 2.0, 5.0]
RHOS = [1.0, 2.0, 5.0, 10.0]

# Artifact file-name prefix of the one-pass runs (noise-floor rank tolerance and
# ρ-continuation, both on by default in the runner).
ARTIFACT_PREFIX = "mp_pinn_reacdiff_ds2"


def _fmt(x: float) -> str:
    return str(int(x)) + ".0" if float(x).is_integer() else str(x)


def _cache_path(nu: float, rho: float, idx: int) -> Path:
    return ARTIFACTS_DIR / f"{ARTIFACT_PREFIX}_nu{_fmt(nu)}_rho{_fmt(rho)}_idx{idx}.h5"


def train_one(nu: float, rho: float, idx: int) -> None:
    subprocess.run(
        ["julia", f"--project={REPO_ROOT}", str(JULIA_RUNNER), str(nu), str(rho), str(idx),
         str(_cache_path(nu, rho, idx))],
        check=True,
    )


def load_cached(nu: float, rho: float, idx: int) -> dict:
    ref = reference_solution(nu, rho, idx)
    u_ref = ref["u_ref"].T  # (n_x, n_t)
    with h5py.File(_cache_path(nu, rho, idx), "r") as f:
        pred_u = f["pred_u"][:].T
        stored_u = f["true_u"][:].T
        rel_l2_ref = float(np.linalg.norm(pred_u - u_ref) / np.linalg.norm(u_ref))
        rel_l2_stored = float(np.linalg.norm(pred_u - stored_u) / np.linalg.norm(stored_u))
        return dict(
            nu=nu, rho=rho, idx=idx,
            Kx=int(f.attrs["Kx"]), Kt=int(f.attrs["Kt"]),
            nt=int(f.attrs["nt"]),
            epochs=int(f.attrs["epochs"]),
            train_time_seconds=float(f.attrs["train_time_seconds"]),
            predict_time_seconds=float(f.attrs["predict_time_seconds"]),
            x=f["x"][:], t=f["t"][:],
            u_ref=u_ref,             # float64 nominal-PDE reference (n_x, n_t)
            stored_u=stored_u,       # the PDEBench file's own (frozen-tail) trajectory
            pred_u=pred_u,
            pred_var=f["pred_var"][:].T,
            residual=f["residual"][:].T,
            rel_l2_ref=rel_l2_ref,
            rel_l2_stored=rel_l2_stored,
        )


def run_or_load(nu: float, rho: float, idx: int) -> dict:
    if not _cache_path(nu, rho, idx).exists():
        train_one(nu, rho, idx)
    return load_cached(nu, rho, idx)
