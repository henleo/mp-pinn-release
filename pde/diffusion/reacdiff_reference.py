"""
Float64 reference solutions of the NOMINAL 1D Fisher-KPP equation
    u_t = nu * u_xx + rho * u * (1 - u),   x in [0,1) periodic,
from the PDEBench dataset's own initial conditions.

Why this exists: the stored PDEBench `1D_ReacDiff` trajectories STOP EVOLVING once diffusion
homogenizes the field -- they freeze at values like u=0.4329 (nu=0.5, rho=1.0) or u=0.625
(nu=5.0, rho=1.0) where the nominal reaction term still demands du/dt = rho*u*(1-u) ~ 0.2.
The generator advances the reaction with an exact-logistic substep at the diffusion-CFL step
size dt = 0.2*dx^2/nu ~ 4e-7 in float32; at (nu=5, rho=1, u=0.625) that float32 map is an
exact one-step fixed point (verified), so the reaction stalls numerically. Evaluating any PDE
solver against the frozen tail punishes it for not reproducing a float32 artifact; this module
re-solves the nominal equation in float64 (spectral diffusion, Strang splitting with the exact
logistic flow, no stability limit) and is validated two ways in `validate()`:
  (1) against the stored data at EARLY times (before the freeze), where the two must agree;
  (2) self-convergence under dt -> dt/2.

Results are cached per (nu, rho, idx) under artifacts/reacdiff_ref/. The same solver (on [0, 2pi))
also gives the reference for the Gaussian-bump IC of PDEBench's single-trajectory generator
(`gaussian_bump_reference`, cached per (nu, rho) as ref_gauss_*.npz).
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = REPO_ROOT / "data" / "1D" / "ReactionDiffusion" / "Train"
CACHE_DIR = REPO_ROOT / "artifacts" / "reacdiff_ref"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DT = 1e-4


def _fmt(x: float) -> str:
    return str(int(x)) + ".0" if float(x).is_integer() else str(x)


def dataset_path(nu: float, rho: float) -> Path:
    return DATA_DIR / f"ReacDiff_Nu{_fmt(nu)}_Rho{_fmt(rho)}.hdf5"


def load_dataset_traj(nu: float, rho: float, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stored trajectory as float64: u (n_t, n_x), x, t."""
    with h5py.File(dataset_path(nu, rho), "r") as f:
        u = f["tensor"][idx].astype(np.float64)
        x = f["x-coordinate"][:].astype(np.float64)
        t = f["t-coordinate"][: u.shape[0]].astype(np.float64)
    return u, x, t


def solve_fisher(u0: np.ndarray, nu: float, rho: float, t_save: np.ndarray, dt: float = DT,
                 L: float = 1.0) -> np.ndarray:
    """Strang splitting: exact logistic half-step, exact spectral diffusion full step, exact
    logistic half-step. Periodic domain [0, L) (L=1 for the dataset). Returns (len(t_save), n_x)."""
    n = u0.shape[0]
    k = 2 * np.pi * np.fft.rfftfreq(n, d=L / n)
    diffusion_factor = np.exp(-nu * k**2 * dt)

    def logistic(u, h):
        # exact flow of u' = rho*u*(1-u); handles u=0 and u=1 exactly
        e = np.exp(-rho * h)
        return u / (u + (1.0 - u) * e)

    u = u0.astype(np.float64).copy()
    out = np.empty((len(t_save), n))
    t = 0.0
    i_save = 0
    if abs(t_save[0]) < 1e-12:
        out[0] = u
        i_save = 1
    n_steps = int(round((t_save[-1] - t_save[0]) / dt))
    for step in range(n_steps):
        u = logistic(u, dt / 2)
        u = np.fft.irfft(np.fft.rfft(u) * diffusion_factor, n)
        u = logistic(u, dt / 2)
        t += dt
        while i_save < len(t_save) and t >= t_save[i_save] - 1e-9:
            out[i_save] = u
            i_save += 1
    while i_save < len(t_save):  # guard against float roundoff on the last frame
        out[i_save] = u
        i_save += 1
    return out


def _cache_path(nu: float, rho: float, idx: int) -> Path:
    return CACHE_DIR / f"ref_nu{_fmt(nu)}_rho{_fmt(rho)}_idx{idx}.npz"


def reference_solution(nu: float, rho: float, idx: int) -> dict:
    """Cached float64 reference for one dataset trajectory. Returns dict with u_ref (n_t, n_x),
    u_data (the stored trajectory, same shape), x, t."""
    p = _cache_path(nu, rho, idx)
    if p.exists():
        z = np.load(p)
        return {key: z[key] for key in z.files}
    u_data, x, t = load_dataset_traj(nu, rho, idx)
    u_ref = solve_fisher(u_data[0], nu, rho, t)
    np.savez_compressed(p, u_ref=u_ref, u_data=u_data, x=x, t=t)
    return dict(u_ref=u_ref, u_data=u_data, x=x, t=t)


def validate(nu: float, rho: float, idx: int) -> None:
    r = reference_solution(nu, rho, idx)
    u_ref, u_data, t = r["u_ref"], r["u_data"], r["t"]
    # (1) early-time agreement with the stored data (before the float32 freeze)
    for tt in (0.01, 0.03, 0.05, 0.1):
        j = int(round(tt / (t[1] - t[0])))
        d = np.abs(u_ref[j] - u_data[j]).max()
        print(f"  t={tt}: max|ref - data| = {d:.3e}")
    # (2) dt self-convergence
    u_half = solve_fisher(u_data[0], nu, rho, t, dt=DT / 2)
    print(f"  dt-convergence: max|ref(dt) - ref(dt/2)| = {np.abs(u_ref - u_half).max():.3e}")
    # (3) the artifact itself: late-time drift of data vs nominal dynamics
    print(f"  late (t=1): data mean {u_data[-1].mean():.4f}, nominal-PDE mean {u_ref[-1].mean():.4f}")


# --- Gaussian-bump initial condition (PDEBench's single-trajectory generator) ------------------
# PDEBench's `pdebench/data_gen/data_gen_NLE/ReactionDiffusionEq/reaction_diffusion_Hydra.py`
# with its default config (init_mode="react"): one deterministic
# trajectory, u0(x) = exp(-0.5 (x - pi)^2 / (pi/4)^2) on [0, 2 pi) periodic, nx = 1024 cell
# centres, saved every dt_save = 0.01 on t in [0, 1]. Re-solved here in float64 for the same
# reason as the dataset (the generator uses the same float32 exact-logistic substep).
GAUSS_L = 2 * np.pi
GAUSS_NX = 1024
GAUSS_T = np.linspace(0.0, 1.0, 101)


def gaussian_bump_ic() -> tuple[np.ndarray, np.ndarray]:
    """(u0, x): the generator's own IC at its own cell centres (`init` in PDEBench's
    `pdebench/data_gen/data_gen_NLE/utils.py`, mode 'react')."""
    dx = GAUSS_L / GAUSS_NX
    x = (np.arange(GAUSS_NX) + 0.5) * dx
    return np.exp(-0.5 * (x - np.pi) ** 2 / (0.25 * np.pi) ** 2), x


def _gauss_cache_path(nu: float, rho: float) -> Path:
    return CACHE_DIR / f"ref_gauss_nu{_fmt(nu)}_rho{_fmt(rho)}.npz"


def gaussian_bump_reference(nu: float, rho: float) -> dict:
    """Cached float64 reference for the Gaussian-bump IC. Returns dict with u_ref (n_t, n_x), x, t."""
    p = _gauss_cache_path(nu, rho)
    if p.exists():
        z = np.load(p)
        return {key: z[key] for key in z.files}
    u0, x = gaussian_bump_ic()
    u_ref = solve_fisher(u0, nu, rho, GAUSS_T, L=GAUSS_L)
    np.savez_compressed(p, u_ref=u_ref, x=x, t=GAUSS_T)
    return dict(u_ref=u_ref, x=x, t=GAUSS_T)


def validate_gaussian(nu: float, rho: float) -> None:
    """dt self-convergence of the Gaussian-bump reference (no stored data to compare with)."""
    r = gaussian_bump_reference(nu, rho)
    u_half = solve_fisher(r["u_ref"][0], nu, rho, r["t"], dt=DT / 2, L=GAUSS_L)
    print(f"  dt-convergence: max|ref(dt) - ref(dt/2)| = {np.abs(r['u_ref'] - u_half).max():.3e}")


if __name__ == "__main__":
    import sys

    if sys.argv[1] == "gauss":
        nu, rho = float(sys.argv[2]), float(sys.argv[3])
        print(f"validating float64 Fisher Gaussian-bump reference at nu={nu}, rho={rho}")
        validate_gaussian(nu, rho)
        sys.exit()
    nu, rho, idx = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
    print(f"validating float64 Fisher reference at nu={nu}, rho={rho}, idx={idx}")
    validate(nu, rho, idx)
