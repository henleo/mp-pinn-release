"""
Loaders (cache-or-train per (nu, rho, traj_idx)) for every method run on the PDEBench-dataset
Fisher-KPP (reaction-diffusion) problems, mirroring the advection sweep modules:

  method   runner (pde/diffusion/)               artifact (artifacts/)
  Ours     mp_pinn_reacdiff_dataset_run.jl       mp_pinn_reacdiff_ds2_nu{nu}_rho{rho}_idx{idx}.h5
  HMC      bpinn_run_reacdiff_multichain.jl      bpinn_mc_reacdiff_nu{nu}_rho{rho}_idx{idx}.h5
  VI       vi_run_reacdiff.jl                    vi_reacdiff_nu{nu}_rho{rho}_idx{idx}.h5
  GD       gd_run_reacdiff.jl                    gd_pinn_reacdiff_nu{nu}_rho{rho}_idx{idx}.h5
  wide     mp_pinn_reacdiff_wide_run.jl          mp_pinn_reacdiff_wideH{H}_nu{nu}_rho{rho}_idx{idx}.h5

All runners build the identical problem (`reacdiff_lib.jl: reacdiff_dataset_problem`) and graph
(same rng seed), so their eigenbasis coordinates `w_eigen_*` are comparable index for index;
every artifact stores `v_fingerprint` (column sums of V) and `assert_same_basis` checks it.

Ground truth for error metrics is the float64 re-solve of the nominal PDE
(`reacdiff_reference.py`), returned as `u_ref`; the stored PDEBench trajectory is
returned as `true_u` (the same array every runner saved) for transparency. Every loader returns
(n_x, n_t)-shaped grids (HDF5.jl wrote them column-major; h5py reads the bytes back transposed).

The HMC loader adds the per-coordinate convergence diagnostics (split-R-hat, bulk ESS) computed
by `bpinn_multichain_sweep.py` from the saved per-chain sample matrices.
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
sys.path.insert(0, str(REPO_ROOT / "pde" / "advection"))

import mp_pinn_reacdiff_dataset_sweep as ds_sweep  # noqa: E402
from bpinn_multichain_sweep import bulk_ess, split_rhat  # noqa: E402
from reacdiff_reference import reference_solution  # noqa: E402

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

NUS = ds_sweep.NUS
RHOS = ds_sweep.RHOS

RUNNERS = {
    "Ours": SCRIPT_DIR / "mp_pinn_reacdiff_dataset_run.jl",
    "HMC": SCRIPT_DIR / "bpinn_run_reacdiff_multichain.jl",
    "VI": SCRIPT_DIR / "vi_run_reacdiff.jl",
    "GD": SCRIPT_DIR / "gd_run_reacdiff.jl",
}
PREFIX = {
    "Ours": ds_sweep.ARTIFACT_PREFIX,
    "HMC": "bpinn_mc_reacdiff",
    "VI": "vi_reacdiff",
    "GD": "gd_pinn_reacdiff",
}


def _fmt(x: float) -> str:
    return str(int(x)) + ".0" if float(x).is_integer() else str(x)


def cache_path(method: str, nu: float, rho: float, idx: int, H: int | None = None) -> Path:
    if method == "wide":
        return ARTIFACTS_DIR / f"mp_pinn_reacdiff_wideH{H}_nu{_fmt(nu)}_rho{_fmt(rho)}_idx{idx}.h5"
    return ARTIFACTS_DIR / f"{PREFIX[method]}_nu{_fmt(nu)}_rho{_fmt(rho)}_idx{idx}.h5"


def train_one(method: str, nu: float, rho: float, idx: int, *, extra_args: list[str] | None = None) -> None:
    """Invoke the Julia runner of `method` for one (nu, rho, idx); it writes its own HDF5 cache."""
    args = ["julia", f"--project={REPO_ROOT}", str(RUNNERS[method]), str(nu), str(rho), str(idx),
            str(cache_path(method, nu, rho, idx))]
    if extra_args:
        args += [str(a) for a in extra_args]
    subprocess.run(args, check=True)


def _common(f: h5py.File, nu: float, rho: float, idx: int) -> dict:
    ref = reference_solution(nu, rho, idx)
    u_ref = ref["u_ref"].T  # (n_x, n_t)
    pred_u = f["pred_u"][:].T
    return dict(
        nu=nu, rho=rho, idx=int(f.attrs["idx"]),
        nu_norm=float(f.attrs["nu_norm"]), rho_norm=float(f.attrs["rho_norm"]),
        Kx=int(f.attrs["Kx"]), Kt=int(f.attrs["Kt"]),
        train_time_seconds=float(f.attrs["train_time_seconds"]),
        predict_time_seconds=float(f.attrs["predict_time_seconds"]),
        x=f["x"][:], t=f["t"][:], tn=f["tn"][:],
        true_u=f["true_u"][:].T,       # the stored PDEBench trajectory (frozen tail, see reacdiff_reference.py)
        u_ref=u_ref,                   # float64 nominal-PDE reference: the ground truth for errors
        pred_u=pred_u,
        residual=f["residual"][:].T,   # the model's own residual at the posterior-mean / point estimate
        rel_l2_ref=float(np.linalg.norm(pred_u - u_ref) / np.linalg.norm(u_ref)),
        rel_l2_stored=float(np.linalg.norm(pred_u - f["true_u"][:].T) / np.linalg.norm(f["true_u"][:].T)),
        v_fingerprint=f["v_fingerprint"][:] if "v_fingerprint" in f else None,
    )


def load_cached(method: str, nu: float, rho: float, idx: int) -> dict:
    with h5py.File(cache_path(method, nu, rho, idx), "r") as f:
        r = _common(f, nu, rho, idx)
        if method == "Ours":
            r.update(epochs=int(f.attrs["epochs"]), pred_var=f["pred_var"][:].T,
                     w_eigen_mean=f["w_eigen_mean"][:], w_eigen_std=f["w_eigen_std"][:],
                     loss_x=f["loss_x"][:], loss_y=f["loss_y"][:],
                     rung_epochs=[int(e) for e in np.atleast_1d(f.attrs["rung_epochs"])])
        elif method == "HMC":
            n_chains = int(f.attrs["n_chains"])
            chains = np.stack([f[f"samples_chain{c}"][:] for c in range(1, n_chains + 1)], axis=0)  # (C, draws, K)
            r.update(
                n_chains=n_chains, num_samples=int(f.attrs["num_samples"]), burn=int(f.attrs["burn"]),
                leapfrog_L=int(f.attrs["leapfrog_L"]), step_size=float(f.attrs["step_size"]),
                accept_rates=f["accept_rates"][:], accept_rates_kept=f["accept_rates_kept"][:],
                chain_step_sizes=f["chain_step_sizes"][:], chain_times=f["chain_times"][:],
                map_time_seconds=float(f.attrs["map_time_seconds"]),
                pred_var=f["pred_var"][:].T,
                w_eigen_mean=f["w_eigen_mean"][:], w_eigen_std=f["w_eigen_std"][:],
                chains=chains, rhat=split_rhat(chains), ess=bulk_ess(chains),
            )
        elif method == "VI":
            r.update(N_z=int(f.attrs["N_z"]), iters=int(f.attrs["iters"]), pred_var=f["pred_var"][:].T,
                     w_eigen_mean=f["w_eigen_mean"][:], w_eigen_std=f["w_eigen_std"][:],
                     loss_x=f["loss_x"][:], loss_y=f["loss_y"][:])
        elif method == "GD":
            r.update(iters=int(f.attrs["iters"]), w_eigen=f["w_eigen"][:],
                     loss_x=f["loss_x"][:], loss_y=f["loss_y"][:])
        else:
            raise ValueError(method)
        return r


def load_wide(nu: float, rho: float, idx: int, H: int) -> dict:
    """The wide one-pass artifact (u = sum_h a_h g(w_h.phi), fixed readout a, H hidden units,
    `mp_pinn_reacdiff_wide_run.jl`): the single-neuron keys plus `H`, `a`, per-neuron posteriors
    `w_eigen_mean` / `w_eigen_std` as (K, H) arrays, raw weights `w` (K_raw, H) and the neuron-usage
    summaries. Missing artifacts raise FileNotFoundError (the figure scripts never train)."""
    path = cache_path("wide", nu, rho, idx, H)
    if not path.exists():
        raise FileNotFoundError(f"missing wide artifact {path} (nu={nu}, rho={rho}, idx={idx}, H={H}); "
                                "generate it with pde/diffusion/mp_pinn_reacdiff_wide_run.jl")
    with h5py.File(path, "r") as f:
        r = _common(f, nu, rho, idx)
        r.update(
            H=int(f.attrs["H"]), a=f["a"][:], epochs=int(f.attrs["epochs"]),
            rung_epochs=[int(e) for e in np.atleast_1d(f.attrs["rung_epochs"])],
            pred_var=f["pred_var"][:].T,
            # Julia wrote (K_raw, H) / (K, H); h5py reads the same bytes back as (H, K_raw) / (H, K).
            w=f["w"][:].T, w_eigen_mean=f["w_eigen_mean"][:].T, w_eigen_std=f["w_eigen_std"][:].T,
            loss_x=f["loss_x"][:], loss_y=f["loss_y"][:],
            neuron_z_min=f["neuron_z_min"][:], neuron_z_max=f["neuron_z_max"][:],
            neuron_z_rms=f["neuron_z_rms"][:], neuron_out_rms=f["neuron_out_rms"][:],
            neuron_var_mean=f["neuron_var_mean"][:],
        )
        return r


def run_or_load(method: str, nu: float, rho: float, idx: int) -> dict:
    if not cache_path(method, nu, rho, idx).exists():
        train_one(method, nu, rho, idx)
    return load_cached(method, nu, rho, idx)


def assert_same_basis(*records: dict, rtol: float = 1e-9) -> None:
    """Every record must carry the same eigenbasis fingerprint (same V, same sign of every
    eigenvector); raises otherwise, also when a record carries no fingerprint."""
    fps = [r["v_fingerprint"] for r in records]
    if any(fp is None for fp in fps):
        raise ValueError("an artifact has no v_fingerprint; regenerate it with its runner in pde/diffusion/")
    for fp in fps[1:]:
        if fp.shape != fps[0].shape or not np.allclose(fp, fps[0], rtol=rtol, atol=1e-12):
            raise ValueError("eigenbasis fingerprints differ between methods (V or eigenvector signs differ, e.g. with BLAS threading)")
