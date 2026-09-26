"""
Library for the multi-chain, mass-preconditioned HMC reference (`bpinn_run_beta_multichain.jl`):
cache-or-train per (beta, idx), plus the convergence diagnostics that make it a defensible
gold standard -- split-R-hat and bulk ESS per eigenbasis coordinate (Vehtari, Gelman, Simpson,
Carpenter & Bürkner 2021), computed from the saved per-chain sample matrices.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import h5py
import numpy as np

import mp_pinn_beta_sweep as mp_sweep

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
JULIA_RUNNER = SCRIPT_DIR / "bpinn_run_beta_multichain.jl"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


def _cache_path(beta: float, idx: int) -> Path:
    return ARTIFACTS_DIR / f"bpinn_mc_beta{beta}_idx{idx}.h5"


def train_one(beta: float, idx: int, *, extra_args: list[str] | None = None) -> None:
    args = ["julia", f"--project={REPO_ROOT}", str(JULIA_RUNNER), str(beta), str(idx), str(_cache_path(beta, idx))]
    if extra_args:
        args += [str(a) for a in extra_args]
    subprocess.run(args, check=True)


def split_rhat(chains: np.ndarray) -> np.ndarray:
    """Split-R-hat per coordinate. `chains`: (n_chains, n_draws, K). Each chain is split in
    half (so 2C sequences), then the classic between/within variance ratio is computed."""
    c, n, k = chains.shape
    half = n // 2
    seqs = np.concatenate([chains[:, :half, :], chains[:, half : 2 * half, :]], axis=0)  # (2C, half, K)
    m, n_h = seqs.shape[0], seqs.shape[1]
    seq_means = seqs.mean(axis=1)                       # (2C, K)
    seq_vars = seqs.var(axis=1, ddof=1)                 # (2C, K)
    w = seq_vars.mean(axis=0)                           # within
    b = n_h * seq_means.var(axis=0, ddof=1)             # between
    var_plus = (n_h - 1) / n_h * w + b / n_h
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(var_plus / w)


def bulk_ess(chains: np.ndarray) -> np.ndarray:
    """Bulk ESS per coordinate via FFT autocorrelations combined across chains (Geyer's
    initial monotone positive sequence, as in Vehtari et al. 2021, without rank-normalization)."""
    c, n, k = chains.shape
    ess = np.empty(k)
    for j in range(k):
        x = chains[:, :, j]
        chain_means = x.mean(axis=1, keepdims=True)
        xc = x - chain_means
        # per-chain autocovariance via FFT
        nfft = 1 << int(np.ceil(np.log2(2 * n)))
        f = np.fft.rfft(xc, nfft, axis=1)
        acov = np.fft.irfft(f * np.conj(f), nfft, axis=1)[:, :n].real / n
        w = acov[:, 0].mean()  # within-chain variance (biased)
        var_plus = w * n / (n - 1) + chain_means.var(ddof=1) if c > 1 else w * n / (n - 1)
        if var_plus <= 0 or not np.isfinite(var_plus):
            ess[j] = np.nan
            continue
        rho = 1.0 - (w - acov.mean(axis=0)) / var_plus  # combined autocorrelation, rho[0]=1
        # Geyer's initial monotone positive sequence: pairs P_i = rho_{2i} + rho_{2i+1},
        # keep while positive, enforce monotone decrease; tau = -1 + 2·Σ P_i.
        pair_sums = []
        t = 0
        while t + 1 < len(rho):
            p = rho[t] + rho[t + 1]
            if p <= 0:
                break
            pair_sums.append(p)
            t += 2
        for i in range(1, len(pair_sums)):
            pair_sums[i] = min(pair_sums[i], pair_sums[i - 1])
        tau = -1.0 + 2.0 * sum(pair_sums)
        ess[j] = c * n / max(tau, 1e-12)
    return ess


def load_cached(beta: float, idx: int) -> dict:
    with h5py.File(_cache_path(beta, idx), "r") as f:
        x, tn = f["x"][:], f["tn"][:]
        residual = mp_sweep.compute_residual(
            x, tn, f["w"][:],
            Kx=int(f.attrs["Kx"]), Kt=int(f.attrs["Kt"]), c_norm=float(f.attrs["c_norm"]),
            alpha=float(f.attrs["alpha"]), piecewise_beta=float(f.attrs["piecewise_beta"]),
        )
        n_chains = int(f.attrs["n_chains"])
        # Julia wrote (K, n_draws); h5py reads back transposed as (n_draws, K).
        chain_list = [f[f"samples_chain{c}"][:] for c in range(1, n_chains + 1)]
        chains = np.stack(chain_list, axis=0)  # (C, n_draws, K)
        return dict(
            beta=beta,
            idx=int(f.attrs["idx"]),
            n_chains=n_chains,
            num_samples=int(f.attrs["num_samples"]),
            burn=int(f.attrs["burn"]),
            step_size=float(f.attrs["step_size"]),
            leapfrog_L=int(f.attrs["leapfrog_L"]),
            accept_rates=f["accept_rates"][:],
            # Warm-up step adaptation: acceptance over the retained draws only, the per-chain
            # initial and adapted step sizes, and the MAP negative log posterior before / after the
            # Gauss-Newton polish; None if the artifact does not contain the field.
            accept_rates_kept=f["accept_rates_kept"][:] if "accept_rates_kept" in f else None,
            chain_step_sizes=f["chain_step_sizes"][:],
            chain_init_step_sizes=f["chain_init_step_sizes"][:] if "chain_init_step_sizes" in f else None,
            step_size_protocol=(f.attrs["step_size_protocol"].decode() if isinstance(f.attrs["step_size_protocol"], bytes)
                                else str(f.attrs["step_size_protocol"])) if "step_size_protocol" in f.attrs
                               else "not recorded",
            map_nlp_adam=float(f.attrs["map_nlp_adam"]) if "map_nlp_adam" in f.attrs else None,
            map_nlp=float(f.attrs["map_nlp"]) if "map_nlp" in f.attrs else None,
            chain_times=f["chain_times"][:],
            map_time_seconds=float(f.attrs["map_time_seconds"]),
            train_time_seconds=float(f.attrs["train_time_seconds"]),
            predict_time_seconds=float(f.attrs["predict_time_seconds"]),
            x=x,
            t=f["t"][:],
            true_u=f["true_u"][:].T,
            pred_u=f["pred_u"][:].T,
            pred_var=f["pred_var"][:].T,
            residual=residual,
            w_eigen_mean=f["w_eigen_mean"][:],
            w_eigen_std=f["w_eigen_std"][:],
            chains=chains,
            rhat=split_rhat(chains),
            ess=bulk_ess(chains),
        )


def run_or_load(beta: float, idx: int, *, extra_args: list[str] | None = None) -> dict:
    if not _cache_path(beta, idx).exists():
        train_one(beta, idx, extra_args=extra_args)
    return load_cached(beta, idx)
