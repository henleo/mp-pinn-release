"""
Training-time AND inference-time comparison of MP-PINN ("Ours") against the multi-chain HMC
(Hamiltonian Monte Carlo sampling) reference and the VI (variational inference) baseline, over the same ten trajectories per wave speed as
`posterior_recovery_comparison.py` (whose cached runs this reuses). One point per
(method, beta, trajectory); lines join per-beta medians. A companion LaTeX table
(`training_inference_time_table.tex`) holds the underlying medians and Ours-relative speedups.

Inference time is the full posterior-predictive grid (mean AND variance on the complete
1024x201 (x,t) grid). All three methods use their FAIR inference path: message passing and
the VI baseline both read the closed-form predictive off the diagonal eigenbasis posterior
deterministically (`predict_moments_grid`) --
the VI posterior is a diagonal Gaussian over the same coordinates, so sampling it would be an
implementation choice, not a method property. HMC has no closed form (its samples ARE its
posterior), so it evaluates the network across all 3000 pooled draws, with a
~1.3% predictive-sigma Monte-Carlo error (sampling noise from a finite draw count).
Training/sampling is excluded from every method's inference time alike.

Timing protocol. Panel (a), training: cold-start wall-clock in a fresh Julia
process, so it includes JIT (just-in-time, i.e. first-call) compilation of the method's own code
path (about 1.5 s for Ours and VI) -- identical protocol for all three methods; training runs once
per problem, so end-to-end time is the relevant cost, and the compile share is largest for the
fastest method (our training speedups are lower bounds). Panel (b), inference: STEADY-STATE
(compiled) times, since inference is the step that is repeated and its cold-start numbers are
dominated by compilation at b <= 2 (e.g. Ours at b=0.1: 0.09 s steady state vs 1.65 s cold):
the same timed block re-timed from the saved posteriors/chains on an idle machine, minimum over 7
repeated calls for the closed-form methods (measurement order of Ours/VI alternated per
trajectory) and over 3 for HMC, after one untimed warm-up call, single-threaded BLAS
(`training_inference_time_steady_state.csv`, produced by the warm re-timing scripts in
pde/advection/diagnostics/steady_state_timing/ via `python run_experiments.py timing`; every
re-timed path reproduces the artifact's saved pred_var to machine precision). The CSV is
required and must cover every plotted run (an error otherwise). Sub-second single-shot timings
are noise-dominated (up to +-50 % for an identical function), which is why the min-of-k protocol
is used. The companion table holds both: the cold-start training and inference times (with
their speedups) and the steady-state inference block (panel (b)).

"Ours" is single-pass EP (expectation propagation: one sweep of message passing over the factor
graph) throughout.

Run: python figures_onepass/advection/training_inference_time.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = Path(__file__).resolve().parent
REPO_ROOT = FIGURES_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "pde" / "advection"))

import mp_pinn_beta_sweep as mp_sweep
import bpinn_multichain_sweep as hmc_sweep
import vi_baseline_sweep as vi_sweep
from trajectory_selection import idxs

OUTPUT_PDF = FIGURES_DIR / "training_inference_time.pdf"
OUTPUT_TEX = FIGURES_DIR / "training_inference_time_table.tex"
STEADY_CSV = FIGURES_DIR / "training_inference_time_steady_state.csv"
STEADY_FAMILY = {"onepass": "Ours", "hmc": "HMC", "vi": "VI"}

BETAS = [0.1, 0.4, 1.0, 2.0, 7.0]
IDXS = idxs(10)
COLORS = {"Ours": "#4a3aa7", "HMC": "#eb6834", "VI": "#008300"}
MARKERS = {"Ours": "o", "HMC": "s", "VI": "^"}


def _have(beta: float, idx: int) -> bool:
    """All three method caches present for this (beta, idx)? This figure is load-only: a missing
    cache is reported and skipped, never trained inline (a b=7 run is ~1 h single-threaded)."""
    return (mp_sweep._cache_path(beta, idx).exists()
            and hmc_sweep._cache_path(beta, idx).exists()
            and vi_sweep._cache_path(beta, idx).exists())


def collect_times() -> tuple[dict, dict]:
    times = {m: {b: {"train": [], "predict": []} for b in BETAS} for m in COLORS}
    hmc_singlechain = {b: [] for b in BETAS}  # MAP (maximum-a-posteriori warm start) + one average chain (minimal HMC variant)
    for beta in BETAS:
        avail = [i for i in IDXS if _have(beta, i)]
        missing = [i for i in IDXS if i not in avail]
        if missing:
            print(f"  [b={beta}] using {len(avail)}/{len(IDXS)} trajectories; "
                  f"missing caches for idx {missing}")
        for idx in avail:
            mp_r = mp_sweep.load_cached(beta, idx)
            hmc_r = hmc_sweep.load_cached(beta, idx)
            vi_r = vi_sweep.load_cached(beta, idx)
            for method, r in (("Ours", mp_r), ("HMC", hmc_r), ("VI", vi_r)):
                times[method][beta]["train"].append(r["train_time_seconds"])
                times[method][beta]["predict"].append(r["predict_time_seconds"])
            hmc_singlechain[beta].append(hmc_r["map_time_seconds"] + float(np.mean(hmc_r["chain_times"])))
    return times, hmc_singlechain


def _med(vals) -> float:
    return float(np.median(vals)) if len(vals) else float("nan")


def load_steady_state() -> dict | None:
    """Steady-state (compiled) inference times from the warm re-timing CSV, restricted to the
    same canonical trajectories the cold numbers use: {method: {beta: [seconds, ...]}}."""
    if not STEADY_CSV.exists():
        return None
    import csv
    rows = [r for r in csv.DictReader(STEADY_CSV.open())
            if STEADY_FAMILY.get(r["family"]) and float(r["beta"]) in BETAS and int(r["idx"]) in IDXS]
    # Same trajectory set for all three methods per beta (strict parity with the cold block,
    # where a (beta, idx) enters only if all three caches exist).
    have = {(STEADY_FAMILY[r["family"]], float(r["beta"]), int(r["idx"])) for r in rows}
    common = {b: [i for i in IDXS if all((m, b, i) in have for m in COLORS)] for b in BETAS}
    out = {m: {b: [] for b in BETAS} for m in COLORS}
    for r in rows:
        m, b, i = STEADY_FAMILY[r["family"]], float(r["beta"]), int(r["idx"])
        if i in common[b]:
            out[m][b].append(float(r["warm_predict_seconds"]))
    return out


def plot_panel(ax, times, key: str, title: str) -> None:
    x_pos = np.arange(len(BETAS))
    rng = np.random.RandomState(1)
    for method in ("Ours", "HMC", "VI"):
        per_beta = [np.asarray(times[method][b][key]) for b in BETAS]
        for i, vals in enumerate(per_beta):
            if not len(vals):
                continue
            ax.scatter(np.full(vals.shape, x_pos[i]) + rng.uniform(-0.09, 0.09, vals.shape),
                       vals, s=30, alpha=0.45, color=COLORS[method], edgecolors="none")
        med = [_med(v) for v in per_beta]
        ax.plot(x_pos, med, marker=MARKERS[method], color=COLORS[method],
                markersize=11, linewidth=2.2, label=method)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(b) for b in BETAS], fontsize=15)
    ax.set_xlabel(r"$b$", fontsize=20)
    ax.set_yscale("log")
    ax.set_ylabel(r"$\mathrm{time\ (s)}$", fontsize=20)
    ax.set_title(title, fontsize=22)
    ax.grid(True, axis="y", which="both", alpha=0.3)
    ax.tick_params(axis="both", labelsize=14)


def write_table(times: dict, hmc_singlechain: dict, n_used: dict, path: Path,
                steady: dict | None = None) -> None:
    """LaTeX companion table: per-beta median train/inference times (s) for all three methods,
    plus Ours-relative speedups (cold-start; panel (a) plots the training column), and -- when the
    warm re-timing CSV exists -- steady-state (compiled) inference times and their speedups (what
    panel (b) plots)."""
    def row_times(key):
        out = []
        for b in BETAS:
            o = _med(times["Ours"][b][key]); h = _med(times["HMC"][b][key]); v = _med(times["VI"][b][key])
            out.append((b, o, h, v))
        return out

    lines = [
        r"% Auto-generated by figures_onepass/advection/training_inference_time.py",
        r"% Median over the canonical trajectories (n per beta in the last column).",
        r"% Ours = single-pass EP (expectation propagation); inference: Ours & VI use the deterministic closed form,",
        r"% HMC averages over all 3000 pooled draws (~1.3% predictive-sigma Monte-Carlo error).",
        r"% Figure: panel (a) = the cold-start training times below; panel (b) = the STEADY-STATE inference times",
        r"% (last blocks); the cold-start inference column is kept for completeness (it includes JIT compilation).",
        r"\begin{tabular}{r rrr rrr r}",
        r"\toprule",
        r" & \multicolumn{3}{c}{Training time (s)} & \multicolumn{3}{c}{Inference time (s)} & \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"$b$ & Ours & HMC & VI & Ours & HMC & VI & $n$ \\",
        r"\midrule",
    ]
    tr = {b: t for b, *t in row_times("train")}
    pr = {b: t for b, *t in row_times("predict")}
    for b in BETAS:
        ot, ht, vt = tr[b]
        op, hp, vp = pr[b]
        lines.append(f"{b:g} & {ot:.1f} & {ht:.1f} & {vt:.1f} & {op:.2f} & {hp:.2f} & {vp:.2f} & {n_used[b]} \\\\")
    lines += [
        r"\midrule",
        r"\multicolumn{8}{l}{\emph{Median speedup of Ours} (HMC/Ours, VI/Ours):}\\",
    ]
    for b in BETAS:
        ot, ht, vt = tr[b]; op, hp, vp = pr[b]
        lines.append(
            f"{b:g} & \\multicolumn{{3}}{{c}}{{train: {ht/ot:.1f}$\\times$ HMC, {vt/ot:.1f}$\\times$ VI}} & "
            f"\\multicolumn{{3}}{{c}}{{infer: {hp/op:.1f}$\\times$ HMC, {vp/op:.1f}$\\times$ VI}} & \\\\")
    # Steady-state (compiled) inference: same timed block re-timed after a warm-up call.
    if steady is not None:
        lines += [r"\midrule",
                  r"\multicolumn{8}{l}{\emph{Steady-state (compiled) inference time, s}: same block re-timed from the saved posterior/chains on an idle machine,}\\",
                  r"\multicolumn{8}{l}{min over 7 repeated calls (Ours, VI; measurement order alternated) / 3 (HMC) after a warm-up call; cold-start above includes JIT (first-call compilation):}\\"]
        for b in BETAS:
            so, sh, sv = (_med(steady[m][b]) for m in ("Ours", "HMC", "VI"))
            ns = min(len(steady[m][b]) for m in ("Ours", "HMC", "VI"))
            lines.append(f"{b:g} & \\multicolumn{{3}}{{c}}{{}} & {so:.2f} & {sh:.2f} & {sv:.2f} & {ns} \\\\")
        lines.append(r"\multicolumn{8}{l}{\emph{Steady-state speedup of Ours} (HMC/Ours, VI/Ours):}\\")
        for b in BETAS:
            so, sh, sv = (_med(steady[m][b]) for m in ("Ours", "HMC", "VI"))
            lines.append(f"{b:g} & \\multicolumn{{3}}{{c}}{{}} & \\multicolumn{{3}}{{c}}{{{sh/so:.1f}$\\times$ HMC, {sv/so:.1f}$\\times$ VI}} & \\\\")
    else:
        lines += [r"% steady-state inference columns omitted: " + STEADY_CSV.name + " not found"]

    # Minimal single-chain HMC training variant (MAP + one average chain), for the text.
    # Commas, not `&`: this text sits inside one \multicolumn cell (an `&` would be read as a
    # column break and fail to compile).
    lines += [r"\midrule", r"\multicolumn{8}{l}{\emph{HMC single-chain train (MAP warm start + 1 chain), s:}}\\"]
    sc = ",\\ ".join(f"{b:g}: {_med(hmc_singlechain[b]):.1f}" for b in BETAS)
    lines.append(rf"\multicolumn{{8}}{{l}}{{{sc}}}\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    path.write_text("\n".join(lines))
    print(f"Saved table to {path}")


def main() -> None:
    times, hmc_singlechain = collect_times()
    n_used = {b: len(times["Ours"][b]["train"]) for b in BETAS}
    steady = load_steady_state()
    if steady is None:
        raise FileNotFoundError(f"{STEADY_CSV} is required: panel (b) plots steady-state inference times")
    short = {(m, b): (len(steady[m][b]), n_used[b]) for m in COLORS for b in BETAS if len(steady[m][b]) != n_used[b]}
    if short:
        raise ValueError(f"steady-state times do not cover every plotted run, (method, b): (have, need) = {short}")
    steady_times = {m: {b: {"predict": steady[m][b]} for b in BETAS} for m in COLORS}
    plt.rcParams["mathtext.fontset"] = "cm"
    fig, (ax_train, ax_pred) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    plot_panel(ax_train, times, "train", r"$\mathrm{(a)\ Training\ time}$")
    plot_panel(ax_pred, steady_times, "predict", r"$\mathrm{(b)\ Inference\ time}$")
    handles, labels = ax_train.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=17, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {OUTPUT_PDF} (panel (a) cold-start training, panel (b) steady-state inference)")

    write_table(times, hmc_singlechain, n_used, OUTPUT_TEX, steady)
    if steady is not None:
        print("\nMedian STEADY-STATE (compiled) inference time [s] Ours / HMC / VI:")
        for beta in BETAS:
            so = _med(steady['Ours'][beta])
            print(f"  b={beta}: {so:.2f} / {_med(steady['HMC'][beta]):.2f} / {_med(steady['VI'][beta]):.2f}   "
                  f"(speedup of Ours {_med(steady['HMC'][beta]) / so:.1f}x vs HMC, {_med(steady['VI'][beta]) / so:.1f}x vs VI)")
    else:
        print(f"\n(no {STEADY_CSV.name}; steady-state columns omitted)")

    for key in ("train", "predict"):
        print(f"\nMedian COLD-START {key} time speedups of Ours (n per beta = {n_used}):")
        for beta in BETAS:
            ours = _med(times["Ours"][beta][key])
            print(f"  b={beta}: {_med(times['HMC'][beta][key]) / ours:.1f}x vs HMC, "
                  f"{_med(times['VI'][beta][key]) / ours:.1f}x vs VI")

    print("\nMedian HMC single-chain (MAP + 1 chain) train time per beta:")
    for beta in BETAS:
        print(f"  b={beta}: {_med(hmc_singlechain[beta]):.1f}s "
              f"(vs Ours {_med(times['Ours'][beta]['train']):.1f}s)")


if __name__ == "__main__":
    main()
