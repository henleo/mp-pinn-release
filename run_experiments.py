"""
Regenerates every artifact behind the figures in figures_onepass/advection and
figures_onepass/diffusion, then the figures themselves.

Every Julia run is one process `julia --project=<repo> <runner.jl> <args...> <artifact.h5>` with
OPENBLAS_NUM_THREADS=1 (eigenvector signs of the shared basis depend on the BLAS threading, and the
figures compare posteriors coordinate by coordinate across methods). Runs are dispatched by
pde/advection/julia_jobs.py: at most --jobs processes at once (at most --big-jobs of them at the
memory-heavy wave speed b = 7), each writing to a temporary file that is renamed to the artifact on
success, with its console output in artifacts/<artifact>.runlog. Existing artifacts are skipped, so
an interrupted step can simply be restarted.

Steps, in the order `all` runs them:
  references    float64 reference solutions of the Fisher-KPP problems (needs the reacdiff data)
  adv-ours      one-pass message passing, advection (b x 10 trajectories, plus trajectory 6369)
  adv-gd        gradient-descent PINN baseline (Adam; "SGD" in the paper), advection
  adv-hmc       multichain HMC reference, advection
  adv-vi        mean-field VI baseline, advection
  adv-wide      one-pass message passing with H = 2, 3, 4 hidden units, advection
  adv-ablation  subgraph ablation (prior only / initial condition only / residual only), advection
  adv-epsilon   PDE-noise (epsilon) sweep, advection
  diff-ours     one-pass message passing, Fisher-KPP (16 settings x 5 trajectories)
  diff-gd       gradient-descent PINN baseline (Adam; "SGD" in the paper), Fisher-KPP
  diff-hmc      multichain HMC reference, Fisher-KPP (needs diff-ours: one MAP start is its mean)
  diff-vi       mean-field VI baseline, Fisher-KPP
  diff-wide     one-pass message passing with H = 2, 3, 4 hidden units, Fisher-KPP
  diff-gauss    one-pass message passing on the Gaussian-bump initial condition (no data needed)
  timing        steady-state inference re-timing -> figures_onepass/*/training_inference_time_steady_state.csv
                (after all training steps, on an otherwise idle machine)
  figures       runs every figure script in figures_onepass/advection and figures_onepass/diffusion,
                then the main results table (figures_onepass/main_results_table.py)

Usage: python run_experiments.py <step> [<step> ...] [--jobs N] [--big-jobs M] [--dry-run]
       python run_experiments.py all --jobs 4 --big-jobs 2
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
ADV = REPO_ROOT / "pde" / "advection"
DIFF = REPO_ROOT / "pde" / "diffusion"
sys.path.insert(0, str(ADV))
sys.path.insert(0, str(DIFF))

import bpinn_multichain_sweep as hmc_sweep  # noqa: E402
import gd_pinn_baseline_sweep as gd_sweep  # noqa: E402
import mp_pinn_beta_sweep as mp_sweep  # noqa: E402
import mp_pinn_epsilon_sweep as eps_sweep  # noqa: E402
import mp_pinn_subgraph_ablation_sweep as abl_sweep  # noqa: E402
import mp_pinn_wide_sweep as wide_sweep  # noqa: E402
import reacdiff_baselines_sweep as rd_sweep  # noqa: E402
import reacdiff_reference  # noqa: E402
import vi_baseline_sweep as vi_sweep  # noqa: E402
from julia_jobs import run_jobs, temp_path  # noqa: E402
from trajectory_selection import idxs  # noqa: E402

ARTIFACTS = REPO_ROOT / "artifacts"
JULIA = ["julia", "--heap-size-hint=2200M", f"--project={REPO_ROOT}"]
JULIA_VERSION = "1.12.6"                    # the version Manifest.toml was resolved with

BETAS = [0.1, 0.4, 1.0, 2.0, 7.0]           # advection wave speeds
BIG_BETA = 7.0
ADV_IDXS = idxs(10)                         # ten canonical advection trajectories
GRID_IDX = 6369                             # the extra trajectory of the advection solution-grid figure
NUS = [0.5, 1.0, 2.0, 5.0]                  # Fisher-KPP diffusion coefficients
RHOS = [1.0, 2.0, 5.0, 10.0]                # Fisher-KPP reaction strengths
DIFF_IDXS = idxs(5)                         # five canonical Fisher-KPP trajectories
DIFF_CELLS = [(nu, rho) for nu in NUS for rho in RHOS]

# Fisher-KPP wide models: H = 2 on every run; H = 3 and H = 4 on the subsets reported in the paper
# (the first trajectory on every setting, the second on every setting (H = 3) or on nu = 0.5,
# rho <= 5 (H = 4), plus one further run for H = 3).
DIFF_WIDE_RUNS = (
    [(2, nu, rho, i) for i in DIFF_IDXS for nu, rho in DIFF_CELLS]
    + [(3, nu, rho, i) for i in DIFF_IDXS[:2] for nu, rho in DIFF_CELLS] + [(3, 0.5, 1.0, DIFF_IDXS[2])]
    + [(4, nu, rho, DIFF_IDXS[0]) for nu, rho in DIFF_CELLS] + [(4, 0.5, rho, DIFF_IDXS[1]) for rho in (1.0, 2.0, 5.0)]
)


def gauss_path(nu: float, rho: float) -> Path:
    return ARTIFACTS / f"mp_pinn_reacdiff_gauss_nu{rd_sweep._fmt(nu)}_rho{rd_sweep._fmt(rho)}.h5"


def julia_run(runner: Path, *args) -> callable:
    """Command builder for a runner whose argument list is `args` with the output path at the
    position of the placeholder None."""
    def command(item, out: Path) -> list[str]:
        return JULIA + [str(runner)] + [str(out) if a is None else str(a) for a in args]
    return command


# Each Julia step: a list of (label, artifact path, command builder(out path), is_big).
def adv_step(runner: str, path_fn, idx_list=ADV_IDXS):
    return [(f"b={b:g} idx={i}", path_fn(b, i), julia_run(ADV / runner, b, i, None), b == BIG_BETA)
            for b in BETAS for i in idx_list]


def diff_step(runner: str, method: str):
    return [(f"nu={nu:g} rho={rho:g} idx={i}", rd_sweep.cache_path(method, nu, rho, i),
             julia_run(DIFF / runner, nu, rho, i, None), False)
            for nu, rho in DIFF_CELLS for i in DIFF_IDXS]


def julia_steps() -> dict[str, list]:
    return {
        "adv-ours": adv_step("mp_pinn_run_beta.jl", mp_sweep._cache_path, ADV_IDXS + [GRID_IDX]),
        "adv-gd": adv_step("mp_pinn_gd_baseline_run_beta.jl", gd_sweep._cache_path),
        "adv-hmc": adv_step("bpinn_run_beta_multichain.jl", hmc_sweep._cache_path),
        "adv-vi": adv_step("vi_run_beta.jl", vi_sweep._cache_path),
        "adv-wide": [(f"H={H} b={b:g} idx={i}", wide_sweep._cache_path(b, i, H),
                      julia_run(ADV / "mp_pinn_run_beta_wide.jl", b, i, None, H), b == BIG_BETA)
                     for H in (2, 3, 4) for b in BETAS for i in ADV_IDXS],
        "adv-ablation": [(f"b={b:g} idx={i}", abl_sweep.cache_path(b, i),
                          julia_run(ADV / "mp_pinn_subgraph_ablation.jl", b, i, None, ",".join(abl_sweep.RUN_REGIMES)),
                          b == BIG_BETA)
                         for b in abl_sweep.BETAS for i in abl_sweep.IDXS],
        "adv-epsilon": [(f"b={b:g} eps={e:g} idx={i}", eps_sweep.cache_path(b, e, i),
                         julia_run(ADV / "mp_pinn_run_beta.jl", b, i, None, eps_sweep.SIGMA_W, eps_sweep.BETA_IC, e),
                         b == BIG_BETA)
                        for b in eps_sweep.BETAS for e in eps_sweep.EPSILONS for i in eps_sweep.IDXS
                        if e != eps_sweep.DEFAULT_EPS],  # the default epsilon is the adv-ours artifact
        "diff-ours": diff_step("mp_pinn_reacdiff_dataset_run.jl", "Ours"),
        "diff-gd": diff_step("gd_run_reacdiff.jl", "GD"),
        "diff-hmc": diff_step("bpinn_run_reacdiff_multichain.jl", "HMC"),
        "diff-vi": diff_step("vi_run_reacdiff.jl", "VI"),
        "diff-wide": [(f"H={H} nu={nu:g} rho={rho:g} idx={i}", rd_sweep.cache_path("wide", nu, rho, i, H),
                       julia_run(DIFF / "mp_pinn_reacdiff_wide_run.jl", nu, rho, i, None, H), False)
                      for H, nu, rho, i in DIFF_WIDE_RUNS],
        "diff-gauss": [(f"nu={nu:g} rho={rho:g}", gauss_path(nu, rho),
                        julia_run(DIFF / "mp_pinn_reacdiff_gaussian_run.jl", nu, rho, None), False)
                       for nu, rho in DIFF_CELLS],
    }


def check_julia() -> None:
    """Every Julia process started from here uses the juliaup channel JULIA_VERSION (juliaup reads
    JULIAUP_CHANNEL; a plain Julia binary ignores it). Warns if a different Julia version runs."""
    os.environ.setdefault("JULIAUP_CHANNEL", JULIA_VERSION)
    result = subprocess.run(JULIA[:1] + ["--startup-file=no", "-e", "print(VERSION)"], cwd=REPO_ROOT,
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"cannot start Julia {JULIA_VERSION}:\n{result.stderr.strip()}\n"
                         f"(with juliaup: juliaup add {JULIA_VERSION})")
    if result.stdout.strip() != JULIA_VERSION:
        print(f"WARNING: Julia {result.stdout.strip()} runs, but Manifest.toml was resolved with Julia {JULIA_VERSION}",
              flush=True)


def run_julia_step(name: str, runs: list, args) -> None:
    by_label = {label: (path, command, big) for label, path, command, big in runs}
    todo = [label for label, path, _, _ in runs if not path.exists()]
    print(f"== {name}: {len(runs)} runs, {len(todo)} missing", flush=True)
    if args.dry_run:
        for label in todo:
            path, command, _ = by_label[label]
            print("  OPENBLAS_NUM_THREADS=1 " + shlex.join(command(label, temp_path(path))) + f"   # -> {path.name}")
        return
    failed = run_jobs(todo, path_for=lambda lb: by_label[lb][0], command_for=lambda lb, out: by_label[lb][1](lb, out),
                      is_big=lambda lb: by_label[lb][2], label_for=lambda lb: lb, cwd=REPO_ROOT,
                      jobs=args.jobs, big_jobs=args.big_jobs)
    if failed:
        raise SystemExit(f"{name}: {len(failed)} runs failed (see the .runlog files in artifacts/)")


def check_prerequisites(name: str) -> None:
    if name == "diff-hmc":
        # One of the two MAP starts of the Fisher-KPP HMC reference is the one-pass posterior mean;
        # without it the runner silently falls back to the start from zero.
        missing = [p.name for nu, rho in DIFF_CELLS for i in DIFF_IDXS
                   if not (p := rd_sweep.cache_path("Ours", nu, rho, i)).exists()]
        if missing:
            raise SystemExit(f"diff-hmc needs the diff-ours artifacts first; {len(missing)} missing, e.g. {missing[0]}")
    if name == "diff-gauss":
        for nu, rho in DIFF_CELLS:
            reacdiff_reference.gaussian_bump_reference(nu, rho)


def check_hmc_map_starts() -> None:
    skipped = [p.name for nu, rho in DIFF_CELLS for i in DIFF_IDXS
               if (p := rd_sweep.cache_path("HMC", nu, rho, i).with_suffix(".runlog")).exists()
               and "MAP start (ii) skipped" in p.read_text(errors="replace")]
    if skipped:
        raise SystemExit(f"diff-hmc: the one-pass MAP start was skipped in {len(skipped)} runs, e.g. {skipped[0]}")


def references(args) -> None:
    print(f"== references: {len(DIFF_CELLS) * len(DIFF_IDXS)} dataset trajectories + {len(DIFF_CELLS)} Gaussian-bump", flush=True)
    if args.dry_run:
        return
    for nu, rho in DIFF_CELLS:
        for i in DIFF_IDXS:
            reacdiff_reference.reference_solution(nu, rho, i)
        reacdiff_reference.gaussian_bump_reference(nu, rho)


def timing(args) -> None:
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1")
    tmp = ARTIFACTS / "timing"
    tmp.mkdir(parents=True, exist_ok=True)
    adv_dir = ADV / "diagnostics" / "steady_state_timing"
    closed_form, hmc, reacdiff = tmp / "warm_closed_form.csv", tmp / "warm_hmc.csv", tmp / "warm_reacdiff.csv"
    adv_merged = tmp / "advection_steady_state.csv"
    adv_csv = REPO_ROOT / "figures_onepass" / "advection" / "training_inference_time_steady_state.csv"
    diff_csv = REPO_ROOT / "figures_onepass" / "diffusion" / "training_inference_time_steady_state.csv"
    commands = [
        JULIA + [str(adv_dir / "warm_retime_closed_form_v3.jl"), str(closed_form)],
        JULIA + [str(adv_dir / "warm_retime_hmc_v2.jl"), str(hmc)],
        JULIA + [str(DIFF / "diagnostics" / "steady_state_timing" / "warm_retime_reacdiff.jl"), str(reacdiff)],
    ]
    print("== timing (run this on an otherwise idle machine)", flush=True)
    for command in commands:
        print("  OPENBLAS_NUM_THREADS=1 " + shlex.join(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True, env=env, cwd=REPO_ROOT)
    if args.dry_run:
        return
    merge_advection_timing(closed_form, hmc, adv_merged)
    # The figures read these CSVs; they are replaced only by complete results.
    for src, dst, n_expected in ((adv_merged, adv_csv, len(BETAS) * len(ADV_IDXS)),
                                 (reacdiff, diff_csv, len(DIFF_CELLS) * len(DIFF_IDXS))):
        counts = timing_counts(src)
        if any(n != n_expected for n in counts.values()):
            raise SystemExit(f"{src.name}: {counts}, expected {n_expected} rows per family -- are all training steps complete?")
        os.replace(src, dst)
        print(f"  wrote {dst.relative_to(REPO_ROOT)}: {counts}", flush=True)


def merge_advection_timing(closed_form: Path, hmc: Path, out: Path) -> None:
    """The closed-form rows (Ours, VI) of the min-of-7 script, then the HMC rows of the min-of-3 script."""
    with open(closed_form, newline="") as f:
        rows = list(csv.reader(f))
    with open(hmc, newline="") as f:
        rows += [r for r in list(csv.reader(f))[1:] if r[1] == "hmc"]
    with open(out, "w", newline="") as f:
        csv.writer(f).writerows(rows)


def timing_counts(path: Path) -> dict[str, int]:
    with open(path, newline="") as f:
        families = [r["family"] for r in csv.DictReader(f)]
    return {fam: families.count(fam) for fam in ("onepass", "vi", "hmc")}


SUBSET_SWITCHES = ("REACDIFF_CELLS", "REACDIFF_WIDTHS", "WIDER_MODEL_BETAS", "WIDER_MODEL_PNG", "WIDER_MODEL_ALL_PANELS")

# Alphabetical order runs mp_vs_gd_scatter_grid.py before mp_vs_gd_scatter_grid_mse.py, which compares
# its numbers with that figure's PDF; the main results table reads the same runs as the figures.
FIGURE_SCRIPTS = sorted((REPO_ROOT / "figures_onepass" / "advection").glob("*.py")) + \
    sorted((REPO_ROOT / "figures_onepass" / "diffusion").glob("*.py")) + \
    [REPO_ROOT / "figures_onepass" / "main_results_table.py"]


def missing_outputs() -> dict[str, int]:
    """Number of missing outputs per step, for the steps whose outputs the figure scripts read."""
    missing = {}
    refs = [reacdiff_reference._cache_path(nu, rho, i) for nu, rho in DIFF_CELLS for i in DIFF_IDXS] + \
        [reacdiff_reference._gauss_cache_path(nu, rho) for nu, rho in DIFF_CELLS]
    missing["references"] = sum(not p.exists() for p in refs)
    for name, runs in julia_steps().items():
        missing[name] = sum(not path.exists() for _, path, _, _ in runs)
    missing["timing"] = sum(not (REPO_ROOT / "figures_onepass" / d / "training_inference_time_steady_state.csv").exists()
                            for d in ("advection", "diffusion"))
    return {step: n for step, n in missing.items() if n}


def figures(args) -> None:
    # The *_ALL_PANELS switches also write the additional panel sets (calibration_comparison.pdf,
    # variance_consistency.pdf, space_analysis_all_panels.pdf) next to the main figures. Every
    # script's console output is saved as <script>.stdout.log next to it.
    env = dict(os.environ, MPLBACKEND="Agg", OPENBLAS_NUM_THREADS="1",
               CALIBRATION_ALL_PANELS="1", SUBGRAPH_ALL_PANELS="1")
    for key in SUBSET_SWITCHES:   # switches that make single scripts plot only a subset of the runs
        env.pop(key, None)
    print(f"== figures: {len(FIGURE_SCRIPTS)} scripts", flush=True)
    missing = missing_outputs()
    if missing and not args.dry_run:
        raise SystemExit("figures: outputs of earlier steps are missing (" +
                         ", ".join(f"{step}: {n}" for step, n in missing.items()) +
                         "); run first: python run_experiments.py " + " ".join(missing))
    failed = []
    for script in FIGURE_SCRIPTS:
        print(f"  {script.relative_to(REPO_ROOT)}", flush=True)
        if args.dry_run:
            continue
        with open(script.with_suffix(".stdout.log"), "w") as log:
            result = subprocess.run([sys.executable, str(script)], env=env, cwd=REPO_ROOT,
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            failed.append(script.relative_to(REPO_ROOT))
            print(f"    FAILED (see {script.with_suffix('.stdout.log').relative_to(REPO_ROOT)})", flush=True)
    if failed:
        raise SystemExit(f"figures: {len(failed)} scripts failed: {', '.join(map(str, failed))}")


ORDER = ["references", "adv-ours", "adv-gd", "adv-hmc", "adv-vi", "adv-wide", "adv-ablation", "adv-epsilon",
         "diff-ours", "diff-gd", "diff-hmc", "diff-vi", "diff-wide", "diff-gauss", "timing", "figures"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("steps", nargs="+", choices=ORDER + ["all"], metavar="step", help="see the list below, or 'all'")
    ap.add_argument("--jobs", type=int, default=4, help="concurrent Julia processes (default 4)")
    ap.add_argument("--big-jobs", type=int, default=2, help="of which at most this many at b = 7 (default 2)")
    ap.add_argument("--dry-run", action="store_true", help="print what would run, run nothing")
    args = ap.parse_args()
    steps = ORDER if "all" in args.steps else [s for s in ORDER if s in args.steps]
    # SIGTERM (e.g. `kill <pid>`) exits through SystemExit, so running Julia processes are stopped too.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(128 + signum))
    if not args.dry_run and any(s not in ("references", "figures") for s in steps):
        check_julia()
    julia = julia_steps()
    for name in steps:
        if name == "references":
            references(args)
        elif name == "timing":
            timing(args)
        elif name == "figures":
            figures(args)
        else:
            if not args.dry_run:
                check_prerequisites(name)
            run_julia_step(name, julia[name], args)
            if name == "diff-hmc" and not args.dry_run:
                check_hmc_map_starts()


if __name__ == "__main__":
    main()
