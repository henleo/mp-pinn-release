"""
A small job dispatcher for batches of Julia runner calls (used by the epsilon-sweep and the
subgraph-ablation drivers and by run_experiments.py): at most `jobs` processes at once, of which
at most `big_jobs` are "big" (the memory-heavy b = 7 runs). A free slot takes a big run only while fewer than
`big_jobs` of them are running, and a small one otherwise, so the slots never idle while big
runs wait. Every run writes to a temporary path that is renamed to the artifact only when the
process exits with 0; its console output goes to `<artifact>.runlog`. Single-threaded BLAS for
every run (the signs of the LAPACK eigenvectors that define the basis coordinates can depend on
BLAS threading). A run whose artifact appeared in the meantime is skipped at launch.

The limits can be changed while a batch runs: if `limits_file` exists, it is re-read every few
seconds as JSON {"jobs": N, "big_jobs": M} (running processes are never stopped; a lower limit
only delays new launches).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Callable, Sequence


def temp_path(path: Path) -> Path:
    return path.with_name(path.stem + ".tmp.h5")


def run_jobs(todo: Sequence, *, path_for: Callable[[object], Path], command_for: Callable[[object, Path], list[str]],
             is_big: Callable[[object], bool], label_for: Callable[[object], str], cwd: Path,
             jobs: int, big_jobs: int, limits_file: Path | None = None) -> list:
    """Runs every item of `todo` (big ones in their given order first, then the small ones);
    returns the failed items."""

    def read_limits(jobs: int, big_jobs: int) -> tuple[int, int]:
        if limits_file is None or not limits_file.exists():
            return jobs, big_jobs
        try:
            lim = json.loads(limits_file.read_text())
            return int(lim.get("jobs", jobs)), int(lim.get("big_jobs", big_jobs))
        except (ValueError, OSError):
            return jobs, big_jobs

    big_queue = [r for r in todo if is_big(r)]
    small_queue = [r for r in todo if not is_big(r)]
    print(f"{len(todo)} runs to do ({len(big_queue)} big); jobs={jobs}, big-jobs={big_jobs}", flush=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1")
    running, failed, n_done, n_skipped = [], [], 0, 0

    def launch(item):
        path = path_for(item)
        tmp = temp_path(path)
        cmd = command_for(item, tmp)
        fh = open(path.with_suffix(".runlog"), "w")
        fh.write(" ".join(cmd) + "\n")
        fh.flush()
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=cwd)
        running.append(dict(item=item, proc=proc, fh=fh, tmp=tmp, path=path, start=time.time()))

    while big_queue or small_queue or running:
        for job in [j for j in running if j["proc"].poll() is not None]:
            running.remove(job)
            job["fh"].close()
            if job["proc"].returncode == 0 and job["tmp"].exists():
                job["tmp"].rename(job["path"])
                status = "ok"
            else:
                failed.append(job["item"])
                status = f"FAILED (exit {job['proc'].returncode}, see {job['path'].with_suffix('.runlog')})"
            n_done += 1
            print(f"[{n_done}/{len(todo)}] {label_for(job['item'])}: {status} ({time.time() - job['start']:.0f} s)",
                  flush=True)
        new_limits = read_limits(jobs, big_jobs)
        if new_limits != (jobs, big_jobs):
            jobs, big_jobs = new_limits
            print(f"limits now jobs={jobs}, big-jobs={big_jobs}", flush=True)
        while len(running) < jobs:
            n_big = sum(is_big(j["item"]) for j in running)
            if big_queue and n_big < big_jobs:
                item = big_queue.pop(0)
            elif small_queue:
                item = small_queue.pop(0)
            else:
                break
            if path_for(item).exists():
                n_skipped += 1
                print(f"skip {label_for(item)}: artifact already exists", flush=True)
                continue
            launch(item)
        time.sleep(2)
    print(f"finished: {len(todo) - len(failed) - n_skipped} ok, {n_skipped} skipped, {len(failed)} failed",
          flush=True)
    for item in failed:
        print(f"  failed: {label_for(item)}")
    return failed
