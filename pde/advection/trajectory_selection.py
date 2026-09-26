"""
Canonical seeded trajectory selection shared by every multi-trajectory advection experiment.

One fixed random draw (seed 0, without replacement) from the 10,000 trajectories per beta
file; scripts take prefixes of it (`idxs(10)` for the Bayesian-comparison set, `idxs(N)` for
the accuracy sweep), so growing N extends the existing sample instead of resampling, and every
figure uses nested subsets of the same trajectories. The development trajectory (idx=-1, the
file's last one, index 9999) is excluded: hyperparameters were chosen while looking at it.
"""

from __future__ import annotations

import numpy as np

N_TRAJ_TOTAL = 10_000
_DEV_IDX = 9_999  # idx=-1 in the runners' convention

_DRAW = np.random.RandomState(0).choice(N_TRAJ_TOTAL, size=200, replace=False)
_DRAW = _DRAW[_DRAW != _DEV_IDX]


def idxs(n: int) -> list[int]:
    """First `n` canonical trajectory indices (order fixed, dev trajectory excluded)."""
    if n > len(_DRAW):
        raise ValueError(f"only {len(_DRAW)} canonical indices pre-drawn, asked for {n}")
    return [int(i) for i in _DRAW[:n]]


if __name__ == "__main__":
    print(idxs(10))
