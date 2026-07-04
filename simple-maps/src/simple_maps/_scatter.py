"""Vectorized scatter-add.

``np.add.at`` handles duplicate indices correctly but is notoriously slow
(it dispatches per element). A per-column ``np.bincount`` computes the
same sum 5-10x faster and is the hot path of every optimizer here.
"""

from __future__ import annotations

import numpy as np


def scatter_add(out: np.ndarray, idx: np.ndarray, vals: np.ndarray) -> None:
    """out[idx] += vals with duplicate-index accumulation, in place.

    Parameters
    ----------
    out : (n, d) float array
    idx : (m,) int array, values in [0, n)
    vals : (m, d) float array
    """
    n = out.shape[0]
    for c in range(out.shape[1]):
        out[:, c] += np.bincount(idx, weights=vals[:, c], minlength=n)
