"""Distance metrics implemented in pure NumPy.

All functions operate on 2-D float arrays of shape ``(n, d)`` and are
vectorized. ``pairwise_distances`` is the single entry point used by the
rest of the library, so adding a metric here makes it available to every
embedding algorithm.
"""

from __future__ import annotations

import numpy as np

METRICS = ("euclidean", "sqeuclidean", "cosine", "manhattan")


def _as_float2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {X.shape}")
    return X


def pairwise_distances(
    X: np.ndarray, Y: np.ndarray | None = None, metric: str = "euclidean"
) -> np.ndarray:
    """Dense pairwise distance matrix between rows of X and rows of Y.

    Parameters
    ----------
    X : (n, d) array
    Y : (m, d) array or None
        If None, distances are computed between rows of X.
    metric : one of ``simple_maps.distances.METRICS``

    Returns
    -------
    (n, m) array of distances.
    """
    X = _as_float2d(X)
    Y = X if Y is None else _as_float2d(Y)
    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"dimension mismatch: {X.shape[1]} vs {Y.shape[1]}")

    if metric == "sqeuclidean":
        return _sqeuclidean(X, Y)
    if metric == "euclidean":
        return np.sqrt(_sqeuclidean(X, Y))
    if metric == "cosine":
        return _cosine(X, Y)
    if metric == "manhattan":
        return _manhattan(X, Y)
    raise ValueError(f"unknown metric {metric!r}; available: {METRICS}")


def _sqeuclidean(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    # ||x - y||^2 = ||x||^2 - 2 x.y + ||y||^2, clipped at 0 to absorb
    # floating point cancellation for near-identical points.
    x2 = np.einsum("ij,ij->i", X, X)[:, None]
    y2 = np.einsum("ij,ij->i", Y, Y)[None, :]
    d2 = x2 + y2 - 2.0 * (X @ Y.T)
    np.maximum(d2, 0.0, out=d2)
    return d2


def _cosine(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    xn = np.linalg.norm(X, axis=1)
    yn = np.linalg.norm(Y, axis=1)
    xn = np.where(xn == 0.0, 1.0, xn)
    yn = np.where(yn == 0.0, 1.0, yn)
    sim = (X / xn[:, None]) @ (Y / yn[:, None]).T
    d = 1.0 - sim
    np.clip(d, 0.0, 2.0, out=d)
    return d


def _manhattan(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    # Broadcasting materializes an (n, m, d) block; callers that need to
    # scale should go through neighbors.knn, which processes X in blocks.
    return np.abs(X[:, None, :] - Y[None, :, :]).sum(axis=2)
