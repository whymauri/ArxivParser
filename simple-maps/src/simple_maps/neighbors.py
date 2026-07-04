"""Nearest-neighbor search.

v0 ships an exact, block-wise brute-force kNN. It is vectorized, memory
bounded (never materializes more than ``block_size * n`` distances), and
has zero dependencies beyond NumPy. An approximate index (NN-descent or
random-projection forest) is a planned drop-in replacement behind the
same ``knn`` signature for large N.
"""

from __future__ import annotations

import numpy as np

from .distances import _as_float2d, pairwise_distances


def knn(
    X: np.ndarray,
    n_neighbors: int,
    metric: str = "euclidean",
    block_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact k nearest neighbors of every row of X within X (self excluded).

    Returns
    -------
    indices : (n, n_neighbors) int array, sorted from nearest to farthest.
    distances : (n, n_neighbors) float array, aligned with ``indices``.
    """
    X = _as_float2d(X)
    n = X.shape[0]
    if not 1 <= n_neighbors < n:
        raise ValueError(f"n_neighbors must be in [1, {n - 1}], got {n_neighbors}")

    k = n_neighbors
    indices = np.empty((n, k), dtype=np.int64)
    distances = np.empty((n, k), dtype=np.float64)

    for start in range(0, n, block_size):
        stop = min(start + block_size, n)
        d = pairwise_distances(X[start:stop], X, metric=metric)
        # Exclude self-matches by construction rather than assuming the
        # smallest distance is self (duplicate rows break that assumption).
        rows = np.arange(start, stop)
        d[rows - start, rows] = np.inf

        part = np.argpartition(d, k - 1, axis=1)[:, :k]
        part_d = np.take_along_axis(d, part, axis=1)
        order = np.argsort(part_d, axis=1)
        indices[start:stop] = np.take_along_axis(part, order, axis=1)
        distances[start:stop] = np.take_along_axis(part_d, order, axis=1)

    return indices, distances


def knn_query(
    X_train: np.ndarray,
    X_query: np.ndarray,
    n_neighbors: int,
    metric: str = "euclidean",
    block_size: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact kNN of each query row against a fixed training set."""
    X_train = _as_float2d(X_train)
    X_query = _as_float2d(X_query)
    n_train = X_train.shape[0]
    if not 1 <= n_neighbors <= n_train:
        raise ValueError(f"n_neighbors must be in [1, {n_train}], got {n_neighbors}")

    k = n_neighbors
    m = X_query.shape[0]
    indices = np.empty((m, k), dtype=np.int64)
    distances = np.empty((m, k), dtype=np.float64)

    for start in range(0, m, block_size):
        stop = min(start + block_size, m)
        d = pairwise_distances(X_query[start:stop], X_train, metric=metric)
        if k < n_train:
            part = np.argpartition(d, k - 1, axis=1)[:, :k]
        else:
            part = np.broadcast_to(np.arange(n_train), (stop - start, n_train)).copy()
        part_d = np.take_along_axis(d, part, axis=1)
        order = np.argsort(part_d, axis=1)
        indices[start:stop] = np.take_along_axis(part, order, axis=1)
        distances[start:stop] = np.take_along_axis(part_d, order, axis=1)

    return indices, distances
