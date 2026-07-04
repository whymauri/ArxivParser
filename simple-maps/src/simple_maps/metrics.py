"""Embedding quality metrics used by tests and benchmarks."""

from __future__ import annotations

import numpy as np

from .neighbors import knn


def knn_preservation(
    X: np.ndarray, Y: np.ndarray, n_neighbors: int = 10, metric: str = "euclidean"
) -> float:
    """Fraction of each point's high-dim kNN preserved in the embedding.

    1.0 means the embedding keeps every k-neighborhood intact; a random
    embedding scores about k / (n - 1).
    """
    idx_x, _ = knn(X, n_neighbors, metric=metric)
    idx_y, _ = knn(Y, n_neighbors, metric="euclidean")
    overlap = 0
    for a, b in zip(idx_x, idx_y):
        overlap += len(np.intersect1d(a, b, assume_unique=True))
    return overlap / idx_x.size
