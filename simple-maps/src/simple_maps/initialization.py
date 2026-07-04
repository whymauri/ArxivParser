"""Embedding initializations."""

from __future__ import annotations

import numpy as np


def pca_init(X: np.ndarray, n_components: int) -> np.ndarray:
    """Project X onto its top principal components (economy SVD, no deps).

    Output is normalized to unit standard deviation per component so each
    algorithm can apply its own preferred scale.
    """
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=0)
    if X.shape[1] > X.shape[0]:
        # Wide data: eigendecompose the (n, n) Gram matrix instead.
        gram = Xc @ Xc.T
        vals, vecs = np.linalg.eigh(gram)
        order = np.argsort(vals)[::-1][:n_components]
        Y = vecs[:, order] * np.sqrt(np.maximum(vals[order], 1e-12))
    else:
        U, S, _ = np.linalg.svd(Xc, full_matrices=False)
        Y = U[:, :n_components] * S[:n_components]
    std = Y.std(axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return Y / std


def random_init(
    n: int, n_components: int, rng: np.random.Generator, scale: float = 10.0
) -> np.ndarray:
    return rng.uniform(-scale, scale, size=(n, n_components))
