"""TriMap (Amid & Warmuth, 2019) in pure NumPy.

Large-scale structure via triplet constraints: for anchor i, neighbor j and
outlier k, the loss w_ijk * s_ik / (s_ij + s_ik) with the heavy-tailed
similarity s = 1/(1 + ||y_i - y_j||^2) pushes j closer to i than k, weighted
by how strongly the high-dimensional data supports the triplet.

Deviation from the reference implementation: optimization uses full-batch
Adam rather than delta-bar-delta (simpler, and Adam is what the reference
package recommends for most datasets anyway).
"""

from __future__ import annotations

import numpy as np

from ._base import BaseEmbedding
from ._scatter import scatter_add
from .initialization import pca_init, random_init
from .neighbors import knn
from .optim import Adam


def _scaled_similarities(dists: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-point scale sigma_i (mean distance to neighbors 4-6) and
    similarity exp(-d^2 / (sigma_i * sigma_j)) for the kNN graph."""
    hi = min(6, dists.shape[1])
    lo = min(3, hi - 1)
    sigma = np.maximum(dists[:, lo:hi].mean(axis=1), 1e-10)
    return sigma, dists


def sample_triplets(
    X: np.ndarray,
    n_inliers: int,
    n_outliers: int,
    n_random: int,
    metric: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (anchor, inlier, outlier) triplets and their weights.

    Weights follow the reference recipe: similarity ratios, normalized by
    the max, then log-damped with weight_adj=500.
    """
    n = X.shape[0]
    k = min(n_inliers, n - 2)
    k_extra = min(k + 5, n - 1)  # extra columns so sigma uses neighbors 4-6
    nbr_idx, nbr_d = knn(X, k_extra, metric=metric)
    sigma, _ = _scaled_similarities(nbr_d)

    def sim(i: np.ndarray, j: np.ndarray, d: np.ndarray) -> np.ndarray:
        return np.exp(-(d**2) / (sigma[i] * sigma[j]))

    anchors = np.repeat(np.arange(n, dtype=np.int64), k)
    inliers = nbr_idx[:, :k].reshape(-1)
    p_in = sim(anchors, inliers, nbr_d[:, :k].reshape(-1))

    # kNN triplets: for each (anchor, inlier) edge, n_outliers random points.
    anchors_t = np.repeat(anchors, n_outliers)
    inliers_t = np.repeat(inliers, n_outliers)
    p_in_t = np.repeat(p_in, n_outliers)
    outliers_t = rng.integers(0, n, size=anchors_t.shape[0])
    # Re-draw collisions with the anchor or the inlier once; anything left
    # is dropped rather than looped on.
    bad = (outliers_t == anchors_t) | (outliers_t == inliers_t)
    outliers_t[bad] = rng.integers(0, n, size=int(bad.sum()))
    ok = (outliers_t != anchors_t) & (outliers_t != inliers_t)
    anchors_t, inliers_t, p_in_t, outliers_t = (
        anchors_t[ok],
        inliers_t[ok],
        p_in_t[ok],
        outliers_t[ok],
    )

    d_out = np.linalg.norm(X[anchors_t] - X[outliers_t], axis=1)
    p_out = sim(anchors_t, outliers_t, d_out)

    # If a sampled "outlier" is actually closer than the inlier, swap roles.
    swap = p_out > p_in_t
    inliers_t[swap], outliers_t[swap] = outliers_t[swap], inliers_t[swap]
    p_in_t[swap], p_out[swap] = p_out[swap], p_in_t[swap]

    weights = p_in_t / np.maximum(p_out, 1e-20)

    # Random triplets for global structure.
    if n_random > 0:
        ra = np.repeat(np.arange(n, dtype=np.int64), n_random)
        rj = rng.integers(0, n, size=ra.shape[0])
        rk = rng.integers(0, n, size=ra.shape[0])
        ok = (rj != ra) & (rk != ra) & (rj != rk)
        ra, rj, rk = ra[ok], rj[ok], rk[ok]
        dj = np.linalg.norm(X[ra] - X[rj], axis=1)
        dk = np.linalg.norm(X[ra] - X[rk], axis=1)
        pj = sim(ra, rj, dj)
        pk = sim(ra, rk, dk)
        swap = pk > pj
        rj[swap], rk[swap] = rk[swap], rj[swap]
        pj[swap], pk[swap] = pk[swap], pj[swap]
        w_r = pj / np.maximum(pk, 1e-20)

        anchors_t = np.concatenate([anchors_t, ra])
        inliers_t = np.concatenate([inliers_t, rj])
        outliers_t = np.concatenate([outliers_t, rk])
        weights = np.concatenate([weights, w_r])

    weights /= weights.max()
    weights = np.log(1.0 + 500.0 * weights)
    triplets = np.stack([anchors_t, inliers_t, outliers_t], axis=1)
    return triplets, weights


def _triplet_grad(
    Y: np.ndarray, triplets: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, float]:
    i, j, k = triplets[:, 0], triplets[:, 1], triplets[:, 2]
    yij = Y[i] - Y[j]
    yik = Y[i] - Y[k]
    s_ij = 1.0 / (1.0 + np.einsum("ij,ij->i", yij, yij))
    s_ik = 1.0 / (1.0 + np.einsum("ij,ij->i", yik, yik))
    denom = s_ij + s_ik
    loss = float(np.sum(weights * s_ik / denom))

    # d loss / d ||y_i - y_j||^2 and / d ||y_i - y_k||^2
    dl_du = weights * s_ik * s_ij**2 / denom**2
    dl_dv = -weights * s_ij * s_ik**2 / denom**2

    g_ij = 2.0 * dl_du[:, None] * yij
    g_ik = 2.0 * dl_dv[:, None] * yik

    grad = np.zeros_like(Y)
    scatter_add(
        grad,
        np.concatenate([i, j, k]),
        np.concatenate([g_ij + g_ik, -g_ij, -g_ik]),
    )
    return grad, loss


class TriMap(BaseEmbedding):
    """Triplet-based manifold embedding."""

    _config_keys = (
        "n_components",
        "n_inliers",
        "n_outliers",
        "n_random",
        "metric",
        "n_iters",
        "learning_rate",
        "init",
        "random_state",
    )

    def __init__(
        self,
        n_components: int = 2,
        n_inliers: int = 12,
        n_outliers: int = 4,
        n_random: int = 3,
        metric: str = "euclidean",
        n_iters: int = 400,
        learning_rate: float = 0.1,
        init: str = "pca",
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.n_inliers = n_inliers
        self.n_outliers = n_outliers
        self.n_random = n_random
        self.metric = metric
        self.n_iters = n_iters
        self.learning_rate = learning_rate
        self.init = init
        self.random_state = random_state

    def _fit_embedding(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        triplets, weights = sample_triplets(
            X, self.n_inliers, self.n_outliers, self.n_random, self.metric, rng
        )

        if self.init == "pca":
            Y = pca_init(X, self.n_components) * 0.01
        elif self.init == "random":
            Y = random_init(X.shape[0], self.n_components, rng, scale=0.01)
        else:
            raise ValueError(f"unknown init {self.init!r}")

        opt = Adam(Y.shape, lr=self.learning_rate)
        for _ in range(self.n_iters):
            grad, _ = _triplet_grad(Y, triplets, weights)
            Y += opt.step(grad)
        return Y
