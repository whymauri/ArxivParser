"""PaCMAP (Wang, Huang, Rudin & Shaposhnik, 2021) in pure NumPy.

Three pair sets — neighbors (local structure), mid-near pairs (global
structure), and further points (repulsion) — optimized with Adam under the
paper's three-phase weight schedule. The whole gradient is a handful of
vectorized gathers/scatters, so this maps almost 1:1 onto a GPU backend
later.
"""

from __future__ import annotations

import numpy as np

from ._base import BaseEmbedding
from ._scatter import scatter_add
from .initialization import pca_init, random_init
from .neighbors import knn
from .optim import Adam


def _phase_weights(it: int, n_iters_phase1: int, n_iters_phase2: int) -> tuple[float, float, float]:
    """(w_neighbors, w_mid_near, w_further) for iteration ``it``."""
    if it < n_iters_phase1:
        t = it / n_iters_phase1
        return 2.0, 1000.0 * (1.0 - t) + 3.0 * t, 1.0
    if it < n_iters_phase1 + n_iters_phase2:
        return 3.0, 3.0, 1.0
    return 1.0, 0.0, 1.0


def sample_pairs(
    X: np.ndarray,
    n_neighbors: int,
    mn_ratio: float,
    fp_ratio: float,
    metric: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Neighbor, mid-near, and further pairs, each as an (m, 2) int array."""
    n = X.shape[0]
    k = min(n_neighbors, n - 2)

    # Neighbor pairs are the k nearest by *scaled* distance d^2/(sig_i sig_j),
    # selected among the k + 50 nearest euclidean candidates.
    k_cand = min(k + 50, n - 1)
    cand_idx, cand_d = knn(X, k_cand, metric=metric)
    hi = min(6, k_cand)
    lo = min(3, hi - 1)
    sig = np.maximum(cand_d[:, lo:hi].mean(axis=1), 1e-10)
    scaled = cand_d**2 / (sig[:, None] * sig[cand_idx])
    order = np.argsort(scaled, axis=1)[:, :k]
    nbr = np.take_along_axis(cand_idx, order, axis=1)
    pair_nb = np.stack(
        [np.repeat(np.arange(n, dtype=np.int64), k), nbr.reshape(-1)], axis=1
    )

    # Mid-near pairs: sample 6 random points, keep the second closest.
    n_mn = max(int(round(k * mn_ratio)), 1)
    anchors = np.repeat(np.arange(n, dtype=np.int64), n_mn)
    cand = rng.integers(0, n, size=(anchors.shape[0], 6))
    d = np.linalg.norm(X[anchors][:, None, :] - X[cand], axis=2)
    d[cand == anchors[:, None]] = np.inf
    second = np.argsort(d, axis=1)[:, 1]
    pair_mn = np.stack([anchors, cand[np.arange(anchors.shape[0]), second]], axis=1)

    # Further pairs: uniform random non-self points.
    n_fp = max(int(round(k * fp_ratio)), 1)
    anchors = np.repeat(np.arange(n, dtype=np.int64), n_fp)
    far = rng.integers(0, n, size=anchors.shape[0])
    clash = far == anchors
    far[clash] = (far[clash] + 1) % n
    pair_fp = np.stack([anchors, far], axis=1)

    return pair_nb, pair_mn, pair_fp


def _pair_grad_attract(
    Y: np.ndarray, pairs: np.ndarray, denom_const: float, weight: float, grad: np.ndarray
) -> None:
    """Accumulate gradient of w * d~/(c + d~), d~ = 1 + ||y_i - y_j||^2."""
    i, j = pairs[:, 0], pairs[:, 1]
    diff = Y[i] - Y[j]
    d_tilde = 1.0 + np.einsum("ij,ij->i", diff, diff)
    coeff = weight * 2.0 * denom_const / (denom_const + d_tilde) ** 2
    g = coeff[:, None] * diff
    scatter_add(grad, np.concatenate([i, j]), np.concatenate([g, -g]))


def _pair_grad_repel(
    Y: np.ndarray, pairs: np.ndarray, weight: float, grad: np.ndarray
) -> None:
    """Accumulate gradient of w * 1/(1 + d~), d~ = 1 + ||y_i - y_j||^2."""
    i, j = pairs[:, 0], pairs[:, 1]
    diff = Y[i] - Y[j]
    d_tilde = 1.0 + np.einsum("ij,ij->i", diff, diff)
    coeff = -weight * 2.0 / (1.0 + d_tilde) ** 2
    g = coeff[:, None] * diff
    scatter_add(grad, np.concatenate([i, j]), np.concatenate([g, -g]))


class PaCMAP(BaseEmbedding):
    """Pairwise Controlled Manifold Approximation Projection."""

    _config_keys = (
        "n_components",
        "n_neighbors",
        "mn_ratio",
        "fp_ratio",
        "metric",
        "n_iters",
        "learning_rate",
        "init",
        "random_state",
    )

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
        mn_ratio: float = 0.5,
        fp_ratio: float = 2.0,
        metric: str = "euclidean",
        n_iters: int = 450,
        learning_rate: float = 1.0,
        init: str = "pca",
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.mn_ratio = mn_ratio
        self.fp_ratio = fp_ratio
        self.metric = metric
        self.n_iters = n_iters
        self.learning_rate = learning_rate
        self.init = init
        self.random_state = random_state

    def _fit_embedding(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        pair_nb, pair_mn, pair_fp = sample_pairs(
            X, self.n_neighbors, self.mn_ratio, self.fp_ratio, self.metric, rng
        )

        if self.init == "pca":
            Y = pca_init(X, self.n_components) * 0.01
        elif self.init == "random":
            Y = random_init(X.shape[0], self.n_components, rng, scale=0.01)
        else:
            raise ValueError(f"unknown init {self.init!r}")

        # Phase lengths follow the paper's 100/100/250 split, scaled to n_iters.
        p1 = int(round(self.n_iters * 100 / 450))
        p2 = int(round(self.n_iters * 100 / 450))

        opt = Adam(Y.shape, lr=self.learning_rate)
        for it in range(self.n_iters):
            w_nb, w_mn, w_fp = _phase_weights(it, p1, p2)
            grad = np.zeros_like(Y)
            _pair_grad_attract(Y, pair_nb, 10.0, w_nb, grad)
            if w_mn > 0.0:
                _pair_grad_attract(Y, pair_mn, 10000.0, w_mn, grad)
            _pair_grad_repel(Y, pair_fp, w_fp, grad)
            Y += opt.step(grad)
        return Y
