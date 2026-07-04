"""UMAP (McInnes, Healy & Melville, 2018) in pure NumPy.

Faithful to the reference algorithm: fuzzy simplicial set construction via
per-point bandwidth calibration, probabilistic set union symmetrization,
and stochastic gradient descent on the cross-entropy with the smooth
``1 / (1 + a * d^(2b))`` low-dimensional kernel and negative sampling.

The optimizer is vectorized per epoch (edge masks + ``np.add.at`` scatter)
instead of umap-learn's numba per-edge loop.
"""

from __future__ import annotations

import numpy as np

from ._base import BaseEmbedding
from ._scatter import scatter_add
from .initialization import pca_init, random_init
from .neighbors import knn

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3


def find_ab_params(spread: float = 1.0, min_dist: float = 0.1) -> tuple[float, float]:
    """Fit (a, b) of the low-dim kernel 1/(1 + a d^(2b)) to the target curve.

    The target is 1 for d < min_dist and exp(-(d - min_dist)/spread) beyond,
    matching umap-learn. Fitting is plain Adam on log-parameters (no scipy).
    """
    x = np.linspace(0.0, spread * 3.0, 300)
    y = np.where(x < min_dist, 1.0, np.exp(-(x - min_dist) / spread))

    log_a, log_b = 0.0, 0.0
    m = np.zeros(2)
    v = np.zeros(2)
    for t in range(1, 3001):
        a, b = np.exp(log_a), np.exp(log_b)
        xa = a * x ** (2.0 * b)
        f = 1.0 / (1.0 + xa)
        r = f - y
        # d f / d log_a = -f^2 * xa ;  d f / d log_b = -f^2 * xa * 2b log x
        with np.errstate(divide="ignore"):
            logx = np.where(x > 0.0, np.log(x), 0.0)
        g = np.array(
            [
                np.sum(2.0 * r * (-(f**2)) * xa),
                np.sum(2.0 * r * (-(f**2)) * xa * 2.0 * b * logx),
            ]
        )
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        mh = m / (1.0 - 0.9**t)
        vh = v / (1.0 - 0.999**t)
        step = 0.01 * mh / (np.sqrt(vh) + 1e-8)
        log_a -= step[0]
        log_b -= step[1]
    return float(np.exp(log_a)), float(np.exp(log_b))


def smooth_knn_dist(
    knn_dists: np.ndarray, n_neighbors: int, n_iter: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Per-point (rho, sigma) so that sum_j exp(-(d_ij - rho_i)/sigma_i) = log2(k).

    Vectorized binary search across all points at once.
    """
    n = knn_dists.shape[0]
    target = np.log2(n_neighbors)

    rho = knn_dists[:, 0].copy()
    lo = np.zeros(n)
    hi = np.full(n, np.inf)
    mid = np.ones(n)

    adjusted = np.maximum(knn_dists - rho[:, None], 0.0)
    for _ in range(n_iter):
        psum = np.exp(-adjusted / mid[:, None]).sum(axis=1)
        err = psum - target
        done = np.abs(err) < SMOOTH_K_TOLERANCE
        too_high = (err > 0.0) & ~done
        too_low = (err < 0.0) & ~done

        hi = np.where(too_high, mid, hi)
        lo = np.where(too_low, mid, lo)
        mid = np.where(too_high, (lo + hi) / 2.0, mid)
        mid = np.where(too_low & np.isinf(hi), mid * 2.0, mid)
        mid = np.where(too_low & ~np.isinf(hi), (lo + hi) / 2.0, mid)
        if done.all():
            break

    sigma = mid
    # Guard against degenerate bandwidths for points in dense duplicates.
    mean_d = knn_dists.mean()
    mean_row = knn_dists.mean(axis=1)
    floor = np.where(rho > 0.0, MIN_K_DIST_SCALE * mean_row, MIN_K_DIST_SCALE * mean_d)
    sigma = np.maximum(sigma, floor)
    return rho, sigma


def fuzzy_simplicial_set(
    knn_indices: np.ndarray, knn_dists: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Symmetrized fuzzy graph as directed COO arrays (head, tail, weight).

    Symmetrization is the probabilistic t-conorm P + P^T - P o P^T. As in
    umap-learn, both directions of every undirected edge are kept: the
    optimizer applies repulsion from the head of each edge, so each
    endpoint must appear as a head.
    """
    n, k = knn_indices.shape
    rho, sigma = smooth_knn_dist(knn_dists, k)

    vals = np.exp(-np.maximum(knn_dists - rho[:, None], 0.0) / sigma[:, None])
    rows = np.repeat(np.arange(n, dtype=np.int64), k)
    cols = knn_indices.reshape(-1)
    vals = vals.reshape(-1)

    # Look up the transposed entry for each directed edge via sorted keys.
    keys = rows * n + cols
    order = np.argsort(keys)
    sorted_keys = keys[order]
    sorted_vals = vals[order]
    t_keys = cols * n + rows
    pos = np.searchsorted(sorted_keys, t_keys)
    pos = np.clip(pos, 0, len(sorted_keys) - 1)
    t_vals = np.where(sorted_keys[pos] == t_keys, sorted_vals[pos], 0.0)

    sym = vals + t_vals - vals * t_vals

    # Support of the symmetric matrix: union of both edge directions,
    # deduplicated where i in kNN(j) and j in kNN(i) overlap.
    all_keys = np.concatenate([keys, t_keys])
    all_heads = np.concatenate([rows, cols])
    all_tails = np.concatenate([cols, rows])
    all_sym = np.concatenate([sym, sym])
    _, uniq_idx = np.unique(all_keys, return_index=True)
    return all_heads[uniq_idx], all_tails[uniq_idx], all_sym[uniq_idx]


def _optimize_layout(
    Y: np.ndarray,
    head: np.ndarray,
    tail: np.ndarray,
    weights: np.ndarray,
    n_epochs: int,
    a: float,
    b: float,
    learning_rate: float,
    negative_sample_rate: int,
    rng: np.random.Generator,
) -> np.ndarray:
    n = Y.shape[0]
    # Strongest edge is sampled every epoch; edge e every (w_max / w_e) epochs.
    weights = weights / weights.max()
    epochs_per_sample = 1.0 / np.maximum(weights, 1e-12)
    next_sample = epochs_per_sample.copy()

    for epoch in range(n_epochs):
        alpha = learning_rate * (1.0 - epoch / n_epochs)
        active = next_sample <= (epoch + 1.0)
        if not active.any():
            continue
        h = head[active]
        t = tail[active]
        next_sample[active] += epochs_per_sample[active]

        # Attractive updates along sampled edges; both endpoints move.
        diff = Y[h] - Y[t]
        d2 = np.maximum(np.einsum("ij,ij->i", diff, diff), 1e-12)
        coeff = (-2.0 * a * b * d2 ** (b - 1.0)) / (a * d2**b + 1.0)
        grad = np.clip(coeff[:, None] * diff, -4.0, 4.0) * alpha
        scatter_add(Y, np.concatenate([h, t]), np.concatenate([grad, -grad]))

        # Repulsive updates against uniform negative samples; head only.
        neg = rng.integers(0, n, size=(h.shape[0], negative_sample_rate))
        diff_n = Y[h][:, None, :] - Y[neg]
        d2n = np.maximum(np.einsum("ijk,ijk->ij", diff_n, diff_n), 1e-12)
        coeff_n = (2.0 * b) / ((0.001 + d2n) * (a * d2n**b + 1.0))
        grad_n = np.clip(coeff_n[:, :, None] * diff_n, -4.0, 4.0).sum(axis=1) * alpha
        scatter_add(Y, h, grad_n)

    return Y


class UMAP(BaseEmbedding):
    """Uniform Manifold Approximation and Projection.

    Parameters mirror umap-learn's core knobs. ``init`` is "pca" (default,
    deterministic and usually faster to converge) or "random".
    """

    _config_keys = (
        "n_neighbors",
        "n_components",
        "metric",
        "min_dist",
        "spread",
        "n_epochs",
        "learning_rate",
        "negative_sample_rate",
        "init",
        "random_state",
    )

    def __init__(
        self,
        n_neighbors: int = 15,
        n_components: int = 2,
        metric: str = "euclidean",
        min_dist: float = 0.1,
        spread: float = 1.0,
        n_epochs: int | None = None,
        learning_rate: float = 1.0,
        negative_sample_rate: int = 5,
        init: str = "pca",
        random_state: int | None = None,
    ):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.metric = metric
        self.min_dist = min_dist
        self.spread = spread
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.negative_sample_rate = negative_sample_rate
        self.init = init
        self.random_state = random_state

    def _fit_embedding(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = X.shape[0]
        # umap-learn's heuristic: more epochs pay off on small datasets.
        n_epochs = self.n_epochs if self.n_epochs is not None else (500 if n < 10000 else 200)
        k = min(self.n_neighbors, n - 1)
        knn_indices, knn_dists = knn(X, k, metric=self.metric)
        head, tail, weights = fuzzy_simplicial_set(knn_indices, knn_dists)

        # Drop negligible edges, as umap-learn does relative to n_epochs.
        keep = weights >= weights.max() / max(n_epochs, 1)
        head, tail, weights = head[keep], tail[keep], weights[keep]

        if self.init == "pca":
            Y = pca_init(X, self.n_components) * 10.0
            Y += rng.normal(scale=1e-4, size=Y.shape)
        elif self.init == "random":
            Y = random_init(n, self.n_components, rng)
        else:
            raise ValueError(f"unknown init {self.init!r}")

        a, b = find_ab_params(self.spread, self.min_dist)
        return _optimize_layout(
            np.ascontiguousarray(Y),
            head,
            tail,
            weights,
            n_epochs,
            a,
            b,
            self.learning_rate,
            self.negative_sample_rate,
            rng,
        )
