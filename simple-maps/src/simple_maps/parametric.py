"""Parametric UMAP: an MLP encoder trained on the UMAP objective.

This is the first parametric formulation from the spec's roadmap. Instead
of optimizing embedding coordinates directly, a small ReLU MLP maps data
to embedding space and the UMAP cross-entropy gradients are backpropagated
through it (manual backprop — still NumPy-only).

Why parametric matters here:
- ``transform`` is a forward pass: no training data, no kNN query, no
  per-query optimization. Inference cost is O(d * hidden) per point.
- ``save`` stores just the network weights and input scaling — the file
  size is independent of the training set size.

Training samples mini-batches of graph edges per step; only the rows
touched by a batch are forward/backpropagated, so step cost is bounded by
``batch_edges`` regardless of n.
"""

from __future__ import annotations

import numpy as np

from ._base import BaseEmbedding
from .neighbors import knn
from .umap_ import find_ab_params, fuzzy_simplicial_set

_WEIGHT_KEYS = ("W1", "b1", "W2", "b2", "W3", "b3")


class ParametricUMAP(BaseEmbedding):
    """UMAP with a learned encoder network (d -> hidden -> hidden -> n_components)."""

    _config_keys = (
        "n_neighbors",
        "n_components",
        "metric",
        "min_dist",
        "spread",
        "hidden_dim",
        "n_steps",
        "batch_edges",
        "learning_rate",
        "negative_sample_rate",
        "random_state",
    )

    def __init__(
        self,
        n_neighbors: int = 15,
        n_components: int = 2,
        metric: str = "euclidean",
        min_dist: float = 0.1,
        spread: float = 1.0,
        hidden_dim: int = 64,
        n_steps: int = 1000,
        batch_edges: int = 2048,
        learning_rate: float = 1e-3,
        negative_sample_rate: int = 5,
        random_state: int | None = None,
    ):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.metric = metric
        self.min_dist = min_dist
        self.spread = spread
        self.hidden_dim = hidden_dim
        self.n_steps = n_steps
        self.batch_edges = batch_edges
        self.learning_rate = learning_rate
        self.negative_sample_rate = negative_sample_rate
        self.random_state = random_state

    # -- network -------------------------------------------------------

    def _init_weights(self, d_in: int, rng: np.random.Generator) -> dict:
        h, d_out = self.hidden_dim, self.n_components
        he = lambda fan_in: np.sqrt(2.0 / fan_in)
        return {
            "W1": rng.normal(scale=he(d_in), size=(d_in, h)),
            "b1": np.zeros(h),
            "W2": rng.normal(scale=he(h), size=(h, h)),
            "b2": np.zeros(h),
            "W3": rng.normal(scale=he(h), size=(h, d_out)),
            "b3": np.zeros(d_out),
        }

    def _forward(self, Xn: np.ndarray):
        p = self._weights
        z1 = Xn @ p["W1"] + p["b1"]
        a1 = np.maximum(z1, 0.0)
        z2 = a1 @ p["W2"] + p["b2"]
        a2 = np.maximum(z2, 0.0)
        Y = a2 @ p["W3"] + p["b3"]
        return Y, (Xn, z1, a1, z2, a2)

    def _backward(self, dY: np.ndarray, cache) -> dict:
        p = self._weights
        Xn, z1, a1, z2, a2 = cache
        dW3 = a2.T @ dY
        db3 = dY.sum(axis=0)
        dz2 = (dY @ p["W3"].T) * (z2 > 0.0)
        dW2 = a1.T @ dz2
        db2 = dz2.sum(axis=0)
        dz1 = (dz2 @ p["W2"].T) * (z1 > 0.0)
        dW1 = Xn.T @ dz1
        db1 = dz1.sum(axis=0)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}

    # -- training --------------------------------------------------------

    def _fit_embedding(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        from ._scatter import scatter_add
        from .optim import Adam

        n = X.shape[0]
        self._mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        self._sigma = np.where(sigma == 0.0, 1.0, sigma)
        Xn = (X - self._mu) / self._sigma

        k = min(self.n_neighbors, n - 1)
        knn_indices, knn_dists = knn(X, k, metric=self.metric)
        head, tail, weights = fuzzy_simplicial_set(knn_indices, knn_dists)
        # Sample edges uniformly, weight gradients by membership strength.
        weights = weights / weights.max()
        a, b = find_ab_params(self.spread, self.min_dist)

        self._weights = self._init_weights(X.shape[1], rng)
        opts = {key: Adam(w.shape, lr=self.learning_rate) for key, w in self._weights.items()}

        m = head.shape[0]
        nsr = self.negative_sample_rate
        for _ in range(self.n_steps):
            e = rng.integers(0, m, size=min(self.batch_edges, m))
            h, t, w_e = head[e], tail[e], weights[e]
            neg = rng.integers(0, n, size=(h.shape[0], nsr))

            # Forward only the rows this batch touches.
            rows, inv = np.unique(
                np.concatenate([h, t, neg.reshape(-1)]), return_inverse=True
            )
            Y, cache = self._forward(Xn[rows])
            hi = inv[: h.shape[0]]
            ti = inv[h.shape[0] : 2 * h.shape[0]]
            ni = inv[2 * h.shape[0] :].reshape(neg.shape)

            dY = np.zeros_like(Y)

            # Attractive term of the cross entropy, weighted by membership.
            diff = Y[hi] - Y[ti]
            d2 = np.maximum(np.einsum("ij,ij->i", diff, diff), 1e-12)
            coeff = (2.0 * a * b * d2 ** (b - 1.0)) / (a * d2**b + 1.0) * w_e
            g = np.clip(coeff[:, None] * diff, -4.0, 4.0)
            scatter_add(dY, np.concatenate([hi, ti]), np.concatenate([g, -g]))

            # Repulsive term against uniform negatives (head side only).
            diff_n = Y[hi][:, None, :] - Y[ni]
            d2n = np.maximum(np.einsum("ijk,ijk->ij", diff_n, diff_n), 1e-12)
            coeff_n = (2.0 * b) / ((0.001 + d2n) * (a * d2n**b + 1.0))
            g_n = np.clip(coeff_n[:, :, None] * diff_n, -4.0, 4.0).sum(axis=1)
            scatter_add(dY, hi, -g_n)

            grads = self._backward(dY, cache)
            for key, gval in grads.items():
                self._weights[key] += opts[key].step(gval)

        return self._forward(Xn)[0]

    # -- inference ---------------------------------------------------------

    def transform(self, X: np.ndarray, n_neighbors: int | None = None) -> np.ndarray:
        if not hasattr(self, "_weights"):
            raise RuntimeError("call fit or fit_transform before transform")
        X = np.asarray(X, dtype=np.float64)
        return self._forward((X - self._mu) / self._sigma)[0]

    # -- serialization -----------------------------------------------------

    def _check_fitted(self) -> None:
        if not hasattr(self, "_weights"):
            raise RuntimeError("model is not fitted")

    def _state_arrays(self) -> dict:
        """Weights-only serialization: file size is independent of n."""
        self._check_fitted()
        state = {"mu": self._mu, "sigma": self._sigma}
        state.update({key: self._weights[key].astype(np.float32) for key in _WEIGHT_KEYS})
        return state

    def _load_state(self, data, prefix: str = "") -> None:
        self._mu = data[prefix + "mu"].astype(np.float64)
        self._sigma = data[prefix + "sigma"].astype(np.float64)
        self._weights = {
            key: data[prefix + key].astype(np.float64) for key in _WEIGHT_KEYS
        }
