"""Parametric manifold embeddings: MLP encoders trained on the UMAP,
TriMap, and PaCMAP objectives.

Instead of optimizing embedding coordinates directly, a small ReLU MLP
(`d -> hidden -> hidden -> n_components`) maps data to embedding space and
the loss gradients are backpropagated through it — manual backprop, still
NumPy-only.

Why parametric matters here:
- ``transform`` is a forward pass: no training data, no kNN query, no
  per-query optimization. Inference cost is O(d * hidden) per point.
- ``save`` stores just the network weights and input scaling — the file
  size is independent of the training set size.

All three share the training scheme in ``ParametricEmbedding``: each step
samples a mini-batch of loss terms (graph edges, triplets, or pairs) and
only the rows that batch touches are forward/backpropagated, so step cost
is bounded by ``batch_size`` regardless of n. Subclasses supply the loss:
``_prepare`` builds the loss structure from the data, ``_sample_batch``
draws index groups for one step, and ``_batch_grad`` computes dL/dY for
the touched rows.
"""

from __future__ import annotations

import numpy as np

from ._base import BaseEmbedding
from ._scatter import scatter_add
from .neighbors import knn
from .optim import Adam

_WEIGHT_KEYS = ("W1", "b1", "W2", "b2", "W3", "b3")


class ParametricEmbedding(BaseEmbedding):
    """Shared encoder network, training loop, and weights-only serialization."""

    def __init__(
        self,
        n_components: int,
        metric: str,
        hidden_dim: int,
        n_steps: int,
        batch_size: int,
        learning_rate: float,
        random_state: int | None,
    ):
        self.n_components = n_components
        self.metric = metric
        self.hidden_dim = hidden_dim
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.random_state = random_state

    # -- loss hooks (implemented by subclasses) ------------------------------

    def _prepare(self, X: np.ndarray, rng: np.random.Generator) -> None:
        """Build the loss structure (graph/triplets/pairs) from the data."""
        raise NotImplementedError

    def _sample_batch(self, rng: np.random.Generator) -> tuple[tuple, object]:
        """Draw one step's loss terms: (groups of global row indices, aux)."""
        raise NotImplementedError

    def _batch_grad(self, Y: np.ndarray, groups: tuple, aux, step: int) -> np.ndarray:
        """dL/dY for the touched rows; ``groups`` hold batch-local indices."""
        raise NotImplementedError

    # -- network -------------------------------------------------------------

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
        self._mu = X.mean(axis=0)
        sigma = X.std(axis=0)
        self._sigma = np.where(sigma == 0.0, 1.0, sigma)
        Xn = (X - self._mu) / self._sigma

        self._prepare(X, rng)
        self._weights = self._init_weights(X.shape[1], rng)
        opts = {key: Adam(w.shape, lr=self.learning_rate) for key, w in self._weights.items()}

        for step in range(self.n_steps):
            groups, aux = self._sample_batch(rng)

            # Forward only the rows this batch touches; hand the loss
            # batch-local indices into that row block.
            flat = np.concatenate([g.reshape(-1) for g in groups])
            rows, inv = np.unique(flat, return_inverse=True)
            local, offset = [], 0
            for g in groups:
                local.append(inv[offset : offset + g.size].reshape(g.shape))
                offset += g.size

            Y, cache = self._forward(Xn[rows])
            dY = self._batch_grad(Y, tuple(local), aux, step)
            grads = self._backward(dY, cache)
            for key, gval in grads.items():
                self._weights[key] += opts[key].step(gval)

        return self._forward(Xn)[0]

    # -- inference -------------------------------------------------------

    def transform(self, X: np.ndarray, n_neighbors: int | None = None) -> np.ndarray:
        self._check_fitted()
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


class ParametricUMAP(ParametricEmbedding):
    """UMAP cross-entropy with negative sampling through the encoder."""

    _config_keys = (
        "n_neighbors",
        "n_components",
        "metric",
        "min_dist",
        "spread",
        "hidden_dim",
        "n_steps",
        "batch_size",
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
        batch_size: int = 2048,
        learning_rate: float = 1e-3,
        negative_sample_rate: int = 5,
        random_state: int | None = None,
    ):
        super().__init__(
            n_components, metric, hidden_dim, n_steps, batch_size, learning_rate, random_state
        )
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.spread = spread
        self.negative_sample_rate = negative_sample_rate

    def _prepare(self, X: np.ndarray, rng: np.random.Generator) -> None:
        from .umap_ import find_ab_params, fuzzy_simplicial_set

        n = X.shape[0]
        k = min(self.n_neighbors, n - 1)
        knn_indices, knn_dists = knn(X, k, metric=self.metric)
        head, tail, weights = fuzzy_simplicial_set(knn_indices, knn_dists)
        # Sample edges uniformly, weight gradients by membership strength.
        self._head, self._tail = head, tail
        self._edge_w = weights / weights.max()
        self._n = n
        self._a, self._b = find_ab_params(self.spread, self.min_dist)

    def _sample_batch(self, rng: np.random.Generator):
        m = self._head.shape[0]
        e = rng.integers(0, m, size=min(self.batch_size, m))
        neg = rng.integers(0, self._n, size=(e.shape[0], self.negative_sample_rate))
        return (self._head[e], self._tail[e], neg), self._edge_w[e]

    def _batch_grad(self, Y: np.ndarray, groups: tuple, aux, step: int) -> np.ndarray:
        hi, ti, ni = groups
        w_e = aux
        a, b = self._a, self._b
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
        return dY


class ParametricTriMap(ParametricEmbedding):
    """TriMap's weighted triplet ratio loss through the encoder."""

    _config_keys = (
        "n_components",
        "n_inliers",
        "n_outliers",
        "n_random",
        "metric",
        "hidden_dim",
        "n_steps",
        "batch_size",
        "learning_rate",
        "random_state",
    )

    def __init__(
        self,
        n_components: int = 2,
        n_inliers: int = 12,
        n_outliers: int = 4,
        n_random: int = 3,
        metric: str = "euclidean",
        hidden_dim: int = 64,
        n_steps: int = 1000,
        batch_size: int = 2048,
        learning_rate: float = 1e-3,
        random_state: int | None = None,
    ):
        super().__init__(
            n_components, metric, hidden_dim, n_steps, batch_size, learning_rate, random_state
        )
        self.n_inliers = n_inliers
        self.n_outliers = n_outliers
        self.n_random = n_random

    def _prepare(self, X: np.ndarray, rng: np.random.Generator) -> None:
        from .trimap_ import sample_triplets

        self._triplets, self._trip_w = sample_triplets(
            X, self.n_inliers, self.n_outliers, self.n_random, self.metric, rng
        )

    def _sample_batch(self, rng: np.random.Generator):
        m = self._triplets.shape[0]
        e = rng.integers(0, m, size=min(self.batch_size, m))
        t = self._triplets[e]
        return (t[:, 0], t[:, 1], t[:, 2]), self._trip_w[e]

    def _batch_grad(self, Y: np.ndarray, groups: tuple, aux, step: int) -> np.ndarray:
        i, j, k = groups
        w = aux
        # Same gradient as trimap_._triplet_grad, scattered batch-locally.
        yij = Y[i] - Y[j]
        yik = Y[i] - Y[k]
        s_ij = 1.0 / (1.0 + np.einsum("ij,ij->i", yij, yij))
        s_ik = 1.0 / (1.0 + np.einsum("ij,ij->i", yik, yik))
        denom = s_ij + s_ik
        dl_du = w * s_ik * s_ij**2 / denom**2
        dl_dv = -w * s_ij * s_ik**2 / denom**2
        g_ij = 2.0 * dl_du[:, None] * yij
        g_ik = 2.0 * dl_dv[:, None] * yik

        dY = np.zeros_like(Y)
        scatter_add(
            dY,
            np.concatenate([i, j, k]),
            np.concatenate([g_ij + g_ik, -g_ij, -g_ik]),
        )
        return dY


class ParametricPaCMAP(ParametricEmbedding):
    """PaCMAP's three pair losses and phase schedule through the encoder."""

    _config_keys = (
        "n_components",
        "n_neighbors",
        "mn_ratio",
        "fp_ratio",
        "metric",
        "hidden_dim",
        "n_steps",
        "batch_size",
        "learning_rate",
        "random_state",
    )

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
        mn_ratio: float = 0.5,
        fp_ratio: float = 2.0,
        metric: str = "euclidean",
        hidden_dim: int = 64,
        n_steps: int = 1000,
        batch_size: int = 4096,
        learning_rate: float = 1e-3,
        random_state: int | None = None,
    ):
        super().__init__(
            n_components, metric, hidden_dim, n_steps, batch_size, learning_rate, random_state
        )
        self.n_neighbors = n_neighbors
        self.mn_ratio = mn_ratio
        self.fp_ratio = fp_ratio

    def _prepare(self, X: np.ndarray, rng: np.random.Generator) -> None:
        from .pacmap_ import sample_pairs

        self._pairs = sample_pairs(
            X, self.n_neighbors, self.mn_ratio, self.fp_ratio, self.metric, rng
        )
        # Paper's 100/100/250 phase split, scaled to n_steps.
        self._p1 = int(round(self.n_steps * 100 / 450))
        self._p2 = int(round(self.n_steps * 100 / 450))

    def _sample_batch(self, rng: np.random.Generator):
        # Sub-batch each pair set proportionally to its size.
        total = sum(p.shape[0] for p in self._pairs)
        groups = []
        for pairs in self._pairs:
            m = pairs.shape[0]
            size = min(m, max(1, round(self.batch_size * m / total)))
            e = rng.integers(0, m, size=size)
            groups.extend([pairs[e, 0], pairs[e, 1]])
        return tuple(groups), None

    def _batch_grad(self, Y: np.ndarray, groups: tuple, aux, step: int) -> np.ndarray:
        from .pacmap_ import _pair_grad_attract, _pair_grad_repel, _phase_weights

        nb_i, nb_j, mn_i, mn_j, fp_i, fp_j = groups
        w_nb, w_mn, w_fp = _phase_weights(step, self._p1, self._p2)

        dY = np.zeros_like(Y)
        _pair_grad_attract(Y, np.stack([nb_i, nb_j], axis=1), 10.0, w_nb, dY)
        if w_mn > 0.0:
            _pair_grad_attract(Y, np.stack([mn_i, mn_j], axis=1), 10000.0, w_mn, dY)
        _pair_grad_repel(Y, np.stack([fp_i, fp_j], axis=1), w_fp, dY)
        return dY
