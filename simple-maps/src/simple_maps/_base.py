"""Shared estimator machinery: fit/transform API and save/load.

Design goals (see repo spec):
- inference (``transform``) must be fast: v0 embeds new points as a
  distance-weighted average of the embeddings of their k nearest training
  points, which is a single blocked kNN query — no per-query optimization
  loop like reference UMAP.
- save/load must be low-dependency and opinionated: a single compressed
  ``.npz`` holding the config, the training data, and the embedding.
  Parametric models will serialize weights instead of training data under
  the same interface.
"""

from __future__ import annotations

import json

import numpy as np

from .neighbors import knn_query

_FORMAT_VERSION = 1


class BaseEmbedding:
    """Base class for the non-parametric embeddings (UMAP/TriMap/PaCMAP).

    Subclasses implement ``_fit_embedding(X, rng) -> (n, n_components)``
    and declare their constructor arguments in ``_config_keys``.
    """

    _config_keys: tuple[str, ...] = ()

    n_components: int
    metric: str
    random_state: int | None

    def _fit_embedding(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    # -- fitting -----------------------------------------------------------

    def fit(self, X: np.ndarray) -> "BaseEmbedding":
        self.fit_transform(X)
        return self

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"expected 2-D data, got shape {X.shape}")
        rng = np.random.default_rng(self.random_state)
        self._X_train = X
        self.embedding_ = self._fit_embedding(X, rng)
        return self.embedding_

    # -- inference ---------------------------------------------------------

    def transform(self, X: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
        """Embed new points via inverse-distance weighted kNN interpolation."""
        if not hasattr(self, "embedding_"):
            raise RuntimeError("call fit or fit_transform before transform")
        X = np.asarray(X, dtype=np.float64)
        k = min(n_neighbors, self._X_train.shape[0])
        idx, dist = knn_query(self._X_train, X, k, metric=self.metric)
        w = 1.0 / (dist + 1e-12)
        w /= w.sum(axis=1, keepdims=True)
        return np.einsum("ijk,ij->ik", self.embedding_[idx], w)

    # -- serialization -----------------------------------------------------

    def get_config(self) -> dict:
        return {key: getattr(self, key) for key in self._config_keys}

    def _check_fitted(self) -> None:
        if not hasattr(self, "embedding_"):
            raise RuntimeError("model is not fitted")

    def _state_arrays(self) -> dict:
        """Arrays that must survive a save/load round trip.

        Non-parametric models need the training data (transform is kNN
        interpolation against it); parametric models override this to
        store weights instead.
        """
        self._check_fitted()
        return {
            "X_train": self._X_train.astype(np.float32),
            "embedding": self.embedding_.astype(np.float32),
        }

    def _load_state(self, data, prefix: str = "") -> None:
        self._X_train = data[prefix + "X_train"].astype(np.float64)
        self.embedding_ = data[prefix + "embedding"].astype(np.float64)

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            format_version=_FORMAT_VERSION,
            estimator=type(self).__name__,
            config=json.dumps(self.get_config()),
            **self._state_arrays(),
        )

    @classmethod
    def _from_npz(cls, data) -> "BaseEmbedding":
        model = cls(**json.loads(str(data["config"])))
        model._load_state(data)
        return model


def load(path: str):
    """Load any fitted simple-maps model saved with ``model.save``."""
    from . import _ESTIMATORS

    with np.load(path, allow_pickle=False) as data:
        name = str(data["estimator"])
        if name not in _ESTIMATORS:
            raise ValueError(f"unknown estimator {name!r} in {path}")
        return _ESTIMATORS[name]._from_npz(data)
