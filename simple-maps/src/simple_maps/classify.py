"""A learned classifier on top of any simple-maps embedding.

The point of this module is that the production path is trivial:

    clf = KNNClassifier(PaCMAP(random_state=0))
    clf.fit(X, y)
    clf.save("clf.npz")            # one file, no pickle
    ...
    clf = simple_maps.load("clf.npz")
    clf.predict(X_new)             # milliseconds, not seconds

Prediction embeds new points with the (fast) embedding ``transform`` and
takes an inverse-distance-weighted vote among the nearest training points
in embedding space.
"""

from __future__ import annotations

import json

import numpy as np

from ._base import _FORMAT_VERSION, BaseEmbedding
from .neighbors import knn_query


class KNNClassifier:
    """k-nearest-neighbor classifier in a learned embedding space."""

    def __init__(self, embedder: BaseEmbedding, n_neighbors: int = 5):
        self.embedder = embedder
        self.n_neighbors = n_neighbors

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        y = np.asarray(y)
        if y.shape[0] != np.asarray(X).shape[0]:
            raise ValueError("X and y must have the same length")
        self.embedder.fit(X)
        # The vote runs against the training embedding, kept here so it
        # round-trips even for parametric embedders that don't store it.
        self._train_embedding = self.embedder.embedding_
        # Map arbitrary labels to contiguous ints; classes_ restores them.
        self.classes_, self._y_idx = np.unique(y, return_inverse=True)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "classes_"):
            raise RuntimeError("call fit before predict")
        Z = self.embedder.transform(X)
        k = min(self.n_neighbors, self._train_embedding.shape[0])
        idx, dist = knn_query(self._train_embedding, Z, k)
        w = 1.0 / (dist + 1e-12)
        votes = np.zeros((Z.shape[0], len(self.classes_)))
        rows = np.repeat(np.arange(Z.shape[0]), k)
        np.add.at(votes, (rows, self._y_idx[idx].reshape(-1)), w.reshape(-1))
        return votes / votes.sum(axis=1, keepdims=True)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == np.asarray(y)))

    # -- serialization -----------------------------------------------------

    def save(self, path: str) -> None:
        if not hasattr(self, "classes_"):
            raise RuntimeError("nothing to save: classifier is not fitted")
        embedder_state = {
            f"embedder_{key}": val for key, val in self.embedder._state_arrays().items()
        }
        np.savez_compressed(
            path,
            format_version=_FORMAT_VERSION,
            estimator=type(self).__name__,
            config=json.dumps({"n_neighbors": self.n_neighbors}),
            embedder_class=type(self.embedder).__name__,
            embedder_config=json.dumps(self.embedder.get_config()),
            classes=self.classes_,
            y_idx=self._y_idx.astype(np.int64),
            train_embedding=self._train_embedding.astype(np.float32),
            **embedder_state,
        )

    @classmethod
    def _from_npz(cls, data) -> "KNNClassifier":
        from . import _ESTIMATORS

        embedder_cls = _ESTIMATORS[str(data["embedder_class"])]
        embedder = embedder_cls(**json.loads(str(data["embedder_config"])))
        embedder._load_state(data, prefix="embedder_")
        model = cls(embedder, **json.loads(str(data["config"])))
        model.classes_ = data["classes"]
        model._y_idx = data["y_idx"]
        model._train_embedding = data["train_embedding"].astype(np.float64)
        return model
