"""Tiny synthetic datasets for tests and benchmarks (no sklearn needed)."""

from __future__ import annotations

import numpy as np


def gaussian_blobs(
    n_samples: int = 600,
    n_features: int = 10,
    n_clusters: int = 3,
    cluster_std: float = 1.0,
    center_scale: float = 10.0,
    random_state: int | None = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Isotropic Gaussian clusters. Returns (X, labels)."""
    rng = np.random.default_rng(random_state)
    centers = rng.normal(scale=center_scale, size=(n_clusters, n_features))
    labels = np.repeat(np.arange(n_clusters), n_samples // n_clusters)
    labels = np.concatenate([labels, rng.integers(0, n_clusters, n_samples - len(labels))])
    X = centers[labels] + rng.normal(scale=cluster_std, size=(n_samples, n_features))
    return X, labels


def swiss_roll(
    n_samples: int = 1000, noise: float = 0.05, random_state: int | None = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Classic swiss roll in 3-D. Returns (X, unrolled coordinate t)."""
    rng = np.random.default_rng(random_state)
    t = 1.5 * np.pi * (1.0 + 2.0 * rng.random(n_samples))
    y = 21.0 * rng.random(n_samples)
    X = np.stack([t * np.cos(t), y, t * np.sin(t)], axis=1)
    X += rng.normal(scale=noise, size=X.shape)
    return X, t
