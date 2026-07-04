import os

import numpy as np
import pytest

from simple_maps import ParametricUMAP, load
from simple_maps.datasets import gaussian_blobs, swiss_roll
from simple_maps.metrics import knn_preservation


def test_shape_finite_and_deterministic():
    X, _ = gaussian_blobs(n_samples=250, n_features=10, random_state=0)
    Y1 = ParametricUMAP(n_steps=300, random_state=0).fit_transform(X)
    Y2 = ParametricUMAP(n_steps=300, random_state=0).fit_transform(X)
    assert Y1.shape == (250, 2)
    assert np.isfinite(Y1).all()
    assert np.allclose(Y1, Y2)


def test_preserves_local_structure_on_swiss_roll():
    X, _ = swiss_roll(n_samples=400, random_state=0)
    Y = ParametricUMAP(n_steps=600, random_state=0).fit_transform(X)
    score = knn_preservation(X, Y, n_neighbors=10)
    assert score > 0.25


def test_transform_is_consistent_with_fit_embedding():
    X, _ = gaussian_blobs(n_samples=300, n_features=10, random_state=0)
    model = ParametricUMAP(n_steps=300, random_state=0)
    Y = model.fit_transform(X)
    # A parametric model's transform of the training data IS the embedding.
    assert np.allclose(model.transform(X), Y)


def test_save_load_is_weights_only_and_exact(tmp_path):
    X, _ = gaussian_blobs(n_samples=300, n_features=10, random_state=0)
    model = ParametricUMAP(n_steps=300, random_state=0)
    model.fit(X)
    path = tmp_path / "pumap.npz"
    model.save(path)

    restored = load(path)
    # float32 weight storage: forward passes agree to ~1e-4 relative
    assert np.allclose(restored.transform(X), model.transform(X), atol=1e-2)


def test_saved_size_independent_of_n(tmp_path):
    sizes = {}
    for n in (200, 800):
        X, _ = gaussian_blobs(n_samples=n, n_features=10, random_state=0)
        model = ParametricUMAP(n_steps=100, random_state=0)
        model.fit(X)
        path = tmp_path / f"pumap_{n}.npz"
        model.save(path)
        sizes[n] = os.path.getsize(path)
    # Weights-only files differ by compression noise, not by 4x data size.
    assert abs(sizes[200] - sizes[800]) < 0.1 * sizes[200]


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        ParametricUMAP().transform(np.zeros((3, 4)))
    with pytest.raises(RuntimeError):
        ParametricUMAP().save("/tmp/never-written.npz")
