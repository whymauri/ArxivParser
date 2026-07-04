import os

import numpy as np
import pytest

from simple_maps import ParametricPaCMAP, ParametricTriMap, ParametricUMAP, load
from simple_maps.datasets import gaussian_blobs, swiss_roll
from simple_maps.metrics import knn_preservation

PARAMETRIC = [ParametricUMAP, ParametricTriMap, ParametricPaCMAP]
IDS = ["ParametricUMAP", "ParametricTriMap", "ParametricPaCMAP"]


@pytest.mark.parametrize("cls", PARAMETRIC, ids=IDS)
def test_shape_finite_and_deterministic(cls):
    X, _ = gaussian_blobs(n_samples=250, n_features=10, random_state=0)
    Y1 = cls(n_steps=300, random_state=0).fit_transform(X)
    Y2 = cls(n_steps=300, random_state=0).fit_transform(X)
    assert Y1.shape == (250, 2)
    assert np.isfinite(Y1).all()
    assert np.allclose(Y1, Y2)


@pytest.mark.parametrize("cls", PARAMETRIC, ids=IDS)
def test_preserves_local_structure_on_swiss_roll(cls):
    X, _ = swiss_roll(n_samples=400, random_state=0)
    Y = cls(n_steps=600, random_state=0).fit_transform(X)
    score = knn_preservation(X, Y, n_neighbors=10)
    assert score > 0.25


@pytest.mark.parametrize("cls", PARAMETRIC, ids=IDS)
def test_separates_clusters(cls):
    X, labels = gaussian_blobs(n_samples=300, n_features=10, n_clusters=3, random_state=0)
    Y = cls(n_steps=400, random_state=0).fit_transform(X)
    centers = np.stack([Y[labels == c].mean(axis=0) for c in np.unique(labels)])
    intra = np.mean(
        [np.linalg.norm(Y[labels == c] - centers[c], axis=1).mean() for c in np.unique(labels)]
    )
    inter = np.mean(
        [
            np.linalg.norm(centers[a] - centers[b])
            for a in range(len(centers))
            for b in range(a + 1, len(centers))
        ]
    )
    assert inter > 2.0 * intra


@pytest.mark.parametrize("cls", PARAMETRIC, ids=IDS)
def test_transform_is_consistent_with_fit_embedding(cls):
    X, _ = gaussian_blobs(n_samples=300, n_features=10, random_state=0)
    model = cls(n_steps=300, random_state=0)
    Y = model.fit_transform(X)
    # A parametric model's transform of the training data IS the embedding.
    assert np.allclose(model.transform(X), Y)


@pytest.mark.parametrize("cls", PARAMETRIC, ids=IDS)
def test_save_load_is_weights_only_and_exact(cls, tmp_path):
    X, _ = gaussian_blobs(n_samples=300, n_features=10, random_state=0)
    model = cls(n_steps=300, random_state=0)
    model.fit(X)
    path = tmp_path / "model.npz"
    model.save(path)

    restored = load(path)
    assert type(restored) is cls
    assert restored.get_config() == model.get_config()
    # float32 weight storage: forward passes agree to ~1e-4 relative
    assert np.allclose(restored.transform(X), model.transform(X), atol=1e-2)


@pytest.mark.parametrize("cls", PARAMETRIC, ids=IDS)
def test_saved_size_independent_of_n(cls, tmp_path):
    sizes = {}
    for n in (200, 800):
        X, _ = gaussian_blobs(n_samples=n, n_features=10, random_state=0)
        model = cls(n_steps=100, random_state=0)
        model.fit(X)
        path = tmp_path / f"model_{n}.npz"
        model.save(path)
        sizes[n] = os.path.getsize(path)
    # Weights-only files differ by compression noise, not by 4x data size.
    assert abs(sizes[200] - sizes[800]) < 0.1 * sizes[200]


@pytest.mark.parametrize("cls", PARAMETRIC, ids=IDS)
def test_unfitted_raises(cls):
    with pytest.raises(RuntimeError):
        cls().transform(np.zeros((3, 4)))
    with pytest.raises(RuntimeError):
        cls().save("/tmp/never-written.npz")
