import numpy as np
import pytest

from simple_maps import KNNClassifier, PaCMAP, ParametricUMAP, TriMap, UMAP, load
from simple_maps.datasets import gaussian_blobs

EMBEDDERS = [
    lambda: UMAP(n_epochs=100, random_state=0),
    lambda: TriMap(n_iters=150, random_state=0),
    lambda: PaCMAP(n_iters=200, random_state=0),
    lambda: ParametricUMAP(n_steps=400, random_state=0),
]
IDS = ["UMAP", "TriMap", "PaCMAP", "ParametricUMAP"]


@pytest.fixture(scope="module")
def split_blobs():
    X, y = gaussian_blobs(n_samples=500, n_features=15, n_clusters=4, random_state=0)
    # labels come out grouped by cluster; shuffle so the split is stratified
    perm = np.random.default_rng(0).permutation(len(X))
    X, y = X[perm], y[perm]
    return X[:400], y[:400], X[400:], y[400:]


@pytest.mark.parametrize("make", EMBEDDERS, ids=IDS)
def test_classifier_learns_and_roundtrips(make, split_blobs, tmp_path):
    X_train, y_train, X_test, y_test = split_blobs
    clf = KNNClassifier(make())
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    assert acc > 0.9

    path = tmp_path / "clf.npz"
    clf.save(path)
    restored = load(path)
    assert type(restored) is KNNClassifier
    assert type(restored.embedder) is type(clf.embedder)
    # the loaded classifier must make identical predictions
    assert np.array_equal(restored.predict(X_test), clf.predict(X_test))
    assert restored.score(X_test, y_test) == pytest.approx(acc)


def test_predict_proba_is_a_distribution(split_blobs):
    X_train, y_train, X_test, _ = split_blobs
    clf = KNNClassifier(PaCMAP(n_iters=200, random_state=0)).fit(X_train, y_train)
    proba = clf.predict_proba(X_test)
    assert proba.shape == (len(X_test), len(np.unique(y_train)))
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0.0).all()


def test_string_labels(tmp_path):
    X, y_int = gaussian_blobs(n_samples=200, n_features=8, n_clusters=2, random_state=1)
    y = np.array(["spam", "ham"])[y_int]
    clf = KNNClassifier(PaCMAP(n_iters=150, random_state=0)).fit(X, y)
    pred = clf.predict(X[:10])
    assert set(pred) <= {"spam", "ham"}

    path = tmp_path / "clf.npz"
    clf.save(path)
    restored = load(path)
    assert np.array_equal(restored.predict(X[:10]), pred)


def test_classifier_input_validation():
    X = np.zeros((10, 3))
    clf = KNNClassifier(PaCMAP(random_state=0))
    with pytest.raises(ValueError):
        clf.fit(X, np.zeros(7))
    with pytest.raises(RuntimeError):
        clf.predict(X)
    with pytest.raises(RuntimeError):
        clf.save("/nonexistent-should-not-matter.npz")
