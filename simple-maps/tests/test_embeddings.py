import numpy as np
import pytest

from simple_maps import PaCMAP, TriMap, UMAP, load
from simple_maps.datasets import gaussian_blobs, swiss_roll
from simple_maps.metrics import knn_preservation
from simple_maps.umap_ import find_ab_params

# Reduced iteration counts keep the suite fast; quality thresholds are set
# accordingly (loose, but far above the random baseline).
ESTIMATORS = [
    lambda: UMAP(n_epochs=100, random_state=0),
    lambda: TriMap(n_iters=200, random_state=0),
    lambda: PaCMAP(n_iters=250, random_state=0),
]
IDS = ["UMAP", "TriMap", "PaCMAP"]


@pytest.fixture(scope="module")
def blobs():
    return gaussian_blobs(n_samples=300, n_features=10, n_clusters=3, random_state=0)


@pytest.mark.parametrize("make", ESTIMATORS, ids=IDS)
def test_embedding_shape_and_finite(make, blobs):
    X, _ = blobs
    Y = make().fit_transform(X)
    assert Y.shape == (X.shape[0], 2)
    assert np.isfinite(Y).all()


@pytest.mark.parametrize("make", ESTIMATORS, ids=IDS)
def test_separates_clusters(make, blobs):
    X, labels = blobs
    Y = make().fit_transform(X)
    # Mean intra-cluster distance must be well below mean inter-cluster
    # distance in the embedding.
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


@pytest.mark.parametrize("make", ESTIMATORS, ids=IDS)
def test_preserves_local_structure_on_swiss_roll(make):
    X, _ = swiss_roll(n_samples=400, random_state=0)
    Y = make().fit_transform(X)
    score = knn_preservation(X, Y, n_neighbors=10)
    random_baseline = 10 / (X.shape[0] - 1)
    assert score > 0.25
    assert score > 5.0 * random_baseline


@pytest.mark.parametrize("make", ESTIMATORS, ids=IDS)
def test_deterministic_given_seed(make, blobs):
    X, _ = blobs
    Y1 = make().fit_transform(X)
    Y2 = make().fit_transform(X)
    assert np.allclose(Y1, Y2)


@pytest.mark.parametrize("make", ESTIMATORS, ids=IDS)
def test_save_load_transform_roundtrip(make, blobs, tmp_path):
    X, _ = blobs
    model = make()
    model.fit(X)
    path = tmp_path / "model.npz"
    model.save(path)

    restored = load(path)
    assert type(restored) is type(model)
    assert restored.get_config() == model.get_config()
    # float32 storage keeps ~1e-6 relative fidelity
    assert np.allclose(restored.embedding_, model.embedding_, atol=1e-4)

    # transform: training points map near their own embedding
    Z = restored.transform(X[:20])
    assert Z.shape == (20, 2)
    assert np.isfinite(Z).all()
    d = np.linalg.norm(Z - restored.embedding_[:20], axis=1)
    spread = restored.embedding_.std()
    assert d.mean() < spread


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        UMAP().transform(np.zeros((3, 4)))


def test_find_ab_params_matches_reference():
    # umap-learn's scipy curve_fit yields a~1.577, b~0.895 for the defaults.
    a, b = find_ab_params(spread=1.0, min_dist=0.1)
    assert a == pytest.approx(1.577, abs=0.05)
    assert b == pytest.approx(0.895, abs=0.05)


def test_small_n_does_not_crash():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 4))
    for make in ESTIMATORS:
        model = make()
        Y = model.fit_transform(X)
        assert Y.shape == (20, 2)
        assert np.isfinite(Y).all()
