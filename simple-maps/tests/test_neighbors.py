import numpy as np
import pytest

from simple_maps.distances import pairwise_distances
from simple_maps.neighbors import knn, knn_query


@pytest.mark.parametrize("metric", ["euclidean", "cosine", "manhattan"])
@pytest.mark.parametrize("block_size", [7, 2048])
def test_knn_exact(metric, block_size):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 8))
    k = 5
    idx, dist = knn(X, k, metric=metric, block_size=block_size)

    D = pairwise_distances(X, metric=metric)
    np.fill_diagonal(D, np.inf)
    expected = np.sort(D, axis=1)[:, :k]
    assert np.allclose(np.sort(dist, axis=1), expected, atol=1e-9)
    assert (idx != np.arange(60)[:, None]).all()
    # distances aligned with indices and sorted ascending
    assert np.allclose(np.take_along_axis(D, idx, axis=1), dist, atol=1e-9)
    assert (np.diff(dist, axis=1) >= -1e-12).all()


def test_knn_query_matches_knn():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 6))
    idx, dist = knn(X, 4)
    # Querying the training set itself returns self as neighbor 0.
    qidx, qdist = knn_query(X, X, 5)
    assert (qidx[:, 0] == np.arange(40)).all()
    assert np.allclose(qdist[:, 0], 0.0, atol=1e-6)
    assert np.array_equal(qidx[:, 1:], idx)
    assert np.allclose(qdist[:, 1:], dist, atol=1e-9)


def test_knn_validates_k():
    X = np.zeros((5, 2))
    with pytest.raises(ValueError):
        knn(X, 5)
    with pytest.raises(ValueError):
        knn(X, 0)
